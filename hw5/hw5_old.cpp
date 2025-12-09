#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <thread>
#include "hip/hip_runtime.h"

#define HIP_CHECK(err) \
    do { \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP Error: %s at %s:%d\n", \
                    hipGetErrorString(err), __FILE__, __LINE__); \
            exit(err); \
        } \
    } while (0)

// 優化參數
#define SMALL_N_THRESHOLD 1024 
#define BATCH_STEPS 10000  // 極大化 batch size，最小化 kernel 啟動
#define BLOCK_SIZE 256
#define SMEM_PAD 8 
#define NUM_GPUS 2

namespace param {
const int n_steps = 200000;
const double dt = 60;
const double eps = 1e-3;
const double eps_sq = eps * eps;
const double G = 6.674e-11;
const double planet_radius = 1e7;
const double planet_radius_sq = planet_radius * planet_radius;
const double missile_speed = 1e6;
const int TYPE_NORMAL = 0;
const int TYPE_DEVICE = 1;

double get_missile_cost(double t) { return 1e5 + 1e3 * t; }
}

struct SimStatus {
    int hit_time_step;
    int destroy_step;
    double min_dist;
    int device_destroyed;
};

// GPU 資源結構
struct GPUResources {
    int gpu_id;
    int scenario_start;      // 此 GPU 負責的 scenario 起始 index (全域)
    int num_scenarios;       // 此 GPU 負責的 scenario 數量
    
    // Device pointers
    double *d_qx_bk, *d_qy_bk, *d_qz_bk, *d_vx_bk, *d_vy_bk, *d_vz_bk, *d_m_bk;
    int *d_type;
    int *d_dev_map;
    SimStatus *d_results;
    double *d_state_1, *d_state_2;
    
    hipStream_t stream;
};

__device__ __forceinline__ double get_device_mass(double m0, double t) {
    double sin_val = sin(t * 0.00016666666666666666);
    return m0 * (1.0 + 0.5 * fabs(sin_val));
}

// =========================================================
// Kernel: Initialize Scenarios
// =========================================================
__global__ void initialize_scenarios_kernel(
    int n, int num_scenarios,
    const double* __restrict__ init_qx, 
    const double* __restrict__ init_qy, 
    const double* __restrict__ init_qz,
    const double* __restrict__ init_vx, 
    const double* __restrict__ init_vy, 
    const double* __restrict__ init_vz,
    const double* __restrict__ init_m,
    const int* __restrict__ init_type,
    double* __restrict__ state_buffer,
    int global_scenario_offset
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y;  // local scenario index
    int global_s = s + global_scenario_offset;

    if (i < n && s < num_scenarios) {
        size_t offset = (size_t)s * 7 * n;
        state_buffer[offset + 0*n + i] = init_qx[i];
        state_buffer[offset + 1*n + i] = init_qy[i];
        state_buffer[offset + 2*n + i] = init_qz[i];
        state_buffer[offset + 3*n + i] = init_vx[i];
        state_buffer[offset + 4*n + i] = init_vy[i];
        state_buffer[offset + 5*n + i] = init_vz[i];
        
        // 全域 scenario 0 的 device 質量設為 0
        if (global_s == 0 && init_type[i] == param::TYPE_DEVICE) {
            state_buffer[offset + 6*n + i] = 0.0;
        } else {
            state_buffer[offset + 6*n + i] = init_m[i];
        }
    }
}

// =========================================================
// Kernel 1: Small N (Shared Memory Batching)
// =========================================================
__global__ void solve_batch_small_n(
    int n, int start_step, int steps_to_run,
    int planet, int asteroid, 
    const int* __restrict__ target_device_ids, 
    const int* __restrict__ type,              
    SimStatus* global_results,
    double* __restrict__ state_backup,
    int global_scenario_offset
) {
    extern __shared__ double smem[];
    int stride = n + SMEM_PAD;
    double* qx = &smem[0];
    double* qy = &smem[stride];
    double* qz = &smem[2*stride];
    double* m  = &smem[3*stride];
    int* t  = (int*)&smem[4*stride];

    int tid = threadIdx.x;
    int scenario_idx = blockIdx.x;  // local scenario index
    int global_scenario_idx = scenario_idx + global_scenario_offset;
    int target_dev_id = target_device_ids[scenario_idx];
    size_t state_offset = (size_t)scenario_idx * 7 * n;

    // 1. Load State from Global
    if (tid < n) {
        qx[tid] = __ldg(&state_backup[state_offset + 0*n + tid]);
        qy[tid] = __ldg(&state_backup[state_offset + 1*n + tid]);
        qz[tid] = __ldg(&state_backup[state_offset + 2*n + tid]);
        m[tid]  = __ldg(&state_backup[state_offset + 6*n + tid]);
        t[tid]  = __ldg(&type[tid]); 
    }
    
    int local_destroyed = global_results[scenario_idx].device_destroyed;
    if (target_dev_id >= 0 && local_destroyed && tid == target_dev_id) m[tid] = 0.0;
    
    __syncthreads();

    double my_qx, my_qy, my_qz, my_vx, my_vy, my_vz, my_m;
    int my_type;

    if (tid < n) {
        my_qx = qx[tid]; 
        my_qy = qy[tid]; 
        my_qz = qz[tid];
        my_vx = __ldg(&state_backup[state_offset + 3*n + tid]);
        my_vy = __ldg(&state_backup[state_offset + 4*n + tid]);
        my_vz = __ldg(&state_backup[state_offset + 5*n + tid]);
        my_m  = m[tid];
        my_type = t[tid];
    }

    double local_min_dist;
    int local_hit_step;
    int local_destroy_step;
    
    if (tid == 0) {
        local_min_dist = global_results[scenario_idx].min_dist;
        local_hit_step = global_results[scenario_idx].hit_time_step;
        local_destroy_step = global_results[scenario_idx].destroy_step;
    }

    // --- Batch Loop ---
    for (int s = 0; s < steps_to_run; ++s) {
        int current_step = start_step + s;
        double current_time = current_step * param::dt;
        
        double ax = 0.0, ay = 0.0, az = 0.0;

        if (tid < n) {
            const double G = param::G;
            const double eps_sq = param::eps_sq;
            
            for (int j = 0; j < n; j++) {
                double mj = m[j]; 
                if (mj > 0.0 && t[j] == param::TYPE_DEVICE) 
                    mj = get_device_mass(mj, current_time);
                
                double dx = qx[j] - my_qx;
                double dy = qy[j] - my_qy;
                double dz = qz[j] - my_qz;
                
                double dist_sq = dx*dx + dy*dy + dz*dz + eps_sq;
                double inv_dist = rsqrt(dist_sq);
                double inv_dist3 = inv_dist * inv_dist * inv_dist;
                
                double mj_factor = (j != tid) ? 1.0 : 0.0;
                double f = G * mj * mj_factor * inv_dist3;
                
                ax += f * dx; 
                ay += f * dy; 
                az += f * dz;
            }
        }
        __syncthreads();

        if (tid < n) {
            my_vx += ax * param::dt;
            my_vy += ay * param::dt;
            my_vz += az * param::dt;
            my_qx += my_vx * param::dt;
            my_qy += my_vy * param::dt;
            my_qz += my_vz * param::dt;
            
            qx[tid] = my_qx;
            qy[tid] = my_qy;
            qz[tid] = my_qz;
        }
        __syncthreads();

        // 4. Check Events
        double dist_to_asteroid = 1e18;
        if (global_scenario_idx == 0 && tid == 0) { 
            double dx = qx[planet] - qx[asteroid];
            double dy = qy[planet] - qy[asteroid];
            double dz = qz[planet] - qz[asteroid];
            dist_to_asteroid = sqrt(dx*dx + dy*dy + dz*dz);
            if (dist_to_asteroid < local_min_dist) local_min_dist = dist_to_asteroid;
        }

        if (tid == 0) {
            double dx = qx[planet] - qx[asteroid];
            double dy = qy[planet] - qy[asteroid];
            double dz = qz[planet] - qz[asteroid];
            double dist_sq = dx*dx + dy*dy + dz*dz;

            if (local_hit_step == -2 && dist_sq < param::planet_radius_sq) 
                local_hit_step = current_step;

            if (target_dev_id >= 0 && local_destroyed == 0) {
                double m_dist = current_step * param::dt * param::missile_speed;
                double d_dx = qx[target_dev_id] - qx[planet];
                double d_dy = qy[target_dev_id] - qy[planet];
                double d_dz = qz[target_dev_id] - qz[planet];
                double d_dist = sqrt(d_dx*d_dx + d_dy*d_dy + d_dz*d_dz);
                
                if (m_dist > d_dist) {
                    local_destroyed = 1;
                    local_destroy_step = current_step;
                    m[target_dev_id] = 0.0;
                }
            }
        }
        __syncthreads(); 
        
        if (tid < n && target_dev_id == tid && local_destroyed) {
            my_m = 0.0;
        }
    }
    
    // Write Back to Global
    if (tid < n) {
        state_backup[state_offset + 0*n + tid] = my_qx;
        state_backup[state_offset + 1*n + tid] = my_qy;
        state_backup[state_offset + 2*n + tid] = my_qz;
        state_backup[state_offset + 3*n + tid] = my_vx;
        state_backup[state_offset + 4*n + tid] = my_vy;
        state_backup[state_offset + 5*n + tid] = my_vz;
        state_backup[state_offset + 6*n + tid] = my_m;
    }
    
    if (tid == 0) {
        global_results[scenario_idx].min_dist = local_min_dist;
        global_results[scenario_idx].hit_time_step = local_hit_step;
        global_results[scenario_idx].device_destroyed = local_destroyed;
        global_results[scenario_idx].destroy_step = local_destroy_step;
    }
}

// =========================================================
// Kernel 2: Large N (Tiled + Parallel Scenarios)
// =========================================================
__global__ void update_physics_large_n_parallel(
    int n, double current_time,
    const int* __restrict__ type,
    const double* __restrict__ state_in,
    double* __restrict__ state_out
) {
    __shared__ double sh_qx[BLOCK_SIZE];
    __shared__ double sh_qy[BLOCK_SIZE];
    __shared__ double sh_qz[BLOCK_SIZE];
    __shared__ double sh_m[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int scenario_idx = blockIdx.y;
    size_t state_offset = (size_t)scenario_idx * 7 * n;

    double my_qx = 0, my_qy = 0, my_qz = 0;
    if (i < n) {
        my_qx = __ldg(&state_in[state_offset + 0*n + i]);
        my_qy = __ldg(&state_in[state_offset + 1*n + i]);
        my_qz = __ldg(&state_in[state_offset + 2*n + i]);
    }

    double acc_x = 0.0, acc_y = 0.0, acc_z = 0.0;

    for (int tile = 0; tile < gridDim.x; tile++) {
        int tile_idx = tile * blockDim.x + tid;
        if (tile_idx < n) {
            sh_qx[tid] = __ldg(&state_in[state_offset + 0*n + tile_idx]);
            sh_qy[tid] = __ldg(&state_in[state_offset + 1*n + tile_idx]);
            sh_qz[tid] = __ldg(&state_in[state_offset + 2*n + tile_idx]);
            double raw_m = __ldg(&state_in[state_offset + 6*n + tile_idx]);
            if (raw_m > 0.0 && type[tile_idx] == param::TYPE_DEVICE) 
                sh_m[tid] = get_device_mass(raw_m, current_time);
            else 
                sh_m[tid] = raw_m;
        } else { sh_m[tid] = 0.0; }
        __syncthreads();

        if (i < n) {
            #pragma unroll 8
            for (int j = 0; j < BLOCK_SIZE; j++) {
                int target = tile * blockDim.x + j;
                if (target >= n) break;
                if (i == target) continue;
                double dx = sh_qx[j] - my_qx; 
                double dy = sh_qy[j] - my_qy; 
                double dz = sh_qz[j] - my_qz;
                double dist_sq = dx*dx + dy*dy + dz*dz + param::eps_sq;
                double inv_dist = rsqrt(dist_sq);
                double inv_dist3 = inv_dist * inv_dist * inv_dist;
                double f = param::G * sh_m[j] * inv_dist3;
                acc_x += f * dx; 
                acc_y += f * dy; 
                acc_z += f * dz;
            }
        }
        __syncthreads();
    }

    if (i < n) {
        double vx = __ldg(&state_in[state_offset + 3*n + i]) + acc_x * param::dt;
        double vy = __ldg(&state_in[state_offset + 4*n + i]) + acc_y * param::dt;
        double vz = __ldg(&state_in[state_offset + 5*n + i]) + acc_z * param::dt;
        double qx = my_qx + vx * param::dt;
        double qy = my_qy + vy * param::dt;
        double qz = my_qz + vz * param::dt;

        state_out[state_offset + 0*n + i] = qx;
        state_out[state_offset + 1*n + i] = qy;
        state_out[state_offset + 2*n + i] = qz;
        state_out[state_offset + 3*n + i] = vx;
        state_out[state_offset + 4*n + i] = vy;
        state_out[state_offset + 5*n + i] = vz;
        state_out[state_offset + 6*n + i] = __ldg(&state_in[state_offset + 6*n + i]); 
    }
}

__global__ void check_events_large_n_parallel(
    int step, int n, int planet, int asteroid, 
    const int* __restrict__ target_device_ids,
    double* __restrict__ state_current, 
    SimStatus* global_results,
    int global_scenario_offset)
{
    int scenario_idx = blockIdx.x;  // local scenario index
    int global_scenario_idx = scenario_idx + global_scenario_offset;
    int target_dev_id = target_device_ids[scenario_idx];
    size_t state_offset = (size_t)scenario_idx * 7 * n;

    if (threadIdx.x == 0) {
        double px = state_current[state_offset + 0*n + planet];
        double py = state_current[state_offset + 1*n + planet];
        double pz = state_current[state_offset + 2*n + planet];
        double ax = state_current[state_offset + 0*n + asteroid];
        double ay = state_current[state_offset + 1*n + asteroid];
        double az = state_current[state_offset + 2*n + asteroid];

        double dx = px - ax; double dy = py - ay; double dz = pz - az;
        double dist_sq = dx*dx + dy*dy + dz*dz;

        if (global_scenario_idx == 0) {
            double dist = sqrt(dist_sq);
            if (dist < global_results[scenario_idx].min_dist) global_results[scenario_idx].min_dist = dist;
        }
        if (global_results[scenario_idx].hit_time_step == -2 && dist_sq < param::planet_radius_sq) {
            global_results[scenario_idx].hit_time_step = step;
        }
        if (target_dev_id >= 0 && global_results[scenario_idx].device_destroyed == 0) {
            double m_dist = step * param::dt * param::missile_speed;
            double dev_x = state_current[state_offset + 0*n + target_dev_id];
            double dev_y = state_current[state_offset + 1*n + target_dev_id];
            double dev_z = state_current[state_offset + 2*n + target_dev_id];
            
            double d_dx = dev_x - px; double d_dy = dev_y - py; double d_dz = dev_z - pz;
            double d_dist = sqrt(d_dx*d_dx + d_dy*d_dy + d_dz*d_dz);
            
            if (m_dist > d_dist) {
                global_results[scenario_idx].device_destroyed = 1;
                global_results[scenario_idx].destroy_step = step;
                state_current[state_offset + 6*n + target_dev_id] = 0.0;
            }
        }
    }
}

void read_input(const char* filename, int& n, int& planet, int& asteroid, 
                std::vector<double>& qx, std::vector<double>& qy, std::vector<double>& qz, 
                std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz, 
                std::vector<double>& m, std::vector<int>& type) {
    std::ifstream fin(filename); fin >> n >> planet >> asteroid;
    qx.resize(n); qy.resize(n); qz.resize(n); vx.resize(n); vy.resize(n); vz.resize(n); m.resize(n); type.resize(n);
    std::string t;
    for (int i = 0; i < n; i++) {
        fin >> qx[i] >> qy[i] >> qz[i] >> vx[i] >> vy[i] >> vz[i] >> m[i] >> t;
        type[i] = (t == "device") ? param::TYPE_DEVICE : param::TYPE_NORMAL;
    }
}

void write_output(const char* filename, double min_dist, int hit_time_step, int dev_id, double cost) {
    std::ofstream fout(filename);
    fout << std::scientific << std::setprecision(16) << min_dist << "\n" << hit_time_step << "\n" << dev_id << " " << cost << "\n";
}

// 初始化 GPU 資源
void init_gpu_resources(GPUResources& res, int n, int total_scenarios,
                        const std::vector<double>& h_qx, const std::vector<double>& h_qy, const std::vector<double>& h_qz,
                        const std::vector<double>& h_vx, const std::vector<double>& h_vy, const std::vector<double>& h_vz,
                        const std::vector<double>& h_m, const std::vector<int>& h_type,
                        const std::vector<int>& host_dev_map) {
    HIP_CHECK(hipSetDevice(res.gpu_id));
    HIP_CHECK(hipStreamCreate(&res.stream));
    
    // 分配初始資料緩衝區
    HIP_CHECK(hipMalloc(&res.d_qx_bk, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&res.d_qy_bk, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&res.d_qz_bk, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&res.d_vx_bk, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&res.d_vy_bk, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&res.d_vz_bk, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&res.d_m_bk, n * sizeof(double)));
    HIP_CHECK(hipMalloc(&res.d_type, n * sizeof(int)));
    
    // 複製初始資料
    HIP_CHECK(hipMemcpyAsync(res.d_qx_bk, h_qx.data(), n * sizeof(double), hipMemcpyHostToDevice, res.stream));
    HIP_CHECK(hipMemcpyAsync(res.d_qy_bk, h_qy.data(), n * sizeof(double), hipMemcpyHostToDevice, res.stream));
    HIP_CHECK(hipMemcpyAsync(res.d_qz_bk, h_qz.data(), n * sizeof(double), hipMemcpyHostToDevice, res.stream));
    HIP_CHECK(hipMemcpyAsync(res.d_vx_bk, h_vx.data(), n * sizeof(double), hipMemcpyHostToDevice, res.stream));
    HIP_CHECK(hipMemcpyAsync(res.d_vy_bk, h_vy.data(), n * sizeof(double), hipMemcpyHostToDevice, res.stream));
    HIP_CHECK(hipMemcpyAsync(res.d_vz_bk, h_vz.data(), n * sizeof(double), hipMemcpyHostToDevice, res.stream));
    HIP_CHECK(hipMemcpyAsync(res.d_m_bk, h_m.data(), n * sizeof(double), hipMemcpyHostToDevice, res.stream));
    HIP_CHECK(hipMemcpyAsync(res.d_type, h_type.data(), n * sizeof(int), hipMemcpyHostToDevice, res.stream));
    
    // 分配此 GPU 負責的 scenarios 的 device map (local)
    std::vector<int> local_dev_map(res.num_scenarios);
    for (int i = 0; i < res.num_scenarios; i++) {
        local_dev_map[i] = host_dev_map[res.scenario_start + i];
    }
    HIP_CHECK(hipMalloc(&res.d_dev_map, res.num_scenarios * sizeof(int)));
    HIP_CHECK(hipMemcpyAsync(res.d_dev_map, local_dev_map.data(), res.num_scenarios * sizeof(int), hipMemcpyHostToDevice, res.stream));
    
    // 分配 results
    HIP_CHECK(hipMalloc(&res.d_results, res.num_scenarios * sizeof(SimStatus)));
    std::vector<SimStatus> init_res(res.num_scenarios, {-2, -1, std::numeric_limits<double>::infinity(), 0});
    HIP_CHECK(hipMemcpyAsync(res.d_results, init_res.data(), res.num_scenarios * sizeof(SimStatus), hipMemcpyHostToDevice, res.stream));
    
    // 分配 state buffer
    size_t state_size = (size_t)res.num_scenarios * 7 * n * sizeof(double);
    HIP_CHECK(hipMalloc(&res.d_state_1, state_size));
    res.d_state_2 = nullptr;  // Large N 時才需要
}

// 釋放 GPU 資源
void free_gpu_resources(GPUResources& res) {
    HIP_CHECK(hipSetDevice(res.gpu_id));
    HIP_CHECK(hipStreamDestroy(res.stream));
    HIP_CHECK(hipFree(res.d_qx_bk));
    HIP_CHECK(hipFree(res.d_qy_bk));
    HIP_CHECK(hipFree(res.d_qz_bk));
    HIP_CHECK(hipFree(res.d_vx_bk));
    HIP_CHECK(hipFree(res.d_vy_bk));
    HIP_CHECK(hipFree(res.d_vz_bk));
    HIP_CHECK(hipFree(res.d_m_bk));
    HIP_CHECK(hipFree(res.d_type));
    HIP_CHECK(hipFree(res.d_dev_map));
    HIP_CHECK(hipFree(res.d_results));
    HIP_CHECK(hipFree(res.d_state_1));
    if (res.d_state_2) HIP_CHECK(hipFree(res.d_state_2));
}

// GPU 執行緒函數 - Small N
void gpu_worker_small_n(GPUResources& res, int n, int planet, int asteroid) {
    HIP_CHECK(hipSetDevice(res.gpu_id));
    
    // 初始化 scenarios
    int block_size_init = 256;
    int blocks_init = (n + block_size_init - 1) / block_size_init;
    dim3 init_grid(blocks_init, res.num_scenarios);
    
    hipLaunchKernelGGL(initialize_scenarios_kernel, init_grid, block_size_init, 0, res.stream,
        n, res.num_scenarios, res.d_qx_bk, res.d_qy_bk, res.d_qz_bk,
        res.d_vx_bk, res.d_vy_bk, res.d_vz_bk, res.d_m_bk, res.d_type,
        res.d_state_1, res.scenario_start);
    
    // 執行模擬
    int launch_block_size = n;
    size_t smem_size = (4 * n * sizeof(double)) + (n * sizeof(int)) + (5 * sizeof(double) * SMEM_PAD);
    
    for (int start = 1; start <= param::n_steps; start += BATCH_STEPS) {
        int steps = std::min(BATCH_STEPS, param::n_steps - start + 1);
        hipLaunchKernelGGL(solve_batch_small_n, res.num_scenarios, launch_block_size, smem_size, res.stream,
            n, start, steps, planet, asteroid, res.d_dev_map, res.d_type, res.d_results, res.d_state_1, res.scenario_start);
    }
    
    HIP_CHECK(hipStreamSynchronize(res.stream));
}

// GPU 執行緒函數 - Large N
void gpu_worker_large_n(GPUResources& res, int n, int planet, int asteroid) {
    HIP_CHECK(hipSetDevice(res.gpu_id));
    
    // 分配第二個 state buffer
    size_t state_size = (size_t)res.num_scenarios * 7 * n * sizeof(double);
    HIP_CHECK(hipMalloc(&res.d_state_2, state_size));
    
    // 初始化 scenarios
    int block_size_init = 256;
    int blocks_init = (n + block_size_init - 1) / block_size_init;
    dim3 init_grid(blocks_init, res.num_scenarios);
    
    hipLaunchKernelGGL(initialize_scenarios_kernel, init_grid, block_size_init, 0, res.stream,
        n, res.num_scenarios, res.d_qx_bk, res.d_qy_bk, res.d_qz_bk,
        res.d_vx_bk, res.d_vy_bk, res.d_vz_bk, res.d_m_bk, res.d_type,
        res.d_state_1, res.scenario_start);
    
    // 執行模擬
    int blocks_per_sim = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(blocks_per_sim, res.num_scenarios);
    
    double *ptr_in = res.d_state_1, *ptr_out = res.d_state_2;
    
    for (int step = 1; step <= param::n_steps; step++) {
        hipLaunchKernelGGL(update_physics_large_n_parallel, grid, BLOCK_SIZE, 0, res.stream,
            n, step * param::dt, res.d_type, ptr_in, ptr_out);
        hipLaunchKernelGGL(check_events_large_n_parallel, res.num_scenarios, 1, 0, res.stream,
            step, n, planet, asteroid, res.d_dev_map, ptr_out, res.d_results, res.scenario_start);
        std::swap(ptr_in, ptr_out);
    }
    
    HIP_CHECK(hipStreamSynchronize(res.stream));
}

int main(int argc, char** argv) {
    if (argc != 3) exit(1);
    
    int n, planet, asteroid;
    std::vector<double> h_qx, h_qy, h_qz, h_vx, h_vy, h_vz, h_m;
    std::vector<int> h_type;
    read_input(argv[1], n, planet, asteroid, h_qx, h_qy, h_qz, h_vx, h_vy, h_vz, h_m, h_type);

    // 識別 devices
    std::vector<int> devices;
    for (int i = 0; i < n; i++) 
        if (h_type[i] == param::TYPE_DEVICE) devices.push_back(i);
    int num_scenarios = 2 + devices.size();

    std::vector<int> host_dev_map(num_scenarios, -1);
    for (size_t i = 0; i < devices.size(); ++i) 
        host_dev_map[2 + i] = devices[i];

    // 檢查可用 GPU 數量
    int num_gpus;
    HIP_CHECK(hipGetDeviceCount(&num_gpus));
    int use_gpus = std::min(num_gpus, NUM_GPUS);
    
    // 如果只有 1 個 GPU 或 scenarios 太少，使用單 GPU
    if (use_gpus < 2 || num_scenarios < 2) {
        use_gpus = 1;
    }

    // 分配 scenarios 到各 GPU
    std::vector<GPUResources> gpu_res(use_gpus);
    int scenarios_per_gpu = num_scenarios / use_gpus;
    int remaining = num_scenarios % use_gpus;
    
    int offset = 0;
    for (int g = 0; g < use_gpus; g++) {
        gpu_res[g].gpu_id = g;
        gpu_res[g].scenario_start = offset;
        gpu_res[g].num_scenarios = scenarios_per_gpu + (g < remaining ? 1 : 0);
        offset += gpu_res[g].num_scenarios;
    }

    // 初始化所有 GPU 資源
    for (int g = 0; g < use_gpus; g++) {
        init_gpu_resources(gpu_res[g], n, num_scenarios, h_qx, h_qy, h_qz, h_vx, h_vy, h_vz, h_m, h_type, host_dev_map);
    }
    
    // 平行執行各 GPU
    std::vector<std::thread> threads;
    
    if (n <= SMALL_N_THRESHOLD) {
        for (int g = 0; g < use_gpus; g++) {
            threads.emplace_back(gpu_worker_small_n, std::ref(gpu_res[g]), n, planet, asteroid);
        }
    } else {
        for (int g = 0; g < use_gpus; g++) {
            threads.emplace_back(gpu_worker_large_n, std::ref(gpu_res[g]), n, planet, asteroid);
        }
    }
    
    // 等待所有 GPU 完成
    for (auto& t : threads) {
        t.join();
    }

    // 收集所有結果
    std::vector<SimStatus> h_results(num_scenarios);
    for (int g = 0; g < use_gpus; g++) {
        HIP_CHECK(hipSetDevice(gpu_res[g].gpu_id));
        HIP_CHECK(hipMemcpy(&h_results[gpu_res[g].scenario_start], gpu_res[g].d_results, 
                           gpu_res[g].num_scenarios * sizeof(SimStatus), hipMemcpyDeviceToHost));
    }

    // 處理結果
    double ans_min_dist = h_results[0].min_dist;
    int ans_hit_step = h_results[1].hit_time_step;
    int ans_dev_id = -1;
    double ans_cost = 0;

    if (ans_hit_step >= 0) {
        double best_cost = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < devices.size(); i++) {
            SimStatus& s = h_results[2 + i];
            if (s.hit_time_step == -2 && s.device_destroyed) {
                double c = param::get_missile_cost(s.destroy_step * param::dt);
                if (c < best_cost) { 
                    best_cost = c; 
                    ans_dev_id = devices[i]; 
                }
            }
        }
        if (ans_dev_id != -1) ans_cost = best_cost;
    }

    write_output(argv[2], ans_min_dist, ans_hit_step, ans_dev_id, ans_cost);

    // 釋放資源
    for (int g = 0; g < use_gpus; g++) {
        free_gpu_resources(gpu_res[g]);
    }

    return 0;
}