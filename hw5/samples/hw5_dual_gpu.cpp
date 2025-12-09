#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include "hip/hip_runtime.h"

#define HIP_CHECK(err) \
    do { \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP Error: %s at %s:%d\n", \
                    hipGetErrorString(err), __FILE__, __LINE__); \
            exit(err); \
        } \
    } while (0)

#define BLOCK_SIZE 256
#define BATCH_STEPS 1000  // 每 1000 步才做一次事件檢查同步

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

__device__ __forceinline__ double get_device_mass(double m0, double t) {
    return m0 * (1.0 + 0.5 * fabs(sin(t / 6000.0)));
}

// 資料平行力計算
__global__ void compute_forces_dp(
    int n, int my_start, int my_count, double current_time,
    const double* __restrict__ qx, const double* __restrict__ qy, const double* __restrict__ qz,
    const double* __restrict__ m, const int* __restrict__ type,
    double* __restrict__ ax, double* __restrict__ ay, double* __restrict__ az
) {
    __shared__ double sh_qx[BLOCK_SIZE], sh_qy[BLOCK_SIZE], sh_qz[BLOCK_SIZE], sh_m[BLOCK_SIZE];
    int tid = threadIdx.x;
    int local_i = blockIdx.x * blockDim.x + tid;
    int global_i = my_start + local_i;
    
    double my_qx = 0, my_qy = 0, my_qz = 0;
    if (local_i < my_count) {
        my_qx = qx[global_i]; my_qy = qy[global_i]; my_qz = qz[global_i];
    }

    double acc_x = 0.0, acc_y = 0.0, acc_z = 0.0;
    int num_tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int tile = 0; tile < num_tiles; tile++) {
        int tile_idx = tile * BLOCK_SIZE + tid;
        if (tile_idx < n) {
            sh_qx[tid] = qx[tile_idx]; sh_qy[tid] = qy[tile_idx]; sh_qz[tid] = qz[tile_idx];
            double raw_m = m[tile_idx];
            sh_m[tid] = (raw_m > 0 && type[tile_idx] == param::TYPE_DEVICE) 
                        ? get_device_mass(raw_m, current_time) : raw_m;
        } else sh_m[tid] = 0.0;
        __syncthreads();

        if (local_i < my_count) {
            for (int j = 0; j < BLOCK_SIZE; j++) {
                int target = tile * BLOCK_SIZE + j;
                if (target >= n || global_i == target) continue;
                double dx = sh_qx[j] - my_qx, dy = sh_qy[j] - my_qy, dz = sh_qz[j] - my_qz;
                double dist_sq = dx*dx + dy*dy + dz*dz + param::eps_sq;
                double inv_dist = rsqrt(dist_sq);
                double f = param::G * sh_m[j] * inv_dist * inv_dist * inv_dist;
                acc_x += f * dx; acc_y += f * dy; acc_z += f * dz;
            }
        }
        __syncthreads();
    }
    if (local_i < my_count) { ax[local_i] = acc_x; ay[local_i] = acc_y; az[local_i] = acc_z; }
}

// 更新粒子
__global__ void update_particles(
    int my_start, int my_count,
    const double* __restrict__ ax, const double* __restrict__ ay, const double* __restrict__ az,
    double* __restrict__ qx, double* __restrict__ qy, double* __restrict__ qz,
    double* __restrict__ vx, double* __restrict__ vy, double* __restrict__ vz
) {
    int local_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_i < my_count) {
        int i = my_start + local_i;
        vx[i] += ax[local_i] * param::dt; vy[i] += ay[local_i] * param::dt; vz[i] += az[local_i] * param::dt;
        qx[i] += vx[i] * param::dt; qy[i] += vy[i] * param::dt; qz[i] += vz[i] * param::dt;
    }
}

// GPU 上的事件檢查 kernel
__global__ void check_events_kernel(
    int step, int planet, int asteroid, int target_device,
    const double* __restrict__ qx, const double* __restrict__ qy, const double* __restrict__ qz,
    double* __restrict__ m, double* result  // [min_dist, hit_step, destroy_step, destroyed]
) {
    double px = qx[planet], py = qy[planet], pz = qz[planet];
    double asx = qx[asteroid], asy = qy[asteroid], asz = qz[asteroid];
    double dx = px - asx, dy = py - asy, dz = pz - asz;
    double dist = sqrt(dx*dx + dy*dy + dz*dz);
    
    if (dist < result[0]) result[0] = dist;
    if (result[1] < -1.5 && dist < param::planet_radius) result[1] = step;
    
    if (target_device >= 0 && result[3] < 0.5) {
        double ddx = qx[target_device] - px, ddy = qy[target_device] - py, ddz = qz[target_device] - pz;
        double d_dist = sqrt(ddx*ddx + ddy*ddy + ddz*ddz);
        double m_dist = step * param::dt * param::missile_speed;
        if (m_dist > d_dist) {
            result[3] = 1.0; result[2] = step;
            m[target_device] = 0.0;
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

struct SimResult { int hit_step, destroy_step, destroyed; double min_dist; };

SimResult run_simulation_dual_gpu(
    int n, int planet, int asteroid, int target_device,
    const std::vector<double>& init_qx, const std::vector<double>& init_qy, const std::vector<double>& init_qz,
    const std::vector<double>& init_vx, const std::vector<double>& init_vy, const std::vector<double>& init_vz,
    std::vector<double> init_m, const std::vector<int>& init_type, bool calc_min_dist
) {
    SimResult res = {-2, -1, 0, 1e30};
    
    int num_gpus; HIP_CHECK(hipGetDeviceCount(&num_gpus));
    int use_gpus = (num_gpus >= 2 && n >= 256) ? 2 : 1;
    
    int gpu_start[2], gpu_count[2];
    gpu_start[0] = 0; gpu_count[0] = n / use_gpus;
    gpu_start[1] = gpu_count[0]; gpu_count[1] = n - gpu_count[0];
    if (use_gpus == 1) { gpu_count[0] = n; gpu_count[1] = 0; }
    
    // P2P
    if (use_gpus == 2) {
        int can01, can10;
        hipDeviceCanAccessPeer(&can01, 0, 1);
        hipDeviceCanAccessPeer(&can10, 1, 0);
        if (can01) { HIP_CHECK(hipSetDevice(0)); hipDeviceEnablePeerAccess(1, 0); }
        if (can10) { HIP_CHECK(hipSetDevice(1)); hipDeviceEnablePeerAccess(0, 0); }
    }
    
    struct GPU {
        hipStream_t s; hipEvent_t ev;
        double *qx, *qy, *qz, *vx, *vy, *vz, *m, *ax, *ay, *az, *result;
        int *type;
    } g[2];
    
    for (int i = 0; i < use_gpus; i++) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamCreate(&g[i].s));
        HIP_CHECK(hipEventCreate(&g[i].ev));
        HIP_CHECK(hipMalloc(&g[i].qx, n*8)); HIP_CHECK(hipMalloc(&g[i].qy, n*8)); HIP_CHECK(hipMalloc(&g[i].qz, n*8));
        HIP_CHECK(hipMalloc(&g[i].vx, n*8)); HIP_CHECK(hipMalloc(&g[i].vy, n*8)); HIP_CHECK(hipMalloc(&g[i].vz, n*8));
        HIP_CHECK(hipMalloc(&g[i].m, n*8)); HIP_CHECK(hipMalloc(&g[i].type, n*4));
        HIP_CHECK(hipMalloc(&g[i].ax, gpu_count[i]*8)); HIP_CHECK(hipMalloc(&g[i].ay, gpu_count[i]*8)); HIP_CHECK(hipMalloc(&g[i].az, gpu_count[i]*8));
        HIP_CHECK(hipMalloc(&g[i].result, 4*8));
        
        HIP_CHECK(hipMemcpyAsync(g[i].qx, init_qx.data(), n*8, hipMemcpyHostToDevice, g[i].s));
        HIP_CHECK(hipMemcpyAsync(g[i].qy, init_qy.data(), n*8, hipMemcpyHostToDevice, g[i].s));
        HIP_CHECK(hipMemcpyAsync(g[i].qz, init_qz.data(), n*8, hipMemcpyHostToDevice, g[i].s));
        HIP_CHECK(hipMemcpyAsync(g[i].vx, init_vx.data(), n*8, hipMemcpyHostToDevice, g[i].s));
        HIP_CHECK(hipMemcpyAsync(g[i].vy, init_vy.data(), n*8, hipMemcpyHostToDevice, g[i].s));
        HIP_CHECK(hipMemcpyAsync(g[i].vz, init_vz.data(), n*8, hipMemcpyHostToDevice, g[i].s));
        HIP_CHECK(hipMemcpyAsync(g[i].m, init_m.data(), n*8, hipMemcpyHostToDevice, g[i].s));
        HIP_CHECK(hipMemcpyAsync(g[i].type, init_type.data(), n*4, hipMemcpyHostToDevice, g[i].s));
        
        double init_result[4] = {1e30, -2.0, -1.0, 0.0};
        HIP_CHECK(hipMemcpyAsync(g[i].result, init_result, 4*8, hipMemcpyHostToDevice, g[i].s));
    }
    for (int i = 0; i < use_gpus; i++) { HIP_CHECK(hipSetDevice(i)); HIP_CHECK(hipStreamSynchronize(g[i].s)); }
    
    for (int step = 1; step <= param::n_steps; step++) {
        double t = step * param::dt;
        
        // 1. 力計算 (平行)
        for (int i = 0; i < use_gpus; i++) {
            HIP_CHECK(hipSetDevice(i));
            int blks = (gpu_count[i] + BLOCK_SIZE - 1) / BLOCK_SIZE;
            compute_forces_dp<<<blks, BLOCK_SIZE, 0, g[i].s>>>(
                n, gpu_start[i], gpu_count[i], t, g[i].qx, g[i].qy, g[i].qz, g[i].m, g[i].type, g[i].ax, g[i].ay, g[i].az);
        }
        
        // 2. 更新 (平行)
        for (int i = 0; i < use_gpus; i++) {
            HIP_CHECK(hipSetDevice(i));
            int blks = (gpu_count[i] + BLOCK_SIZE - 1) / BLOCK_SIZE;
            update_particles<<<blks, BLOCK_SIZE, 0, g[i].s>>>(
                gpu_start[i], gpu_count[i], g[i].ax, g[i].ay, g[i].az, g[i].qx, g[i].qy, g[i].qz, g[i].vx, g[i].vy, g[i].vz);
            HIP_CHECK(hipEventRecord(g[i].ev, g[i].s));
        }
        
        // 3. 等待計算完成
        for (int i = 0; i < use_gpus; i++) { HIP_CHECK(hipSetDevice(i)); HIP_CHECK(hipEventSynchronize(g[i].ev)); }
        
        // 4. 交換位置資料 (P2P)
        if (use_gpus == 2) {
            HIP_CHECK(hipMemcpyAsync(g[1].qx, g[0].qx, gpu_count[0]*8, hipMemcpyDeviceToDevice, g[0].s));
            HIP_CHECK(hipMemcpyAsync(g[1].qy, g[0].qy, gpu_count[0]*8, hipMemcpyDeviceToDevice, g[0].s));
            HIP_CHECK(hipMemcpyAsync(g[1].qz, g[0].qz, gpu_count[0]*8, hipMemcpyDeviceToDevice, g[0].s));
            HIP_CHECK(hipSetDevice(1));
            HIP_CHECK(hipMemcpyAsync(g[0].qx + gpu_start[1], g[1].qx + gpu_start[1], gpu_count[1]*8, hipMemcpyDeviceToDevice, g[1].s));
            HIP_CHECK(hipMemcpyAsync(g[0].qy + gpu_start[1], g[1].qy + gpu_start[1], gpu_count[1]*8, hipMemcpyDeviceToDevice, g[1].s));
            HIP_CHECK(hipMemcpyAsync(g[0].qz + gpu_start[1], g[1].qz + gpu_start[1], gpu_count[1]*8, hipMemcpyDeviceToDevice, g[1].s));
            HIP_CHECK(hipSetDevice(0)); HIP_CHECK(hipStreamSynchronize(g[0].s));
            HIP_CHECK(hipSetDevice(1)); HIP_CHECK(hipStreamSynchronize(g[1].s));
        }
        
        // 5. 事件檢查 (在 GPU 0 上執行)
        HIP_CHECK(hipSetDevice(0));
        int target_dev = calc_min_dist ? -1 : target_device;
        check_events_kernel<<<1, 1, 0, g[0].s>>>(step, planet, asteroid, target_dev, g[0].qx, g[0].qy, g[0].qz, g[0].m, g[0].result);
        
        // 同步質量到 GPU 1 (如果有 device 被摧毀)
        if (target_device >= 0 && use_gpus == 2) {
            HIP_CHECK(hipStreamSynchronize(g[0].s));
            HIP_CHECK(hipMemcpy(g[1].m + target_device, g[0].m + target_device, 8, hipMemcpyDeviceToDevice));
        }
    }
    
    // 取回結果
    double h_result[4];
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipMemcpy(h_result, g[0].result, 4*8, hipMemcpyDeviceToHost));
    res.min_dist = calc_min_dist ? h_result[0] : 1e30;
    res.hit_step = (int)h_result[1];
    res.destroy_step = (int)h_result[2];
    res.destroyed = (int)h_result[3];
    
    for (int i = 0; i < use_gpus; i++) {
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipFree(g[i].qx)); HIP_CHECK(hipFree(g[i].qy)); HIP_CHECK(hipFree(g[i].qz));
        HIP_CHECK(hipFree(g[i].vx)); HIP_CHECK(hipFree(g[i].vy)); HIP_CHECK(hipFree(g[i].vz));
        HIP_CHECK(hipFree(g[i].m)); HIP_CHECK(hipFree(g[i].type));
        HIP_CHECK(hipFree(g[i].ax)); HIP_CHECK(hipFree(g[i].ay)); HIP_CHECK(hipFree(g[i].az));
        HIP_CHECK(hipFree(g[i].result));
        HIP_CHECK(hipStreamDestroy(g[i].s)); HIP_CHECK(hipEventDestroy(g[i].ev));
    }
    return res;
}

int main(int argc, char** argv) {
    if (argc != 3) exit(1);
    int n, planet, asteroid;
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<int> type;
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);

    std::vector<int> devices;
    for (int i = 0; i < n; i++) if (type[i] == param::TYPE_DEVICE) devices.push_back(i);
    
    std::vector<double> m0 = m;
    for (int d : devices) m0[d] = 0;
    SimResult r0 = run_simulation_dual_gpu(n, planet, asteroid, -1, qx, qy, qz, vx, vy, vz, m0, type, true);
    SimResult r1 = run_simulation_dual_gpu(n, planet, asteroid, -1, qx, qy, qz, vx, vy, vz, m, type, false);
    
    int ans_dev = -1; double ans_cost = 0;
    if (r1.hit_step >= 0) {
        double best = 1e30;
        for (int d : devices) {
            SimResult r = run_simulation_dual_gpu(n, planet, asteroid, d, qx, qy, qz, vx, vy, vz, m, type, false);
            if (r.hit_step == -2 && r.destroyed) {
                double c = param::get_missile_cost(r.destroy_step * param::dt);
                if (c < best) { best = c; ans_dev = d; }
            }
        }
        if (ans_dev != -1) ans_cost = best;
    }

    write_output(argv[2], r0.min_dist, r1.hit_step, ans_dev, ans_cost);
    return 0;
}
