#include "hip/hip_runtime.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

#define HIP_CHECK(cmd) \
    do { \
        hipError_t error = (cmd); \
        if (error != hipSuccess) { \
            fprintf(stderr, "HIP error: '%s'(%d) at %s:%d\n", \
                    hipGetErrorString(error), error, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ==============================================================================
// 優化參數
// ==============================================================================
#define BLOCK_SIZE 256

namespace param {
const int n_steps = 200000;
const double dt = 60;
const double eps = 1e-3;
const double eps_sq = eps * eps;
const double G = 6.674e-11;
const double planet_radius = 1e7;
const double planet_radius_sq = planet_radius * planet_radius;
const double missile_speed = 1e6;

__device__ __host__ double gravity_device_mass(double m0, double t) {
    return m0 + 0.5 * m0 * fabs(sin(t / 6000));
}
__device__ __host__ double get_missile_cost(double t) { return 1e5 + 1e3 * t; }
}  // namespace param

// ==============================================================================
// IO 函數
// ==============================================================================
void read_input(const char* filename, int& n, int& planet, int& asteroid,
    std::vector<double>& qx, std::vector<double>& qy, std::vector<double>& qz,
    std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz,
    std::vector<double>& m, std::vector<int>& is_device) {
    std::ifstream fin(filename);
    fin >> n >> planet >> asteroid;
    qx.resize(n); qy.resize(n); qz.resize(n);
    vx.resize(n); vy.resize(n); vz.resize(n);
    m.resize(n); is_device.resize(n);
    for (int i = 0; i < n; i++) {
        std::string type;
        fin >> qx[i] >> qy[i] >> qz[i] >> vx[i] >> vy[i] >> vz[i] >> m[i] >> type;
        is_device[i] = (type == "device") ? 1 : 0;
    }
}

void write_output(const char* filename, double min_dist, int hit_time_step,
    int gravity_device_id, double missile_cost) {
    std::ofstream fout(filename);
    fout << std::scientific
         << std::setprecision(std::numeric_limits<double>::digits10 + 1) << min_dist
         << '\n'
         << hit_time_step << '\n'
         << gravity_device_id << ' ' << missile_cost << '\n';
}

// ==============================================================================
// GPU Kernel: Tiled Force Calculation (適用於任意 N)
// ==============================================================================
__global__ void compute_and_update_tiled(
    int n, double current_time,
    const double* __restrict__ qx_in, const double* __restrict__ qy_in, const double* __restrict__ qz_in,
    const double* __restrict__ vx_in, const double* __restrict__ vy_in, const double* __restrict__ vz_in,
    const double* __restrict__ m, const int* __restrict__ is_device,
    double* __restrict__ qx_out, double* __restrict__ qy_out, double* __restrict__ qz_out,
    double* __restrict__ vx_out, double* __restrict__ vy_out, double* __restrict__ vz_out)
{
    __shared__ double sh_qx[BLOCK_SIZE];
    __shared__ double sh_qy[BLOCK_SIZE];
    __shared__ double sh_qz[BLOCK_SIZE];
    __shared__ double sh_m[BLOCK_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    double my_qx = 0, my_qy = 0, my_qz = 0;
    if (i < n) {
        my_qx = qx_in[i];
        my_qy = qy_in[i];
        my_qz = qz_in[i];
    }
    
    double acc_x = 0.0, acc_y = 0.0, acc_z = 0.0;
    int num_tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int tile = 0; tile < num_tiles; tile++) {
        int tile_idx = tile * BLOCK_SIZE + tid;
        if (tile_idx < n) {
            sh_qx[tid] = qx_in[tile_idx];
            sh_qy[tid] = qy_in[tile_idx];
            sh_qz[tid] = qz_in[tile_idx];
            double raw_m = m[tile_idx];
            if (raw_m > 0.0 && is_device[tile_idx]) {
                sh_m[tid] = param::gravity_device_mass(raw_m, current_time);
            } else {
                sh_m[tid] = raw_m;
            }
        } else {
            sh_m[tid] = 0.0;
        }
        __syncthreads();
        
        if (i < n) {
            #pragma unroll 8
            for (int j = 0; j < BLOCK_SIZE; j++) {
                int target = tile * BLOCK_SIZE + j;
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
        double vx = vx_in[i] + acc_x * param::dt;
        double vy = vy_in[i] + acc_y * param::dt;
        double vz = vz_in[i] + acc_z * param::dt;
        
        qx_out[i] = my_qx + vx * param::dt;
        qy_out[i] = my_qy + vy * param::dt;
        qz_out[i] = my_qz + vz * param::dt;
        vx_out[i] = vx;
        vy_out[i] = vy;
        vz_out[i] = vz;
    }
}

// ==============================================================================
// GPU Kernel: 檢查最小距離 (單一執行緒)
// ==============================================================================
__global__ void check_min_dist(
    int planet, int asteroid,
    const double* __restrict__ qx, const double* __restrict__ qy, const double* __restrict__ qz,
    double* min_dist)
{
    double dx = qx[planet] - qx[asteroid];
    double dy = qy[planet] - qy[asteroid];
    double dz = qz[planet] - qz[asteroid];
    double dist = sqrt(dx*dx + dy*dy + dz*dz);
    if (dist < *min_dist) *min_dist = dist;
}

// ==============================================================================
// GPU Kernel: 檢查碰撞 (單一執行緒)
// ==============================================================================
__global__ void check_collision(
    int step, int planet, int asteroid,
    const double* __restrict__ qx, const double* __restrict__ qy, const double* __restrict__ qz,
    int* hit_step)
{
    if (*hit_step != -2) return;  // 已發現碰撞
    
    double dx = qx[planet] - qx[asteroid];
    double dy = qy[planet] - qy[asteroid];
    double dz = qz[planet] - qz[asteroid];
    if (dx*dx + dy*dy + dz*dz < param::planet_radius_sq) {
        *hit_step = step;
    }
}

// ==============================================================================
// GPU Kernel: 檢查飛彈擊中和碰撞（合併版）
// ==============================================================================
__global__ void check_missile_and_collision(
    int step, int planet, int asteroid, int target_device,
    double planet_x0, double planet_y0, double planet_z0,
    const double* __restrict__ qx, const double* __restrict__ qy, const double* __restrict__ qz,
    double* __restrict__ d_m,
    int* device_destroyed, int* destroy_step, int* collision_prevented)
{
    // 1. 檢查飛彈是否擊中
    if (*device_destroyed == 0 && step > 0) {
        double missile_dist = step * param::dt * param::missile_speed;
        double dx = qx[target_device] - planet_x0;
        double dy = qy[target_device] - planet_y0;
        double dz = qz[target_device] - planet_z0;
        double device_dist = sqrt(dx*dx + dy*dy + dz*dz);
        
        if (missile_dist > device_dist) {
            d_m[target_device] = 0;
            *device_destroyed = 1;
            *destroy_step = step;
        }
    }
    
    // 2. 檢查碰撞
    if (*collision_prevented == 1) {
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        if (dx*dx + dy*dy + dz*dz < param::planet_radius_sq) {
            *collision_prevented = 0;  // 發生碰撞
        }
    }
}

// ==============================================================================
// Problem 1: 計算最小距離 (雙 GPU 版本)
// 使用雙緩衝避免每步同步，只在結尾讀取結果
// ==============================================================================
double solve_problem1(int n, int planet, int asteroid,
    const std::vector<double>& qx, const std::vector<double>& qy, const std::vector<double>& qz,
    const std::vector<double>& vx, const std::vector<double>& vy, const std::vector<double>& vz,
    const std::vector<double>& m_no_device, const std::vector<int>& is_device)
{
    size_t size = n * sizeof(double);
    size_t size_int = n * sizeof(int);
    
    // 分配 GPU 記憶體（雙緩衝）
    double *d_qx[2], *d_qy[2], *d_qz[2];
    double *d_vx[2], *d_vy[2], *d_vz[2];
    double *d_m, *d_min_dist;
    int *d_is_device;
    
    for (int i = 0; i < 2; i++) {
        HIP_CHECK(hipMalloc(&d_qx[i], size));
        HIP_CHECK(hipMalloc(&d_qy[i], size));
        HIP_CHECK(hipMalloc(&d_qz[i], size));
        HIP_CHECK(hipMalloc(&d_vx[i], size));
        HIP_CHECK(hipMalloc(&d_vy[i], size));
        HIP_CHECK(hipMalloc(&d_vz[i], size));
    }
    HIP_CHECK(hipMalloc(&d_m, size));
    HIP_CHECK(hipMalloc(&d_is_device, size_int));
    HIP_CHECK(hipMalloc(&d_min_dist, sizeof(double)));
    
    // 初始化
    HIP_CHECK(hipMemcpy(d_qx[0], qx.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_qy[0], qy.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_qz[0], qz.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vx[0], vx.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vy[0], vy.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vz[0], vz.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_m, m_no_device.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_is_device, is_device.data(), size_int, hipMemcpyHostToDevice));
    
    double inf = std::numeric_limits<double>::infinity();
    HIP_CHECK(hipMemcpy(d_min_dist, &inf, sizeof(double), hipMemcpyHostToDevice));
    
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int cur = 0;
    
    // step 0: 檢查初始距離
    hipLaunchKernelGGL(check_min_dist, 1, 1, 0, 0,
        planet, asteroid, d_qx[0], d_qy[0], d_qz[0], d_min_dist);
    
    for (int step = 1; step <= param::n_steps; step++) {
        int next = 1 - cur;
        double current_time = (step - 1) * param::dt;
        
        // 物理更新
        hipLaunchKernelGGL(compute_and_update_tiled, grid_size, BLOCK_SIZE, 0, 0,
            n, current_time,
            d_qx[cur], d_qy[cur], d_qz[cur],
            d_vx[cur], d_vy[cur], d_vz[cur],
            d_m, d_is_device,
            d_qx[next], d_qy[next], d_qz[next],
            d_vx[next], d_vy[next], d_vz[next]);
        
        // 檢查最小距離（在 GPU 上）
        hipLaunchKernelGGL(check_min_dist, 1, 1, 0, 0,
            planet, asteroid, d_qx[next], d_qy[next], d_qz[next], d_min_dist);
        
        cur = next;
    }
    
    // 讀取結果
    double min_dist;
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(&min_dist, d_min_dist, sizeof(double), hipMemcpyDeviceToHost));
    
    // 釋放記憶體
    for (int i = 0; i < 2; i++) {
        HIP_CHECK(hipFree(d_qx[i])); HIP_CHECK(hipFree(d_qy[i])); HIP_CHECK(hipFree(d_qz[i]));
        HIP_CHECK(hipFree(d_vx[i])); HIP_CHECK(hipFree(d_vy[i])); HIP_CHECK(hipFree(d_vz[i]));
    }
    HIP_CHECK(hipFree(d_m));
    HIP_CHECK(hipFree(d_is_device));
    HIP_CHECK(hipFree(d_min_dist));
    
    return min_dist;
}

// ==============================================================================
// Problem 2: 檢測碰撞時間
// ==============================================================================
int solve_problem2(int n, int planet, int asteroid,
    const std::vector<double>& qx, const std::vector<double>& qy, const std::vector<double>& qz,
    const std::vector<double>& vx, const std::vector<double>& vy, const std::vector<double>& vz,
    const std::vector<double>& m, const std::vector<int>& is_device)
{
    size_t size = n * sizeof(double);
    size_t size_int = n * sizeof(int);
    
    double *d_qx[2], *d_qy[2], *d_qz[2];
    double *d_vx[2], *d_vy[2], *d_vz[2];
    double *d_m;
    int *d_is_device, *d_hit_step;
    
    for (int i = 0; i < 2; i++) {
        HIP_CHECK(hipMalloc(&d_qx[i], size));
        HIP_CHECK(hipMalloc(&d_qy[i], size));
        HIP_CHECK(hipMalloc(&d_qz[i], size));
        HIP_CHECK(hipMalloc(&d_vx[i], size));
        HIP_CHECK(hipMalloc(&d_vy[i], size));
        HIP_CHECK(hipMalloc(&d_vz[i], size));
    }
    HIP_CHECK(hipMalloc(&d_m, size));
    HIP_CHECK(hipMalloc(&d_is_device, size_int));
    HIP_CHECK(hipMalloc(&d_hit_step, sizeof(int)));
    
    HIP_CHECK(hipMemcpy(d_qx[0], qx.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_qy[0], qy.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_qz[0], qz.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vx[0], vx.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vy[0], vy.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vz[0], vz.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_m, m.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_is_device, is_device.data(), size_int, hipMemcpyHostToDevice));
    
    int hit_step_init = -2;
    HIP_CHECK(hipMemcpy(d_hit_step, &hit_step_init, sizeof(int), hipMemcpyHostToDevice));
    
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int cur = 0;
    
    // step 0: 檢查初始碰撞
    hipLaunchKernelGGL(check_collision, 1, 1, 0, 0,
        0, planet, asteroid, d_qx[0], d_qy[0], d_qz[0], d_hit_step);
    
    for (int step = 1; step <= param::n_steps; step++) {
        int next = 1 - cur;
        double current_time = (step - 1) * param::dt;
        
        hipLaunchKernelGGL(compute_and_update_tiled, grid_size, BLOCK_SIZE, 0, 0,
            n, current_time,
            d_qx[cur], d_qy[cur], d_qz[cur],
            d_vx[cur], d_vy[cur], d_vz[cur],
            d_m, d_is_device,
            d_qx[next], d_qy[next], d_qz[next],
            d_vx[next], d_vy[next], d_vz[next]);
        
        hipLaunchKernelGGL(check_collision, 1, 1, 0, 0,
            step, planet, asteroid, d_qx[next], d_qy[next], d_qz[next], d_hit_step);
        
        cur = next;
    }
    
    int hit_step;
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(&hit_step, d_hit_step, sizeof(int), hipMemcpyDeviceToHost));
    
    for (int i = 0; i < 2; i++) {
        HIP_CHECK(hipFree(d_qx[i])); HIP_CHECK(hipFree(d_qy[i])); HIP_CHECK(hipFree(d_qz[i]));
        HIP_CHECK(hipFree(d_vx[i])); HIP_CHECK(hipFree(d_vy[i])); HIP_CHECK(hipFree(d_vz[i]));
    }
    HIP_CHECK(hipFree(d_m));
    HIP_CHECK(hipFree(d_is_device));
    HIP_CHECK(hipFree(d_hit_step));
    
    return hit_step;
}

// ==============================================================================
// Problem 3: 嘗試破壞裝置 (單一裝置模擬)
// ==============================================================================
struct DeviceResult {
    int device_id;
    bool success;
    int destroy_step;
    double cost;
};

DeviceResult try_destroy_device(int device_id, int n, int planet, int asteroid,
    const std::vector<double>& qx, const std::vector<double>& qy, const std::vector<double>& qz,
    const std::vector<double>& vx, const std::vector<double>& vy, const std::vector<double>& vz,
    const std::vector<double>& m, const std::vector<int>& is_device,
    hipStream_t stream)
{
    size_t size = n * sizeof(double);
    size_t size_int = n * sizeof(int);
    
    double *d_qx[2], *d_qy[2], *d_qz[2];
    double *d_vx[2], *d_vy[2], *d_vz[2];
    double *d_m;
    int *d_is_device;
    int *d_device_destroyed, *d_destroy_step, *d_collision_prevented;
    
    for (int i = 0; i < 2; i++) {
        HIP_CHECK(hipMalloc(&d_qx[i], size));
        HIP_CHECK(hipMalloc(&d_qy[i], size));
        HIP_CHECK(hipMalloc(&d_qz[i], size));
        HIP_CHECK(hipMalloc(&d_vx[i], size));
        HIP_CHECK(hipMalloc(&d_vy[i], size));
        HIP_CHECK(hipMalloc(&d_vz[i], size));
    }
    HIP_CHECK(hipMalloc(&d_m, size));
    HIP_CHECK(hipMalloc(&d_is_device, size_int));
    HIP_CHECK(hipMalloc(&d_device_destroyed, sizeof(int)));
    HIP_CHECK(hipMalloc(&d_destroy_step, sizeof(int)));
    HIP_CHECK(hipMalloc(&d_collision_prevented, sizeof(int)));
    
    HIP_CHECK(hipMemcpyAsync(d_qx[0], qx.data(), size, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_qy[0], qy.data(), size, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_qz[0], qz.data(), size, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_vx[0], vx.data(), size, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_vy[0], vy.data(), size, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_vz[0], vz.data(), size, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_m, m.data(), size, hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_is_device, is_device.data(), size_int, hipMemcpyHostToDevice, stream));
    
    int zero = 0, one = 1, neg_one = -1;
    HIP_CHECK(hipMemcpyAsync(d_device_destroyed, &zero, sizeof(int), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_destroy_step, &neg_one, sizeof(int), hipMemcpyHostToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(d_collision_prevented, &one, sizeof(int), hipMemcpyHostToDevice, stream));
    
    double planet_x0 = qx[planet], planet_y0 = qy[planet], planet_z0 = qz[planet];
    
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int cur = 0;
    
    for (int step = 0; step <= param::n_steps; step++) {
        if (step > 0) {
            int next = 1 - cur;
            double current_time = (step - 1) * param::dt;
            
            hipLaunchKernelGGL(compute_and_update_tiled, grid_size, BLOCK_SIZE, 0, stream,
                n, current_time,
                d_qx[cur], d_qy[cur], d_qz[cur],
                d_vx[cur], d_vy[cur], d_vz[cur],
                d_m, d_is_device,
                d_qx[next], d_qy[next], d_qz[next],
                d_vx[next], d_vy[next], d_vz[next]);
            
            cur = next;
        }
        
        hipLaunchKernelGGL(check_missile_and_collision, 1, 1, 0, stream,
            step, planet, asteroid, device_id,
            planet_x0, planet_y0, planet_z0,
            d_qx[cur], d_qy[cur], d_qz[cur], d_m,
            d_device_destroyed, d_destroy_step, d_collision_prevented);
    }
    
    int device_destroyed, destroy_step, collision_prevented;
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpy(&device_destroyed, d_device_destroyed, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&destroy_step, d_destroy_step, sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&collision_prevented, d_collision_prevented, sizeof(int), hipMemcpyDeviceToHost));
    
    for (int i = 0; i < 2; i++) {
        HIP_CHECK(hipFree(d_qx[i])); HIP_CHECK(hipFree(d_qy[i])); HIP_CHECK(hipFree(d_qz[i]));
        HIP_CHECK(hipFree(d_vx[i])); HIP_CHECK(hipFree(d_vy[i])); HIP_CHECK(hipFree(d_vz[i]));
    }
    HIP_CHECK(hipFree(d_m));
    HIP_CHECK(hipFree(d_is_device));
    HIP_CHECK(hipFree(d_device_destroyed));
    HIP_CHECK(hipFree(d_destroy_step));
    HIP_CHECK(hipFree(d_collision_prevented));
    
    DeviceResult result;
    result.device_id = device_id;
    result.success = (collision_prevented == 1 && device_destroyed == 1);
    result.destroy_step = destroy_step;
    result.cost = result.success ? param::get_missile_cost(destroy_step * param::dt) : std::numeric_limits<double>::infinity();
    
    return result;
}

// ==============================================================================
// Main
// ==============================================================================
int main(int argc, char** argv) {
    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }
    
    int n, planet, asteroid;
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<int> is_device;
    
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, is_device);
    
    // 準備無 device 質量版本
    std::vector<double> m_no_device = m;
    for (int i = 0; i < n; i++) {
        if (is_device[i]) m_no_device[i] = 0;
    }
    
    // Problem 1: 計算最小距離
    double min_dist = solve_problem1(n, planet, asteroid, qx, qy, qz, vx, vy, vz, m_no_device, is_device);
    
    // Problem 2: 檢測碰撞
    int hit_time_step = solve_problem2(n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, is_device);
    
    // Problem 3: 嘗試破壞裝置
    int gravity_device_id = -1;
    double missile_cost = 0;
    
    if (hit_time_step >= 0) {
        std::vector<int> devices;
        for (int i = 0; i < n; i++) {
            if (is_device[i]) devices.push_back(i);
        }
        
        // 使用多 GPU 並行處理不同裝置
        int num_gpus;
        HIP_CHECK(hipGetDeviceCount(&num_gpus));
        num_gpus = std::min(num_gpus, 2);  // 最多使用 2 個 GPU
        
        std::vector<hipStream_t> streams(devices.size());
        std::vector<DeviceResult> results(devices.size());
        
        for (size_t i = 0; i < devices.size(); i++) {
            int gpu_id = i % num_gpus;
            HIP_CHECK(hipSetDevice(gpu_id));
            HIP_CHECK(hipStreamCreate(&streams[i]));
        }
        
        // 並行模擬所有裝置
        for (size_t i = 0; i < devices.size(); i++) {
            int gpu_id = i % num_gpus;
            HIP_CHECK(hipSetDevice(gpu_id));
            results[i] = try_destroy_device(devices[i], n, planet, asteroid,
                qx, qy, qz, vx, vy, vz, m, is_device, streams[i]);
        }
        
        // 找最佳結果
        double best_cost = std::numeric_limits<double>::infinity();
        for (const auto& result : results) {
            if (result.success && result.cost < best_cost) {
                best_cost = result.cost;
                gravity_device_id = result.device_id;
            }
        }
        if (gravity_device_id != -1) {
            missile_cost = best_cost;
        }
        
        // 清理 streams
        for (size_t i = 0; i < devices.size(); i++) {
            int gpu_id = i % num_gpus;
            HIP_CHECK(hipSetDevice(gpu_id));
            HIP_CHECK(hipStreamDestroy(streams[i]));
        }
    }
    
    write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);
    return 0;
}
