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

namespace param {
const int n_steps = 200000;
const double dt = 60;
const double eps = 1e-3;
const double G = 6.674e-11;
__device__ __host__ double gravity_device_mass(double m0, double t) {
    return m0 + 0.5 * m0 * fabs(sin(t / 6000));
}
const double planet_radius = 1e7;
const double missile_speed = 1e6;
__device__ __host__ double get_missile_cost(double t) { return 1e5 + 1e3 * t; }
}  // namespace param

void read_input(const char* filename, int& n, int& planet, int& asteroid,
    std::vector<double>& qx, std::vector<double>& qy, std::vector<double>& qz,
    std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz,
    std::vector<double>& m, std::vector<int>& is_device) {
    std::ifstream fin(filename);
    fin >> n >> planet >> asteroid;
    qx.resize(n);
    qy.resize(n);
    qz.resize(n);
    vx.resize(n);
    vy.resize(n);
    vz.resize(n);
    m.resize(n);
    is_device.resize(n);
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

// GPU Kernel: 計算加速度
__global__ void compute_acceleration(
    int n, double current_time,
    const double* __restrict__ qx, const double* __restrict__ qy, const double* __restrict__ qz,
    const double* __restrict__ m, const int* __restrict__ is_device,
    double* __restrict__ ax, double* __restrict__ ay, double* __restrict__ az)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    double ax_sum = 0.0, ay_sum = 0.0, az_sum = 0.0;
    double xi = qx[i], yi = qy[i], zi = qz[i];
    
    for (int j = 0; j < n; j++) {
        if (j == i) continue;
        
        double mj = m[j];
        // 只有當 m[j] > 0 且是 device 時才套用波動公式
        if (mj > 0 && is_device[j]) {
            mj = param::gravity_device_mass(mj, current_time);
        }
        
        double dx = qx[j] - xi;
        double dy = qy[j] - yi;
        double dz = qz[j] - zi;
        double dist_sq = dx * dx + dy * dy + dz * dz + param::eps * param::eps;
        double dist3 = dist_sq * sqrt(dist_sq);
        double factor = param::G * mj / dist3;
        
        ax_sum += factor * dx;
        ay_sum += factor * dy;
        az_sum += factor * dz;
    }
    
    ax[i] = ax_sum;
    ay[i] = ay_sum;
    az[i] = az_sum;
}

// GPU Kernel: 更新速度和位置
__global__ void update_velocity_position(
    int n,
    double* __restrict__ qx, double* __restrict__ qy, double* __restrict__ qz,
    double* __restrict__ vx, double* __restrict__ vy, double* __restrict__ vz,
    const double* __restrict__ ax, const double* __restrict__ ay, const double* __restrict__ az)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // Update velocity
    vx[i] += ax[i] * param::dt;
    vy[i] += ay[i] * param::dt;
    vz[i] += az[i] * param::dt;
    
    // Update position
    qx[i] += vx[i] * param::dt;
    qy[i] += vy[i] * param::dt;
    qz[i] += vz[i] * param::dt;
}

// 在 GPU 上執行一步模擬（與 CPU run_step 邏輯完全相同）
void run_step_gpu(int step, int n,
    double* d_qx, double* d_qy, double* d_qz,
    double* d_vx, double* d_vy, double* d_vz,
    double* d_ax, double* d_ay, double* d_az,
    const double* d_m, const int* d_is_device)
{
    // 計算力的時間點是該 step 的開始 (t = (step-1)*dt)
    double current_time = (step - 1) * param::dt;
    
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    // 計算加速度
    hipLaunchKernelGGL(compute_acceleration, dim3(grid_size), dim3(block_size), 0, 0,
        n, current_time, d_qx, d_qy, d_qz, d_m, d_is_device, d_ax, d_ay, d_az);
    
    // 更新速度和位置
    hipLaunchKernelGGL(update_velocity_position, dim3(grid_size), dim3(block_size), 0, 0,
        n, d_qx, d_qy, d_qz, d_vx, d_vy, d_vz, d_ax, d_ay, d_az);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }
    
    int n, planet, asteroid;
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<int> is_device;
    
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, is_device);
    
    // 分配 GPU 記憶體
    double *d_qx, *d_qy, *d_qz, *d_vx, *d_vy, *d_vz;
    double *d_ax, *d_ay, *d_az, *d_m;
    int *d_is_device;
    
    size_t size = n * sizeof(double);
    size_t size_int = n * sizeof(int);
    
    HIP_CHECK(hipMalloc(&d_qx, size));
    HIP_CHECK(hipMalloc(&d_qy, size));
    HIP_CHECK(hipMalloc(&d_qz, size));
    HIP_CHECK(hipMalloc(&d_vx, size));
    HIP_CHECK(hipMalloc(&d_vy, size));
    HIP_CHECK(hipMalloc(&d_vz, size));
    HIP_CHECK(hipMalloc(&d_ax, size));
    HIP_CHECK(hipMalloc(&d_ay, size));
    HIP_CHECK(hipMalloc(&d_az, size));
    HIP_CHECK(hipMalloc(&d_m, size));
    HIP_CHECK(hipMalloc(&d_is_device, size_int));
    
    // ===== Problem 1: 忽略 device (m=0) =====
    double min_dist = std::numeric_limits<double>::infinity();
    std::vector<double> m_no_device = m;
    for (int i = 0; i < n; i++) {
        if (is_device[i]) m_no_device[i] = 0;
    }
    
    // 複製數據到 GPU
    HIP_CHECK(hipMemcpy(d_qx, qx.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_qy, qy.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_qz, qz.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vx, vx.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vy, vy.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vz, vz.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_m, m_no_device.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_is_device, is_device.data(), size_int, hipMemcpyHostToDevice));
    
    std::vector<double> qx_host(n), qy_host(n), qz_host(n);
    
    for (int step = 0; step <= param::n_steps; step++) {
        if (step > 0) {
            run_step_gpu(step, n, d_qx, d_qy, d_qz, d_vx, d_vy, d_vz, 
                        d_ax, d_ay, d_az, d_m, d_is_device);
        }
        
        // 【關鍵】每一步都檢查距離，與 CPU 版本邏輯一致
        HIP_CHECK(hipMemcpy(qx_host.data(), d_qx, size, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(qy_host.data(), d_qy, size, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(qz_host.data(), d_qz, size, hipMemcpyDeviceToHost));
        
        double dx = qx_host[planet] - qx_host[asteroid];
        double dy = qy_host[planet] - qy_host[asteroid];
        double dz = qz_host[planet] - qz_host[asteroid];
        min_dist = std::min(min_dist, sqrt(dx * dx + dy * dy + dz * dz));
    }
    
    // ===== Problem 2: 正常模擬 =====
    int hit_time_step = -2;
    
    // 重新讀取並複製數據（包含 device）
    HIP_CHECK(hipMemcpy(d_qx, qx.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_qy, qy.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_qz, qz.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vx, vx.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vy, vy.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vz, vz.data(), size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_m, m.data(), size, hipMemcpyHostToDevice));
    
    double planet_radius_sq = param::planet_radius * param::planet_radius;
    
    for (int step = 0; step <= param::n_steps; step++) {
        if (step > 0) {
            run_step_gpu(step, n, d_qx, d_qy, d_qz, d_vx, d_vy, d_vz,
                        d_ax, d_ay, d_az, d_m, d_is_device);
        }
        
        // 【關鍵】每一步都檢查碰撞，與 CPU 版本邏輯一致
        HIP_CHECK(hipMemcpy(qx_host.data(), d_qx, size, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(qy_host.data(), d_qy, size, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(qz_host.data(), d_qz, size, hipMemcpyDeviceToHost));
        
        double dx = qx_host[planet] - qx_host[asteroid];
        double dy = qy_host[planet] - qy_host[asteroid];
        double dz = qz_host[planet] - qz_host[asteroid];
        if (dx * dx + dy * dy + dz * dz < planet_radius_sq) {
            hit_time_step = step;
            break;
        }
    }
    
    // ===== Problem 3 =====
    int gravity_device_id = -1;
    double missile_cost = 0;
    
    if (hit_time_step >= 0) {
        std::vector<int> devices;
        for (int i = 0; i < n; i++) {
            if (is_device[i]) devices.push_back(i);
        }
        
        double best_cost = std::numeric_limits<double>::infinity();
        int best_device = -1;
        
        // 保存原始數據
        std::vector<double> qx_orig = qx, qy_orig = qy, qz_orig = qz;
        std::vector<double> vx_orig = vx, vy_orig = vy, vz_orig = vz;
        std::vector<double> m_orig = m;
        
        for (int device_id : devices) {
            // 重置模擬狀態
            std::vector<double> qx_sim = qx_orig, qy_sim = qy_orig, qz_sim = qz_orig;
            std::vector<double> vx_sim = vx_orig, vy_sim = vy_orig, vz_sim = vz_orig;
            std::vector<double> m_sim = m_orig;
            
            // 複製到 GPU
            HIP_CHECK(hipMemcpy(d_qx, qx_sim.data(), size, hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_qy, qy_sim.data(), size, hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_qz, qz_sim.data(), size, hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_vx, vx_sim.data(), size, hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_vy, vy_sim.data(), size, hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_vz, vz_sim.data(), size, hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_m, m_sim.data(), size, hipMemcpyHostToDevice));
            
            bool collision_prevented = true;
            bool device_destroyed = false;
            int destroy_step = -1;
            
            double planet_x0 = qx_orig[planet];
            double planet_y0 = qy_orig[planet];
            double planet_z0 = qz_orig[planet];
            
            for (int step = 0; step <= param::n_steps; step++) {
                // 1. 先執行物理更新
                if (step > 0) {
                    run_step_gpu(step, n, d_qx, d_qy, d_qz, d_vx, d_vy, d_vz,
                                d_ax, d_ay, d_az, d_m, d_is_device);
                }
                
                // 複製當前位置回 CPU
                HIP_CHECK(hipMemcpy(qx_sim.data(), d_qx, size, hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(qy_sim.data(), d_qy, size, hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(qz_sim.data(), d_qz, size, hipMemcpyDeviceToHost));
                
                // 2. 檢查飛彈是否擊中
                if (!device_destroyed && step > 0) {
                    double missile_dist = step * param::dt * param::missile_speed;
                    
                    double dx = qx_sim[device_id] - planet_x0;
                    double dy = qy_sim[device_id] - planet_y0;
                    double dz = qz_sim[device_id] - planet_z0;
                    double device_dist = sqrt(dx * dx + dy * dy + dz * dz);
                    
                    if (missile_dist > device_dist) {
                        m_sim[device_id] = 0;  // 破壞裝置
                        HIP_CHECK(hipMemcpy(d_m, m_sim.data(), size, hipMemcpyHostToDevice));
                        device_destroyed = true;
                        destroy_step = step;
                    }
                }
                
                // 3. 檢查星球碰撞
                double dx = qx_sim[planet] - qx_sim[asteroid];
                double dy = qy_sim[planet] - qy_sim[asteroid];
                double dz = qz_sim[planet] - qz_sim[asteroid];
                if (dx * dx + dy * dy + dz * dz < planet_radius_sq) {
                    collision_prevented = false;
                    break;
                }
            }
            
            // 只有當「成功防止碰撞」且「確實破壞了裝置」時才考慮此策略
            if (collision_prevented && device_destroyed) {
                double cost = param::get_missile_cost(destroy_step * param::dt);
                if (cost < best_cost) {
                    best_cost = cost;
                    best_device = device_id;
                }
            }
        }
        
        if (best_device != -1) {
            gravity_device_id = best_device;
            missile_cost = best_cost;
        }
    }
    
    // 清理 GPU 記憶體
    HIP_CHECK(hipFree(d_qx));
    HIP_CHECK(hipFree(d_qy));
    HIP_CHECK(hipFree(d_qz));
    HIP_CHECK(hipFree(d_vx));
    HIP_CHECK(hipFree(d_vy));
    HIP_CHECK(hipFree(d_vz));
    HIP_CHECK(hipFree(d_ax));
    HIP_CHECK(hipFree(d_ay));
    HIP_CHECK(hipFree(d_az));
    HIP_CHECK(hipFree(d_m));
    HIP_CHECK(hipFree(d_is_device));
    
    write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);
    return 0;
}
