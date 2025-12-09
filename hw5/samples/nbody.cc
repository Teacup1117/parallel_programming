#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace param {
const int n_steps = 200000;
const double dt = 60;
const double eps = 1e-3;
const double G = 6.674e-11;
double gravity_device_mass(double m0, double t) {
    return m0 + 0.5 * m0 * fabs(sin(t / 6000));
}
const double planet_radius = 1e7;
const double missile_speed = 1e6;
double get_missile_cost(double t) { return 1e5 + 1e3 * t; }
}  // namespace param

void read_input(const char* filename, int& n, int& planet, int& asteroid,
    std::vector<double>& qx, std::vector<double>& qy, std::vector<double>& qz,
    std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz,
    std::vector<double>& m, std::vector<std::string>& type) {
    std::ifstream fin(filename);
    fin >> n >> planet >> asteroid;
    qx.resize(n);
    qy.resize(n);
    qz.resize(n);
    vx.resize(n);
    vy.resize(n);
    vz.resize(n);
    m.resize(n);
    type.resize(n);
    for (int i = 0; i < n; i++) {
        fin >> qx[i] >> qy[i] >> qz[i] >> vx[i] >> vy[i] >> vz[i] >> m[i] >> type[i];
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

void run_step(int step, int n, std::vector<double>& qx, std::vector<double>& qy,
    std::vector<double>& qz, std::vector<double>& vx, std::vector<double>& vy,
    std::vector<double>& vz, const std::vector<double>& m,
    const std::vector<std::string>& type) {
    
    std::vector<double> ax(n, 0.0), ay(n, 0.0), az(n, 0.0);
    
    // 計算力的時間點是該 step 的開始 (t = (step-1)*dt)
    double current_time = (step - 1) * param::dt;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            double mj = m[j];
            
            // 如果裝置已被破壞 (m[j]==0)，這裡就不會計算它的引力
            // 只有當 m[j] > 0 且是 device 時才套用波動公式
            if (mj > 0 && type[j] == "device") {
                mj = param::gravity_device_mass(mj, current_time);
            }

            double dx = qx[j] - qx[i];
            double dy = qy[j] - qy[i];
            double dz = qz[j] - qz[i];
            double dist3 =
                pow(dx * dx + dy * dy + dz * dz + param::eps * param::eps, 1.5);
            ax[i] += param::G * mj * dx / dist3;
            ay[i] += param::G * mj * dy / dist3;
            az[i] += param::G * mj * dz / dist3;
        }
    }

    // update velocities
    for (int i = 0; i < n; i++) {
        vx[i] += ax[i] * param::dt;
        vy[i] += ay[i] * param::dt;
        vz[i] += az[i] * param::dt;
    }

    // update positions
    for (int i = 0; i < n; i++) {
        qx[i] += vx[i] * param::dt;
        qy[i] += vy[i] * param::dt;
        qz[i] += vz[i] * param::dt;
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        throw std::runtime_error("must supply 2 arguments");
    }
    int n, planet, asteroid;
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<std::string> type;

    // Problem 1: 忽略 device (m=0)
    double min_dist = std::numeric_limits<double>::infinity();
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
    for (int i = 0; i < n; i++) {
        if (type[i] == "device") m[i] = 0;
    }
    for (int step = 0; step <= param::n_steps; step++) {
        if (step > 0) run_step(step, n, qx, qy, qz, vx, vy, vz, m, type);
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        min_dist = std::min(min_dist, sqrt(dx * dx + dy * dy + dz * dz));
    }

    // Problem 2: 正常模擬
    int hit_time_step = -2;
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
    for (int step = 0; step <= param::n_steps; step++) {
        if (step > 0) run_step(step, n, qx, qy, qz, vx, vy, vz, m, type);
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius) {
            hit_time_step = step;
            break;
        }
    }

    // Problem 3
    int gravity_device_id = -1;
    double missile_cost = 0;
    
    if (hit_time_step >= 0) {
        int n_orig, planet_orig, asteroid_orig;
        std::vector<double> qx_orig, qy_orig, qz_orig, vx_orig, vy_orig, vz_orig, m_orig;
        std::vector<std::string> type_orig;
        read_input(argv[1], n_orig, planet_orig, asteroid_orig, qx_orig, qy_orig, qz_orig, vx_orig, vy_orig, vz_orig, m_orig, type_orig);

        std::vector<int> devices;
        for (int i = 0; i < n_orig; i++) {
            if (type_orig[i] == "device") devices.push_back(i);
        }
        
        double best_cost = std::numeric_limits<double>::infinity();
        int best_device = -1;
        
        for (int device_id : devices) {
            // 重置模擬狀態
            n = n_orig; planet = planet_orig; asteroid = asteroid_orig;
            qx = qx_orig; qy = qy_orig; qz = qz_orig;
            vx = vx_orig; vy = vy_orig; vz = vz_orig;
            m = m_orig; type = type_orig;
            
            bool collision_prevented = true;
            bool device_destroyed = false;
            int destroy_step = -1;
            
            for (int step = 0; step <= param::n_steps; step++) {
                // 1. 先執行物理更新 (重要：Spec 說距離要看「當前最新」位置)
                if (step > 0) {
                    run_step(step, n, qx, qy, qz, vx, vy, vz, m, type);
                }
                
                // 2. 檢查飛彈是否擊中
                if (!device_destroyed && step > 0) {
                    // Spec [24]: 飛彈距離 = step * dt * speed
                    double missile_dist = step * param::dt * param::missile_speed;
                    
                    // Spec [25]: 目標距離 = Planet 與 Device 在「當前 step」的距離
                    double dx = qx[device_id] - qx[planet];
                    double dy = qy[device_id] - qy[planet];
                    double dz = qz[device_id] - qz[planet];
                    double device_dist = sqrt(dx * dx + dy * dy + dz * dz);
                    
                    // Spec [26]: 飛彈距離 > 裝置距離 則判定擊中
                    if (missile_dist > device_dist) {
                        m[device_id] = 0; // 破壞裝置
                        device_destroyed = true;
                        destroy_step = step; // 紀錄是這一步結束時破壞的
                    }
                }
                
                // 3. 檢查星球碰撞
                double dx = qx[planet] - qx[asteroid];
                double dy = qy[planet] - qy[asteroid];
                double dz = qz[planet] - qz[asteroid];
                if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius) {
                    // 如果即使裝置被破壞了，還是發生碰撞，則此策略失敗
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

    write_output(argv[2], min_dist, hit_time_step, gravity_device_id, missile_cost);
}