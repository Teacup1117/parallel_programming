#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <lodepng.h>

#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

#define pi 3.14159265358979323846f

typedef glm::vec2 vec2;
typedef glm::vec3 vec3;
typedef glm::vec4 vec4;
typedef glm::mat3 mat3;

// Constants (use __constant__ memory for better performance)
__constant__ float d_power = 8.0f;
__constant__ int d_md_iter = 24;
__constant__ int d_ray_step = 10000;
__constant__ int d_shadow_step = 1500;
__constant__ float d_step_limiter = 0.2f;
__constant__ float d_ray_multiplier = 0.1f;
__constant__ float d_bailout = 2.0f;
__constant__ float d_eps = 0.0005f;
__constant__ float d_FOV = 1.5f;
__constant__ float d_far_plane = 100.f;
__constant__ int d_AA = 3;

__constant__ vec3 d_camera_pos;
__constant__ vec3 d_target_pos;
__constant__ vec2 d_iResolution;

// Device helper functions
__device__ float length_vec3(vec3 v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ vec3 normalize_vec3(vec3 v) {
    float len = length_vec3(v);
    return len > 0 ? vec3(v.x / len, v.y / len, v.z / len) : v;
}

__device__ float dot_vec3(vec3 a, vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ vec3 cross_vec3(vec3 a, vec3 b) {
    return vec3(a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x);
}

__device__ float clamp_float(float x, float minVal, float maxVal) {
    return fminf(fmaxf(x, minVal), maxVal);
}

__device__ vec3 clamp_vec3(vec3 v, float minVal, float maxVal) {
    return vec3(clamp_float(v.x, minVal, maxVal),
                clamp_float(v.y, minVal, maxVal),
                clamp_float(v.z, minVal, maxVal));
}

__device__ vec3 pow_vec3(vec3 v, vec3 p) {
    return vec3(pow(v.x, p.x), pow(v.y, p.y), pow(v.z, p.z));
}

__device__ vec3 cos_vec3(vec3 v) {
    return vec3(cos(v.x), cos(v.y), cos(v.z));
}

// mandelbulb distance function (DE)
__device__ float md(vec3 p, float& trap) {
    vec3 v = p;
    float dr = 1.f;
    float r = length_vec3(v);
    trap = r;

    for (int i = 0; i < d_md_iter; ++i) {
        float theta = atan2(v.y, v.x) * d_power;
        float phi = asin(v.z / r) * d_power;
        dr = d_power * pow(r, d_power - 1.f) * dr + 1.f;
        v = p + pow(r, d_power) *
                vec3(cos(theta) * cos(phi), cos(phi) * sin(theta), -sin(phi));

        trap = fminf(trap, r);

        r = length_vec3(v);
        if (r > d_bailout) break;
    }
    return 0.5f * log(r) * r / dr;
}

// scene mapping
__device__ float map(vec3 p, float& trap, int& ID) {
    vec2 rt = vec2(cos(pi / 2.f), sin(pi / 2.f));
    // Rotation matrix
    vec3 rp = vec3(p.x,
                   rt.x * p.y - rt.y * p.z,
                   rt.y * p.y + rt.x * p.z);
    ID = 1;
    return md(rp, trap);
}

// dummy function for normal calculation
__device__ float map(vec3 p) {
    float dmy;
    int dmy2;
    return map(p, dmy, dmy2);
}

// simple palette function
__device__ vec3 pal(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b * cos_vec3(2.f * pi * (c * t + d));
}

// soft shadow
__device__ float softshadow(vec3 ro, vec3 rd, float k) {
    float res = 1.0f;
    float t = 0.001f;  // start with small positive value to avoid division by zero
    for (int i = 0; i < d_shadow_step; ++i) {
        float h = map(ro + rd * t);
        res = fminf(res, k * h / t);
        if (res < 0.02f) return 0.02f;
        t += clamp_float(h, .001f, d_step_limiter);
    }
    return clamp_float(res, .02f, 1.f);
}

// calculate surface normal
__device__ vec3 calcNor(vec3 p) {
    vec2 e = vec2(d_eps, 0.f);
    return normalize_vec3(vec3(
        map(p + vec3(e.x, e.y, e.y)) - map(p - vec3(e.x, e.y, e.y)),
        map(p + vec3(e.y, e.x, e.y)) - map(p - vec3(e.y, e.x, e.y)),
        map(p + vec3(e.y, e.y, e.x)) - map(p - vec3(e.y, e.y, e.x))
    ));
}

// ray marching
__device__ float trace(vec3 ro, vec3 rd, float& trap, int& ID) {
    float t = 0;
    float len = 0;

    for (int i = 0; i < d_ray_step; ++i) {
        len = map(ro + rd * t, trap, ID);
        if (fabs(len) < d_eps || t > d_far_plane) break;
        t += len * d_ray_multiplier;
    }
    return t < d_far_plane ? t : -1.f;
}

// Original static kernel (baseline)
// __launch_bounds__ 提示編譯器優化：每個 block 256 threads, 每個 SM 最少 2 blocks
__global__ void __launch_bounds__(256, 2)
render_kernel_static(unsigned char* d_image, unsigned int width, unsigned int height) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= height || j >= width) return;

    // 使用共享記憶體快取相機向量（所有 threads 共用）
    __shared__ vec3 shared_cf, shared_cs, shared_cu;
    
    // 只讓第一個 thread 計算，其他 threads 重用
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        vec3 ro = d_camera_pos;
        vec3 ta = d_target_pos;
        shared_cf = normalize_vec3(ta - ro);
        shared_cs = normalize_vec3(cross_vec3(shared_cf, vec3(0., 1., 0.)));
        shared_cu = normalize_vec3(cross_vec3(shared_cs, shared_cf));
    }
    __syncthreads();

    float fcol_r = 0.0f;
    float fcol_g = 0.0f;
    float fcol_b = 0.0f;

    // anti aliasing - 提示編譯器完全展開這個小迴圈（3x3=9 次迭代）
    #pragma unroll
    for (int m = 0; m < d_AA; ++m) {
        #pragma unroll
        for (int n = 0; n < d_AA; ++n) {
            vec2 p = vec2(j, i) + vec2(m, n) / (float)d_AA;

            // convert screen space coordinate to (-ap~ap, -1~1)
            vec2 uv = vec2((-d_iResolution.x + 2.f * p.x) / d_iResolution.y,
                          (-d_iResolution.y + 2.f * p.y) / d_iResolution.y);
            uv.y *= -1;

            // create camera - 直接使用共享記憶體中的向量
            vec3 ro = d_camera_pos;
            vec3 rd = normalize_vec3(uv.x * shared_cs + uv.y * shared_cu + d_FOV * shared_cf);

            // marching
            float trap;
            int objID;
            float d = trace(ro, rd, trap, objID);

            // lighting
            vec3 col(0.f);
            vec3 sd = normalize_vec3(d_camera_pos);
            vec3 sc = vec3(1.f, .9f, .717f);

            // coloring
            if (d < 0.f) {
                col = vec3(0.f);
            } else {
                vec3 pos = ro + rd * d;
                vec3 nr = calcNor(pos);
                vec3 hal = normalize_vec3(sd - rd);

                // use orbit trap to get the color
                col = pal(trap - .4f, vec3(.5f), vec3(.5f), vec3(1.f), vec3(.0f, .1f, .2f));
                vec3 ambc = vec3(0.3f);
                float gloss = 32.f;

                // simple blinn phong lighting model
                float amb = (0.7f + 0.3f * nr.y) *
                            (0.2f + 0.8f * clamp_float(0.05f * log(trap), 0.0f, 1.0f));
                float sdw = softshadow(pos + .001f * nr, sd, 16.f);
                float dif = clamp_float(dot_vec3(sd, nr), 0.f, 1.f) * sdw;
                float spe = pow(clamp_float(dot_vec3(nr, hal), 0.f, 1.f), gloss) * dif;

                vec3 lin(0.f);
                lin += ambc * (.05f + .95f * amb);
                lin += sc * dif * 0.8f;
                col = vec3(col.x * lin.x, col.y * lin.y, col.z * lin.z);

                col = pow_vec3(col, vec3(.7f, .9f, 1.f));
                col += spe * 0.8f;
            }

            col = clamp_vec3(pow_vec3(col, vec3(.4545f)), 0.f, 1.f);
            fcol_r += col.x;
            fcol_g += col.y;
            fcol_b += col.z;
        }
    }

    // average
    fcol_r /= (float)(d_AA * d_AA);
    fcol_g /= (float)(d_AA * d_AA);
    fcol_b /= (float)(d_AA * d_AA);

    // convert to unsigned char and write as uchar4 for vectorized memory access
    int idx = (i * width + j) * 4;
    uchar4 pixel = make_uchar4(
        (unsigned char)(fcol_r * 255.0f),
        (unsigned char)(fcol_g * 255.0f),
        (unsigned char)(fcol_b * 255.0f),
        255
    );
    // 使用 uchar4* 指標進行向量化寫入（一次寫入 4 bytes）
    reinterpret_cast<uchar4*>(d_image)[i * width + j] = pixel;
}

// Tile-based Dynamic Work Queue kernel - optimized version
// Reduces atomic operations by fetching tiles instead of individual pixels
// Benefits:
// 1. Reduces atomicAdd calls from (width*height) to (tiles_x*tiles_y)
// 2. Better cache locality - threads in a block process nearby pixels
// 3. Better ray coherence - adjacent rays likely hit similar geometry
// 修正後的 Tile-based 動態工作佇列核心
// __launch_bounds__ 提示編譯器優化：每個 block 256 threads, 每個 SM 最少 2 blocks
__global__ void __launch_bounds__(256, 2)
render_kernel_dynamic(
    unsigned char* d_image,
    unsigned int width,
    unsigned int height,
    unsigned int* d_task_counter)
{
    const int TILE_SIZE = 16;
    const unsigned int tiles_x = (width + TILE_SIZE - 1) / TILE_SIZE;
    const unsigned int tiles_y = (height + TILE_SIZE - 1) / TILE_SIZE;
    const unsigned int total_tiles = tiles_x * tiles_y;

    // 使用共享記憶體來儲存 Block 領取的 tile_id
    __shared__ unsigned int shared_tile_id;
    
    // 優化：使用共享記憶體快取相機向量（所有 threads 共用）
    __shared__ vec3 shared_cf, shared_cs, shared_cu;
    
    // 只讓第一個 thread 計算相機向量
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        vec3 ro = d_camera_pos;
        vec3 ta = d_target_pos;
        shared_cf = normalize_vec3(ta - ro);
        shared_cs = normalize_vec3(cross_vec3(shared_cf, vec3(0., 1., 0.)));
        shared_cu = normalize_vec3(cross_vec3(shared_cs, shared_cf));
    }
    __syncthreads();

    // 每個 Block 進入一個迴圈，不斷領取新的 tile 任務
    while (true) {
        // KEY FIX: 只讓 Block 中的 (0,0) 執行緒去領取任務 ID
        // 確保只有一個 thread 執行 atomicAdd
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            shared_tile_id = atomicAdd(d_task_counter, 1);
        }

        // 2. 同步 Block 中的所有執行緒，確保它們都看到了同一個 shared_tile_id
        __syncthreads();

        // 3. 如果所有 tile 都已經被分配完畢，則整個 Block 退出
        if (shared_tile_id >= total_tiles) {
            break;
        }

        // 將一維的 tile_id 解碼回二維的 tile 座標
        int tile_x = shared_tile_id % tiles_x;
        int tile_y = shared_tile_id / tiles_x;

        // KEY FIX: 直接使用 2D 的 threadIdx，不再需要 % 和 /
        // 這樣可以保證同一個 warp 內的 thread 存取連續記憶體
        int local_x = threadIdx.x;  // 0-15
        int local_y = threadIdx.y;  // 0-15

        // 計算出最終要渲染的像素的全域座標 (i, j)
        int j = tile_x * TILE_SIZE + local_x; // column
        int i = tile_y * TILE_SIZE + local_y; // row

        // 邊界檢查，防止寫入越界
        // 注意：不使用 continue，而是用 if-else 來避免 __syncthreads 死結
        if (i < height && j < width) {

        // ========== 渲染邏輯 (與原始程式碼完全相同) ==========
        float fcol_r = 0.0f;
        float fcol_g = 0.0f;
        float fcol_b = 0.0f;

        // anti aliasing - 提示編譯器完全展開這個小迴圈（3x3=9 次迭代）
        #pragma unroll
        for (int m = 0; m < d_AA; ++m) {
            #pragma unroll
            for (int n = 0; n < d_AA; ++n) {
                vec2 p = vec2(j, i) + vec2(m, n) / (float)d_AA;

                vec2 uv = vec2((-d_iResolution.x + 2.f * p.x) / d_iResolution.y,
                              (-d_iResolution.y + 2.f * p.y) / d_iResolution.y);
                uv.y *= -1;

                // create camera - 使用共享記憶體中的相機向量
                vec3 ro = d_camera_pos;
                vec3 rd = normalize_vec3(uv.x * shared_cs + uv.y * shared_cu + d_FOV * shared_cf);

                float trap;
                int objID;
                float d = trace(ro, rd, trap, objID);

                vec3 col(0.f);
                vec3 sd = normalize_vec3(d_camera_pos);
                vec3 sc = vec3(1.f, .9f, .717f);

                if (d < 0.f) {
                    col = vec3(0.f);
                } else {
                    vec3 pos = ro + rd * d;
                    vec3 nr = calcNor(pos);
                    vec3 hal = normalize_vec3(sd - rd);
                    col = pal(trap - .4f, vec3(.5f), vec3(.5f), vec3(1.f), vec3(.0f, .1f, .2f));
                    vec3 ambc = vec3(0.3f);
                    float gloss = 32.f;
                    float amb = (0.7f + 0.3f * nr.y) *
                                (0.2f + 0.8f * clamp_float(0.05f * log(trap), 0.0f, 1.0f));
                    float sdw = softshadow(pos + .001f * nr, sd, 16.f);
                    float dif = clamp_float(dot_vec3(sd, nr), 0.f, 1.f) * sdw;
                    float spe = pow(clamp_float(dot_vec3(nr, hal), 0.f, 1.f), gloss) * dif;
                    vec3 lin(0.f);
                    lin += ambc * (.05f + .95f * amb);
                    lin += sc * dif * 0.8f;
                    col = vec3(col.x * lin.x, col.y * lin.y, col.z * lin.z);
                    col = pow_vec3(col, vec3(.7f, .9f, 1.f));
                    col += spe * 0.8f;
                }
                col = clamp_vec3(pow_vec3(col, vec3(.4545f)), 0.f, 1.f);
                fcol_r += col.x;
                fcol_g += col.y;
                fcol_b += col.z;
            }
        }

        fcol_r /= (float)(d_AA * d_AA);
        fcol_g /= (float)(d_AA * d_AA);
        fcol_b /= (float)(d_AA * d_AA);

        // 使用向量化寫入優化記憶體存取
        uchar4 pixel = make_uchar4(
            (unsigned char)(fcol_r * 255.0f),
            (unsigned char)(fcol_g * 255.0f),
            (unsigned char)(fcol_b * 255.0f),
            255
        );
        reinterpret_cast<uchar4*>(d_image)[i * width + j] = pixel;
        }
        // Out-of-bounds threads do nothing but must reach __syncthreads
    }
}

// Wrapper kernel for backward compatibility
__global__ void render_kernel(unsigned char* d_image, unsigned int width, unsigned int height) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= height || j >= width) return;
    
    // Call the static version
    // (In practice, this will be replaced by the dynamic version in main)
}

// save raw_image to PNG file
void write_png(const char* filename, unsigned char* raw_image, unsigned int width, unsigned int height) {
    unsigned error = lodepng_encode32_file(filename, raw_image, width, height);
    if (error) printf("png error %u: %s\n", error, lodepng_error_text(error));
}

int main(int argc, char** argv) {
    assert(argc == 10);

    // init arguments
    vec3 camera_pos = vec3(atof(argv[1]), atof(argv[2]), atof(argv[3]));
    vec3 target_pos = vec3(atof(argv[4]), atof(argv[5]), atof(argv[6]));
    unsigned int width = atoi(argv[7]);
    unsigned int height = atoi(argv[8]);
    vec2 iResolution = vec2(width, height);

    // copy constants to device
    cudaMemcpyToSymbol(d_camera_pos, &camera_pos, sizeof(vec3));
    cudaMemcpyToSymbol(d_target_pos, &target_pos, sizeof(vec3));
    cudaMemcpyToSymbol(d_iResolution, &iResolution, sizeof(vec2));

    // allocate device memory
    unsigned char* d_image;
    size_t image_size = width * height * 4 * sizeof(unsigned char);
    cudaMalloc(&d_image, image_size);

    // Load balancing mode:
    // 0 = Static (original 2D grid mapping)
    // 1 = Dynamic (work queue with atomic counter)
    int lb_mode = 1;  // Default: dynamic load balancing (ALWAYS ENABLED)
    
    // Optional: Environment variable to disable if needed
    char* env_lb = getenv("DISABLE_LB");
    if (env_lb != NULL && atoi(env_lb) == 1) {
        lb_mode = 0;  // Disable dynamic load balancing
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    if (lb_mode == 0) {
        // Static 2D grid mapping (original method)
        dim3 blockDim(16, 16);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                     (height + blockDim.y - 1) / blockDim.y);
        
        render_kernel_static<<<gridDim, blockDim>>>(d_image, width, height);
        
    } // 在 main 函式中...
    else {
        // Tile-based dynamic work queue (optimized)
        const int TILE_SIZE = 16;
        unsigned int tiles_x = (width + TILE_SIZE - 1) / TILE_SIZE;
        unsigned int tiles_y = (height + TILE_SIZE - 1) / TILE_SIZE;
        unsigned int total_tiles = tiles_x * tiles_y;
    
        unsigned int* d_task_counter;
        cudaMalloc(&d_task_counter, sizeof(unsigned int));
        cudaMemset(d_task_counter, 0, sizeof(unsigned int));
    
    // **** 優化啟動配置以提升 GPU 佔用率 ****

    // 1. 查詢 GPU 屬性
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        int sm_count = prop.multiProcessorCount;
        int max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    
    // 2. 計算最佳啟動參數
    // TILE_SIZE x TILE_SIZE = 256 threads per block
    // 每個 SM 可以容納更多 blocks 以隱藏延遲
        dim3 blockDim(TILE_SIZE, TILE_SIZE);  // 256 threads/block
        
        // 計算每個 SM 可同時執行的 block 數量
        int blocks_per_sm = max_threads_per_sm / (TILE_SIZE * TILE_SIZE);
        
        // 啟動比實際 SM 更多的 blocks，確保工作飽和
        // 使用 2-4 倍的 oversubscription 以隱藏記憶體延遲
        int oversubscription = 3;
        int gridSize = min(sm_count * blocks_per_sm * oversubscription, 2048);

        printf("GPU: %d SMs, %d blocks/SM, launching %d blocks (%.1fx tiles)\n", 
               sm_count, blocks_per_sm, gridSize, (float)gridSize / total_tiles);
    
        render_kernel_dynamic<<<gridSize, blockDim>>>(d_image, width, height, d_task_counter);
    
        cudaFree(d_task_counter);
    }
    
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Print timing info
    printf("Render time: %.2f ms (mode=%s)\n", milliseconds, 
           lb_mode == 0 ? "static" : "dynamic");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return 1;
    }

    // copy result back to host
    unsigned char* raw_image = new unsigned char[width * height * 4];
    cudaMemcpy(raw_image, d_image, image_size, cudaMemcpyDeviceToHost);

    // save image
    write_png(argv[9], raw_image, width, height);

    // cleanup
    cudaFree(d_image);
    delete[] raw_image;

    return 0;
}