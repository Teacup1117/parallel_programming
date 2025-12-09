#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <array>
#include <tuple>
#include <cassert>
#include <atomic>
#include <omp.h>
#include <mpi.h>

#include "sift.hpp"
#include "image.hpp"



ScaleSpacePyramid generate_gaussian_pyramid(const Image& img, float sigma_min,
                                            int num_octaves, int scales_per_octave)
{
    // assume initial sigma is 1.0 (after resizing) and smooth
    // the image with sigma_diff to reach requried base_sigma
    float base_sigma = sigma_min / MIN_PIX_DIST;
    Image base_img = img.resize(img.width*2, img.height*2, Interpolation::BILINEAR);
    float sigma_diff = std::sqrt(base_sigma*base_sigma - 1.0f);
    base_img = gaussian_blur(base_img, sigma_diff);

    int imgs_per_octave = scales_per_octave + 3;

    // determine sigma values for bluring
    float k = std::pow(2, 1.0/scales_per_octave);
    std::vector<float> sigma_vals {base_sigma};
    for (int i = 1; i < imgs_per_octave; i++) {
        float sigma_prev = base_sigma * std::pow(k, i-1);
        float sigma_total = k * sigma_prev;
        sigma_vals.push_back(std::sqrt(sigma_total*sigma_total - sigma_prev*sigma_prev));
    }

    // create a scale space pyramid of gaussian images
    // images in each octave are half the size of images in the previous one
    ScaleSpacePyramid pyramid = {
        num_octaves,
        imgs_per_octave,
        std::vector<std::vector<Image>>(num_octaves)
    };
    
    // 預先分配所有空間並初始化
    for (int i = 0; i < num_octaves; i++) {
        pyramid.octaves[i].resize(imgs_per_octave);
    }
    
    pyramid.octaves[0][0] = std::move(base_img);
    
    // ===== 真正的流水線並行：使用 OpenMP 任務依賴 =====
    // 策略：明確指定任務之間的依賴關係，讓 OpenMP 運行時自動調度
    // 這允許不同 octave 的計算重疊執行
    
    int critical_scale = imgs_per_octave - 3;
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            // Octave 0 的所有 scales（串行處理，但立即開始）
            for (int j = 1; j < imgs_per_octave; j++) {
                Image* prev = &pyramid.octaves[0][j-1];
                Image* curr = &pyramid.octaves[0][j];
                float sig = sigma_vals[j];
                
                #pragma omp task depend(in: prev[0]) depend(out: curr[0]) shared(pyramid, sigma_vals) firstprivate(j, sig)
                {
                    pyramid.octaves[0][j] = gaussian_blur(pyramid.octaves[0][j-1], sig);
                }
            }
            
            // 其他 octaves：一旦依賴準備好就開始
            for (int i = 1; i < num_octaves; i++) {
                Image* critical_img = &pyramid.octaves[i-1][critical_scale];
                Image* base_img_new = &pyramid.octaves[i][0];
                
                // 創建新 octave 的 base（依賴前一個 octave 的 critical_scale）
                #pragma omp task depend(in: critical_img[0]) depend(out: base_img_new[0]) shared(pyramid) firstprivate(i, critical_scale)
                {
                    pyramid.octaves[i][0] = pyramid.octaves[i-1][critical_scale].resize(
                        pyramid.octaves[i-1][critical_scale].width / 2,
                        pyramid.octaves[i-1][critical_scale].height / 2,
                        Interpolation::NEAREST);
                }
                
                // 這個 octave 的所有後續 scales
                for (int j = 1; j < imgs_per_octave; j++) {
                    Image* prev = &pyramid.octaves[i][j-1];
                    Image* curr = &pyramid.octaves[i][j];
                    float sig = sigma_vals[j];
                    
                    #pragma omp task depend(in: prev[0]) depend(out: curr[0]) shared(pyramid, sigma_vals) firstprivate(i, j, sig)
                    {
                        pyramid.octaves[i][j] = gaussian_blur(pyramid.octaves[i][j-1], sig);
                    }
                }
            }
            
            // 等待所有任務完成
            #pragma omp taskwait
        }
    }
    
    return pyramid;
}

// generate pyramid of difference of gaussians (DoG) images
ScaleSpacePyramid generate_dog_pyramid(const ScaleSpacePyramid& img_pyramid)
{
    ScaleSpacePyramid dog_pyramid = {
        img_pyramid.num_octaves,
        img_pyramid.imgs_per_octave - 1,
        std::vector<std::vector<Image>>(img_pyramid.num_octaves)
    };
    
    // 預先分配所有 octave 的空間
    for (int i = 0; i < dog_pyramid.num_octaves; i++) {
        dog_pyramid.octaves[i].reserve(dog_pyramid.imgs_per_octave);
        for (int j = 0; j < dog_pyramid.imgs_per_octave; j++) {
            dog_pyramid.octaves[i].emplace_back(img_pyramid.octaves[i][0].width, 
                                                 img_pyramid.octaves[i][0].height, 1);
        }
    }
    
    // 使用 collapse(2) 並行處理所有 DoG 圖像
    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (int i = 0; i < dog_pyramid.num_octaves; i++) {
        for (int j = 0; j < dog_pyramid.imgs_per_octave; j++) {
            const float* src1 = img_pyramid.octaves[i][j+1].data;
            const float* src0 = img_pyramid.octaves[i][j].data;
            float* dst = dog_pyramid.octaves[i][j].data;
            int size = dog_pyramid.octaves[i][j].size;
            
            // 使用 SIMD 優化的像素運算
            #pragma omp simd
            for (int pix_idx = 0; pix_idx < size; pix_idx++) {
                dst[pix_idx] = src1[pix_idx] - src0[pix_idx];
            }
        }
    }
    return dog_pyramid;
}

bool point_is_extremum(const std::vector<Image>& octave, int scale, int x, int y)
{
    const Image& img = octave[scale];
    const Image& prev = octave[scale-1];
    const Image& next = octave[scale+1];

    bool is_min = true, is_max = true;
    float val = img.get_pixel(x, y, 0), neighbor;

    for (int dx : {-1,0,1}) {
        for (int dy : {-1,0,1}) {
            neighbor = prev.get_pixel(x+dx, y+dy, 0);
            if (neighbor > val) is_max = false;
            if (neighbor < val) is_min = false;

            neighbor = next.get_pixel(x+dx, y+dy, 0);
            if (neighbor > val) is_max = false;
            if (neighbor < val) is_min = false;

            neighbor = img.get_pixel(x+dx, y+dy, 0);
            if (neighbor > val) is_max = false;
            if (neighbor < val) is_min = false;

            if (!is_min && !is_max) return false;
        }
    }
    return true;
}

// fit a quadratic near the discrete extremum,
// update the keypoint (interpolated) extremum value
// and return offsets of the interpolated extremum from the discrete extremum
std::tuple<float, float, float> fit_quadratic(Keypoint& kp,
                                              const std::vector<Image>& octave,
                                              int scale)
{
    const Image& img = octave[scale];
    const Image& prev = octave[scale-1];
    const Image& next = octave[scale+1];

    float g1, g2, g3;
    float h11, h12, h13, h22, h23, h33;
    int x = kp.i, y = kp.j;

    // gradient 
    g1 = (next.get_pixel(x, y, 0) - prev.get_pixel(x, y, 0)) * 0.5;
    g2 = (img.get_pixel(x+1, y, 0) - img.get_pixel(x-1, y, 0)) * 0.5;
    g3 = (img.get_pixel(x, y+1, 0) - img.get_pixel(x, y-1, 0)) * 0.5;

    // hessian
    h11 = next.get_pixel(x, y, 0) + prev.get_pixel(x, y, 0) - 2*img.get_pixel(x, y, 0);
    h22 = img.get_pixel(x+1, y, 0) + img.get_pixel(x-1, y, 0) - 2*img.get_pixel(x, y, 0);
    h33 = img.get_pixel(x, y+1, 0) + img.get_pixel(x, y-1, 0) - 2*img.get_pixel(x, y, 0);
    h12 = (next.get_pixel(x+1, y, 0) - next.get_pixel(x-1, y, 0)
          -prev.get_pixel(x+1, y, 0) + prev.get_pixel(x-1, y, 0)) * 0.25;
    h13 = (next.get_pixel(x, y+1, 0) - next.get_pixel(x, y-1, 0)
          -prev.get_pixel(x, y+1, 0) + prev.get_pixel(x, y-1, 0)) * 0.25;
    h23 = (img.get_pixel(x+1, y+1, 0) - img.get_pixel(x+1, y-1, 0)
          -img.get_pixel(x-1, y+1, 0) + img.get_pixel(x-1, y-1, 0)) * 0.25;
    
    // invert hessian
    float hinv11, hinv12, hinv13, hinv22, hinv23, hinv33;
    float det = h11*h22*h33 - h11*h23*h23 - h12*h12*h33 + 2*h12*h13*h23 - h13*h13*h22;
    hinv11 = (h22*h33 - h23*h23) / det;
    hinv12 = (h13*h23 - h12*h33) / det;
    hinv13 = (h12*h23 - h13*h22) / det;
    hinv22 = (h11*h33 - h13*h13) / det;
    hinv23 = (h12*h13 - h11*h23) / det;
    hinv33 = (h11*h22 - h12*h12) / det;

    // find offsets of the interpolated extremum from the discrete extremum
    float offset_s = -hinv11*g1 - hinv12*g2 - hinv13*g3;
    float offset_x = -hinv12*g1 - hinv22*g2 - hinv23*g3;
    float offset_y = -hinv13*g1 - hinv23*g3 - hinv33*g3;

    float interpolated_extrema_val = img.get_pixel(x, y, 0)
                                   + 0.5*(g1*offset_s + g2*offset_x + g3*offset_y);
    kp.extremum_val = interpolated_extrema_val;
    return {offset_s, offset_x, offset_y};
}

bool point_is_on_edge(const Keypoint& kp, const std::vector<Image>& octave, float edge_thresh=C_EDGE)
{
    const Image& img = octave[kp.scale];
    float h11, h12, h22;
    int x = kp.i, y = kp.j;
    h11 = img.get_pixel(x+1, y, 0) + img.get_pixel(x-1, y, 0) - 2*img.get_pixel(x, y, 0);
    h22 = img.get_pixel(x, y+1, 0) + img.get_pixel(x, y-1, 0) - 2*img.get_pixel(x, y, 0);
    h12 = (img.get_pixel(x+1, y+1, 0) - img.get_pixel(x+1, y-1, 0)
          -img.get_pixel(x-1, y+1, 0) + img.get_pixel(x-1, y-1, 0)) * 0.25;

    float det_hessian = h11*h22 - h12*h12;
    float tr_hessian = h11 + h22;
    float edgeness = tr_hessian*tr_hessian / det_hessian;

    if (edgeness > std::pow(edge_thresh+1, 2)/edge_thresh)
        return true;
    else
        return false;
}

void find_input_img_coords(Keypoint& kp, float offset_s, float offset_x, float offset_y,
                                   float sigma_min=SIGMA_MIN,
                                   float min_pix_dist=MIN_PIX_DIST, int n_spo=N_SPO)
{
    kp.sigma = std::pow(2, kp.octave) * sigma_min * std::pow(2, (offset_s+kp.scale)/n_spo);
    kp.x = min_pix_dist * std::pow(2, kp.octave) * (offset_x+kp.i);
    kp.y = min_pix_dist * std::pow(2, kp.octave) * (offset_y+kp.j);
}

bool refine_or_discard_keypoint(Keypoint& kp, const std::vector<Image>& octave,
                                float contrast_thresh, float edge_thresh)
{
    int k = 0;
    bool kp_is_valid = false; 
    while (k++ < MAX_REFINEMENT_ITERS) {
        auto [offset_s, offset_x, offset_y] = fit_quadratic(kp, octave, kp.scale);

        float max_offset = std::max({std::abs(offset_s),
                                     std::abs(offset_x),
                                     std::abs(offset_y)});
        // find nearest discrete coordinates
        kp.scale += std::round(offset_s);
        kp.i += std::round(offset_x);
        kp.j += std::round(offset_y);
        if (kp.scale >= octave.size()-1 || kp.scale < 1)
            break;

        bool valid_contrast = std::abs(kp.extremum_val) > contrast_thresh;
        if (max_offset < 0.6 && valid_contrast && !point_is_on_edge(kp, octave, edge_thresh)) {
            find_input_img_coords(kp, offset_s, offset_x, offset_y);
            kp_is_valid = true;
            break;
        }
    }
    return kp_is_valid;
}

// in sift.cpp
#include <vector>

std::vector<Keypoint> find_keypoints(const ScaleSpacePyramid& dog_pyramid, float contrast_thresh,
                                     float edge_thresh)
{
    std::vector<Keypoint> keypoints;
    
    // 使用 OpenMP 的 thread-local 容器技術來並行化，避免執行緒競爭
    #pragma omp parallel
    {
        std::vector<Keypoint> local_keypoints;
        // 使用 collapse(2) 將外層的 octave 和 scale 迴圈攤平，為 OpenMP 提供更多可並行的任務
        #pragma omp for schedule(dynamic) collapse(2)
        for (int i = 0; i < dog_pyramid.num_octaves; i++) {
            for (int j = 1; j < dog_pyramid.imgs_per_octave-1; j++) {
                const std::vector<Image>& octave = dog_pyramid.octaves[i];
                const Image& img = octave[j];
                // 內部 x, y 迴圈保持循序，以保證負載平衡
                for (int x = 1; x < img.width-1; x++) {
                    for (int y = 1; y < img.height-1; y++) {
                        if (std::abs(img.get_pixel(x, y, 0)) < 0.8*contrast_thresh) {
                            continue;
                        }
                        if (point_is_extremum(octave, j, x, y)) {
                            Keypoint kp = {x, y, i, j, -1, -1, -1, -1};
                            if (refine_or_discard_keypoint(kp, octave, contrast_thresh, edge_thresh)) {
                                local_keypoints.push_back(kp);
                            }
                        }
                    }
                }
            }
        }
        
        // 所有執行緒計算完畢後，在 critical section 中安全地將本地結果合併到全域容器中
        #pragma omp critical
        {
            keypoints.insert(keypoints.end(), local_keypoints.begin(), local_keypoints.end());
        }
    }
    
    return keypoints;
}

// calculate x and y derivatives for all images in the input pyramid
ScaleSpacePyramid generate_gradient_pyramid(const ScaleSpacePyramid& pyramid)
{
    ScaleSpacePyramid grad_pyramid = {
        pyramid.num_octaves,
        pyramid.imgs_per_octave,
        std::vector<std::vector<Image>>(pyramid.num_octaves)
    };
    
    // 預先分配所有 octave 的空間
    for (int i = 0; i < pyramid.num_octaves; i++) {
        int width = pyramid.octaves[i][0].width;
        int height = pyramid.octaves[i][0].height;
        grad_pyramid.octaves[i].reserve(pyramid.imgs_per_octave);
        for (int j = 0; j < pyramid.imgs_per_octave; j++) {
            grad_pyramid.octaves[i].emplace_back(width, height, 2);
        }
    }
    
    // 使用 collapse(2) 並行處理所有梯度圖像
    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (int i = 0; i < pyramid.num_octaves; i++) {
        for (int j = 0; j < pyramid.imgs_per_octave; j++) {
            Image& grad = grad_pyramid.octaves[i][j];
            const Image& src = pyramid.octaves[i][j];
            
            int w = grad.width;
            int h = grad.height;
            
            // 計算梯度（避免邊界），使用直接內存訪問
            for (int y = 1; y < h-1; y++) {
                for (int x = 1; x < w-1; x++) {
                    float gx = (src.get_pixel(x+1, y, 0) - src.get_pixel(x-1, y, 0)) * 0.5f;
                    float gy = (src.get_pixel(x, y+1, 0) - src.get_pixel(x, y-1, 0)) * 0.5f;
                    int idx = y * w + x;
                    grad.data[idx] = gx;           // channel 0
                    grad.data[w*h + idx] = gy;     // channel 1
                }
            }
        }
    }
    return grad_pyramid;
}

// convolve 6x with box filter
void smooth_histogram(float hist[N_BINS])
{
    float tmp_hist[N_BINS];
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < N_BINS; j++) {
            int prev_idx = (j-1+N_BINS)%N_BINS;
            int next_idx = (j+1)%N_BINS;
            tmp_hist[j] = (hist[prev_idx] + hist[j] + hist[next_idx]) / 3;
        }
        for (int j = 0; j < N_BINS; j++) {
            hist[j] = tmp_hist[j];
        }
    }
}

std::vector<float> find_keypoint_orientations(Keypoint& kp, 
                                              const ScaleSpacePyramid& grad_pyramid,
                                              float lambda_ori, float lambda_desc)
{
    float pix_dist = MIN_PIX_DIST * std::pow(2, kp.octave);
    const Image& img_grad = grad_pyramid.octaves[kp.octave][kp.scale];

    // 預計算常量
    const float sqrt2 = 1.41421356237f;
    float threshold = sqrt2 * lambda_desc * kp.sigma;
    
    // discard kp if too close to image borders 
    float min_dist_from_border = std::min({kp.x, kp.y, pix_dist*img_grad.width-kp.x,
                                           pix_dist*img_grad.height-kp.y});
    if (min_dist_from_border <= threshold) {
        return {};
    }

    float hist[N_BINS] = {};
    int bin;
    float gx, gy, grad_norm, weight, theta;
    float patch_sigma = lambda_ori * kp.sigma;
    float patch_radius = 3 * patch_sigma;
    
    // 預計算循環邊界
    int x_start = std::round((kp.x - patch_radius)/pix_dist);
    int x_end = std::round((kp.x + patch_radius)/pix_dist);
    int y_start = std::round((kp.y - patch_radius)/pix_dist);
    int y_end = std::round((kp.y + patch_radius)/pix_dist);

    // 預計算權重相關常量
    float inv_2sigma2 = -1.0f / (2.0f * patch_sigma * patch_sigma);
    float bin_scale = N_BINS / (2.0f * M_PI);

    // accumulate gradients in orientation histogram
    for (int x = x_start; x <= x_end; x++) {
        float dx = x * pix_dist - kp.x;
        float dx2 = dx * dx;
        for (int y = y_start; y <= y_end; y++) {
            gx = img_grad.get_pixel(x, y, 0);
            gy = img_grad.get_pixel(x, y, 1);
            grad_norm = std::sqrt(gx*gx + gy*gy);
            
            float dy = y * pix_dist - kp.y;
            weight = std::exp((dx2 + dy*dy) * inv_2sigma2);
            
            theta = std::fmod(std::atan2(gy, gx)+2*M_PI, 2*M_PI);
            bin = (int)std::round(bin_scale * theta) % N_BINS;
            hist[bin] += weight * grad_norm;
        }
    }

    smooth_histogram(hist);

    // extract reference orientations
    float ori_thresh = 0.8, ori_max = 0;
    std::vector<float> orientations;
    for (int j = 0; j < N_BINS; j++) {
        if (hist[j] > ori_max) {
            ori_max = hist[j];
        }
    }
    for (int j = 0; j < N_BINS; j++) {
        if (hist[j] >= ori_thresh * ori_max) {
            float prev = hist[(j-1+N_BINS)%N_BINS], next = hist[(j+1)%N_BINS];
            if (prev > hist[j] || next > hist[j])
                continue;
            float theta = 2*M_PI*(j+1)/N_BINS + M_PI/N_BINS*(prev-next)/(prev-2*hist[j]+next);
            orientations.push_back(theta);
        }
    }
    return orientations;
}

void update_histograms(float hist[N_HIST][N_HIST][N_ORI], float x, float y,
                       float contrib, float theta_mn, float lambda_desc)
{
    float x_i, y_j;
    for (int i = 1; i <= N_HIST; i++) {
        x_i = (i-(1+(float)N_HIST)/2) * 2*lambda_desc/N_HIST;
        if (std::abs(x_i-x) > 2*lambda_desc/N_HIST)
            continue;
        for (int j = 1; j <= N_HIST; j++) {
            y_j = (j-(1+(float)N_HIST)/2) * 2*lambda_desc/N_HIST;
            if (std::abs(y_j-y) > 2*lambda_desc/N_HIST)
                continue;
            
            float hist_weight = (1 - N_HIST*0.5/lambda_desc*std::abs(x_i-x))
                               *(1 - N_HIST*0.5/lambda_desc*std::abs(y_j-y));

            for (int k = 1; k <= N_ORI; k++) {
                float theta_k = 2*M_PI*(k-1)/N_ORI;
                float theta_diff = std::fmod(theta_k-theta_mn+2*M_PI, 2*M_PI);
                if (std::abs(theta_diff) >= 2*M_PI/N_ORI)
                    continue;
                float bin_weight = 1 - N_ORI*0.5/M_PI*std::abs(theta_diff);
                hist[i-1][j-1][k-1] += hist_weight*bin_weight*contrib;
            }
        }
    }
}

void hists_to_vec(float histograms[N_HIST][N_HIST][N_ORI], std::array<uint8_t, 128>& feature_vec)
{
    int size = N_HIST*N_HIST*N_ORI;
    float *hist = reinterpret_cast<float *>(histograms);

    // 移除並行化：對於128個元素，順序執行更快
    float norm = 0;
    for (int i = 0; i < size; i++) {
        norm += hist[i] * hist[i];
    }
    norm = std::sqrt(norm);
    
    float norm_threshold = 0.2f * norm;
    float norm2 = 0;
    for (int i = 0; i < size; i++) {
        hist[i] = std::min(hist[i], norm_threshold);
        norm2 += hist[i] * hist[i];
    }
    norm2 = std::sqrt(norm2);
    
    float inv_norm2 = 512.0f / norm2;
    for (int i = 0; i < size; i++) {
        float val = std::floor(hist[i] * inv_norm2);
        feature_vec[i] = std::min((int)val, 255);
    }
}

void compute_keypoint_descriptor(Keypoint& kp, float theta,
                                 const ScaleSpacePyramid& grad_pyramid,
                                 float lambda_desc)
{
    float pix_dist = MIN_PIX_DIST * std::pow(2, kp.octave);
    const Image& img_grad = grad_pyramid.octaves[kp.octave][kp.scale];
    float histograms[N_HIST][N_HIST][N_ORI] = {0};

    // 預計算三角函數和常量
    float cos_t = std::cos(theta), sin_t = std::sin(theta);
    float inv_sigma = 1.0f / kp.sigma;
    float patch_sigma = lambda_desc * kp.sigma;
    float inv_2sigma2 = -1.0f / (2.0f * patch_sigma * patch_sigma);
    
    //find start and end coords for loops over image patch
    float half_size = std::sqrt(2)*lambda_desc*kp.sigma*(N_HIST+1.)/N_HIST;
    int x_start = std::round((kp.x-half_size) / pix_dist);
    int x_end = std::round((kp.x+half_size) / pix_dist);
    int y_start = std::round((kp.y-half_size) / pix_dist);
    int y_end = std::round((kp.y+half_size) / pix_dist);

    float max_dist = lambda_desc*(N_HIST+1.)/N_HIST;
    
    //accumulate samples into histograms
    for (int m = x_start; m <= x_end; m++) {
        float m_offset = m * pix_dist - kp.x;
        for (int n = y_start; n <= y_end; n++) {
            float n_offset = n * pix_dist - kp.y;
            
            // find normalized coords w.r.t. kp position and reference orientation
            float x = (m_offset * cos_t + n_offset * sin_t) * inv_sigma;
            float y = (-m_offset * sin_t + n_offset * cos_t) * inv_sigma;

            // verify (x, y) is inside the description patch
            if (std::max(std::abs(x), std::abs(y)) > max_dist)
                continue;

            float gx = img_grad.get_pixel(m, n, 0), gy = img_grad.get_pixel(m, n, 1);
            float theta_mn = std::fmod(std::atan2(gy, gx)-theta+4*M_PI, 2*M_PI);
            float grad_norm = std::sqrt(gx*gx + gy*gy);
            float weight = std::exp((m_offset*m_offset + n_offset*n_offset) * inv_2sigma2);
            float contribution = weight * grad_norm;

            update_histograms(histograms, x, y, contribution, theta_mn, lambda_desc);
        }
    }

    // build feature vector (descriptor) from histograms
    hists_to_vec(histograms, kp.descriptor);
}

std::vector<Keypoint> find_keypoints_and_descriptors(const Image& img, float sigma_min,
                                                     int num_octaves, int scales_per_octave, 
                                                     float contrast_thresh, float edge_thresh, 
                                                     float lambda_ori, float lambda_desc)
{
    assert(img.channels == 1 || img.channels == 3);

    const Image& input = img.channels == 1 ? img : rgb_to_grayscale(img);
    ScaleSpacePyramid gaussian_pyramid = generate_gaussian_pyramid(input, sigma_min, num_octaves,
                                                                   scales_per_octave);
    ScaleSpacePyramid dog_pyramid = generate_dog_pyramid(gaussian_pyramid);
    std::vector<Keypoint> tmp_kps = find_keypoints(dog_pyramid, contrast_thresh, edge_thresh);
    ScaleSpacePyramid grad_pyramid = generate_gradient_pyramid(gaussian_pyramid);
    
    std::vector<Keypoint> kps;

    #pragma omp parallel
    {
        // 每個 thread 預留空間，減少記憶體重新分配
        std::vector<Keypoint> local_kps;
        local_kps.reserve(tmp_kps.size() * 2 / omp_get_num_threads()); // 預估空間
        
        #pragma omp for schedule(dynamic) nowait
        for (int i = 0; i < tmp_kps.size(); i++) {
            Keypoint& kp_tmp = tmp_kps[i];
            std::vector<float> orientations = find_keypoint_orientations(kp_tmp, grad_pyramid,
                                                                         lambda_ori, lambda_desc);
            for (float theta : orientations) {
                Keypoint kp = kp_tmp;
                compute_keypoint_descriptor(kp, theta, grad_pyramid, lambda_desc);
                local_kps.push_back(kp);
            }
        }
        
        // 單次 critical section 進行合併
        #pragma omp critical
        {
            kps.insert(kps.end(), local_kps.begin(), local_kps.end());
        }
    }
    return kps;
}

float euclidean_dist(std::array<uint8_t, 128>& a, std::array<uint8_t, 128>& b)
{
    // 移除並行化：對於128個元素，順序執行更快，且避免sqrt
    int dist = 0;
    for (int i = 0; i < 128; i++) {
        int di = (int)a[i] - b[i];
        dist += di * di;
    }
    return std::sqrt((float)dist);
}

Image draw_keypoints(const Image& img, const std::vector<Keypoint>& kps)
{
    Image res(img);
    if (img.channels == 1) {
        res = grayscale_to_rgb(res);
    }
    for (auto& kp : kps) {
        draw_point(res, kp.x, kp.y, 5);
    }
    return res;
}

// MPI 版本 - 完整實現：任務分配、結果回收、排序保證一致性

// ===================================================================================
// ================ 請用下面的程式碼完整替換您現有的 mpi 函式 =======================
// ===================================================================================
std::vector<Keypoint> find_keypoints_and_descriptors_mpi(const Image& img, int rank, int size,
                                                         float sigma_min,
                                                         int num_octaves, int scales_per_octave, 
                                                         float contrast_thresh, float edge_thresh, 
                                                         float lambda_ori, float lambda_desc)
{
    assert(img.channels == 1 || img.channels == 3);
    const Image& input = img.channels == 1 ? img : rgb_to_grayscale(img);
    
    // 階段 1: 所有進程並行計算 Pyramids (保留)
    ScaleSpacePyramid gaussian_pyramid = generate_gaussian_pyramid(input, sigma_min, num_octaves, scales_per_octave);
    ScaleSpacePyramid dog_pyramid = generate_dog_pyramid(gaussian_pyramid);
    ScaleSpacePyramid grad_pyramid = generate_gradient_pyramid(gaussian_pyramid);

    // 階段 2: 【修正】只有 Rank 0 偵測關鍵點，避免重複工作
    std::vector<Keypoint> tmp_kps;
    if (rank == 0) {
        tmp_kps = find_keypoints(dog_pyramid, contrast_thresh, edge_thresh);
    }

    // 階段 3: 【關鍵通訊】Rank 0 將任務總數和內容廣播給所有 Rank
    int total_kps = 0;
    if (rank == 0) {
        total_kps = tmp_kps.size();
    }
    // 廣播任務總數
    MPI_Bcast(&total_kps, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 其他 Rank 根據總數準備接收容器
    if (rank != 0) {
        tmp_kps.resize(total_kps);
    }
    // 廣播任務內容 (所有關鍵點的資訊)
    MPI_Bcast(tmp_kps.data(), total_kps * sizeof(Keypoint), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    // 階段 4: 【核心邏輯 - 分工】所有 Ranks 計算各自的任務範圍
    int kps_per_rank = (total_kps + size - 1) / size; // 向上取整，保證所有任務都被覆蓋
    int start_idx = rank * kps_per_rank;
    int end_idx = std::min(start_idx + kps_per_rank, total_kps);
    
    std::vector<Keypoint> local_kps; // 每個 rank 的本地結果容器

    // 【核心邏輯 - 執行】所有 Ranks 都執行這段迴圈，但只處理自己的分片！
    // 這裡沒有 if (rank == 0) !!!
    #pragma omp parallel
    {
        std::vector<Keypoint> thread_local_kps;
        if (end_idx > start_idx) { // 確保有工作才預分配
            thread_local_kps.reserve((end_idx - start_idx) * 2 / omp_get_num_threads() + 1);
        }

        #pragma omp for schedule(dynamic) nowait
        // 注意！迴圈範圍是 start_idx 到 end_idx
        for (int i = start_idx; i < end_idx; i++) {
            Keypoint& kp_tmp = tmp_kps[i];
            std::vector<float> orientations = find_keypoint_orientations(kp_tmp, grad_pyramid, lambda_ori, lambda_desc);
            for (float theta : orientations) {
                Keypoint kp = kp_tmp;
                compute_keypoint_descriptor(kp, theta, grad_pyramid, lambda_desc);
                thread_local_kps.push_back(kp);
            }
        }
        
        #pragma omp critical
        {
            local_kps.insert(local_kps.end(), thread_local_kps.begin(), thread_local_kps.end());
        }
    }

    // 階段 5: 【關鍵通訊】結果收集到 Rank 0
    if (rank == 0) {
        std::vector<Keypoint> all_kps = local_kps; // 先放入自己的結果

        // 從其他所有 Rank 接收它們的結果
        for (int source_rank = 1; source_rank < size; ++source_rank) {
            int incoming_count;
            MPI_Status status;
            MPI_Recv(&incoming_count, 1, MPI_INT, source_rank, 0, MPI_COMM_WORLD, &status);
            
            if (incoming_count > 0) {
                std::vector<Keypoint> incoming_kps(incoming_count);
                MPI_Recv(incoming_kps.data(), incoming_count * sizeof(Keypoint), MPI_BYTE, source_rank, 1, MPI_COMM_WORLD, &status);
                all_kps.insert(all_kps.end(), incoming_kps.begin(), incoming_kps.end());
            }
        }
        
        // 【關鍵處理】排序！確保每次執行的輸出順序都和 golden file 一致
        std::sort(all_kps.begin(), all_kps.end(), [](const Keypoint& a, const Keypoint& b) {
            if (a.octave != b.octave) return a.octave < b.octave;
            if (a.scale != b.scale) return a.scale < b.scale;
            if (a.y != b.y) return a.y < b.y;
            return a.x < b.x;
        });
        
        return all_kps; // Rank 0 返回最終的完整結果

    } else {
        // 其他 Ranks 將自己的結果發送給 Rank 0
        int local_count = local_kps.size();
        MPI_Send(&local_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        if (local_count > 0) {
            MPI_Send(local_kps.data(), local_count * sizeof(Keypoint), MPI_BYTE, 0, 1, MPI_COMM_WORLD);
        }
        return {}; // 其他 ranks 返回空 vector
    }
}