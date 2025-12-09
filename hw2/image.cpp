#include <cmath>
#include <iostream>
#include <cassert>
#include <utility>
#include <omp.h>

#include "image.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

Image::Image(std::string file_path)
{
    unsigned char *img_data = stbi_load(file_path.c_str(), &width, &height, &channels, 0);
    if (img_data == nullptr) {
        const char *error_msg = stbi_failure_reason();
        std::cerr << "Failed to load image: " << file_path.c_str() << "\n";
        std::cerr << "Error msg (stb_image): " << error_msg << "\n";
        std::exit(1);
    }

    size = width * height * channels;
    data = new float[size]; 
    
    // 使用 dynamic 調度，在多核服務器上更好
    #pragma omp parallel for schedule(dynamic, 64)
    for (int idx = 0; idx < size; idx++) {
        int c = idx / (height * width);
        int remainder = idx % (height * width);
        int y = remainder / width;
        int x = remainder % width;
        int src_idx = y*width*channels + x*channels + c;
        data[idx] = img_data[src_idx] / 255.f;
    }
    if (channels == 4)
        channels = 3; //ignore alpha channel
    stbi_image_free(img_data);
}

Image::Image(int w, int h, int c)
    :width {w},
     height {h},
     channels {c},
     size {w*h*c},
     data {new float[w*h*c]()}
{
}

Image::Image()
    :width {0},
     height {0},
     channels {0},
     size {0},
     data {nullptr} 
{
}

Image::~Image()
{
    delete[] this->data;
}

Image::Image(const Image& other)
    :width {other.width},
     height {other.height},
     channels {other.channels},
     size {other.size},
     data {new float[other.size]}
{
    //std::cout << "copy constructor\n";
    #pragma omp parallel for
    for (int i = 0; i < size; i++)
        data[i] = other.data[i];
}

Image& Image::operator=(const Image& other)
{
    if (this != &other) {
        delete[] data;
        //std::cout << "copy assignment\n";
        width = other.width;
        height = other.height;
        channels = other.channels;
        size = other.size;
        data = new float[other.size];
        #pragma omp parallel for
        for (int i = 0; i < other.size; i++)
            data[i] = other.data[i];
    }
    return *this;
}

Image::Image(Image&& other)
    :width {other.width},
     height {other.height},
     channels {other.channels},
     size {other.size},
     data {other.data}
{
    //std::cout << "move constructor\n";
    other.data = nullptr;
    other.size = 0;
}

Image& Image::operator=(Image&& other)
{
    //std::cout << "move assignment\n";
    delete[] data;
    data = other.data;
    width = other.width;
    height = other.height;
    channels = other.channels;
    size = other.size;

    other.data = nullptr;
    other.size = 0;
    return *this;
}

//save image as jpg file
bool Image::save(std::string file_path)
{
    unsigned char *out_data = new unsigned char[width*height*channels]; 
    
    // 使用 dynamic 調度，在多核服務器上更好
    #pragma omp parallel for schedule(dynamic, 64)
    for (int idx = 0; idx < width*height*channels; idx++) {
        int c = idx / (height * width);
        int remainder = idx % (height * width);
        int y = remainder / width;
        int x = remainder % width;
        int dst_idx = y*width*channels + x*channels + c;
        out_data[dst_idx] = std::roundf(data[idx] * 255.f);
    }
    bool success = stbi_write_jpg(file_path.c_str(), width, height, channels, out_data, 100);
    if (!success)
        std::cerr << "Failed to save image: " << file_path << "\n";

    delete[] out_data;
    return true;
}

void Image::set_pixel(int x, int y, int c, float val)
{
    if (x >= width || x < 0 || y >= height || y < 0 || c >= channels || c < 0) {
        std::cerr << "set_pixel() error: Index out of bounds.\n";
        std::exit(1);
    }
    data[c*width*height + y*width + x] = val;
}

float Image::get_pixel(int x, int y, int c) const
{
    if (x < 0)
        x = 0;
    if (x >= width)
        x = width - 1;
    if (y < 0)
        y = 0;
    if (y >= height)
        y = height - 1;
    return data[c*width*height + y*width + x];
}

void Image::clamp()
{
    int total_size = width * height * channels;
    // 使用 dynamic 調度，適合多核服務器
    #pragma omp parallel for schedule(dynamic, 1024)
    for (int i = 0; i < total_size; i++) {
        float val = data[i];
        data[i] = (val > 1.0f) ? 1.0f : ((val < 0.0f) ? 0.0f : val);
    }
}

//map coordinate from 0-current_max range to 0-new_max range
float map_coordinate(float new_max, float current_max, float coord)
{
    float a = new_max / current_max;
    float b = -0.5 + a*0.5;
    return a*coord + b;
}

Image Image::resize(int new_w, int new_h, Interpolation method) const
{
    Image resized(new_w, new_h, this->channels);
    
    // 使用 dynamic 調度，適合多核服務器
    #pragma omp parallel for schedule(dynamic, 32)
    for (int idx = 0; idx < new_w * new_h * resized.channels; idx++) {
        int c = idx / (new_h * new_w);
        int remainder = idx % (new_h * new_w);
        int y = remainder / new_w;
        int x = remainder % new_w;
        
        float old_x = map_coordinate(this->width, new_w, x);
        float old_y = map_coordinate(this->height, new_h, y);
        float value;
        if (method == Interpolation::BILINEAR)
            value = bilinear_interpolate(*this, old_x, old_y, c);
        else if (method == Interpolation::NEAREST)
            value = nn_interpolate(*this, old_x, old_y, c);
        resized.data[idx] = value;
    }
    return resized;
}

float bilinear_interpolate(const Image& img, float x, float y, int c)
{
    float p1, p2, p3, p4, q1, q2;
    float x_floor = std::floor(x), y_floor = std::floor(y);
    float x_ceil = x_floor + 1, y_ceil = y_floor + 1;
    p1 = img.get_pixel(x_floor, y_floor, c);
    p2 = img.get_pixel(x_ceil, y_floor, c);
    p3 = img.get_pixel(x_floor, y_ceil, c);
    p4 = img.get_pixel(x_ceil, y_ceil, c);
    q1 = (y_ceil-y)*p1 + (y-y_floor)*p3;
    q2 = (y_ceil-y)*p2 + (y-y_floor)*p4;
    return (x_ceil-x)*q1 + (x-x_floor)*q2;
}

float nn_interpolate(const Image& img, float x, float y, int c)
{
    return img.get_pixel(std::round(x), std::round(y), c);
}

Image rgb_to_grayscale(const Image& img)
{
    assert(img.channels == 3);
    Image gray(img.width, img.height, 1);
    
    // 使用 dynamic 調度和直接內存訪問
    int total_pixels = img.width * img.height;
    #pragma omp parallel for schedule(dynamic, 256)
    for (int idx = 0; idx < total_pixels; idx++) {
        int y = idx / img.width;
        int x = idx % img.width;
        int base_idx = y * img.width + x;
        float red = img.data[base_idx];
        float green = img.data[img.width * img.height + base_idx];
        float blue = img.data[2 * img.width * img.height + base_idx];
        gray.data[base_idx] = 0.299f*red + 0.587f*green + 0.114f*blue;
    }
    return gray;
}

Image grayscale_to_rgb(const Image& img)
{
    assert(img.channels == 1);
    Image rgb(img.width, img.height, 3);
    
    // 使用 dynamic 調度和直接內存訪問
    int total_pixels = img.width * img.height;
    #pragma omp parallel for schedule(dynamic, 256)
    for (int idx = 0; idx < total_pixels; idx++) {
        float gray_val = img.data[idx];
        rgb.data[idx] = gray_val;                              // R
        rgb.data[total_pixels + idx] = gray_val;               // G
        rgb.data[2 * total_pixels + idx] = gray_val;           // B
    }
    return rgb;
}

// separable 2D gaussian blur for 1 channel image
Image gaussian_blur(const Image& img, float sigma)
{
    assert(img.channels == 1);

    int kernel_size = static_cast<int>(std::ceil(6 * sigma));
    if (kernel_size % 2 == 0) kernel_size++;
    int center = kernel_size / 2;

    // 計算高斯核
    std::vector<float> kernel(kernel_size);
    float sum = 0.0f;
    for (int i = 0; i < kernel_size; ++i) {
        float x = static_cast<float>(i - center);
        kernel[i] = std::exp(-(x * x) / (2.0f * sigma * sigma));
        sum += kernel[i];
    }
    for (int i = 0; i < kernel_size; ++i) {
        kernel[i] /= sum;
    }

    Image tmp(img.width, img.height, 1);
    Image filtered(img.width, img.height, 1);
    
    int w = img.width;
    int h = img.height;
    
    // ===== 激進優化策略 =====
    // 1. 完全展開小核心以減少循環開銷
    // 2. 使用編譯器 SIMD 向量化提示
    // 3. 減少條件分支
    // 4. 改善記憶體訪問模式
    
    const float* src_data = img.data;
    float* tmp_data = tmp.data;
    float* dst_data = filtered.data;
    
    // 垂直卷積 - 優化：展開循環 + 向量化
    #pragma omp parallel for schedule(static) collapse(2)
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float sum_val = 0.0f;
            int base_idx = y * w + x;
            
            // 展開小核心情況以減少循環開銷
            if (kernel_size <= 7) {
                // 手動展開小核心
                for (int k = 0; k < kernel_size; ++k) {
                    int sy = y + k - center;
                    sy = (sy < 0) ? 0 : ((sy >= h) ? h - 1 : sy);
                    sum_val += src_data[sy * w + x] * kernel[k];
                }
            } else {
                // 大核心使用向量化
                #pragma omp simd reduction(+:sum_val)
                for (int k = 0; k < kernel_size; ++k) {
                    int sy = y + k - center;
                    sy = (sy < 0) ? 0 : ((sy >= h) ? h - 1 : sy);
                    sum_val += src_data[sy * w + x] * kernel[k];
                }
            }
            tmp_data[base_idx] = sum_val;
        }
    }
    
    // 水平卷積 - 優化：批次處理 + 向量化
    #pragma omp parallel
    {
        // 每個線程的行緩衝區
        std::vector<float> row_buffer;
        if (kernel_size > 7) {
            row_buffer.resize(w + 2 * center);
        }
        
        #pragma omp for schedule(static)
        for (int y = 0; y < h; ++y) {
            const float* tmp_row = tmp_data + y * w;
            float* filtered_row = dst_data + y * w;
            
            if (kernel_size <= 7) {
                // 小核心：直接計算，完全向量化
                #pragma omp simd
                for (int x = 0; x < w; ++x) {
                    float sum_val = 0.0f;
                    for (int k = 0; k < kernel_size; ++k) {
                        int sx = x + k - center;
                        sx = (sx < 0) ? 0 : ((sx >= w) ? w - 1 : sx);
                        sum_val += tmp_row[sx] * kernel[k];
                    }
                    filtered_row[x] = sum_val;
                }
            } else {
                // 大核心：使用預填充緩衝區
                for (int k = 0; k < center; ++k) {
                    row_buffer[k] = tmp_row[0];
                }
                std::copy(tmp_row, tmp_row + w, row_buffer.begin() + center);
                for (int k = 0; k < center; ++k) {
                    row_buffer[center + w + k] = tmp_row[w - 1];
                }
                
                #pragma omp simd
                for (int x = 0; x < w; ++x) {
                    float sum_val = 0.0f;
                    for (int k = 0; k < kernel_size; ++k) {
                        sum_val += row_buffer[x + k] * kernel[k];
                    }
                    filtered_row[x] = sum_val;
                }
            }
        }
    }
    
    return filtered;
}
void draw_point(Image& img, int x, int y, int size)
{
    for (int i = x-size/2; i <= x+size/2; i++) {
        for (int j = y-size/2; j <= y+size/2; j++) {
            if (i < 0 || i >= img.width) continue;
            if (j < 0 || j >= img.height) continue;
            if (std::abs(i-x) + std::abs(j-y) > size/2) continue;
            if (img.channels == 3) {
                img.set_pixel(i, j, 0, 1.f);
                img.set_pixel(i, j, 1, 0.f);
                img.set_pixel(i, j, 2, 0.f);
            } else {
                img.set_pixel(i, j, 0, 1.f);
            }
        }
    }
}

void draw_line(Image& img, int x1, int y1, int x2, int y2)
{
    if (x2 < x1) {
        std::swap(x1, x2);
        std::swap(y1, y2);
    }
    int dx = x2 - x1, dy = y2 - y1;
    for (int x = x1; x < x2; x++) {
        int y = y1 + dy*(x-x1)/dx;
        if (img.channels == 3) {
            img.set_pixel(x, y, 0, 0.f);
            img.set_pixel(x, y, 1, 1.f);
            img.set_pixel(x, y, 2, 0.f);
        } else {
            img.set_pixel(x, y, 0, 1.f);
        }
    }
}
