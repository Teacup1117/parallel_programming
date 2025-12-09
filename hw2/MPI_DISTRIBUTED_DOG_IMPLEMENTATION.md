# MPI 分散式 DoG 和 Gradient Pyramid 實作說明

## 🎯 修改目標

將 SIFT 演算法中**獨立的計算步驟**（DoG 和 Gradient Pyramid）分配到不同的 MPI 節點，實現真正的分散式計算。

---

## 📊 修改前後對比

### **修改前（原始實作）**

```
所有進程執行相同的工作:
├─ Rank 0: 計算完整的 Gaussian Pyramid → DoG Pyramid → Gradient Pyramid
├─ Rank 1: 計算完整的 Gaussian Pyramid → DoG Pyramid → Gradient Pyramid
├─ Rank 2: 計算完整的 Gaussian Pyramid → DoG Pyramid → Gradient Pyramid
└─ Rank 3: 計算完整的 Gaussian Pyramid → DoG Pyramid → Gradient Pyramid

結果: 只用 Rank 0 的結果，其他進程的計算被丟棄
加速: 來自 OpenMP 在每個進程內部的並行化
```

### **修改後（分散式實作）**

```
工作分配到不同進程:
├─ Gaussian Pyramid: 所有進程計算（有依賴性，無法分割）
├─ DoG Pyramid: 分散計算
│   ├─ Rank 0: 計算 DoG[0-13]
│   ├─ Rank 1: 計算 DoG[14-27]
│   ├─ Rank 2: 計算 DoG[28-41]
│   └─ Rank 3: 計算 DoG[42-55]
├─ Gradient Pyramid: 分散計算
│   ├─ Rank 0: 計算 Gradient[0-15]
│   ├─ Rank 1: 計算 Gradient[16-31]
│   ├─ Rank 2: 計算 Gradient[32-47]
│   └─ Rank 3: 計算 Gradient[48-63]
└─ 透過 MPI_Bcast 共享結果

加速: MPI 層面的真正分散計算 + OpenMP 內部並行化
```

---

## 🔧 新增函數

### **1. `generate_dog_pyramid_mpi()`**

```cpp
ScaleSpacePyramid generate_dog_pyramid_mpi(const ScaleSpacePyramid& img_pyramid, 
                                            int rank, int size)
```

**功能**：
- 將 DoG Pyramid 的計算工作分配到不同的 MPI 進程
- 每個進程計算部分 DoG 圖像
- 透過 MPI_Bcast 共享所有結果

**工作分配邏輯**：

```cpp
// 總共 56 張 DoG 圖像 (8 octaves × 7 scales)
int total_dogs = num_octaves * imgs_per_octave;  // 56

// 每個 rank 處理的圖像數量
int dogs_per_rank = (total_dogs + size - 1) / size;  // 向上取整

// 範例: 4 個進程
// Rank 0: 圖像 0-13   (14 張)
// Rank 1: 圖像 14-27  (14 張)
// Rank 2: 圖像 28-41  (14 張)
// Rank 3: 圖像 42-55  (14 張)
```

**計算流程**：

```cpp
// 1. 每個 rank 只計算分配給它的 DoG 圖像
#pragma omp parallel for schedule(dynamic)
for (int idx = start_idx; idx < end_idx; idx++) {
    int i = idx / imgs_per_octave;  // octave
    int j = idx % imgs_per_octave;  // scale
    
    // 計算 DoG
    dst[pix_idx] = src1[pix_idx] - src0[pix_idx];
}

// 2. 透過 MPI_Bcast 共享結果
for (每個 DoG 圖像) {
    int owner_rank = idx / dogs_per_rank;  // 誰計算的
    MPI_Bcast(dog_data, size, MPI_FLOAT, owner_rank, MPI_COMM_WORLD);
}
```

### **2. `generate_gradient_pyramid_mpi()`**

```cpp
ScaleSpacePyramid generate_gradient_pyramid_mpi(const ScaleSpacePyramid& pyramid,
                                                 int rank, int size)
```

**功能**：
- 將 Gradient Pyramid 的計算工作分配到不同的 MPI 進程
- 每個進程計算部分 Gradient 圖像
- 透過 MPI_Bcast 共享所有結果

**工作分配邏輯**：

```cpp
// 總共 64 張 Gradient 圖像 (8 octaves × 8 scales)
int total_grads = num_octaves * imgs_per_octave;  // 64

// 範例: 4 個進程
// Rank 0: 圖像 0-15   (16 張)
// Rank 1: 圖像 16-31  (16 張)
// Rank 2: 圖像 32-47  (16 張)
// Rank 3: 圖像 48-63  (16 張)
```

---

## 📈 性能分析

### **計算複雜度**

假設圖像大小為 2000×2000：

| 階段 | 圖像數量 | 每張計算量 | 總計算量 |
|------|---------|-----------|---------|
| Gaussian Pyramid | 64 | 高 (Gaussian Blur) | ~600ms |
| **DoG Pyramid** | 56 | 低 (相減) | ~50ms |
| **Gradient Pyramid** | 64 | 中 (梯度計算) | ~100ms |

### **加速效果預估**

#### **修改前**：
```
Gaussian: 600ms (所有進程重複計算)
DoG:      50ms  (所有進程重複計算)
Gradient: 100ms (所有進程重複計算)
Total:    750ms
```

#### **修改後（4 進程）**：
```
Gaussian: 600ms (仍需完整計算，有依賴性)
DoG:      12.5ms (50ms ÷ 4 = 12.5ms + 通訊開銷 ~5ms)
Gradient: 25ms   (100ms ÷ 4 = 25ms + 通訊開銷 ~8ms)
Total:    650.5ms

理論加速: 750ms → 650.5ms ≈ 1.15x
```

### **通訊開銷分析**

**DoG Pyramid 通訊**：
```
56 張圖像，每張需要一次 MPI_Bcast
平均圖像大小: 500KB (octave 0) → 1KB (octave 7)
總資料量: ~10MB
預估通訊時間: ~5ms (取決於網路)
```

**Gradient Pyramid 通訊**：
```
64 張圖像，每張有 2 個通道
總資料量: ~12MB
預估通訊時間: ~8ms
```

---

## 🎯 為什麼只分散 DoG 和 Gradient？

### **Gaussian Pyramid 不能分散**

```cpp
for (int i = 0; i < num_octaves; i++) {
    pyramid.octaves[i].push_back(std::move(base_img));
    
    // Scale 之間有依賴
    for (int j = 1; j < imgs_per_octave; j++) {
        const Image& prev_img = pyramid.octaves[i].back();  // 依賴前一個！
        pyramid.octaves[i].push_back(gaussian_blur(prev_img, sigma_vals[j]));
    }
    
    // Octave 之間有依賴
    base_img = next_base_img.resize(...);  // 為下一個 octave 準備
}
```

**依賴圖**：
```
Octave 0 → Octave 1 → Octave 2 → ... (順序依賴)
  ↓           ↓           ↓
Scale 0     Scale 0     Scale 0
  ↓           ↓           ↓
Scale 1     Scale 1     Scale 1    (Scale 內部依賴)
  ↓           ↓           ↓
Scale 2     Scale 2     Scale 2
  ...         ...         ...
```

### **DoG 和 Gradient 可以分散**

```cpp
// DoG: 每張圖像完全獨立
DoG[i][j] = Gaussian[i][j+1] - Gaussian[i][j]

// Gradient: 每張圖像完全獨立
Gradient[i][j] = compute_gradient(Gaussian[i][j])
```

**獨立性圖**：
```
DoG[0][0]   DoG[0][1]   DoG[0][2]   ... (完全獨立)
DoG[1][0]   DoG[1][1]   DoG[1][2]   ... (完全獨立)
   ↓           ↓           ↓
可分配到   可分配到   可分配到
Rank 0     Rank 1     Rank 2
```

---

## 🔍 實作細節

### **1. 線性索引到 2D 座標的轉換**

```cpp
// 將 DoG 的線性索引轉換為 (octave, scale)
int idx = 25;  // 想要計算第 25 張 DoG 圖像
int imgs_per_octave = 7;

int octave = idx / imgs_per_octave;  // 25 / 7 = 3
int scale = idx % imgs_per_octave;   // 25 % 7 = 4

// 結果: DoG[3][4]
```

### **2. MPI_Bcast 的使用**

```cpp
// 每個 DoG 圖像由一個特定的 rank 計算
int owner_rank = idx / dogs_per_rank;

// 從 owner_rank 廣播到所有其他 ranks
MPI_Bcast(
    dog_pyramid.octaves[i][j].data,  // 資料指標
    dog_pyramid.octaves[i][j].size,  // 資料大小
    MPI_FLOAT,                       // 資料類型
    owner_rank,                      // 來源 rank
    MPI_COMM_WORLD                   // 通訊器
);
```

**工作流程**：
```
初始狀態:
Rank 0: DoG[0-13] 已計算, DoG[14-55] 空白
Rank 1: DoG[0-13] 空白, DoG[14-27] 已計算, DoG[28-55] 空白
Rank 2: DoG[0-27] 空白, DoG[28-41] 已計算, DoG[42-55] 空白
Rank 3: DoG[0-41] 空白, DoG[42-55] 已計算

經過 56 次 MPI_Bcast 後:
所有 Ranks: DoG[0-55] 全部有效
```

### **3. 混合並行：MPI + OpenMP**

```cpp
// MPI 層面: 分配不同的圖像給不同進程
for (int idx = start_idx; idx < end_idx; idx++) {
    
    // OpenMP 層面: 在每個進程內部並行處理像素
    #pragma omp simd
    for (int pix_idx = 0; pix_idx < size; pix_idx++) {
        dst[pix_idx] = src1[pix_idx] - src0[pix_idx];
    }
}
```

**並行層級**：
```
Level 1 (MPI): 4 個進程，各計算 14 張圖像
Level 2 (OpenMP 外層): 每個進程用 6 個線程並行處理 14 張圖像
Level 3 (OpenMP SIMD): 每個線程用 SIMD 指令並行處理像素

總並行度: 4 (MPI) × 6 (OpenMP) = 24 個工作單元
```

---

## 🚀 使用方式

### **編譯**

```bash
make clean && make
```

### **執行**

```bash
# 使用 4 個進程，每個 6 個線程
srun -N 2 -n 4 -c 6 ./hw2 input.jpg output.jpg output.txt
```

### **執行流程**

```
1. Rank 0 讀取圖像並廣播給所有 ranks
2. 所有 ranks 並行計算 Gaussian Pyramid
3. Ranks 分散計算 DoG Pyramid 並透過 MPI_Bcast 共享
4. Ranks 分散計算 Gradient Pyramid 並透過 MPI_Bcast 共享
5. 所有 ranks 找到相同的 keypoints
6. 只有 Rank 0 計算 descriptors 並輸出結果
```

---

## 📊 優勢與限制

### **優勢**

1. **真正的分散計算**
   - DoG 和 Gradient 不再重複計算
   - 充分利用多節點資源

2. **保持結果一致性**
   - 透過 MPI_Bcast 確保所有進程有相同資料
   - Keypoint 檢測結果完全一致

3. **混合並行架構**
   - MPI: 圖像級並行
   - OpenMP: 像素級並行
   - SIMD: 向量級並行

### **限制**

1. **通訊開銷**
   - 需要多次 MPI_Bcast（56 + 64 = 120 次）
   - 總資料量約 20MB

2. **有限的加速比**
   - DoG 和 Gradient 只佔總時間的 ~20%
   - 主要瓶頸仍在 Gaussian Pyramid（佔 ~80%）

3. **網路依賴**
   - 需要高速網路才能降低通訊開銷
   - 慢速網路可能抵消加速效果

---

## 💡 總結

這次修改實現了：

✅ **DoG Pyramid 的分散式計算** - 每個進程計算不同的 DoG 圖像  
✅ **Gradient Pyramid 的分散式計算** - 每個進程計算不同的 Gradient 圖像  
✅ **透過 MPI_Bcast 共享結果** - 確保所有進程有完整資料  
✅ **保持數值一致性** - 與 golden file 完全相符  
✅ **混合並行架構** - MPI + OpenMP + SIMD 三層並行  

**預期效果**：在高速網路環境下，相比原始 MPI 實作，可獲得額外 10-15% 的性能提升。
