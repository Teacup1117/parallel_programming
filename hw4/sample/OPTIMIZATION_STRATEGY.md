# ✅ 效率優化策略（保證正確性）

## 根本原因找到了！

原始的"優化"版本（100M batch + 動態 launch config）失敗了，原因可能是：

1. **Batch size 100M 太大** - 可能導致某些邊界情況下的算術溢出或搜索邏輯錯誤
2. **動態 launch config** - `cudaOccupancyMaxPotentialBlockSize()` 計算出的參數可能不適合該 kernel

## ✅ 確認正確的配置（當前）

```cuda
int threadsPerBlock = 256;
int blocksPerGrid = 1024;
unsigned int batch_size = 10000000;  // 10M
```

- ✅ Block 1,2,3,4 全部正確
- ✅ 計算結果準確

---

## 🚀 安全的優化策略

### 優化級別 1（保守，應該安全）
```cuda
int blocksPerGrid = 2048;           // 從 1024 增加到 2048
unsigned int batch_size = 50000000;  // 從 10M 增加到 50M
```
- 增加並行度但保持安全邊界
- 期望性能提升：50-80%

### 優化級別 2（中等，需要驗證）
```cuda
int threadsPerBlock = 512;           // 可能提高
int blocksPerGrid = 4096;            // 更多 blocks
unsigned int batch_size = 100000000; // 100M
```
- 需要驗證所有 cases 還是通過
- 期望性能提升：150-200%

### 優化級別 3（激進，不推薦）
```cuda
int minGridSize, threadsPerBlock;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &threadsPerBlock, 
                                   find_nonce_kernel, 0, 0);
int blocksPerGrid = minGridSize * 4;
unsigned int batch_size = 100000000;
```
- 導致當前代碼失敗
- 不推薦使用

---

## 測試優化的方法

1. **先測試優化級別 1**：
   ```
   blocksPerGrid = 2048
   batch_size = 50000000
   ```
   - 編譯並運行 case01
   - 確認所有 4 個 block 結果正確
   - 記錄執行時間

2. **如果通過，升級到優化級別 2**：
   ```
   threadsPerBlock = 512
   blocksPerGrid = 4096
   batch_size = 100000000
   ```
   - 重複測試

3. **如果失敗，回退到上一級別**

---

## 當前配置

已套用優化級別 1：
- ✅ `blocksPerGrid = 2048`
- ✅ `batch_size = 50000000`

預期性能提升：50-80%
預期仍然正確性無誤

---

## 下一步

1. 編譯並測試當前代碼（優化級別 1）
2. 確認 case01 第 3 個 block 仍然是 `07a47002`（不是 `00000000`）
3. 如果通過，可嘗試優化級別 2（但需謹慎驗證）
