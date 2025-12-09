# Case01 Block 3 不匹配診斷

## 數據對比

### 你的輸出 vs 標準答案

Block 1: `5ea01346` ✅ 
Block 2: `1dac2b7c` ✅
Block 3: `00000000` ❌ (應該是 `07a47002`)
Block 4: `85274beb` ✅

## 問題分析

第 3 個 block 輸出 `00000000` 表示：
1. **Nonce 為 0** - 這是初始值，表示根本沒有找到任何滿足條件的 nonce
2. **搜索過程中沒有更新 found_nonce**

可能的原因：

### 1. ⚠️ **Nonce 搜索範圍問題**
- 可能搜索的 nonce 範圍不覆蓋正確答案 `07a47002`
- Batch 大小或搜索循環有問題

### 2. ⚠️ **Target 值計算錯誤**
- 第 3 個 block 的 nbits 可能導致不同的 target 計算
- Target 計算邏輯可能有 bug

### 3. ⚠️ **Hash 比較邏輯錯誤**
- Little-endian 比較的邏輯可能在某些情況下失效
- 邊界條件處理不當

### 4. ⚠️ **GPU 內存問題**
- 第 3 個 block 的 device memory 可能沒有正確清理
- 前面 blocks 的結果可能干擾了 block 3

### 5. ⚠️ **Atomic 操作問題**
- 如果沒有找到答案，atomicCAS 可能導致 found_nonce 保持初始值

---

## 關鍵觀察

1. **Block 1,2,4 都正確** → 基本邏輯沒問題
2. **只有 Block 3 失敗** → 可能是該 block 的特定數據觸發了 bug
3. **輸出 0** → 搜索過程根本沒找到任何滿足條件的 nonce

---

## 建議診斷步驟

### 立即檢查的代碼：

1. **GPU 內存初始化**
   ```cuda
   // 確保每個 block 都重新初始化
   int found_flag = 0;
   unsigned int found_nonce = 0;
   cudaMemcpy(d_found_flag, &found_flag, sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(d_found_nonce, &found_nonce, sizeof(unsigned int), cudaMemcpyHostToDevice);
   ```

2. **Nonce 初始值**
   ```cuda
   // 確保每個 block 的搜索都從 0 開始
   unsigned int start_nonce = 0;
   block.nonce = 0;
   ```

3. **Batch 大小和搜索循環**
   - 檢查是否所有 nonce 都被檢查到
   - 確認 start_nonce 的更新邏輯

4. **Target 值計算**
   - 打印 Block 3 的 nbits 和計算出的 target
   - 驗證位移計算是否正確

---

## 最可能的根本原因

🔴 **GPU 內存狀態污染** - 前面 blocks 的 d_found_nonce 或 d_found_flag 沒有正確重置

在 solve() 函數開始時：
```cuda
// 需要為每個 block 重新初始化
int found_flag = 0;
unsigned int found_nonce = 0;
cudaMemcpy(d_found_flag, &found_flag, sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_found_nonce, &found_nonce, sizeof(unsigned int), cudaMemcpyHostToDevice);
```

這應該在每次 solve() 調用時都執行，但如果遺漏或只執行一次，就會導致後續 blocks 使用前一個 block 的結果。
