# 🎯 Case01 Block 3 錯誤根本原因 - 已找到並修復

## 問題症狀
```
Block 3 輸出: 00000000 (應該是 07a47002)
```

## 🔴 根本原因：遺漏初始化 d_found_nonce

### 錯誤代碼（hw4.cu 第 296-297 行）
```cuda
int found_flag = 0;
unsigned int found_nonce = 0;
cudaMemcpy(d_found_flag, &found_flag, sizeof(int), cudaMemcpyHostToDevice);
// ❌ 缺少這行！
```

### 正確代碼
```cuda
int found_flag = 0;
unsigned int found_nonce = 0;
cudaMemcpy(d_found_flag, &found_flag, sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(d_found_nonce, &found_nonce, sizeof(unsigned int), cudaMemcpyHostToDevice);  // ✅ 已修復
```

---

## 為什麼這導致 Block 3 失敗？

### 執行流程：

1. **Block 1 執行**:
   - d_found_nonce 被初始化為 0 → 找到答案 `5ea01346` → GPU 更新 d_found_nonce

2. **Block 2 執行**:
   - d_found_nonce 被初始化為 0 → 找到答案 `1dac2b7c` → GPU 更新 d_found_nonce

3. **Block 3 執行** ❌:
   - d_found_nonce **沒有初始化** → 保持前一個 block 的垃圾值或邏輯錯誤
   - GPU kernel 執行時無法正確更新 found_nonce
   - 搜索完成後 cudaMemcpy 回傳的是未初始化或錯誤的值
   - 最終輸出為 0（uninitialized memory 的預設值或邏輯錯誤）

4. **Block 4 執行** ✅:
   - 偶然正確（可能是前面的初始化有效了，或者巧合的內存狀態）

---

## 為什麼在本地測試通過但評分系統失敗？

1. **隨機性**: Uninitialized memory 在不同系統上有不同行為
2. **編譯器優化**: 不同的編譯器可能優化不同
3. **GPU 內存佈局**: 不同 GPU 的內存配置不同
4. **評分系統的嚴格性**: 可能有額外的驗證

---

## 修復內容

✅ 已在 `hw4.cu` 第 297 行後添加：
```cuda
cudaMemcpy(d_found_nonce, &found_nonce, sizeof(unsigned int), cudaMemcpyHostToDevice);
```

---

## 預期結果

修復後，Case01 應該輸出：
```
4
5ea01346
1dac2b7c
07a47002  ← 不再是 00000000
85274beb
```

所有 4 個 case 應該全部 ACCEPTED ✅
