# 詳細錯誤診斷分析 - Case01 Wrong Answer

## 問題症狀
- Case00, Case02, Case03: ✅ ACCEPTED
- Case01: ❌ WRONG ANSWER

## 已檢查的修正
1. ✅ Block 結構複製方式：已改為 `HashBlock local_block = *blocks;`
2. ✅ CUDA kernel 邏輯：基本結構看起來正確

## 可能的根本原因列表

### 1. **SHA256 計算邏輯錯誤** ⚠️ 高度懷疑
   - Device 版本與 Host 版本可能有差異
   - 優化的 MACRO 版本可能有 bug
   - Loop unrolling 導致的狀態管理問題

### 2. **Byte Order 和 Endianness 問題** ⚠️ 高度懷疑
   - Block 結構中的填充（padding）可能導致 byte 順序不對
   - 最終輸出的字節順序可能不正確

### 3. **Target 值計算錯誤**
   - nbits 到 target 的轉換邏輯可能有誤
   - 位移操作的邏輯可能不對

### 4. **Merkle Root 計算錯誤**
   - 合併邏輯不正確
   - 字節順序轉換不對

### 5. **Block 初始化問題**
   - version, ntime, nbits 的轉換可能不對
   - memcpy 的使用可能不安全

### 6. **GPU 內存問題**
   - Device memory 沒有正確初始化
   - 結構體的 padding 導致內存佈局不同

### 7. **Nonce 搜索邏輯**
   - Nonce 的起始值或範圍不正確
   - Batch 分割導致某些 nonce 被跳過

## 關鍵差異對比

### 目前版本 vs 備份版本

#### sha256.h Device Macro 方式
```cuda
#define SHA256_ROUND(a,b,c,d,e,f,g,h,i) \
{ \
    d += temp1;  // 問題：直接修改 d
    h = temp1 + temp2;
}
```

#### 標準 SHA256 方式
```cuda
h = g;
g = f;
f = e;
e = d + temp1;  // 正確的循環移位
d = c;
c = b;
b = a;
a = temp1 + temp2;
```

**⚠️ 這可能是導致 SHA256 計算錯誤的根本原因！**

## 建議診斷步驟

1. 比較 SHA256 device 版本的 MACRO 實現與標準循環實現
2. 驗證 Block 結構內存佈局（sizeof, offsetof）
3. 檢查 Endianness 轉換的完整性
4. 在小的測試用例上逐步驗證每個計算步驟

## 優先修復順序

1. **最優先**: 修復 SHA256 device MACRO 實現的狀態管理邏輯
2. **次優先**: 驗證 Block 結構和初始化
3. **第三優先**: 檢查 Endianness 處理
