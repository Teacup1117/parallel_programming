# 🔧 修復總結

## 找到並修復的 2 個重大 Bug

### Bug #1: Block 複製不完整 (hw4.cu)
- **已修復**: ✅ 從手動欄位複製改為 `HashBlock local_block = *blocks;`

### Bug #2: SHA256 Device MACRO 邏輯致命錯誤 (sha256.h) ⭐ 最關鍵
- **原因**: 優化的 MACRO 實現完全破壞了 SHA256 的狀態管理邏輯
- **症狀**: SHA256 計算完全不正確，導致無法找到正確的 nonce
- **修復**: 替換為標準的循環實現，確保每個 round 都進行正確的 8 個變數循環移位
- **文件**: `sha256.h` 第 45-100 行

---

## 核心差異 (SHA256 Device Transform)

### ❌ 錯誤版本 (MACRO)
```cuda
d += temp1;              // 只修改 d
h = temp1 + temp2;       // 只修改 h
// 沒有對 a, b, c, e, f, g 進行正確的循環移位
```

### ✅ 正確版本 (標準循環)
```cuda
h = g;
g = f;
f = e;
e = d + temp1;
d = c;
c = b;
b = a;
a = temp1 + temp2;
// 所有 8 個變數都正確循環移位
```

---

## 預期結果
修復後，Case01 應該從 "wrong answer" 變為 "accepted" ✅
