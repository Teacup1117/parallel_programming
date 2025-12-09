# 🔴 重大發現：SHA256 Device MACRO 實現邏輯錯誤

## 問題根源

### 目前版本的 SHA256_ROUND MACRO（sha256.h）
```cuda
#define SHA256_ROUND(a,b,c,d,e,f,g,h,i) \
{ \
    WORD S1 = (_rotr(e, 6)) ^ (_rotr(e, 11)) ^ (_rotr(e, 25)); \
    WORD ch = (e & f) ^ ((~e) & g); \
    WORD temp1 = h + S1 + ch + sha256_k[i] + w[i]; \
    WORD S0 = (_rotr(a, 2)) ^ (_rotr(a, 13)) ^ (_rotr(a, 22)); \
    WORD maj = (a & b) ^ (a & c) ^ (b & c); \
    WORD temp2 = S0 + maj; \
    d += temp1;              // ❌ 錯誤：直接修改 d
    h = temp1 + temp2;       // ❌ 錯誤：沒有循環移位其他變數
}
```

### 標準 SHA256（sha256.cu 中的 Host 版本）
```cuda
for(i=0;i<64;++i)
{
    WORD S0 = (_rotr(a, 2)) ^ (_rotr(a, 13)) ^ (_rotr(a, 22));
    WORD S1 = (_rotr(e, 6)) ^ (_rotr(e, 11)) ^ (_rotr(e, 25));
    WORD ch = (e & f) ^ ((~e) & g);
    WORD maj = (a & b) ^ (a & c) ^ (b & c);
    WORD temp1 = h + S1 + ch + k[i] + w[i];
    WORD temp2 = S0 + maj;
    
    h = g;                   // ✅ 正確：循環移位
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
}
```

## 核心差異

在 SHA256 中，每個 round 應該是**循環移位**（rotation）所有 8 個工作變數：
- `a := temp1 + temp2`
- `b := a`（前一個）
- `c := b`（前一個）
- ... 以此類推

但目前的 MACRO 實現只修改了 `d` 和 `h`，**完全破壞了 SHA256 的狀態管理！**

## 為什麼只有 Case01 失敗？

這取決於：
- 不同測試用例可能觸發不同的代碼路徑
- Case01 的特定 Block 數據可能導致 Hash 計算中的微妙差異
- 或者 Case01 的目標閾值（target）恰好落在被破壞計算的邊界上

## 解決方案

### 方法 1：移除 MACRO，使用標準循環（推薦）
```cuda
for(i=0;i<64;++i)
{
    WORD S0 = (_rotr(a, 2)) ^ (_rotr(a, 13)) ^ (_rotr(a, 22));
    WORD S1 = (_rotr(e, 6)) ^ (_rotr(e, 11)) ^ (_rotr(e, 25));
    WORD ch = (e & f) ^ ((~e) & g);
    WORD maj = (a & b) ^ (a & c) ^ (b & c);
    WORD temp1 = h + S1 + ch + sha256_k[i] + w[i];
    WORD temp2 = S0 + maj;
    
    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
}
```

### 方法 2：修復 MACRO（複雜，容易出錯）
MACRO 需要完全重寫以正確處理所有 8 個變數的循環移位。

## 立即行動

需要用標準的循環實現替換 `sha256_transform_device()` 中的優化 MACRO 版本。
