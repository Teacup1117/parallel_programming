# 深度診斷：Block 3 的 Target 和搜索範圍分析

## Block 3 的輸入數據

```
nbits: 00000000  (第 10 行)
ntime: dd24a8e3f419006cd2f7cb3336b605fa6c2898accecac3d0a844c2ae8f5f53d9 (merkle tree)
```

等等，讓我重新讀取正確的 block 3 數據...

Block 結構應該是：
1. Block 1: 行 2-8
2. Block 2: 行 9-15
3. Block 3: 行 16-22
4. Block 4: 行 23-29

---

## 關鍵問題發現：Nonce 初始化

在 solve() 函數中：
```cuda
block.nonce = 0;  // 初始值
```

然後搜索找到新的 nonce 值。

但問題是：**如果 target 計算錯誤，或者比較邏輯錯誤，就找不到任何滿足條件的 nonce。**

---

## 可能的根本原因

### 1. ❌ SHA256 Device 版本仍然有 Bug
- 即使修復了 MACRO，可能還有其他問題
- Device 版本的 SHA256 與 Host 版本的結果不一致

### 2. ❌ Target 計算邏輯
```cuda
unsigned int exp = block.nbits >> 24;
unsigned int mant = block.nbits & 0xffffff;
unsigned int shift = 8 * (exp - 3);
unsigned int sb = shift / 8;
unsigned int rb = shift % 8;

target_hex[sb    ] = (mant << rb);
target_hex[sb + 1] = (mant >> (8-rb));
target_hex[sb + 2] = (mant >> (16-rb));
target_hex[sb + 3] = (mant >> (24-rb));
```

對於 Block 3 的 nbits: `00000000`
- exp = 0 >> 24 = 0
- mant = 0 & 0xffffff = 0
- shift = 8 * (0 - 3) = -24 (負數！)
- sb = -24 / 8 = -3 (負索引！)

⚠️ **這會導致緩衝區溢出！**

### 3. ❌ Hash 比較邏輯
```cuda
if(little_endian_bit_comparison_device(sha256_ctx.b, target_hex, 32) < 0)
```

可能在某些邊界情況下失效。

---

## 最可能的原因：Block 3 的 nbits 為 0x00000000

如果 nbits 為 0，這表示：
- Target 的難度沒有定義
- 或者這個 block 沒有有效的 target
- 導致 target 計算邏輯失效

**應該檢查：**
1. Block 3 的真實 nbits 值是什麼？
2. 如果 nbits 為 0，應該有特殊處理

---

## 立即行動

需要檢查：
1. 打印 Block 3 的 nbits 實際值
2. 打印計算出的 target_hex
3. 驗證是否有邊界情況（exp < 3）
