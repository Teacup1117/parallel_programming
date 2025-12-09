# GPU 優化摘要

## 目標
從原本的 14 秒優化到 7-9 秒（150%-200% 加速效果）

## 主要優化技術

### 1. SHA256 計算核心優化 (sha256.h)
- **循環展開 (#pragma unroll)**: 
  - 消息排程陣列 (w[]) 的計算完全展開
  - 主循環使用 8 路展開配合 SHA256_ROUND 宏
  - 減少分支預測失敗和循環控制開銷
  
- **SHA256_ROUND 宏優化**:
  - 消除不必要的臨時變數賦值
  - 減少寄存器壓力
  - 直接進行 d += temp1，減少一次變數移動

### 2. Kernel 啟動配置優化 (hw4.cu)
- **動態計算最佳配置**:
  - 使用 `cudaOccupancyMaxPotentialBlockSize` 自動計算最佳 thread/block 配置
  - 4x 過度訂閱 (oversubscription) 以隱藏延遲
  - 自動適應不同 GPU 架構

### 3. 早期退出優化 (hw4.cu - kernel)
- **Block 層級早期退出**:
  - 使用 shared memory (`block_found`) 讓同一 block 內的 thread 快速感知解答
  - 減少不必要的計算
  
- **快速比較優化**:
  - 先檢查最後一個 byte（最有可能不同）
  - 只有在初步檢查通過後才進行完整比較
  - 大幅減少比較運算

### 4. 記憶體訪問優化 (hw4.cu - kernel)
- **結構化複製**:
  - 使用 #pragma unroll 展開陣列複製
  - 減少記憶體訪問延遲
  
- **局部變數優化**:
  - 明確複製 HashBlock 的各個欄位
  - 幫助編譯器進行更好的寄存器分配

### 5. Batch Size 策略 (hw4.cu)
- **大 Batch Size**: 從 10M 增加到 100M
  - 減少 kernel 啟動開銷（每次啟動約 5-10 微秒）
  - 更好地攤銷啟動成本
  
- **減少同步頻率**:
  - 每 5 個 batch 才檢查一次結果
  - 減少 Host-Device 通訊開銷

### 6. Pinned Memory 優化 (hw4.cu)
- **使用 cudaHostAlloc**:
  - 分配頁鎖定記憶體 (pinned memory)
  - 加速 Host-Device 傳輸（約 2-3x 速度提升）
  - 支援非同步傳輸

### 7. CUDA Stream 異步執行 (hw4.cu)
- **異步操作**:
  - 使用 cudaMemcpyAsync 進行非阻塞傳輸
  - 重疊計算與通訊
  - 減少總體等待時間

## 預期效能提升分析

| 優化項目 | 預期加速 | 累積加速 |
|---------|---------|---------|
| SHA256 循環展開 | 1.3x | 1.3x |
| 動態配置 + 過度訂閱 | 1.2x | 1.56x |
| 早期退出 + 快速比較 | 1.15x | 1.79x |
| Batch Size 優化 | 1.05x | 1.88x |
| Pinned Memory + Stream | 1.05x | **1.97x** |

**總預期加速**: ~2x (200%)
**預期執行時間**: 14秒 / 2 = **7秒**

## 編譯建議

使用以下編譯選項以獲得最佳性能：

```bash
nvcc -O3 -arch=sm_70 -use_fast_math hw4.cu sha256.cu -o hw4
```

或者針對您的具體 GPU 架構（例如 sm_80 for A100, sm_86 for RTX 3090）。

## 正確性保證

所有優化都保持：
1. 完全相同的 SHA256 計算邏輯
2. 原子操作保證只記錄一個正確的 nonce
3. 完整的搜索範圍覆蓋
4. CPU 端驗證計算結果

## 使用方式

編譯並執行即可，無需修改參數：
```bash
make clean
make
./hw4 testcases/case00.in output.txt
```
