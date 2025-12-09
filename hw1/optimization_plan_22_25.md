# Sokoban 複雜測資 22-25 優化策略

## 當前性能分析

基於實際測試結果：
- **測資22**: 15,478,520 nodes, 58.032秒 ✅
- **測資24**: 8,890,879 nodes, 47.886秒 ✅  
- **測資25**: 執行中，可能需要更長時間 ⚠️

## 核心問題診斷

1. **記憶體瓶頸**: A* 算法維護巨大的 visited_states 集合
2. **搜索效率**: 節點展開數量達千萬級別
3. **啟發函數**: 當前啟發函數對複雜測資指導性不足
4. **死鎖檢測**: 可能存在未檢測到的死鎖狀態

## 優化策略（按優先級排序）

### 🔥 1. IDA* 算法替換 (最高優先級)

**問題**: A* 記憶體消耗 O(b^d)，visited_states 可能占用數GB
**解決**: IDA* 記憶體消耗 O(d)，適合深度搜索

**實現要點**:
```cpp
// 核心 IDA* 搜索函數
int ida_search(BitState state, int g, int threshold, string& path) {
    int f = g + heuristic(state);
    if (f > threshold) return f;
    if (is_goal(state)) return FOUND;
    
    int min_t = INT_MAX;
    for (auto [next_state, move] : get_successors(state)) {
        int t = ida_search(next_state, g+1, threshold, path + move);
        if (t == FOUND) return FOUND;
        if (t < min_t) min_t = t;
    }
    return min_t;
}
```

**優勢**:
- 記憶體使用降低 90%+
- 避免重複狀態存儲
- 更適合深搜索空間

### 🔥 2. 增強啟發函數 (高優先級)

**當前問題**: 簡單的匈牙利算法對複雜佈局指導性不足

**改進方案**:

#### A. 模式數據庫 (Pattern Database)
```cpp
// 預計算小型子問題的精確距離
unordered_map<uint64_t, int> pattern_db;

int pattern_heuristic(const BitState& state) {
    // 將狀態分解為多個小模式
    // 查表獲得每個模式的精確代價
    // 取最大值作為啟發值
}
```

#### B. 死角懲罰啟發
```cpp
int corner_penalty_heuristic(const BitState& state) {
    int penalty = 0;
    for (int box_pos : get_box_positions(state)) {
        if (is_in_corner(box_pos) && !is_target(box_pos)) {
            penalty += 100; // 重懲罰死角中的箱子
        }
    }
    return penalty + hungarian_heuristic(state);
}
```

### 🔥 3. 增強死鎖檢測 (高優先級)

**新增檢測規則**:

#### A. 2x2方塊死鎖
```cpp
bool check_2x2_deadlock(const BitState& state) {
    for (int y = 0; y < height-1; y++) {
        for (int x = 0; x < width-1; x++) {
            if (has_box(y,x) && has_box(y,x+1) && 
                has_box(y+1,x) && has_box(y+1,x+1)) {
                // 檢查是否所有4個位置都不是目標
                if (!all_are_targets({(y,x), (y,x+1), (y+1,x), (y+1,x+1)})) {
                    return true;
                }
            }
        }
    }
    return false;
}
```

#### B. 線性衝突檢測
```cpp
bool check_linear_conflict(const BitState& state) {
    // 檢查同一行/列中箱子的相對位置衝突
    // 如果兩個箱子在同一行且目標位置相反，增加額外代價
}
```

### 🚀 4. 宏移動優化 (中優先級)

**當前問題**: 每次只考慮單步移動，搜索樹過深

**改進**:
```cpp
vector<pair<BitState, string>> generate_enhanced_macro_moves(const BitState& state) {
    vector<pair<BitState, string>> moves;
    
    // 1. 標準推箱子宏移動
    for (auto box_push : get_box_pushes(state)) {
        moves.push_back(box_push);
    }
    
    // 2. 多步推箱子組合（如果不產生死鎖）
    for (auto combo : get_safe_push_combinations(state)) {
        moves.push_back(combo);
    }
    
    return moves;
}
```

### ⚡ 5. 雙向搜索 (中優先級)

**原理**: 從起點和終點同時搜索，在中間相遇

```cpp
string bidirectional_ida_search(const BitState& start, const BitState& goal) {
    // 前向搜索
    auto forward_result = ida_search_forward(start, max_depth/2);
    
    // 後向搜索  
    auto backward_result = ida_search_backward(goal, max_depth/2);
    
    // 尋找相交狀態
    for (auto& state : forward_result) {
        if (backward_result.contains(state)) {
            return construct_path(forward_path[state] + backward_path[state]);
        }
    }
}
```

### 🔧 6. 技術優化 (中低優先級)

#### A. 轉置表優化
```cpp
// 使用 LRU 或時間戳管理轉置表大小
struct TTEntry {
    BitState state;
    int depth_bound;
    uint32_t timestamp;
};
```

#### B. 移動排序
```cpp
// 按啟發值排序後繼狀態，優先搜索有希望的分支
sort(successors.begin(), successors.end(), [](const auto& a, const auto& b) {
    return heuristic(a.state) < heuristic(b.state);
});
```

#### C. 記憶體池
```cpp
// 預分配記憶體池避免頻繁 malloc/free
class MemoryPool {
    vector<AStarNode> node_pool;
    size_t next_index = 0;
public:
    AStarNode* allocate() { return &node_pool[next_index++]; }
    void reset() { next_index = 0; }
};
```

## 實現優先級建議

### Phase 1 (立即實現)
1. **IDA* 替換 A*** - 解決記憶體瓶頸
2. **增強死鎖檢測** - 減少無效搜索
3. **模式數據庫啟發** - 提高搜索效率

### Phase 2 (後續實現)  
1. **雙向搜索** - 進一步減少搜索深度
2. **宏移動增強** - 減少搜索樹深度
3. **技術優化** - 提升實現效率

## 預期效果

- **記憶體使用**: 降低 80-90%
- **搜索節點**: 減少 50-70% 
- **執行時間**: 複雜測資提速 2-5x
- **成功率**: 25號等超難測資有望在合理時間內解決

## 實現注意事項

1. **逐步實現**: 每次只改一個優化，確保正確性
2. **性能測試**: 在簡單測資上驗證後再用於複雜測資
3. **降級機制**: IDA* 如果超時可降級到 A*
4. **參數調節**: 啟發函數權重需要實驗調整

通過這些優化，預期能將測資22-25的解決時間從分鐘級降低到秒級。
