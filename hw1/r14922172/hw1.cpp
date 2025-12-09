#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <climits>
#include <functional>
#include <bitset>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

using namespace std;

// Hash specialization for pair<int,int>
namespace std {
    template<>
    struct hash<pair<int, int>> {
        size_t operator()(const pair<int, int>& p) const {
            return hash<int>{}(p.first) ^ (hash<int>{}(p.second) << 1);
        }
    };
}

// Global variables for state compression
static vector<string> g_base_map;  // Base map with walls and targets, no boxes or player
static vector<pair<int,int>> g_movable_cells;  // All non-wall cells where boxes can be
static unordered_map<pair<int,int>, int, hash<pair<int,int>>> g_cell_to_index;  // Maps (y,x) to index in movable_cells

// Global variables to track target boundaries
int min_target_x = INT_MAX;
int max_target_x = INT_MIN;
int min_target_y = INT_MAX;
int max_target_y = INT_MIN;

// Precomputed target presence per row/column for fast deadlock pruning
static int g_height = 0;
static int g_width = 0;
static vector<bool> row_has_target;
static vector<bool> col_has_target;
// Precomputed targets and distances (per target -> grid distances)
static vector<pair<int,int>> g_all_targets;             // all goal cells
static unordered_map<long long,int> g_target_index;     // key(y,x) -> target idx
static vector<vector<int>> g_target_dists;              // size: targets x (H*W), INF if unreachable/wall

static inline long long keyYX(int y, int x) { return (static_cast<long long>(y) << 32) ^ static_cast<unsigned int>(x); }

static vector<int> bfs_dist_from_target(const vector<string>& m, int ty, int tx) {
    const int H = (int)m.size();
    const int W = H ? (int)m[0].size() : 0;
    const int INF = 1e9;
    vector<int> dist(H * W, INF);
    auto inb = [&](int y, int x){ return y>=0 && y<H && x>=0 && x<W; };
    if (!inb(ty, tx) || m[ty][tx] == '#') return dist;
    deque<pair<int,int>> dq;
    dq.emplace_back(ty, tx);
    dist[ty*W + tx] = 0;
    const int DY[4] = {-1,0,1,0};
    const int DX[4] = {0,-1,0,1};
    while (!dq.empty()) {
        auto [y,x] = dq.front(); dq.pop_front();
        int d = dist[y*W + x] + 1;
        for (int k=0;k<4;++k) {
            int ny = y + DY[k], nx = x + DX[k];
            if (!inb(ny,nx) || m[ny][nx] == '#') continue;
            int &ref = dist[ny*W + nx];
            if (d < ref) { ref = d; dq.emplace_back(ny,nx); }
        }
    }
    return dist;
}

class Oops : public runtime_error {
public:
    Oops(const string& msg) : runtime_error(msg) {}
};

// Maximum number of movable cells - increased for larger maps
const int MAX_MOVABLE_CELLS = 512;

// Bitmap-based state representation using bitsets
struct BitState {
    std::bitset<MAX_MOVABLE_CELLS> boxes_on_floor;    // Boxes on floor positions
    std::bitset<MAX_MOVABLE_CELLS> boxes_on_target;   // Boxes on target positions
    int player_pos;  // Player position as index in movable_cells array
    
    bool operator==(const BitState& other) const {
        return boxes_on_floor == other.boxes_on_floor && 
               boxes_on_target == other.boxes_on_target && 
               player_pos == other.player_pos;
    }
    
    // Get total number of boxes
    int box_count() const {
        return (int)(boxes_on_floor.count() + boxes_on_target.count());
    }
    
    // Check if position has a box
    bool has_box(int index) const {
        return boxes_on_floor[index] || boxes_on_target[index];
    }
    
    // Get box type at position: 0=no box, 1=box on floor, 2=box on target
    int get_box_type(int index) const {
        if (boxes_on_target[index]) return 2;
        if (boxes_on_floor[index]) return 1;
        return 0;
    }
    
    // Set box at position
    void set_box(int index, int type) {
        boxes_on_floor[index] = (type == 1);
        boxes_on_target[index] = (type == 2);
    }
    
    // Remove box at position
    void remove_box(int index) {
        boxes_on_floor[index] = false;
        boxes_on_target[index] = false;
    }
    
    // Move box from one position to another
    void move_box(int from_index, int to_index) {
        int box_type = get_box_type(from_index);
        remove_box(from_index);
        
        // Determine new box type based on target cell
        if (box_type > 0 && to_index < (int)g_movable_cells.size()) {
            int y = g_movable_cells[to_index].first;
            int x = g_movable_cells[to_index].second;
            bool is_target = (g_base_map[y][x] == '.');
            set_box(to_index, is_target ? 2 : 1);
        }
    }
};

struct BitStateHash {
    size_t operator()(const BitState& state) const {
        // Much simpler and faster hash - just use the underlying bitset hash
        // Modern std::bitset hash is already optimized
        auto h1 = std::hash<std::bitset<MAX_MOVABLE_CELLS>>{}(state.boxes_on_floor);
        auto h2 = std::hash<std::bitset<MAX_MOVABLE_CELLS>>{}(state.boxes_on_target);
        auto h3 = std::hash<int>{}(state.player_pos);
        
        // Simple combine - avoid complex bit operations
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

// Minimal GameState for legacy function compatibility - only used during loading and conversion
struct GameState {
    vector<string> map;
    int player_pos;  // Player position as integer index in movable_cells array
    
    bool operator==(const GameState& other) const {
        return map == other.map && player_pos == other.player_pos;
    }
};

// Optimized A* search node - only use BitState for minimal memory usage
struct AStarNode {
    // Only store the minimal bitmap representation
    BitState bit_state;
    
    // Metadata for search
    string path;
    int g_cost;    // Distance from start
    int h_cost;    // Heuristic to goal
    
    // Lightweight metadata for pruning immediate backtracking
    char last_move = 0;      // 'W','A','S','D' or 0 for none
    bool last_was_push = false;
    
    int f_cost() const { return g_cost + h_cost; }
    
    bool operator>(const AStarNode& other) const {
        if (f_cost() != other.f_cost()) {
            return f_cost() > other.f_cost();
        }
        return h_cost > other.h_cost;  // Tie-breaker: prefer lower h_cost
    }
};

// Convert GameState to bitmap representation
BitState compress_to_bitstate(const GameState& state) {
    BitState bit_state;
    bit_state.player_pos = state.player_pos;
    
    // Clear all bits first
    bit_state.boxes_on_floor.reset();
    bit_state.boxes_on_target.reset();
    
    // Set box positions based on map
    for (int y = 0; y < (int)state.map.size(); ++y) {
        for (int x = 0; x < (int)state.map[y].size(); ++x) {
            char c = state.map[y][x];
            if (c == 'x' || c == 'X') {  // Box on floor or target
                auto it = g_cell_to_index.find({y, x});
                if (it != g_cell_to_index.end()) {
                    int index = it->second;
                    if (c == 'x') {
                        bit_state.boxes_on_floor[index] = true;
                    } else {
                        bit_state.boxes_on_target[index] = true;
                    }
                }
            }
        }
    }
    
    return bit_state;
}

// Convert bitmap representation to GameState
GameState decompress_from_bitstate(const BitState& bit_state) {
    GameState state;
    state.map = g_base_map;  // Copy base map
    state.player_pos = bit_state.player_pos;
    
    // Place boxes based on bitmap
    for (int i = 0; i < (int)g_movable_cells.size() && i < MAX_MOVABLE_CELLS; ++i) {
        if (bit_state.boxes_on_floor[i] || bit_state.boxes_on_target[i]) {
            int y = g_movable_cells[i].first;
            int x = g_movable_cells[i].second;
            
            if (bit_state.boxes_on_floor[i]) {
                state.map[y][x] = 'x';  // Box on floor
            } else {
                state.map[y][x] = 'X';  // Box on target
            }
        }
    }
    
    // Update player position on map
    if (bit_state.player_pos >= 0 && bit_state.player_pos < (int)g_movable_cells.size()) {
        int py = g_movable_cells[bit_state.player_pos].first;
        int px = g_movable_cells[bit_state.player_pos].second;
        
        char base_cell = g_base_map[py][px];
        if (base_cell == '.') {
            state.map[py][px] = 'O';  // Player on target
        } else {
            state.map[py][px] = 'o';  // Player on floor
        }
    }
    
    return state;
}

// Initialize compression data structures
void init_compression(const GameState& initial_state) {
    g_base_map = initial_state.map;
    g_movable_cells.clear();
    g_cell_to_index.clear();
    
    // Create base map (walls and targets only, no boxes or player)
    for (int y = 0; y < (int)g_base_map.size(); ++y) {
        for (int x = 0; x < (int)g_base_map[y].size(); ++x) {
            char c = g_base_map[y][x];
            if (c != '#') {  // Not a wall
                g_cell_to_index[{y, x}] = g_movable_cells.size();
                g_movable_cells.push_back({y, x});
                
                // Set base map cell
                if (c == 'x') {
                    g_base_map[y][x] = ' ';  // Box on floor -> floor
                } else if (c == 'X') {
                    g_base_map[y][x] = '.';  // Box on target -> target
                } else if (c == 'o') {
                    g_base_map[y][x] = ' ';  // Player on floor -> floor
                } else if (c == 'O') {
                    g_base_map[y][x] = '.';  // Player on target -> target
                }
            }
        }
    }
}

// Direction mappings
const vector<pair<char, pair<int, int>>> DYDX = {
    {'W', {-1, 0}},  // Up
    {'A', {0, -1}},  // Left
    {'S', {1, 0}},   // Down
    {'D', {0, 1}}    // Right
};

static inline char opposite_dir(char c) {
    switch (c) {
        case 'W': return 'S';
        case 'S': return 'W';
        case 'A': return 'D';
        case 'D': return 'A';
        default: return 0;
    }
}

// Enhanced heuristic using Hungarian algorithm (exact matching) + player proximity
// - Boxes considered: only unsolved boxes 'x'
// - Targets considered: only free targets '.' or 'O' (player on goal is still free)
// This yields an admissible lower bound of required pushes; we also add
// the Manhattan distance from player to the nearest unsolved box (also admissible).
static int hungarian_min_cost(const vector<vector<int>>& a) {
    // Implementation of the O(n^3) Hungarian algorithm for square cost matrix
    // Assumes a.size() == a[0].size() == n and entries are non-negative ints
    int n = (int)a.size();
    if (n == 0) return 0;
    vector<int> u(n + 1), v(n + 1), p(n + 1), way(n + 1);
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        vector<int> minv(n + 1, INT_MAX);
        vector<char> used(n + 1, false);
        do {
            used[j0] = true;
            int i0 = p[j0], delta = INT_MAX, j1 = 0;
            for (int j = 1; j <= n; ++j) if (!used[j]) {
                int cur = a[i0 - 1][j - 1] - u[i0] - v[j];
                if (cur < minv[j]) minv[j] = cur, way[j] = j0;
                if (minv[j] < delta) delta = minv[j], j1 = j;
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) u[p[j]] += delta, v[j] -= delta;
                else minv[j] -= delta;
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    int cost = -v[0];
    return cost;
}

// Removed - see BitState version below

// Forward declarations
int calculate_heuristic_bitstate_enhanced(const BitState& bit_state);
int calculate_heuristic_bitstate_hungarian(const BitState& bit_state);
int calculate_heuristic_bitstate_adaptive(const BitState& bit_state);

// (Old adaptive heuristic removed - replaced by new implementation further below)

// Hungarian algorithm based heuristic for BitState
// Helper: gather unsolved boxes and free target indices
static inline void collect_boxes_and_free_targets(const BitState &bs, vector<int> &unsolved_indices, vector<int> &free_target_indices) {
    unsolved_indices.clear();
    free_target_indices.clear();
    // mark occupied targets by solved boxes
    vector<char> target_taken(g_all_targets.size(), 0);
    for (int i = 0; i < (int)g_movable_cells.size(); ++i) {
        if (bs.boxes_on_target[i]) {
            int y = g_movable_cells[i].first;
            int x = g_movable_cells[i].second;
            auto it = g_target_index.find(keyYX(y,x));
            if (it != g_target_index.end()) target_taken[it->second] = 1;
        }
    }
    for (int i = 0; i < (int)g_movable_cells.size(); ++i) if (bs.boxes_on_floor[i]) unsolved_indices.push_back(i);
    for (int t = 0; t < (int)g_all_targets.size(); ++t) if (!target_taken[t]) free_target_indices.push_back(t);
}

// Adaptive heuristic with intelligent strategy selection
int calculate_heuristic_bitstate_hungarian(const BitState& bit_state) {
    if (bit_state.boxes_on_floor.none()) return 0;
    vector<int> boxes_idx, free_targets; 
    boxes_idx.reserve(32); 
    free_targets.reserve(32);
    collect_boxes_and_free_targets(bit_state, boxes_idx, free_targets);
    if (boxes_idx.empty()) return 0;
    if (boxes_idx.size() != free_targets.size()) {
        // Fallback: return simple count if mismatch
        return (int)bit_state.boxes_on_floor.count();
    }
    
    int n = (int)boxes_idx.size();
    
    // Fast path for very small instances
    if (n == 1) {
        const auto &cell = g_movable_cells[boxes_idx[0]]; 
        int flat = cell.first * g_width + cell.second;
        return g_target_dists[free_targets[0]][flat];
    }
    
    // Use greedy for instances > 4 to balance speed and quality
    if (n > 4) {
        // Optimized greedy matching with better strategy
        int total = 0; 
        vector<char> used(n, 0);
        
        // Sort boxes by minimum distance to any target (process harder boxes first)
        vector<pair<int, int>> box_min_dist; // (min_dist, box_idx_in_array)
        box_min_dist.reserve(n);
        
        for (int i = 0; i < n; ++i) {
            const auto &cell = g_movable_cells[boxes_idx[i]]; 
            int flat = cell.first * g_width + cell.second;
            int min_d = INT_MAX;
            for (int j = 0; j < n; ++j) {
                min_d = min(min_d, g_target_dists[free_targets[j]][flat]);
            }
            box_min_dist.emplace_back(min_d, i);
        }
        
        sort(box_min_dist.begin(), box_min_dist.end(), greater<pair<int,int>>());
        
        for (const auto &[_, bi_idx] : box_min_dist) {
            const auto &cell = g_movable_cells[boxes_idx[bi_idx]]; 
            int flat = cell.first * g_width + cell.second;
            int best = INT_MAX, bestj = -1;
            
            for (int j = 0; j < n; ++j) {
                if (!used[j]) {
                    int dist = g_target_dists[free_targets[j]][flat];
                    if (dist < best) { 
                        best = dist; 
                        bestj = j; 
                    }
                }
            }
            
            if (bestj >= 0) { 
                used[bestj] = 1; 
                total += best; 
            }
        }
        return total;
    }
    
    // Hungarian algorithm for small-medium instances (2-6 boxes) - best quality
    vector<vector<int>> cost(n, vector<int>(n));
    
    // Build cost matrix: box-to-target distances
    for (int i = 0; i < n; ++i) {
        const auto &cell = g_movable_cells[boxes_idx[i]]; 
        int flat = cell.first * g_width + cell.second;
        for (int j = 0; j < n; ++j) {
            cost[i][j] = g_target_dists[free_targets[j]][flat];
        }
    }
    
    return hungarian_min_cost(cost);
}

// Simple greedy matching heuristic
int calculate_heuristic_bitstate_enhanced(const BitState& bit_state) {
    if (bit_state.boxes_on_floor.none()) return 0;
    vector<int> boxes_idx, free_targets; boxes_idx.reserve(32); free_targets.reserve(32);
    collect_boxes_and_free_targets(bit_state, boxes_idx, free_targets);
    if (boxes_idx.empty()) return 0;
    
    // Simple greedy matching - fast and effective
    int total = 0; 
    vector<char> used(free_targets.size(), 0);
    
    // For each box, find the best target
    for (int bi : boxes_idx) {
        const auto &cell = g_movable_cells[bi]; 
        int flat = cell.first * g_width + cell.second;
        int best = INT_MAX, bestj = -1;
        
        for (int j = 0; j < (int)free_targets.size(); ++j) {
            if (!used[j]) {
                int dist = g_target_dists[free_targets[j]][flat];
                if (dist < best) { 
                    best = dist; 
                    bestj = j; 
                }
            }
        }
        
        if (bestj >= 0) { 
            used[bestj] = 1; 
            total += best; 
        }
    }
    
    return total;
}

// Simple count-based heuristic for better performance
int calculate_heuristic_bitstate_simple(const BitState& bit_state) { return (int)bit_state.boxes_on_floor.count(); }

// Adaptive heuristic - always use Hungarian for optimal heuristic quality
int calculate_heuristic_bitstate_adaptive(const BitState& bit_state) {
    if (bit_state.boxes_on_floor.none()) return 0;
    
    // Always use Hungarian algorithm for best admissible heuristic
    return calculate_heuristic_bitstate_hungarian(bit_state);
}

// Ultra-simple deadlock detection for stability (restored full function)
bool is_deadlock(const GameState& state) {
    auto tunnel_deadlock = [&](int y, int x) -> bool {
        if (state.map[y][x] != 'x') return false;
        int H = (int)state.map.size();
        int W = (int)state.map[0].size();
        auto is_goal = [&](char c){ return c == '.' || c == 'X' || c == 'O'; };
        auto inb = [&](int yy, int xx){ return yy >= 0 && yy < H && xx >= 0 && xx < W; };
        if (inb(y, x-1) && inb(y, x+1) && state.map[y][x-1] == '#' && state.map[y][x+1] == '#') {
            int y0 = y, y1 = y; bool ok_up = true, ok_dn = true; bool any_goal = is_goal(state.map[y][x]); int yy = y;
            while (yy-1 >= 0 && state.map[yy-1][x] != '#' && inb(yy-1, x-1) && inb(yy-1, x+1) && state.map[yy-1][x-1] == '#' && state.map[yy-1][x+1] == '#') { yy--; any_goal = any_goal || is_goal(state.map[yy][x]); }
            if (!(yy-1 >= 0 && state.map[yy-1][x] == '#')) ok_up = false; y0 = yy;
            yy = y;
            while (yy+1 < H && state.map[yy+1][x] != '#' && inb(yy+1, x-1) && inb(yy+1, x+1) && state.map[yy+1][x-1] == '#' && state.map[yy+1][x+1] == '#') { yy++; any_goal = any_goal || is_goal(state.map[yy][x]); }
            if (!(yy+1 < H && state.map[yy+1][x] == '#')) ok_dn = false; y1 = yy;
            if (ok_up && ok_dn && !any_goal) return true;
        }
        if (inb(y-1, x) && inb(y+1, x) && state.map[y-1][x] == '#' && state.map[y+1][x] == '#') {
            int x0 = x, x1 = x; bool ok_lt = true, ok_rt = true; bool any_goal = is_goal(state.map[y][x]); int xx = x;
            while (xx-1 >= 0 && state.map[y][xx-1] != '#' && inb(y-1, xx-1) && inb(y+1, xx-1) && state.map[y-1][xx-1] == '#' && state.map[y+1][xx-1] == '#') { xx--; any_goal = any_goal || is_goal(state.map[y][xx]); }
            if (!(xx-1 >= 0 && state.map[y][xx-1] == '#')) ok_lt = false; x0 = xx;
            xx = x;
            while (xx+1 < W && state.map[y][xx+1] != '#' && inb(y-1, xx+1) && inb(y+1, xx+1) && state.map[y-1][xx+1] == '#' && state.map[y+1][xx+1] == '#') { xx++; any_goal = any_goal || is_goal(state.map[y][xx]); }
            if (!(xx+1 < W && state.map[y][xx+1] == '#')) ok_rt = false; x1 = xx;
            if (ok_lt && ok_rt && !any_goal) return true;
        }
        return false;
    };
    for (int y = 1; y < (int)state.map.size() - 1; y++) {
        for (int x = 1; x < (int)state.map[y].size() - 1; x++) {
            if (state.map[y][x] == 'x') {
                char up = state.map[y-1][x]; char down = state.map[y+1][x]; char left = state.map[y][x-1]; char right = state.map[y][x+1];
                char upleft = state.map[y-1][x-1]; char upright = state.map[y-1][x+1]; char downleft = state.map[y+1][x-1]; char downright = state.map[y+1][x+1];
                if ((up == '#' && left == '#') || (up == '#' && right == '#') || (down == '#' && left == '#') || (down == '#' && right == '#')) return true;
                if (up == '#') {
                    if (left == '#' || ((left == 'x' || left == 'X') && (upleft == '#' || downleft == '#'))) return true;
                    if (right == '#' || ((right == 'x' || right == 'X') && (upright == '#' || downright == '#'))) return true;
                    bool canyon_detected = true;
                    for (int i = x; i > 0 && (state.map[y][i] != '#'); i--) if (state.map[y-1][i] != '#') { canyon_detected = false; break; }
                    for (int i = x; i < (int)state.map[y].size() - 1 && (state.map[y][i] != '#'); i++) if (state.map[y-1][i] != '#') { canyon_detected = false; break; }
                    if (canyon_detected && max_target_y < y) return true;
                }
                if (left == '#') {
                    if (up == '#' || ((up == 'x' || up == 'X') && (upleft == '#' || upright == '#'))) return true;
                    if (down == '#' || ((down == 'x' || down == 'X') && (downleft == '#' || downright == '#'))) return true;
                    bool canyon_detected = true;
                    for (int i = y; i > 0 && (state.map[i][x] != '#'); i--) if (state.map[i][x-1] != '#') { canyon_detected = false; break; }
                    for (int i = y; i < (int)state.map.size() - 1 && (state.map[i][x] != '#'); i++) if (state.map[i][x-1] != '#') { canyon_detected = false; break; }
                    if (canyon_detected && min_target_x > x) return true;
                }
                if (right == '#') {
                    if (up == '#' || ((up == 'x' || up == 'X') && (upright == '#' || upleft == '#'))) return true;
                    if (down == '#' || ((down == 'x' || down == 'X') && (downright == '#' || downleft == '#'))) return true;
                    bool canyon_detected = true;
                    for (int i = y; i > 0 && (state.map[i][x] != '#'); i--) if (state.map[i][x+1] != '#') { canyon_detected = false; break; }
                    for (int i = y; i < (int)state.map.size() - 1 && (state.map[i][x] != '#'); i++) if (state.map[i][x+1] != '#') { canyon_detected = false; break; }
                    if (canyon_detected && max_target_x < x) return true;
                }
                if (down == '#') {
                    if (left == '#' || ((left == 'x' || left == 'X') && (downleft == '#' || upleft == '#'))) return true;
                    if (right == '#' || ((right == 'x' || right == 'X') && (downright == '#' || upright == '#'))) return true;
                    bool canyon_detected = true;
                    for (int i = x; i > 0 && (state.map[y][i] != '#'); i--) if (state.map[y+1][i] != '#') { canyon_detected = false; break; }
                    for (int i = x; i < (int)state.map[y].size() - 1 && (state.map[y][i] != '#'); i++) if (state.map[y+1][i] != '#') { canyon_detected = false; break; }
                    if (canyon_detected && min_target_y > y) return true;
                }
                if ((up == 'x' || up == 'X' || up == '#') && (left == 'x' || left == 'X' || left == '#') && (upleft == '#' || upleft == 'x' || upleft == 'X')) return true;
                if ((up == 'x' || up == 'X' || up == '#') && (right == 'x' || right == 'X' || right == '#') && (upright == '#' || upright == 'x' || upright == 'X')) return true;
                if ((down == 'x' || down == 'X' || down == '#') && (left == 'x' || left == 'X' || left == '#') && (downleft == '#' || downleft == 'x' || downleft == 'X')) return true;
                if ((down == 'x' || down == 'X' || down == '#') && (right == 'x' || right == 'X' || right == '#') && (downright == '#' || downright == 'x' || downright == 'X')) return true;
            }
        }
    }
    return false;
}

// Load state from file
GameState loadstate(const string& filename) {
    vector<string> m;
    pair<int, int> player_pos = {-1, -1};
    vector<int> widths;
    unordered_map<char, int> stats;
    
    ifstream file(filename);
    if (!file.is_open()) {
        throw Oops("Cannot open file: " + filename);
    }
    
    string line;
    int y = 0;
    while (getline(file, line)) {
        while (!line.empty() && isspace(line.back())) {
            line.pop_back();
        }
        
        for (int x = 0; x < line.length(); x++) {
            char c = line[x];
            if (c == 'o' || c == '!' || c == 'O') {
                player_pos = {y, x};
            }
            // Update target boundaries for target positions
            if (c == '.' || c == 'X' || c == 'O') {
                min_target_x = min(min_target_x, x);
                max_target_x = max(max_target_x, x);
                min_target_y = min(min_target_y, y);
                max_target_y = max(max_target_y, y);
            }
        }
        
        widths.push_back(line.length());
        m.push_back(line);
        
        for (char c : line) {
            stats[c]++;
        }
        y++;
    }
    
    if (m.empty()) {
        throw Oops("input file is empty");
    }
    
    // Validation
    unordered_set<char> valid_chars = {'x', 'X', 'o', 'O', '.', ' ', '#', '@', '!'};
    for (const auto& [c, count] : stats) {
        if (valid_chars.find(c) == valid_chars.end()) {
            throw Oops("input file contains invalid characters: " + string(1, c));
        }
    }
    
    int boxes = stats['x'] + stats['X'];
    int targets = stats['.'] + stats['X'] + stats['O'];
    if (boxes != targets) {
        throw Oops("got " + to_string(boxes) + " boxes and " + to_string(targets) + " targets in input");
    }
    
    int nplayers = stats['o'] + stats['O'] + stats['!'];
    if (nplayers != 1) {
        throw Oops("got " + to_string(nplayers) + " players in input");
    }
    
    if (!widths.empty()) {
        int first_width = widths[0];
        for (int width : widths) {
            if (width != first_width) {
                throw Oops("input rows having different widths");
            }
        }
    }
    
    if (player_pos.first == -1 || player_pos.second == -1) {
        throw Oops("player position not found");
    }

    // Precompute dimensions and row/column target presence
    g_height = (int)m.size();
    g_width = (int)(!m.empty() ? m[0].size() : 0);
    row_has_target.assign(g_height, false);
    col_has_target.assign(g_width, false);
    g_all_targets.clear();
    g_target_index.clear();
    for (int yy = 0; yy < g_height; ++yy) {
        for (int xx = 0; xx < g_width; ++xx) {
            char c = m[yy][xx];
            if (c == '.' || c == 'X' || c == 'O') {
                row_has_target[yy] = true;
                col_has_target[xx] = true;
                // Collect all goal cells for distance precomputation
                g_target_index[keyYX(yy,xx)] = (int)g_all_targets.size();
                g_all_targets.emplace_back(yy,xx);
            }
        }
    }

    // Precompute BFS distances from each target to every cell (ignoring boxes, respecting walls)
    g_target_dists.clear();
    g_target_dists.reserve(g_all_targets.size());
    for (auto &t : g_all_targets) {
        g_target_dists.emplace_back(bfs_dist_from_target(m, t.first, t.second));
    }

    // Initialize compression mapping first
    g_movable_cells.clear();
    g_cell_to_index.clear();
    
    // Create compression mapping (walls and targets only, no boxes or player)
    for (int y = 0; y < (int)m.size(); ++y) {
        for (int x = 0; x < (int)m[y].size(); ++x) {
            char c = m[y][x];
            if (c != '#') {  // Not a wall
                if ((int)g_movable_cells.size() >= MAX_MOVABLE_CELLS) {
                    throw Oops("Map too large: exceeds MAX_MOVABLE_CELLS (" + to_string(MAX_MOVABLE_CELLS) + ")");
                }
                g_cell_to_index[{y, x}] = g_movable_cells.size();
                g_movable_cells.push_back({y, x});
            }
        }
    }

    // Convert player_pos pair to integer index
    int player_index = -1;
    if (player_pos.first != -1 && player_pos.second != -1) {
        auto it = g_cell_to_index.find(player_pos);
        if (it != g_cell_to_index.end()) {
            player_index = it->second;
        }
    }

    return {m, player_index};
}

// Check if the game is solved using BitState
bool is_solved(const BitState& bit_state) {
    // Game is solved when all boxes are on targets (no boxes on floor)
    return bit_state.boxes_on_floor.none();
}

// Ultra-simple deadlock detection using BitState
bool is_deadlock(const BitState& bit_state) {
    // Convert to GameState for legacy deadlock detection
    // TODO: Implement direct BitState deadlock detection for better performance
    GameState state = decompress_from_bitstate(bit_state);
    return is_deadlock(state);
}

// Calculate heuristic using BitState
int calculate_heuristic(const BitState& bit_state) { return calculate_heuristic_bitstate_adaptive(bit_state); }

// Check if player can reach target position without moving any boxes
bool can_player_reach(const BitState& bit_state, int target_pos) {
    if (bit_state.player_pos == target_pos) {
        return true;
    }
    
    vector<bool> visited(g_movable_cells.size(), false);
    queue<int> q;
    q.push(bit_state.player_pos);
    visited[bit_state.player_pos] = true;
    
    while (!q.empty()) {
        int pos = q.front();
        q.pop();
        
        if (pos == target_pos) {
            return true;
        }
        
        int y = g_movable_cells[pos].first;
        int x = g_movable_cells[pos].second;
        
        const int DY[4] = {-1, 0, 1, 0};  // Up, Left, Down, Right
        const int DX[4] = {0, -1, 0, 1};
        
        for (int dir = 0; dir < 4; dir++) {
            int ny = y + DY[dir];
            int nx = x + DX[dir];
            
            // Check bounds
            if (ny < 0 || ny >= g_height || nx < 0 || nx >= g_width) {
                continue;
            }
            
            // Check if it's a wall
            if (g_base_map[ny][nx] == '#') {
                continue;
            }
            
            // Find index of new position
            auto it = g_cell_to_index.find({ny, nx});
            if (it == g_cell_to_index.end()) {
                continue;
            }
            int new_pos = it->second;
            
            // Skip if there's a box at this position
            if (bit_state.has_box(new_pos)) {
                continue;
            }
            
            // Skip if already visited
            if (visited[new_pos]) {
                continue;
            }
            
            visited[new_pos] = true;
            q.push(new_pos);
        }
    }
    
    return false;
}

// Generate macro moves - only return states where boxes are actually moved
// This compresses all pure player movements into the resulting push states
// Helper function to find path from one position to another
// Step 1 Optimization: integrated path reconstruction inside macro move generation
// (Removes need for a secondary BFS per push.)
vector<pair<BitState, string>> generate_macro_moves(const BitState& bit_state) {
    vector<pair<BitState, string>> push_states;
    
    const size_t N = g_movable_cells.size();
    vector<int> parent(N, -1);
    vector<char> move_dir(N, 0);
    vector<char> reachable(N, 0); // use char for compactness

    const int start = bit_state.player_pos;
    queue<int> q; q.push(start);
    parent[start] = start; reachable[start] = 1;

    static constexpr int DY[4] = {-1, 0, 1, 0};  // Up, Left, Down, Right
    static constexpr int DX[4] = {0, -1, 0, 1};
    static constexpr char DIRS[4] = {'W', 'A', 'S', 'D'};

    auto build_path = [&](int to_pos){
        string path; path.reserve(32);
        for (int cur = to_pos; cur != start; cur = parent[cur]) path.push_back(move_dir[cur]);
        reverse(path.begin(), path.end());
        return path;
    };

    while (!q.empty()) {
        int pos = q.front(); q.pop();
        int y = g_movable_cells[pos].first;
        int x = g_movable_cells[pos].second;

        for (int dir = 0; dir < 4; ++dir) {
            int ny = y + DY[dir];
            int nx = x + DX[dir];
            if ((unsigned)ny >= (unsigned)g_height || (unsigned)nx >= (unsigned)g_width) continue;
            if (g_base_map[ny][nx] == '#') continue;
            auto it = g_cell_to_index.find({ny,nx});
            if (it == g_cell_to_index.end()) continue;
            int next_pos = it->second;

            if (bit_state.has_box(next_pos)) {
                // Attempt a push
                int box_ny = ny + DY[dir];
                int box_nx = nx + DX[dir];
                if ((unsigned)box_ny >= (unsigned)g_height || (unsigned)box_nx >= (unsigned)g_width) continue;
                if (g_base_map[box_ny][box_nx] == '#') continue;
                if (g_base_map[box_ny][box_nx] == '@' || g_base_map[box_ny][box_nx] == '!') continue;
                auto box_it = g_cell_to_index.find({box_ny, box_nx});
                if (box_it == g_cell_to_index.end()) continue;
                int box_dest = box_it->second;
                if (bit_state.has_box(box_dest)) continue; // blocked
                if (!reachable[pos]) continue; // cannot stand to push

                BitState new_state = bit_state;
                new_state.move_box(next_pos, box_dest);
                new_state.player_pos = next_pos; // player moves into box original spot

                string path = build_path(pos);
                path.push_back(DIRS[dir]);
                push_states.emplace_back(std::move(new_state), std::move(path));
            } else {
                if (!reachable[next_pos]) {
                    reachable[next_pos] = 1;
                    parent[next_pos] = pos;
                    move_dir[next_pos] = DIRS[dir];
                    q.push(next_pos);
                }
            }
        }
    }
    return push_states;
}

// Check if the game is solved using GameState (legacy)
bool is_solved(const GameState& state) {
    for (const string& row : state.map) {
        for (char c : row) {
            if (c == 'x') {  // Unsolved box
                return false;
            }
        }
    }
    return true;
}

// A* solver with BitState optimization and enhanced node ordering
string astar_solver(const string& filename) {
    GameState initial_state = loadstate(filename);
    // Normalize base map (already mostly done in loadstate but ensure consistency)
    g_base_map = initial_state.map;
    for (int y = 0; y < (int)g_base_map.size(); ++y) {
        for (int x = 0; x < (int)g_base_map[y].size(); ++x) {
            char &c = g_base_map[y][x];
            if (c == 'x') c = ' ';
            else if (c == 'X') c = '.';
            else if (c == 'o') c = ' ';
            else if (c == 'O') c = '.';
        }
    }
    priority_queue<AStarNode, vector<AStarNode>, greater<AStarNode>> open_set;
    unordered_set<BitState, BitStateHash> visited_states;
    AStarNode start_node; 
    start_node.bit_state = compress_to_bitstate(initial_state);
    start_node.path = ""; 
    start_node.g_cost = 0;
    start_node.h_cost = calculate_heuristic_bitstate_adaptive(start_node.bit_state);
    start_node.last_move = 0; 
    start_node.last_was_push = false;
    open_set.push(start_node);
    visited_states.insert(start_node.bit_state);
    
    int nodes_expanded = 0;
    const int DEADLOCK_CHECK_INTERVAL = 5; // Check deadlock every N nodes for performance
    
    while (!open_set.empty()) {
        AStarNode current = open_set.top(); 
        open_set.pop();
        ++nodes_expanded;
        
        if (is_solved(current.bit_state)) {
            cerr << "A* solved in " << nodes_expanded << " nodes" << endl;
            return current.path;
        }
        
        // Periodic deadlock checking to reduce overhead
        if (nodes_expanded % DEADLOCK_CHECK_INTERVAL == 0 && is_deadlock(current.bit_state)) {
            continue;
        }
        
        auto successors = generate_macro_moves(current.bit_state);
        
        // Score and sort successors for better node ordering
        vector<pair<int,int>> scored; 
        scored.reserve(successors.size());
        
        for (int i = 0; i < (int)successors.size(); ++i) {
            const BitState &nb = successors[i].first;
            if (visited_states.find(nb) != visited_states.end()) continue;
            
            // Quick deadlock check for obviously bad states
            if (is_deadlock(nb)) continue;
            
            int h = calculate_heuristic_bitstate_adaptive(nb);
            scored.emplace_back(h, i);
        }
        
        // Sort by heuristic (best first)
        sort(scored.begin(), scored.end());
        
        // Limit expansion to top candidates for large branching factors
        int expansion_limit = min((int)scored.size(), 50);
        
        for (int idx = 0; idx < expansion_limit; ++idx) {
            int i = scored[idx].second; 
            const auto &pairRef = successors[i];
            const BitState &nb = pairRef.first; 
            const string &mp = pairRef.second;
            
            if (!visited_states.insert(nb).second) continue;
            
            AStarNode nn; 
            nn.bit_state = nb; 
            nn.path = current.path + mp; 
            nn.g_cost = current.g_cost + 1; 
            nn.h_cost = scored[idx].first;
            nn.last_move = mp.back(); 
            nn.last_was_push = true; 
            open_set.push(std::move(nn));
        }
    }
    throw Oops("A* search could not find solution");
}

// Optimized Parallel A* - streamlined for maximum throughput
string astar_solver_parallel(const string& filename, int num_threads) {
    GameState initial_state = loadstate(filename);
    g_base_map = initial_state.map;
    for (int y = 0; y < (int)g_base_map.size(); ++y) {
        for (int x = 0; x < (int)g_base_map[y].size(); ++x) {
            char &c = g_base_map[y][x];
            if (c == 'x') c = ' ';
            else if (c == 'X') c = '.';
            else if (c == 'o') c = ' ';
            else if (c == 'O') c = '.';
        }
    }
    
    using PQ = priority_queue<AStarNode, vector<AStarNode>, greater<AStarNode>>;
    PQ open_set; 
    unordered_set<BitState, BitStateHash> visited;
    
    AStarNode start; 
    start.bit_state = compress_to_bitstate(initial_state);
    start.path = ""; 
    start.g_cost = 0; 
    start.h_cost = calculate_heuristic_bitstate_adaptive(start.bit_state);
    open_set.push(start); 
    visited.insert(start.bit_state);
    
    mutex pq_mutex, visited_mutex; 
    condition_variable pq_cv; 
    atomic<bool> solved(false);
    string solution_path; 
    atomic<long long> nodes_expanded{0};
    
    auto worker = [&](){
        constexpr int CHUNK_SIZE = 64; // Smaller batches for better responsiveness
        constexpr int DEADLOCK_CHECK_INTERVAL = 7;
        constexpr int MAX_SUCCESSORS = 60;
        
        vector<AStarNode> batch; 
        batch.reserve(CHUNK_SIZE);
        vector<AStarNode> new_nodes; 
        new_nodes.reserve(256);
        vector<BitState> local_to_check; 
        local_to_check.reserve(256);
        
        while (!solved.load(memory_order_relaxed)) {
            batch.clear();
            
            // Fetch batch of nodes
            {
                unique_lock<mutex> lk(pq_mutex);
                pq_cv.wait(lk, [&]{
                    return solved.load(memory_order_relaxed) || !open_set.empty();
                });
                
                if (solved.load(memory_order_relaxed)) return;
                
                int actual_batch = CHUNK_SIZE;
                // Adaptive batch size based on queue depth
                if (open_set.size() > 1000) actual_batch = 64;
                else if (open_set.size() < 100) actual_batch = 16;
                
                for (int i = 0; i < actual_batch && !open_set.empty(); ++i) { 
                    batch.push_back(open_set.top()); 
                    open_set.pop(); 
                }
                
                if (batch.empty()) continue;
            }
            
            // Process batch without locks
            local_to_check.clear();
            new_nodes.clear();
            
            for (auto &current : batch) {
                if (solved.load(memory_order_relaxed)) break;
                
                long long expanded = nodes_expanded.fetch_add(1, memory_order_relaxed);
                
                if (is_solved(current.bit_state)) { 
                    solution_path = current.path; 
                    solved.store(true, memory_order_release); 
                    pq_cv.notify_all(); 
                    return; 
                }
                
                // Periodic deadlock checking
                if (expanded % DEADLOCK_CHECK_INTERVAL == 0 && is_deadlock(current.bit_state)) {
                    continue;
                }
                
                auto successors = generate_macro_moves(current.bit_state);
                
                // Score successors
                vector<pair<int, int>> scored;
                scored.reserve(successors.size());
                
                for (int i = 0; i < (int)successors.size(); ++i) {
                    if (solved.load(memory_order_relaxed)) break;
                    
                    const BitState &nb = successors[i].first;
                    
                    // Skip obvious deadlocks early
                    if (is_deadlock(nb)) continue;
                    
                    int h = calculate_heuristic_bitstate_adaptive(nb);
                    scored.emplace_back(h, i);
                }
                
                // Sort and limit expansion
                sort(scored.begin(), scored.end());
                int limit = min((int)scored.size(), MAX_SUCCESSORS);
                
                for (int idx = 0; idx < limit; ++idx) {
                    if (solved.load(memory_order_relaxed)) break;
                    
                    int i = scored[idx].second;
                    const auto &pr = successors[i];
                    const BitState &nb = pr.first; 
                    const string &mp = pr.second;
                    
                    local_to_check.push_back(nb);
                    
                    AStarNode nn; 
                    nn.bit_state = nb; 
                    nn.path = current.path + mp; 
                    nn.g_cost = current.g_cost + 1; 
                    nn.h_cost = scored[idx].first; 
                    nn.last_move = mp.back(); 
                    nn.last_was_push = true;
                    new_nodes.push_back(std::move(nn));
                }
            }
            
            // Batch visited check and insertion
            if (!local_to_check.empty()) {
                vector<bool> should_insert(local_to_check.size(), false);
                
                {
                    lock_guard<mutex> g(visited_mutex);
                    for (size_t i = 0; i < local_to_check.size(); ++i) {
                        if (visited.find(local_to_check[i]) == visited.end()) {
                            visited.insert(local_to_check[i]);
                            should_insert[i] = true;
                        }
                    }
                }
                
                // Filter new nodes based on visited check
                size_t write_pos = 0;
                for (size_t i = 0; i < new_nodes.size() && i < should_insert.size(); ++i) {
                    if (should_insert[i]) {
                        if (write_pos != i) {
                            new_nodes[write_pos] = std::move(new_nodes[i]);
                        }
                        ++write_pos;
                    }
                }
                new_nodes.resize(write_pos);
            }
            
            // Batch insert into priority queue
            if (!new_nodes.empty()) {
                {
                    lock_guard<mutex> lk(pq_mutex);
                    for (auto &n : new_nodes) {
                        open_set.push(std::move(n));
                    }
                }
                pq_cv.notify_all();
            }
        }
    };
    
    vector<thread> threads; 
    threads.reserve(num_threads);
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back(worker);
    }
    
    for (auto &t : threads) {
        t.join();
    }
    
    if (!solved.load(memory_order_relaxed)) {
        throw Oops("Parallel A* failed to find solution");
    }
    
    return solution_path;
}

// Main solver using only A*
string main_solver(const string& filename) {
    return astar_solver(filename);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <input_file>" << endl;
        return 1;
    }
    constexpr int THREADS = 6; // 固定 6 執行緒
    try {
        auto start_time = chrono::high_resolution_clock::now();
        string result = astar_solver_parallel(argv[1], THREADS);
        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
        cout << result << endl;
        cerr << "Solved in " << duration.count() << " ms (threads=" << THREADS << ")" << endl;
        return 0;
    } catch (const Oops& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    } catch (const exception& e) {
        cerr << "Unexpected error: " << e.what() << endl;
        return 1;
    }
}
