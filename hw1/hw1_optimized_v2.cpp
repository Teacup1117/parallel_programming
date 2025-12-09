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

// Calculate push distance from player to push box to target
int calculate_push_distance(const GameState& state, int box_y, int box_x, int target_y, int target_x) {
    const int H = (int)state.map.size();
    const int W = H ? (int)state.map[0].size() : 0;
    
    int py = g_movable_cells[state.player_pos].first;
    int px = g_movable_cells[state.player_pos].second;
    
    // Calculate Manhattan distance from box to target (push distance)
    int push_dist = abs(box_y - target_y) + abs(box_x - target_x);
    
    // Calculate minimum distance for player to reach any position where they can push the box toward target
    int min_player_dist = INT_MAX;
    
    // Check all 4 directions from box position to find valid push positions
    const int DY[4] = {-1, 0, 1, 0}; // up, right, down, left
    const int DX[4] = {0, 1, 0, -1};
    
    for (int dir = 0; dir < 4; ++dir) {
        int push_to_y = box_y + DY[dir];
        int push_to_x = box_x + DX[dir];
        
        // Check if pushing in this direction moves box toward target
        int new_box_dist = abs(push_to_y - target_y) + abs(push_to_x - target_x);
        if (new_box_dist >= push_dist) continue; // This direction doesn't help
        
        // Player needs to be at the opposite position to push
        int player_need_y = box_y - DY[dir];
        int player_need_x = box_x - DX[dir];
        
        // Check if this position is valid (not wall, not out of bounds)
        if (player_need_y < 0 || player_need_y >= H || player_need_x < 0 || player_need_x >= W) continue;
        if (state.map[player_need_y][player_need_x] == '#') continue;
        
        // Check if push destination is valid
        if (push_to_y < 0 || push_to_y >= H || push_to_x < 0 || push_to_x >= W) continue;
        if (state.map[push_to_y][push_to_x] == '#') continue;
        if (state.map[push_to_y][push_to_x] == 'x' || state.map[push_to_y][push_to_x] == 'X') continue; // Another box
        
        // Calculate player distance to this position (Manhattan as lower bound)
        int player_dist = abs(py - player_need_y) + abs(px - player_need_x);
        min_player_dist = min(min_player_dist, player_dist);
    }
    
    if (min_player_dist == INT_MAX) {
        // If no valid push position found, use simple player-to-box distance
        min_player_dist = abs(py - box_y) + abs(px - box_x);
    }
    
    return min_player_dist + push_dist;
}

// Forward declarations
int calculate_heuristic_bitstate_enhanced(const BitState& bit_state);
int calculate_heuristic_bitstate_hungarian(const BitState& bit_state);
int calculate_heuristic_bitstate_adaptive(const BitState& bit_state);

// Adaptive heuristic that chooses method based on problem size
int calculate_heuristic_bitstate_adaptive(const BitState& bit_state) {
    if (bit_state.boxes_on_floor.none()) return 0;  // Already solved
    
    const int num_boxes = (int)bit_state.boxes_on_floor.count();
    
    // For small problems, use Hungarian algorithm for optimal matching
    if (num_boxes <= 8) {
        return calculate_heuristic_bitstate_hungarian(bit_state);
    }
    
    // For medium problems, use enhanced greedy with distance precomputation
    if (num_boxes <= 15) {
        return calculate_heuristic_bitstate_enhanced(bit_state);
    }
    
    // For large problems, use fast heuristics with additional pruning
    
    // Collect unsolved box positions
    vector<pair<int,int>> boxes;
    for (int i = 0; i < (int)g_movable_cells.size(); ++i) {
        if (bit_state.boxes_on_floor[i]) {
            boxes.push_back(g_movable_cells[i]);
        }
    }
    
    // Use clustered greedy assignment - sort boxes by position to improve cache locality
    sort(boxes.begin(), boxes.end());
    
    int total_cost = 0;
    vector<bool> target_used(g_all_targets.size(), false);
    
    // Quick pruning: if any box is outside target boundary, add penalty
    int boundary_penalty = 0;
    for (const auto& box : boxes) {
        if (box.first < min_target_y || box.first > max_target_y ||
            box.second < min_target_x || box.second > max_target_x) {
            boundary_penalty += 5;  // Penalty for being outside target area
        }
    }
    
    for (const auto& box : boxes) {
        int best_cost = INT_MAX;
        int best_target = -1;
        
        // Find closest available target, but limit search for speed
        const int max_targets_to_check = min(5, (int)g_all_targets.size());
        int targets_checked = 0;
        
        for (int t = 0; t < (int)g_all_targets.size() && targets_checked < max_targets_to_check; ++t) {
            if (target_used[t]) continue;
            targets_checked++;
            
            int box_flat = box.first * g_width + box.second;
            int dist = g_target_dists[t][box_flat];
            
            if (dist < best_cost) {
                best_cost = dist;
                best_target = t;
            }
        }
        
        if (best_target != -1) {
            target_used[best_target] = true;
            total_cost += best_cost;
        } else {
            // If no target found in limited search, use Manhattan distance to nearest
            int min_manhattan = INT_MAX;
            for (int t = 0; t < (int)g_all_targets.size(); ++t) {
                if (target_used[t]) continue;
                const auto& target = g_all_targets[t];
                int manhattan = abs(box.first - target.first) + abs(box.second - target.second);
                min_manhattan = min(min_manhattan, manhattan);
            }
            total_cost += min_manhattan;
        }
    }
    
    return total_cost + boundary_penalty;
}

// Hungarian algorithm based heuristic for BitState
int calculate_heuristic_bitstate_hungarian(const BitState& bit_state) {
    if (bit_state.boxes_on_floor.none()) return 0;  // Already solved
    
    // Collect unsolved box positions
    vector<pair<int,int>> boxes;
    for (int i = 0; i < (int)g_movable_cells.size(); ++i) {
        if (bit_state.boxes_on_floor[i]) {
            boxes.push_back(g_movable_cells[i]);
        }
    }
    
    if (boxes.empty()) return 0;
    
    const int n = (int)boxes.size();
    if (n != (int)g_all_targets.size()) {
        // Fallback to greedy if size mismatch
        return calculate_heuristic_bitstate_enhanced(bit_state);
    }
    
    // Build cost matrix using precomputed distances
    vector<vector<int>> cost(n, vector<int>(n));
    for (int i = 0; i < n; ++i) {
        int box_flat = boxes[i].first * g_width + boxes[i].second;
        for (int j = 0; j < n; ++j) {
            cost[i][j] = g_target_dists[j][box_flat];
        }
    }
    
    return hungarian_min_cost(cost);
}

// Enhanced heuristic for BitState using precomputed distances
int calculate_heuristic_bitstate_enhanced(const BitState& bit_state) {
    if (bit_state.boxes_on_floor.none()) return 0;  // Already solved
    
    // Collect unsolved box positions
    vector<pair<int,int>> boxes;
    for (int i = 0; i < (int)g_movable_cells.size(); ++i) {
        if (bit_state.boxes_on_floor[i]) {
            boxes.push_back(g_movable_cells[i]);
        }
    }
    
    if (boxes.empty()) return 0;
    
    // Use greedy assignment with precomputed distances for speed
    int total_cost = 0;
    vector<bool> target_used(g_all_targets.size(), false);
    
    for (const auto& box : boxes) {
        int best_cost = INT_MAX;
        int best_target = -1;
        
        // Find closest available target using precomputed distances
        for (int t = 0; t < (int)g_all_targets.size(); ++t) {
            if (target_used[t]) continue;
            
            int box_flat = box.first * g_width + box.second;
            int dist = g_target_dists[t][box_flat];
            
            if (dist < best_cost) {
                best_cost = dist;
                best_target = t;
            }
        }
        
        if (best_target != -1) {
            target_used[best_target] = true;
            total_cost += best_cost;
        }
    }
    
    return total_cost;
}

// Simple count-based heuristic for better performance
int calculate_heuristic_bitstate_simple(const BitState& bit_state) {
    return (int)bit_state.boxes_on_floor.count();
}

// Legacy heuristic function for GameState (keep for compatibility during transition)
int calculate_heuristic(const GameState& state) {
    vector<pair<int, int>> boxes;    // unsolved boxes
    vector<pair<int, int>> targets;  // free targets

    const int H = (int)state.map.size();
    const int W = H ? (int)state.map[0].size() : 0;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < (int)state.map[y].size(); x++) {
            char c = state.map[y][x];
            if (c == 'x') boxes.push_back({y, x});
            if (c == '.' || c == 'O') targets.push_back({y, x});
        }
    }

    if (boxes.empty()) return 0;

    // Build square cost matrix for Hungarian using push distances
    const int n = (int)boxes.size();
    if ((int)targets.size() != n) {
        // Fallback: if mismatch happens unexpectedly, use greedy safe lower bound
        int greedy = 0;
        vector<bool> used(targets.size(), false);
        for (const auto& b : boxes) {
            int best = INT_MAX, idx = -1;
            for (int i = 0; i < (int)targets.size(); ++i) if (!used[i]) {
                // Use push distance calculation
                int d = calculate_push_distance(state, b.first, b.second, targets[i].first, targets[i].second);
                if (d < best) best = d, idx = i;
            }
            if (idx >= 0) used[idx] = true, greedy += best;
        }
        return greedy;
    }

    vector<vector<int>> cost(n, vector<int>(n, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            // Use push distance instead of simple box-to-target distance
            cost[i][j] = calculate_push_distance(state, boxes[i].first, boxes[i].second, targets[j].first, targets[j].second);
        }
    }
    int push_lower_bound = hungarian_min_cost(cost);

    return push_lower_bound;
}

// Ultra-simple deadlock detection for stability
bool is_deadlock(const GameState& state) {
    auto tunnel_deadlock = [&](int y, int x) -> bool {
        // Only consider unsolved box
        if (state.map[y][x] != 'x') return false;
        int H = (int)state.map.size();
        int W = (int)state.map[0].size();
        auto is_goal = [&](char c){ return c == '.' || c == 'X' || c == 'O'; };
        auto inb = [&](int yy, int xx){ return yy >= 0 && yy < H && xx >= 0 && xx < W; };

        // Vertical tunnel: both sides are walls continuously
        if (inb(y, x-1) && inb(y, x+1) && state.map[y][x-1] == '#' && state.map[y][x+1] == '#') {
            int y0 = y, y1 = y;
            bool ok_up = true, ok_dn = true;
            bool any_goal = is_goal(state.map[y][x]);
            // expand up while left/right are walls
            int yy = y;
            while (yy-1 >= 0 && state.map[yy-1][x] != '#'
                   && inb(yy-1, x-1) && inb(yy-1, x+1)
                   && state.map[yy-1][x-1] == '#' && state.map[yy-1][x+1] == '#') {
                yy--;
                any_goal = any_goal || is_goal(state.map[yy][x]);
            }
            // the cell above yy is either border or wall -> closed
            if (!(yy-1 >= 0 && state.map[yy-1][x] == '#')) ok_up = false;
            y0 = yy;

            // expand down
            yy = y;
            while (yy+1 < H && state.map[yy+1][x] != '#'
                   && inb(yy+1, x-1) && inb(yy+1, x+1)
                   && state.map[yy+1][x-1] == '#' && state.map[yy+1][x+1] == '#') {
                yy++;
                any_goal = any_goal || is_goal(state.map[yy][x]);
            }
            if (!(yy+1 < H && state.map[yy+1][x] == '#')) ok_dn = false;
            y1 = yy;

            if (ok_up && ok_dn && !any_goal) {
                return true;
            }
        }

        // Horizontal tunnel: both vertical neighbors are walls continuously
        if (inb(y-1, x) && inb(y+1, x) && state.map[y-1][x] == '#' && state.map[y+1][x] == '#') {
            int x0 = x, x1 = x;
            bool ok_lt = true, ok_rt = true;
            bool any_goal = is_goal(state.map[y][x]);
            int xx = x;
            while (xx-1 >= 0 && state.map[y][xx-1] != '#'
                   && inb(y-1, xx-1) && inb(y+1, xx-1)
                   && state.map[y-1][xx-1] == '#' && state.map[y+1][xx-1] == '#') {
                xx--;
                any_goal = any_goal || is_goal(state.map[y][xx]);
            }
            if (!(xx-1 >= 0 && state.map[y][xx-1] == '#')) ok_lt = false;
            x0 = xx;

            xx = x;
            while (xx+1 < W && state.map[y][xx+1] != '#'
                   && inb(y-1, xx+1) && inb(y+1, xx+1)
                   && state.map[y-1][xx+1] == '#' && state.map[y+1][xx+1] == '#') {
                xx++;
                any_goal = any_goal || is_goal(state.map[y][xx]);
            }
            if (!(xx+1 < W && state.map[y][xx+1] == '#')) ok_rt = false;
            x1 = xx;

            if (ok_lt && ok_rt && !any_goal) {
                return true;
            }
        }

        return false;
    };

    for (int y = 1; y < state.map.size() - 1; y++) {
        for (int x = 1; x < state.map[y].size() - 1; x++) {
            if (state.map[y][x] == 'x') {  // Unsolved box
                // Tunnel corridor deadlock (disabled for now; refine before enabling)
                // if (tunnel_deadlock(y, x)) return true;
                char up = state.map[y-1][x];
                char down = state.map[y+1][x];
                char left = state.map[y][x-1];
                char right = state.map[y][x+1];
                char upleft = state.map[y-1][x-1];
                char upright = state.map[y-1][x+1];
                char downleft = state.map[y+1][x-1];
                char downright = state.map[y+1][x+1];
                
                // Only detect obvious corner deadlocks
                if ((up == '#' && left == '#') ||
                    (up == '#' && right == '#') ||
                    (down == '#' && left == '#') ||
                    (down == '#' && right == '#')) {
                    return true;
                }
                
                if (up == '#') {
                    if (left == '#' || ((left == 'x' || left == 'X') && (upleft == '#' || downleft == '#'))) {
                        return true;
                    }
                    if (right == '#' || ((right == 'x' || right == 'X') && (upright == '#' || downright == '#'))) {
                        return true;
                    }
                    bool canyon_detected = true;
                    for (int i = x; i > 0 && (state.map[y][i] != '#'); i--) {
                        if (state.map[y-1][i] != '#') {
                            canyon_detected = false;
                            break;
                        }
                    }
                    for (int i = x; i < state.map[y].size() - 1 && (state.map[y][i] != '#'); i++) {
                        if (state.map[y-1][i] != '#') {
                            canyon_detected = false;
                            break;
                        }
                    }
                    if (canyon_detected && max_target_y < y) {
                        return true;
                    }
                }
                if (left == '#') {
                    if (up == '#' || ((up == 'x' || up == 'X') && (upleft == '#' || upright == '#'))) {
                        return true;
                    }
                    if (down == '#' || ((down == 'x' || down == 'X') && (downleft == '#' || downright == '#'))) {
                        return true;
                    }
                    bool canyon_detected = true;
                    for (int i = y; i > 0 && (state.map[i][x] != '#'); i--) {
                        if (state.map[i][x-1] != '#') {
                            canyon_detected = false;
                            break;
                        }
                    }
                    for (int i = y; i < state.map.size() - 1 && (state.map[i][x] != '#'); i++) {
                        if (state.map[i][x-1] != '#') {
                            canyon_detected = false;
                            break;
                        }
                    }
                    if (canyon_detected && min_target_x > x) {
                        return true;
                    }
                }
                if (right == '#') {
                    if (up == '#' || ((up == 'x' || up == 'X') && (upright == '#' || upleft == '#'))) {
                        return true;
                    }
                    if (down == '#' || ((down == 'x' || down == 'X') && (downright == '#' || downleft == '#'))) {
                        return true;
                    }
                    bool canyon_detected = true;
                    for (int i = y; i > 0 && (state.map[i][x] != '#'); i--) {
                        if (state.map[i][x+1] != '#') {
                            canyon_detected = false;
                            break;
                        }
                    }
                    for (int i = y; i < state.map.size() - 1 && (state.map[i][x] != '#'); i++) {
                        if (state.map[i][x+1] != '#') {
                            canyon_detected = false;
                            break;
                        }
                    }
                    if (canyon_detected && max_target_x < x) {
                        return true;
                    }
                }
                if (down == '#') {
                    if (left == '#' || ((left == 'x' || left == 'X') && (downleft == '#' || upleft == '#'))) {
                        return true;
                    }
                    if (right == '#' || ((right == 'x' || right == 'X') && (downright == '#' || upright == '#'))) {
                        return true;
                    }
                    bool canyon_detected = true;
                    for (int i = x; i > 0 && (state.map[y][i] != '#'); i--) {
                        if (state.map[y+1][i] != '#') {
                            canyon_detected = false;
                            break;
                        }
                    }
                    for (int i = x; i < state.map[y].size() - 1 && (state.map[y][i] != '#'); i++) {
                        if (state.map[y+1][i] != '#') {
                            canyon_detected = false;
                            break;
                        }
                    }
                    if (canyon_detected && min_target_y > y) {
                        return true;
                    }
                }
                

                if ((up == 'x' || up == 'X' || up == '#') && (left == 'x' || left == 'X' || left == '#') && (upleft == '#' || upleft == 'x' || upleft == 'X')) {
                    return true;
                }
                if ((up == 'x' || up == 'X' || up == '#') && (right == 'x' || right == 'X' || right == '#') && (upright == '#' || upright == 'x' || upright == 'X')) {
                    return true;
                }
                if ((down == 'x' || down == 'X' || down == '#') && (left == 'x' || left == 'X' || left == '#') && (downleft == '#' || downleft == 'x' || downleft == 'X')) {
                    return true;
                }
                if ((down == 'x' || down == 'X' || down == '#') && (right == 'x' || right == 'X' || right == '#') && (downright == '#' || downright == 'x' || downright == 'X')) {
                    return true;
                }

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
int calculate_heuristic(const BitState& bit_state) {
    // Convert to GameState for legacy heuristic calculation  
    // TODO: Implement direct BitState heuristic for better performance
    GameState state = decompress_from_bitstate(bit_state);
    return calculate_heuristic(state);
}

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
string find_path(int from_pos, int to_pos, const BitState& state) {
    if (from_pos == to_pos) return "";
    
    // Static vectors to avoid repeated allocations (thread-local if needed)
    static vector<int> parent;
    static vector<char> move_dir;
    
    // Resize if needed
    const size_t num_cells = g_movable_cells.size();
    if (parent.size() != num_cells) {
        parent.resize(num_cells);
        move_dir.resize(num_cells);
    }
    
    // Reset parent array (only for cells we might visit)
    fill(parent.begin(), parent.end(), -1);
    
    queue<int> q;
    q.push(from_pos);
    parent[from_pos] = from_pos;
    
    // Constants for directions
    static constexpr int DY[4] = {-1, 0, 1, 0};  // Up, Left, Down, Right  
    static constexpr int DX[4] = {0, -1, 0, 1};
    static constexpr char DIRS[4] = {'W', 'A', 'S', 'D'};
    
    while (!q.empty()) {
        int pos = q.front();
        q.pop();
        
        const auto& cell = g_movable_cells[pos];
        int y = cell.first;
        int x = cell.second;
        
        for (int dir = 0; dir < 4; dir++) {
            int ny = y + DY[dir];
            int nx = x + DX[dir];
            
            // Quick boundary and wall checks
            if (static_cast<unsigned>(ny) >= static_cast<unsigned>(g_height) ||
                static_cast<unsigned>(nx) >= static_cast<unsigned>(g_width) ||
                g_base_map[ny][nx] == '#') continue;
            
            auto it = g_cell_to_index.find({ny, nx});
            if (it == g_cell_to_index.end()) continue;
            int new_pos = it->second;
            
            // Can't move through boxes or already visited
            if (state.has_box(new_pos) || parent[new_pos] != -1) continue;
            
            parent[new_pos] = pos;
            move_dir[new_pos] = DIRS[dir];
            
            // Check if we reached target
            if (new_pos == to_pos) {
                // Reconstruct path more efficiently
                string path;
                path.reserve(32); // Reserve some space to avoid reallocations
                int curr = to_pos;
                while (curr != from_pos) {
                    path += move_dir[curr];
                    curr = parent[curr];
                }
                reverse(path.begin(), path.end());
                return path;
            }
            
            q.push(new_pos);
        }
    }
    
    return ""; // No path found
}

vector<pair<BitState, string>> generate_macro_moves(const BitState& bit_state) {
    vector<pair<BitState, string>> push_states;
    
    // Find all positions where player can push boxes
    vector<bool> player_reachable(g_movable_cells.size(), false);
    queue<int> q;
    q.push(bit_state.player_pos);
    player_reachable[bit_state.player_pos] = true;
    
    // BFS to find all reachable positions for the player
    while (!q.empty()) {
        int pos = q.front();
        q.pop();
        
        int y = g_movable_cells[pos].first;
        int x = g_movable_cells[pos].second;
        
        const int DY[4] = {-1, 0, 1, 0};  // Up, Left, Down, Right
        const int DX[4] = {0, -1, 0, 1};
        const char DIRS[4] = {'W', 'A', 'S', 'D'};
        
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
            
            // If there's a box at this position, check if we can push it
            if (bit_state.has_box(new_pos)) {
                // Check where the box would go
                int box_ny = ny + DY[dir];
                int box_nx = nx + DX[dir];
                
                // Check bounds for box destination
                if (box_ny < 0 || box_ny >= g_height || box_nx < 0 || box_nx >= g_width) {
                    continue;
                }
                
                // Check if box destination is a wall
                if (g_base_map[box_ny][box_nx] == '#') {
                    continue;
                }
                
                // Find index of box destination
                auto box_it = g_cell_to_index.find({box_ny, box_nx});
                if (box_it == g_cell_to_index.end()) {
                    continue;
                }
                int box_dest_pos = box_it->second;
                
                // Check if there's already a box at destination
                if (bit_state.has_box(box_dest_pos)) {
                    continue;
                }
                
                // Only add this push if we can reach the push position
                if (player_reachable[pos]) {
                    // Create new state with box pushed
                    BitState new_state = bit_state;
                    new_state.move_box(new_pos, box_dest_pos);
                    new_state.player_pos = new_pos;  // Player moves to where box was
                    
                    // Generate complete path: player movement + push
                    string path = find_path(bit_state.player_pos, pos, bit_state);
                    path += DIRS[dir];  // Add the push move
                    
                    push_states.push_back({new_state, path});
                }
            } else {
                // No box, player can move here - add to reachable positions
                if (!player_reachable[new_pos]) {
                    player_reachable[new_pos] = true;
                    q.push(new_pos);
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

// A* solver with BitState optimization
string astar_solver(const string& filename) {
    GameState initial_state = loadstate(filename);
    
    // Create base map for compression (done in loadstate now)
    g_base_map = initial_state.map;
    
    // Clean base map from boxes and players for compression
    for (int y = 0; y < (int)g_base_map.size(); ++y) {
        for (int x = 0; x < (int)g_base_map[y].size(); ++x) {
            char c = g_base_map[y][x];
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
    
    priority_queue<AStarNode, vector<AStarNode>, greater<AStarNode>> open_set;
    
    // Use bitmap states for duplicate detection, track player positions per box state
    // Use bitmap states for duplicate detection - store complete states
    unordered_set<BitState, BitStateHash> visited_states;
    
    AStarNode start_node;
    start_node.bit_state = compress_to_bitstate(initial_state);
    start_node.path = "";
    start_node.g_cost = 0;
    start_node.h_cost = calculate_heuristic_bitstate_adaptive(start_node.bit_state);
    start_node.last_move = 0;
    start_node.last_was_push = false;
    
    open_set.push(start_node);
    
    int nodes_expanded = 0;
    
    while (!open_set.empty()) {
        AStarNode current = open_set.top();
        open_set.pop();
        
        nodes_expanded++;
        
        if (is_solved(current.bit_state)) {
            cerr << "A* solved in " << nodes_expanded << " nodes" << endl;
            return current.path;
        }
        
        // Skip deadlock states
        if (is_deadlock(current.bit_state)) {
            continue;
        }
        
        // Generate macro moves - only push states are returned
        vector<pair<BitState, string>> successors = generate_macro_moves(current.bit_state);
        
        // Process and sort successors by heuristic value for better node ordering
        vector<pair<int, int>> scored_indices;  // (heuristic, index)
        
        for (int i = 0; i < (int)successors.size(); ++i) {
            const auto& [new_bit_state, move_path] = successors[i];
            
            // Skip if already visited
            if (visited_states.find(new_bit_state) != visited_states.end()) {
                continue;
            }
            
            // Quick deadlock check before adding to queue
            if (is_deadlock(new_bit_state)) {
                continue;
            }
            
            int h_score = calculate_heuristic_bitstate_adaptive(new_bit_state);
            scored_indices.emplace_back(h_score, i);
        }
        
        // Sort by heuristic score (ascending - better states first)
        sort(scored_indices.begin(), scored_indices.end());
        
        // Add sorted successors to open set
        for (const auto& [h_score, idx] : scored_indices) {
            const auto& [new_bit_state, move_path] = successors[idx];
            
            // Add to visited states
            visited_states.insert(new_bit_state);

            AStarNode new_node;
            new_node.bit_state = new_bit_state;
            new_node.path = current.path + move_path;
            new_node.g_cost = current.g_cost + 1;  // Each macro move counts as 1 step
            new_node.h_cost = h_score;  // Already calculated
            new_node.last_move = move_path.back();  // Last character of the move path
            new_node.last_was_push = true;  // All macro moves result in pushes

            open_set.push(new_node);
        }
    }
    
    throw Oops("A* search could not find solution");
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
    
    try {
        auto start_time = chrono::high_resolution_clock::now();
        string result = main_solver(argv[1]);
        auto end_time = chrono::high_resolution_clock::now();
        
        auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
        cout << result << endl;
        cerr << "Solved in " << duration.count() << " ms" << endl;
        return 0;
    } catch (const Oops& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    } catch (const exception& e) {
        cerr << "Unexpected error: " << e.what() << endl;
        return 1;
    }
}
