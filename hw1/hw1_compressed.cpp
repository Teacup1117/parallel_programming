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

// Compressed state representation
struct CompressedBoxState {
    vector<char> boxes;  // 1D array representing box positions: 'x' (unsolved), 'X' (on target), ' ' (empty floor), '.' (empty target)
    
    bool operator==(const CompressedBoxState& other) const {
        return boxes == other.boxes;
    }
};

struct CompressedBoxStateHash {
    size_t operator()(const CompressedBoxState& state) const {
        size_t hash = 0;
        for (size_t i = 0; i < state.boxes.size(); ++i) {
            hash ^= std::hash<char>{}(state.boxes[i]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

struct GameState {
    vector<string> map;
    pair<int, int> player_pos;
    
    bool operator==(const GameState& other) const {
        return map == other.map && player_pos == other.player_pos;
    }
};

// Build a key for boxes-layout (ignore player position). Normalize player tiles to floor/goal.
static inline string boxes_layout_key(const GameState& state) {
    string key;
    if (state.map.empty()) return key;
    key.reserve(state.map.size() * state.map[0].size());
    for (const auto& row : state.map) {
        for (char c : row) {
            switch (c) {
                case 'o': key.push_back(' '); break;  // player on floor -> floor
                case 'O': key.push_back('.'); break;  // player on goal -> goal
                case '!': key.push_back('@'); break;  // player on wall -> wall (rare)
                default: key.push_back(c); break;     // keep boxes 'x'/'X', walls, goals
            }
        }
    }
    return key;
}

// More efficient hash function
struct GameStateHash {
    size_t operator()(const GameState& state) const {
        size_t h1 = hash<string>{}("");
        for (const auto& row : state.map) {
            h1 ^= hash<string>{}(row) + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
        }
        size_t h2 = hash<int>{}(state.player_pos.first);
        size_t h3 = hash<int>{}(state.player_pos.second);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

// A* search node with compressed state
struct AStarNode {
    GameState state;  // Keep for compatibility, but we'll use compressed representation for storage
    string path;
    int g_cost;    // Distance from start
    int h_cost;    // Heuristic to goal
    // Lightweight metadata for pruning immediate backtracking
    char last_move = 0;      // 'W','A','S','D' or 0 for none
    bool last_was_push = false;
    
    // Compressed state data
    CompressedBoxState compressed_boxes;
    int player_index;
    
    int f_cost() const { return g_cost + h_cost; }
    
    bool operator>(const AStarNode& other) const {
        if (f_cost() != other.f_cost()) {
            return f_cost() > other.f_cost();
        }
        return h_cost > other.h_cost;  // Tie-breaker: prefer lower h_cost
    }
};

// Convert GameState to compressed representation
CompressedBoxState compress_boxes(const GameState& state) {
    CompressedBoxState compressed;
    compressed.boxes.resize(g_movable_cells.size(), ' ');
    
    for (int y = 0; y < (int)state.map.size(); ++y) {
        for (int x = 0; x < (int)state.map[y].size(); ++x) {
            char c = state.map[y][x];
            if (c == 'x' || c == 'X') {  // Box on floor or target
                auto it = g_cell_to_index.find({y, x});
                if (it != g_cell_to_index.end()) {
                    compressed.boxes[it->second] = c;
                }
            }
        }
    }
    
    return compressed;
}

// Convert player position to index
int compress_player(const GameState& state) {
    auto it = g_cell_to_index.find(state.player_pos);
    return (it != g_cell_to_index.end()) ? it->second : -1;
}

// Reconstruct GameState from compressed representation
GameState decompress_state(const CompressedBoxState& boxes, int player_idx) {
    GameState state;
    state.map = g_base_map;  // Copy base map
    
    // Place boxes
    for (int i = 0; i < (int)boxes.boxes.size(); ++i) {
        if (boxes.boxes[i] == 'x' || boxes.boxes[i] == 'X') {
            int y = g_movable_cells[i].first;
            int x = g_movable_cells[i].second;
            state.map[y][x] = boxes.boxes[i];
        }
    }
    
    // Place player
    if (player_idx >= 0 && player_idx < (int)g_movable_cells.size()) {
        int py = g_movable_cells[player_idx].first;
        int px = g_movable_cells[player_idx].second;
        state.player_pos = {py, px};
        
        // Update player position on map
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

// Check if player can reach target position without pushing boxes
bool can_player_reach(const GameState& state, int from_y, int from_x, int to_y, int to_x) {
    if (from_y == to_y && from_x == to_x) return true;
    
    const int H = (int)state.map.size();
    const int W = H ? (int)state.map[0].size() : 0;
    
    // BFS to find if player can reach target
    queue<pair<int,int>> q;
    unordered_set<long long> visited;
    
    q.push({from_y, from_x});
    visited.insert(keyYX(from_y, from_x));
    
    const int DY[4] = {-1, 0, 1, 0};
    const int DX[4] = {0, -1, 0, 1};
    
    while (!q.empty()) {
        auto [y, x] = q.front();
        q.pop();
        
        for (int k = 0; k < 4; ++k) {
            int ny = y + DY[k];
            int nx = x + DX[k];
            
            if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
            
            char c = state.map[ny][nx];
            if (c == '#' || c == 'x' || c == 'X') continue;  // Wall or box
            
            long long key = keyYX(ny, nx);
            if (visited.find(key) != visited.end()) continue;
            
            if (ny == to_y && nx == to_x) return true;
            
            visited.insert(key);
            q.push({ny, nx});
        }
    }
    
    return false;
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

    // Build square cost matrix for Hungarian; sizes should match by Sokoban invariant
    const int n = (int)boxes.size();
    if ((int)targets.size() != n) {
        // Fallback: if mismatch happens unexpectedly, use greedy safe lower bound
        int greedy = 0;
        vector<bool> used(targets.size(), false);
        for (const auto& b : boxes) {
            int best = INT_MAX, idx = -1;
            for (int i = 0; i < (int)targets.size(); ++i) if (!used[i]) {
                // prefer BFS distance; fallback to Manhattan if not available
                int d = INT_MAX;
                long long k = keyYX(targets[i].first, targets[i].second);
                auto it = g_target_index.find(k);
                if (it != g_target_index.end() && !g_target_dists.empty()) {
                    int idxT = it->second;
                    int cell = b.first * g_width + b.second;
                    if (idxT >= 0 && idxT < (int)g_target_dists.size() && cell >= 0 && cell < (int)g_target_dists[idxT].size()) {
                        d = g_target_dists[idxT][cell];
                    }
                }
                if (d == INT_MAX) {
                    d = abs(b.first - targets[i].first) + abs(b.second - targets[i].second);
                }
                if (d < best) best = d, idx = i;
            }
            if (idx >= 0) used[idx] = true, greedy += best;
        }
        // Add player-to-nearest-box distance (admissible)
        int py = state.player_pos.first, px = state.player_pos.second;
        int player_min = INT_MAX;
        for (auto& b : boxes) player_min = min(player_min, abs(py - b.first) + abs(px - b.second));
        if (player_min != INT_MAX) greedy += player_min;
        return greedy;
    }

    vector<vector<int>> cost(n, vector<int>(n, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int d = INT_MAX;
            // Use precomputed BFS distances from target j to cell of box i
            long long k = keyYX(targets[j].first, targets[j].second);
            auto it = g_target_index.find(k);
            if (it != g_target_index.end() && !g_target_dists.empty()) {
                int idxT = it->second;
                int cell = boxes[i].first * g_width + boxes[i].second;
                if (idxT >= 0 && idxT < (int)g_target_dists.size() && cell >= 0 && cell < (int)g_target_dists[idxT].size()) {
                    d = g_target_dists[idxT][cell];
                }
            }
            if (d == INT_MAX) {
                d = abs(boxes[i].first - targets[j].first) + abs(boxes[i].second - targets[j].second);
            }
            cost[i][j] = d;
        }
    }
    int push_lower_bound = hungarian_min_cost(cost);

    // Add player-to-nearest-box distance (ignoring walls). Still a lower bound of future walking.
    int py = state.player_pos.first, px = state.player_pos.second;
    int player_min = INT_MAX;
    for (auto& b : boxes) player_min = min(player_min, abs(py - b.first) + abs(px - b.second));
    if (player_min == INT_MAX) player_min = 0;

    return push_lower_bound + player_min;
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

    return {m, player_pos};
}

// Try to move in the given direction
pair<GameState, bool> try_move(const GameState& state, int dy, int dx) {
    GameState new_state = state;
    auto& m = new_state.map;
    int y = state.player_pos.first;
    int x = state.player_pos.second;
    
    int yy = y + dy;
    int yyy = yy + dy;
    int xx = x + dx;
    int xxx = xx + dx;
    
    // Check bounds
    if (yy < 0 || yy >= m.size() || xx < 0 || xx >= m[yy].length()) {
        return {new_state, false};
    }
    
    char next_cell = m[yy][xx];
    
    if (next_cell == ' ') {
        m[yy][xx] = 'o';
    } else if (next_cell == '.') {
        m[yy][xx] = 'O';
    } else if (next_cell == '@') {
        m[yy][xx] = '!';
    } else if ((next_cell == 'x' || next_cell == 'X')) {
        // Check if we can push the box
        if (yyy < 0 || yyy >= m.size() || xxx < 0 || xxx >= m[yyy].length()) {
            return {new_state, false};
        }
        
        char box_dest = m[yyy][xxx];
        if (box_dest == ' ' || box_dest == '.') {
            // Move player to box position
            if (next_cell == 'x') {
                m[yy][xx] = 'o';
            } else {  // next_cell == 'X'
                m[yy][xx] = 'O';
            }
            
            // Move box to destination
            if (box_dest == ' ') {
                m[yyy][xxx] = 'x';
            } else {  // box_dest == '.'
                m[yyy][xxx] = 'X';
            }
        } else {
            return {new_state, false};
        }
    } else {
        return {new_state, false};
    }
    
    // Update player's previous position
    char current_cell = m[y][x];
    if (current_cell == 'o') {
        m[y][x] = ' ';
    } else if (current_cell == '!') {
        m[y][x] = '@';
    } else {  // current_cell == 'O'
        m[y][x] = '.';
    }
    
    new_state.player_pos = {yy, xx};
    return {new_state, true};
}

// Check if the game is solved
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

// A* solver with state compression and player position optimization
string astar_solver(const string& filename) {
    GameState initial_state = loadstate(filename);
    
    // Initialize compression data structures
    init_compression(initial_state);
    
    priority_queue<AStarNode, vector<AStarNode>, greater<AStarNode>> open_set;
    
    // Use compressed box states for duplicate detection, but track multiple player positions per box state
    unordered_map<CompressedBoxState, unordered_set<int>, CompressedBoxStateHash> visited_box_states;
    
    AStarNode start_node;
    start_node.state = initial_state;
    start_node.path = "";
    start_node.g_cost = 0;
    start_node.h_cost = calculate_heuristic(initial_state);
    start_node.last_move = 0;
    start_node.last_was_push = false;
    start_node.compressed_boxes = compress_boxes(initial_state);
    start_node.player_index = compress_player(initial_state);
    
    open_set.push(start_node);
    
    int nodes_expanded = 0;
    
    while (!open_set.empty()) {
        AStarNode current = open_set.top();
        open_set.pop();
        
        nodes_expanded++;
        
        if (is_solved(current.state)) {
            cerr << "A* solved in " << nodes_expanded << " nodes" << endl;
            return current.path;
        }
        
        // Skip deadlock states
        if (is_deadlock(current.state)) {
            continue;
        }
        
        // Generate successors with immediate backtracking pruning
        for (const auto& [key, direction] : DYDX) {
            // Avoid immediately undoing a non-push move
            if (current.last_move && !current.last_was_push && key == opposite_dir(current.last_move)) {
                continue;
            }

            // Determine if this move would be a push (before applying)
            bool would_push = false;
            int py = current.state.player_pos.first;
            int px = current.state.player_pos.second;
            int ny = py + direction.first;
            int nx = px + direction.second;
            if (ny >= 0 && ny < (int)current.state.map.size() && nx >= 0 && nx < (int)current.state.map[ny].size()) {
                char next_cell = current.state.map[ny][nx];
                if (next_cell == 'x' || next_cell == 'X') would_push = true;
            }

            auto [new_state, valid] = try_move(current.state, direction.first, direction.second);
            if (!valid) {
                continue;
            }

            // Create compressed representation
            CompressedBoxState new_compressed_boxes = compress_boxes(new_state);
            int new_player_index = compress_player(new_state);
            
            // Check if this box configuration with any player position has been visited
            bool skip_node = false;
            auto box_it = visited_box_states.find(new_compressed_boxes);
            if (box_it != visited_box_states.end()) {
                if (would_push) {
                    // For push moves, check if exact player position was visited
                    if (box_it->second.find(new_player_index) != box_it->second.end()) {
                        skip_node = true;
                    }
                } else {
                    // For non-push moves, check if player can reach any previously visited position
                    for (int visited_player_idx : box_it->second) {
                        if (visited_player_idx < (int)g_movable_cells.size()) {
                            int vpy = g_movable_cells[visited_player_idx].first;
                            int vpx = g_movable_cells[visited_player_idx].second;
                            int npy = new_state.player_pos.first;
                            int npx = new_state.player_pos.second;
                            
                            if (can_player_reach(new_state, npy, npx, vpy, vpx)) {
                                skip_node = true;
                                break;
                            }
                        }
                    }
                }
            }
            
            if (skip_node) {
                continue;
            }
            
            // Add to visited states
            visited_box_states[new_compressed_boxes].insert(new_player_index);

            AStarNode new_node;
            new_node.state = new_state;
            new_node.path = current.path + key;
            new_node.g_cost = current.g_cost + 1;
            new_node.h_cost = calculate_heuristic(new_state);
            new_node.last_move = key;
            new_node.last_was_push = would_push;
            new_node.compressed_boxes = new_compressed_boxes;
            new_node.player_index = new_player_index;

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
