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
#include <future>
#include <memory>
#include <numeric>

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
static vector<string> g_base_map;
static vector<pair<int,int>> g_movable_cells;
static unordered_map<pair<int,int>, int, hash<pair<int,int>>> g_cell_to_index;

// Target boundaries
int min_target_x = INT_MAX;
int max_target_x = INT_MIN;
int min_target_y = INT_MAX;
int max_target_y = INT_MIN;

// Precomputed data
static int g_height = 0;
static int g_width = 0;
static vector<bool> row_has_target;
static vector<bool> col_has_target;
static vector<pair<int,int>> g_all_targets;
static unordered_map<long long,int> g_target_index;
static vector<vector<int>> g_target_dists;

// Parallel execution controls
static atomic<bool> solution_found{false};
static atomic<int> active_threads{0};
static mutex solution_mutex;
static string global_solution;

// Thread pool for work stealing
class ThreadPool {
private:
    vector<thread> workers;
    queue<function<void()>> tasks;
    mutex queue_mutex;
    condition_variable condition;
    bool stop;
    
public:
    ThreadPool(size_t threads) : stop(false) {
        for(size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                for(;;) {
                    function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this]{ return this->stop || !this->tasks.empty(); });
                        if(this->stop && this->tasks.empty()) return;
                        task = move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
        }
    }
    
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> future<typename result_of<F(Args...)>::type> {
        using return_type = typename result_of<F(Args...)>::type;
        
        auto task = make_shared<packaged_task<return_type()>>(
            bind(forward<F>(f), forward<Args>(args)...)
        );
        
        future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if(stop) throw runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace([task](){ (*task)(); });
        }
        condition.notify_one();
        return res;
    }
    
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for(thread &worker: workers) worker.join();
    }
};

static inline long long keyYX(int y, int x) { 
    return (static_cast<long long>(y) << 32) ^ static_cast<unsigned int>(x); 
}

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

const int MAX_MOVABLE_CELLS = 512;

struct BitState {
    std::bitset<MAX_MOVABLE_CELLS> boxes_on_floor;
    std::bitset<MAX_MOVABLE_CELLS> boxes_on_target;
    int player_pos;
    
    bool operator==(const BitState& other) const {
        return boxes_on_floor == other.boxes_on_floor && 
               boxes_on_target == other.boxes_on_target && 
               player_pos == other.player_pos;
    }
    
    int box_count() const {
        return (int)(boxes_on_floor.count() + boxes_on_target.count());
    }
    
    bool has_box(int index) const {
        return boxes_on_floor[index] || boxes_on_target[index];
    }
    
    void move_box(int from_index, int to_index) {
        // Determine if source box is on target
        bool was_on_target = boxes_on_target[from_index];
        
        // Remove from source
        boxes_on_floor[from_index] = false;
        boxes_on_target[from_index] = false;
        
        // Determine if destination is a target
        int to_y = g_movable_cells[to_index].first;
        int to_x = g_movable_cells[to_index].second;
        bool dest_is_target = (g_base_map[to_y][to_x] == '.');
        
        // Place at destination
        if (dest_is_target) {
            boxes_on_target[to_index] = true;
        } else {
            boxes_on_floor[to_index] = true;
        }
    }
};

struct BitStateHash {
    size_t operator()(const BitState& state) const {
        size_t h1 = hash<bitset<MAX_MOVABLE_CELLS>>{}(state.boxes_on_floor);
        size_t h2 = hash<bitset<MAX_MOVABLE_CELLS>>{}(state.boxes_on_target);
        size_t h3 = hash<int>{}(state.player_pos);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

struct GameState {
    vector<string> map;
    int player_pos;
};

// Thread-safe visited states using sharded hash tables
class ConcurrentVisited {
private:
    static const int NUM_SHARDS = 64;
    vector<unordered_set<BitState, BitStateHash>> shards;
    mutable vector<mutex> shard_mutexes;
    
    int get_shard(const BitState& state) const {
        return BitStateHash{}(state) % NUM_SHARDS;
    }
    
public:
    ConcurrentVisited() : shards(NUM_SHARDS), shard_mutexes(NUM_SHARDS) {}
    
    bool insert(const BitState& state) {
        int shard_id = get_shard(state);
        std::lock_guard<std::mutex> lock(shard_mutexes[shard_id]);
        return shards[shard_id].insert(state).second;
    }
    
    bool contains(const BitState& state) const {
        int shard_id = get_shard(state);
        std::lock_guard<std::mutex> lock(shard_mutexes[shard_id]);
        return shards[shard_id].find(state) != shards[shard_id].end();
    }
    
    size_t size() const {
        size_t total = 0;
        for (int i = 0; i < NUM_SHARDS; ++i) {
            std::lock_guard<std::mutex> lock(shard_mutexes[i]);
            total += shards[i].size();
        }
        return total;
    }
};

struct AStarNode {
    BitState bit_state;
    string path;
    int g_cost;
    int h_cost;
    char last_move;
    bool last_was_push;
    
    int f_cost() const { return g_cost + h_cost; }
    
    bool operator>(const AStarNode& other) const {
        if (f_cost() != other.f_cost()) return f_cost() > other.f_cost();
        return h_cost > other.h_cost;
    }
};

GameState loadstate(const string& filename) {
    ifstream file(filename);
    if (!file) throw Oops("Cannot open file: " + filename);
    
    GameState state;
    string line;
    int player_count = 0;
    
    while (getline(file, line)) {
        if (line.empty()) continue;
        state.map.push_back(line);
    }
    
    if (state.map.empty()) throw Oops("Empty map");
    
    // Find player and initialize global data
    for (int y = 0; y < (int)state.map.size(); ++y) {
        for (int x = 0; x < (int)state.map[y].size(); ++x) {
            char c = state.map[y][x];
            if (c == 'o' || c == 'O') {
                player_count++;
                state.player_pos = -1; // Will be set during compression init
                if (c == '.') {
                    min_target_x = min(min_target_x, x);
                    max_target_x = max(max_target_x, x);
                    min_target_y = min(min_target_y, y);
                    max_target_y = max(max_target_y, y);
                }
            } else if (c == '.' || c == 'X') {
                min_target_x = min(min_target_x, x);
                max_target_x = max(max_target_x, x);
                min_target_y = min(min_target_y, y);
                max_target_y = max(max_target_y, y);
            }
        }
    }
    
    if (player_count != 1) throw Oops("Map must have exactly one player");
    
    g_height = (int)state.map.size();
    g_width = g_height ? (int)state.map[0].size() : 0;
    
    return state;
}

BitState compress_to_bitstate(const GameState& state) {
    BitState bit_state;
    bit_state.boxes_on_floor.reset();
    bit_state.boxes_on_target.reset();
    bit_state.player_pos = -1;
    
    for (int y = 0; y < (int)state.map.size(); ++y) {
        for (int x = 0; x < (int)state.map[y].size(); ++x) {
            char c = state.map[y][x];
            
            if (c == 'o' || c == 'O') {
                auto it = g_cell_to_index.find({y, x});
                if (it != g_cell_to_index.end()) {
                    bit_state.player_pos = it->second;
                }
            }
            
            if (c == 'x' || c == 'X') {
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

GameState decompress_from_bitstate(const BitState& bit_state) {
    GameState state;
    state.map = g_base_map;
    state.player_pos = bit_state.player_pos;
    
    for (int i = 0; i < (int)g_movable_cells.size() && i < MAX_MOVABLE_CELLS; ++i) {
        if (bit_state.boxes_on_floor[i] || bit_state.boxes_on_target[i]) {
            int y = g_movable_cells[i].first;
            int x = g_movable_cells[i].second;
            
            if (bit_state.boxes_on_floor[i]) {
                state.map[y][x] = 'x';
            } else {
                state.map[y][x] = 'X';
            }
        }
    }
    
    if (bit_state.player_pos >= 0 && bit_state.player_pos < (int)g_movable_cells.size()) {
        int py = g_movable_cells[bit_state.player_pos].first;
        int px = g_movable_cells[bit_state.player_pos].second;
        
        char base_cell = g_base_map[py][px];
        if (base_cell == '.') {
            state.map[py][px] = 'O';
        } else {
            state.map[py][px] = 'o';
        }
    }
    
    return state;
}

void init_compression(const GameState& initial_state) {
    g_base_map = initial_state.map;
    g_movable_cells.clear();
    g_cell_to_index.clear();
    
    for (int y = 0; y < (int)g_base_map.size(); ++y) {
        for (int x = 0; x < (int)g_base_map[y].size(); ++x) {
            char c = g_base_map[y][x];
            if (c != '#') {
                g_cell_to_index[{y, x}] = g_movable_cells.size();
                g_movable_cells.push_back({y, x});
                
                if (c == 'x') {
                    g_base_map[y][x] = ' ';
                } else if (c == 'X') {
                    g_base_map[y][x] = '.';
                } else if (c == 'o') {
                    g_base_map[y][x] = ' ';
                } else if (c == 'O') {
                    g_base_map[y][x] = '.';
                }
            }
        }
    }
}

const vector<pair<char, pair<int, int>>> DYDX = {
    {'W', {-1, 0}},
    {'A', {0, -1}},
    {'S', {1, 0}},
    {'D', {0, 1}}
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

// Parallel Hungarian algorithm for minimum cost assignment
int hungarian_min_cost_parallel(const vector<vector<int>>& cost) {
    int n = cost.size();
    if (n == 0) return 0;
    
    // For small problems, use sequential version
    if (n <= 4) {
        // Simple sequential version for small problems
        vector<int> assignment(n);
        std::iota(assignment.begin(), assignment.end(), 0);
        int min_cost = INT_MAX;
        
        do {
            int total = 0;
            for (int i = 0; i < n; ++i) {
                total += cost[i][assignment[i]];
            }
            min_cost = min(min_cost, total);
        } while (next_permutation(assignment.begin(), assignment.end()));
        
        return min_cost;
    }
    
    // For larger problems, implement Hungarian algorithm
    // This is a simplified version - full implementation would be more complex
    vector<vector<int>> c = cost;
    vector<int> u(n+1), v(n+1), p(n+1), way(n+1);
    
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        vector<int> minv(n+1, INT_MAX);
        vector<bool> used(n+1, false);
        
        do {
            used[j0] = true;
            int i0 = p[j0], delta = INT_MAX, j1;
            
            for (int j = 1; j <= n; ++j) {
                if (!used[j]) {
                    int cur = c[i0-1][j-1] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            
            for (int j = 0; j <= n; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            
            j0 = j1;
        } while (p[j0] != 0);
        
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    
    return -v[0];
}

// Parallel heuristic calculation using thread pool
int calculate_heuristic_bitstate_parallel(const BitState& bit_state, ThreadPool& pool) {
    if (bit_state.boxes_on_floor.none()) return 0;
    
    vector<pair<int, int>> boxes;
    for (int i = 0; i < (int)g_movable_cells.size() && i < MAX_MOVABLE_CELLS; ++i) {
        if (bit_state.boxes_on_floor[i]) {
            boxes.push_back(g_movable_cells[i]);
        }
    }
    
    if (boxes.empty()) return 0;
    
    int num_boxes = boxes.size();
    int num_targets = g_all_targets.size();
    
    if (num_boxes > num_targets) return INT_MAX;
    
    // Parallel distance calculation
    vector<vector<int>> cost(num_boxes, vector<int>(num_targets));
    vector<future<void>> futures;
    
    for (int i = 0; i < num_boxes; ++i) {
        futures.push_back(pool.enqueue([i, &boxes, &cost]() {
            int box_y = boxes[i].first;
            int box_x = boxes[i].second;
            
            for (int t = 0; t < (int)g_all_targets.size(); ++t) {
                int target_y = g_all_targets[t].first;
                int target_x = g_all_targets[t].second;
                
                // Use precomputed distances
                cost[i][t] = g_target_dists[t][box_y * g_width + box_x];
                if (cost[i][t] >= 1e9) {
                    cost[i][t] = abs(box_y - target_y) + abs(box_x - target_x);
                }
            }
        }));
    }
    
    // Wait for all distance calculations
    for (auto& f : futures) {
        f.wait();
    }
    
    return hungarian_min_cost_parallel(cost);
}

// Simplified sequential version for compatibility
int calculate_heuristic_bitstate_adaptive(const BitState& bit_state) {
    if (bit_state.boxes_on_floor.none()) return 0;
    
    vector<pair<int, int>> boxes;
    for (int i = 0; i < (int)g_movable_cells.size() && i < MAX_MOVABLE_CELLS; ++i) {
        if (bit_state.boxes_on_floor[i]) {
            boxes.push_back(g_movable_cells[i]);
        }
    }
    
    if (boxes.empty()) return 0;
    
    int sum_dist = 0;
    for (const auto& box : boxes) {
        int min_dist = INT_MAX;
        for (const auto& target : g_all_targets) {
            int dist = abs(box.first - target.first) + abs(box.second - target.second);
            min_dist = min(min_dist, dist);
        }
        sum_dist += min_dist;
    }
    
    return sum_dist;
}

bool is_deadlock(const BitState& bit_state) {
    GameState state = decompress_from_bitstate(bit_state);
    
    for (int y = 1; y < state.map.size() - 1; y++) {
        for (int x = 1; x < state.map[y].size() - 1; x++) {
            if (state.map[y][x] == 'x') {
                char up = state.map[y-1][x];
                char down = state.map[y+1][x];
                char left = state.map[y][x-1];
                char right = state.map[y][x+1];
                
                if ((up == '#' && left == '#') ||
                    (up == '#' && right == '#') ||
                    (down == '#' && left == '#') ||
                    (down == '#' && right == '#')) {
                    return true;
                }
            }
        }
    }
    
    return false;
}

bool is_deadlock(const GameState& state) {
    for (int y = 1; y < state.map.size() - 1; y++) {
        for (int x = 1; x < state.map[y].size() - 1; x++) {
            if (state.map[y][x] == 'x') {
                char up = state.map[y-1][x];
                char down = state.map[y+1][x];
                char left = state.map[y][x-1];
                char right = state.map[y][x+1];
                
                if ((up == '#' && left == '#') ||
                    (up == '#' && right == '#') ||
                    (down == '#' && left == '#') ||
                    (down == '#' && right == '#')) {
                    return true;
                }
            }
        }
    }
    
    return false;
}

void init_targets_and_distances() {
    g_all_targets.clear();
    g_target_index.clear();
    g_target_dists.clear();
    
    row_has_target.assign(g_height, false);
    col_has_target.assign(g_width, false);
    
    for (int y = 0; y < g_height; ++y) {
        for (int x = 0; x < g_width; ++x) {
            if (g_base_map[y][x] == '.') {
                g_all_targets.emplace_back(y, x);
                g_target_index[keyYX(y, x)] = g_all_targets.size() - 1;
                row_has_target[y] = true;
                col_has_target[x] = true;
            }
        }
    }
    
    for (const auto& t : g_all_targets) {
        g_target_dists.emplace_back(bfs_dist_from_target(g_base_map, t.first, t.second));
    }
}

bool is_solved(const BitState& bit_state) {
    return bit_state.boxes_on_floor.none();
}

bool is_solved(const GameState& state) {
    for (const string& row : state.map) {
        for (char c : row) {
            if (c == 'x') {
                return false;
            }
        }
    }
    return true;
}

string find_path(int from_pos, int to_pos, const BitState& bit_state) {
    if (from_pos == to_pos) return "";
    
    vector<bool> visited(g_movable_cells.size(), false);
    vector<int> parent(g_movable_cells.size(), -1);
    vector<char> move_dir(g_movable_cells.size(), 0);
    queue<int> q;
    
    q.push(from_pos);
    visited[from_pos] = true;
    
    const int DY[4] = {-1, 0, 1, 0};
    const int DX[4] = {0, -1, 0, 1};
    const char DIRS[4] = {'W', 'A', 'S', 'D'};
    
    while (!q.empty()) {
        int pos = q.front();
        q.pop();
        
        int y = g_movable_cells[pos].first;
        int x = g_movable_cells[pos].second;
        
        for (int dir = 0; dir < 4; dir++) {
            int ny = y + DY[dir];
            int nx = x + DX[dir];
            
            if (ny < 0 || ny >= g_height || nx < 0 || nx >= g_width) continue;
            if (g_base_map[ny][nx] == '#') continue;
            
            auto it = g_cell_to_index.find({ny, nx});
            if (it == g_cell_to_index.end()) continue;
            int new_pos = it->second;
            
            if (visited[new_pos] || bit_state.has_box(new_pos)) continue;
            
            visited[new_pos] = true;
            parent[new_pos] = pos;
            move_dir[new_pos] = DIRS[dir];
            
            if (new_pos == to_pos) {
                string path;
                path.reserve(32);
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
    
    return "";
}

vector<pair<BitState, string>> generate_macro_moves(const BitState& bit_state) {
    vector<pair<BitState, string>> push_states;
    
    vector<bool> player_reachable(g_movable_cells.size(), false);
    queue<int> q;
    q.push(bit_state.player_pos);
    player_reachable[bit_state.player_pos] = true;
    
    while (!q.empty()) {
        int pos = q.front();
        q.pop();
        
        int y = g_movable_cells[pos].first;
        int x = g_movable_cells[pos].second;
        
        const int DY[4] = {-1, 0, 1, 0};
        const int DX[4] = {0, -1, 0, 1};
        const char DIRS[4] = {'W', 'A', 'S', 'D'};
        
        for (int dir = 0; dir < 4; dir++) {
            int ny = y + DY[dir];
            int nx = x + DX[dir];
            
            if (ny < 0 || ny >= g_height || nx < 0 || nx >= g_width) continue;
            if (g_base_map[ny][nx] == '#') continue;
            
            auto it = g_cell_to_index.find({ny, nx});
            if (it == g_cell_to_index.end()) continue;
            int new_pos = it->second;
            
            if (bit_state.has_box(new_pos)) {
                int box_ny = ny + DY[dir];
                int box_nx = nx + DX[dir];
                
                if (box_ny < 0 || box_ny >= g_height || box_nx < 0 || box_nx >= g_width) continue;
                if (g_base_map[box_ny][box_nx] == '#') continue;
                
                auto box_it = g_cell_to_index.find({box_ny, box_nx});
                if (box_it == g_cell_to_index.end()) continue;
                int box_dest_pos = box_it->second;
                
                if (bit_state.has_box(box_dest_pos)) continue;
                
                if (player_reachable[pos]) {
                    BitState new_state = bit_state;
                    new_state.move_box(new_pos, box_dest_pos);
                    new_state.player_pos = new_pos;
                    
                    string path = find_path(bit_state.player_pos, pos, bit_state);
                    path += DIRS[dir];
                    
                    push_states.push_back({new_state, path});
                }
            } else {
                if (!player_reachable[new_pos]) {
                    player_reachable[new_pos] = true;
                    q.push(new_pos);
                }
            }
        }
    }
    
    return push_states;
}

// Parallel work-stealing A* solver
class ParallelAStar {
private:
    ThreadPool& pool;
    ConcurrentVisited visited_states;
    mutex open_set_mutex;
    priority_queue<AStarNode, vector<AStarNode>, greater<AStarNode>> open_set;
    atomic<bool> search_complete{false};
    
public:
    ParallelAStar(ThreadPool& tp) : pool(tp) {}
    
    string solve(const BitState& initial_state) {
        AStarNode start_node;
        start_node.bit_state = initial_state;
        start_node.path = "";
        start_node.g_cost = 0;
        start_node.h_cost = calculate_heuristic_bitstate_adaptive(initial_state);
        start_node.last_move = 0;
        start_node.last_was_push = false;
        
        open_set.push(start_node);
        visited_states.insert(initial_state);
        
        const int num_workers = thread::hardware_concurrency();
        vector<future<void>> workers;
        
        for (int i = 0; i < num_workers; ++i) {
            workers.push_back(pool.enqueue([this]() {
                this->worker_thread();
            }));
        }
        
        // Wait for completion
        for (auto& worker : workers) {
            worker.wait();
        }
        
        if (solution_found.load()) {
            std::lock_guard<std::mutex> lock(solution_mutex);
            return global_solution;
        }
        
        throw Oops("Parallel A* search could not find solution");
    }
    
private:
    void worker_thread() {
        active_threads.fetch_add(1);
        
        while (!solution_found.load() && !search_complete.load()) {
            AStarNode current;
            bool has_work = false;
            
            {
                std::lock_guard<std::mutex> lock(open_set_mutex);
                if (!open_set.empty()) {
                    current = open_set.top();
                    open_set.pop();
                    has_work = true;
                }
            }
            
            if (!has_work) {
                // Check if all threads are idle
                if (active_threads.load() == 1) {
                    search_complete.store(true);
                }
                this_thread::sleep_for(chrono::microseconds(10));
                continue;
            }
            
            if (is_solved(current.bit_state)) {
                std::lock_guard<std::mutex> lock(solution_mutex);
                if (!solution_found.load()) {
                    global_solution = current.path;
                    solution_found.store(true);
                }
                break;
            }
            
            if (is_deadlock(current.bit_state)) {
                continue;
            }
            
            // Generate successors in parallel batches
            vector<pair<BitState, string>> successors = generate_macro_moves(current.bit_state);
            
            // Process successors in batches to reduce lock contention
            const int batch_size = 10;
            for (int i = 0; i < (int)successors.size(); i += batch_size) {
                vector<AStarNode> batch;
                
                for (int j = i; j < min(i + batch_size, (int)successors.size()); ++j) {
                    const auto& [new_bit_state, move_path] = successors[j];
                    
                    // Skip if already visited
                    if (visited_states.contains(new_bit_state)) continue;
                    
                    // Quick deadlock check
                    if (is_deadlock(new_bit_state)) continue;
                    
                    // Mark as visited
                    if (!visited_states.insert(new_bit_state)) continue;
                    
                    AStarNode new_node;
                    new_node.bit_state = new_bit_state;
                    new_node.path = current.path + move_path;
                    new_node.g_cost = current.g_cost + 1;
                    new_node.h_cost = calculate_heuristic_bitstate_adaptive(new_bit_state);
                    new_node.last_move = move_path.back();
                    new_node.last_was_push = true;
                    
                    batch.push_back(new_node);
                }
                
                if (!batch.empty()) {
                    std::lock_guard<std::mutex> lock(open_set_mutex);
                    for (const auto& node : batch) {
                        open_set.push(node);
                    }
                }
            }
        }
        
        active_threads.fetch_sub(1);
    }
};

string astar_solver_parallel(const string& filename) {
    GameState initial_state = loadstate(filename);
    
    // Initialize compression
    init_compression(initial_state);
    init_targets_and_distances();
    
    g_base_map = initial_state.map;
    
    for (int y = 0; y < (int)g_base_map.size(); ++y) {
        for (int x = 0; x < (int)g_base_map[y].size(); ++x) {
            char c = g_base_map[y][x];
            if (c == 'x') {
                g_base_map[y][x] = ' ';
            } else if (c == 'X') {
                g_base_map[y][x] = '.';
            } else if (c == 'o') {
                g_base_map[y][x] = ' ';
            } else if (c == 'O') {
                g_base_map[y][x] = '.';
            }
        }
    }
    
    // Create thread pool with optimal number of threads
    const int num_threads = max(2, (int)thread::hardware_concurrency());
    ThreadPool pool(num_threads);
    
    ParallelAStar solver(pool);
    BitState initial_bit_state = compress_to_bitstate(initial_state);
    
    return solver.solve(initial_bit_state);
}

string main_solver(const string& filename) {
    return astar_solver_parallel(filename);
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
        cerr << "Solved in " << duration.count() << " ms using parallel A*" << endl;
        return 0;
    } catch (const Oops& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    } catch (const exception& e) {
        cerr << "Unexpected error: " << e.what() << endl;
        return 1;
    }
}
