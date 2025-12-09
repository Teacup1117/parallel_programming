// IDA* 優化版本 - 基於 hw1.cpp
// 主要改進：使用 IDA* 替換 A* 以節省記憶體，並增強死鎖檢測

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
#include <deque>

using namespace std;

class Ooos : public runtime_error {
public:
    Ooos(const string& msg) : runtime_error(msg) {}
};

// Transposition table for IDA* - stores depth bounds
static unordered_map<BitState, int, BitStateHash> transposition_table;
static int nodes_expanded_ida = 0;

// IDA* search with transposition table
int ida_search(const BitState& state, int g, int threshold, string& solution_path, const string& current_path) {
    int h = calculate_heuristic_bitstate_simple(state);
    int f = g + h;
    
    if (f > threshold) return f;
    
    if (is_solved(state)) {
        solution_path = current_path;
        return -1; // Found solution
    }
    
    nodes_expanded_ida++;
    
    // Transposition table lookup - prune if we've seen this state with better bound
    auto tt_it = transposition_table.find(state);
    if (tt_it != transposition_table.end() && tt_it->second <= threshold) {
        return INT_MAX; // Already explored with same or better threshold
    }
    
    if (is_deadlock(state)) {
        return INT_MAX;
    }
    
    int min_threshold = INT_MAX;
    
    // Generate successors using macro moves
    vector<pair<BitState, string>> successors = generate_macro_moves(state);
    
    // Order moves by heuristic value for better pruning
    sort(successors.begin(), successors.end(), [](const auto& a, const auto& b) {
        return calculate_heuristic_bitstate_simple(a.first) < calculate_heuristic_bitstate_simple(b.first);
    });
    
    for (const auto& [new_state, move_path] : successors) {
        int t = ida_search(new_state, g + 1, threshold, solution_path, current_path + move_path);
        
        if (t == -1) return -1; // Solution found
        if (t < min_threshold) min_threshold = t;
        
        // Early termination if search is taking too long
        if (nodes_expanded_ida > 50000000) { // 5千萬節點上限
            return -1;
        }
    }
    
    // Store in transposition table
    transposition_table[state] = threshold;
    
    return min_threshold;
}

string ida_star_solver(const string& filename) {
    GameState initial_state = loadstate(filename);
    
    // Initialize compression (same as original)
    g_base_map = initial_state.map;
    for (int y = 0; y < (int)g_base_map.size(); ++y) {
        for (int x = 0; x < (int)g_base_map[y].size(); ++x) {
            char c = g_base_map[y][x];
            if (c == 'x') g_base_map[y][x] = ' ';
            else if (c == 'X') g_base_map[y][x] = '.';
            else if (c == 'o') g_base_map[y][x] = ' ';
            else if (c == 'O') g_base_map[y][x] = '.';
        }
    }
    
    BitState start_state = compress_to_bitstate(initial_state);
    
    int threshold = calculate_heuristic_bitstate_simple(start_state);
    string solution;
    
    cerr << "Starting IDA* with initial threshold: " << threshold << endl;
    
    while (threshold != INT_MAX) {
        transposition_table.clear(); // Clear TT for each iteration
        nodes_expanded_ida = 0;
        
        cerr << "IDA* iteration with threshold: " << threshold << endl;
        
        int t = ida_search(start_state, 0, threshold, solution, "");
        
        if (t == -1) {
            cerr << "IDA* solved in " << nodes_expanded_ida << " nodes (total)" << endl;
            return solution;
        }
        
        cerr << "IDA* iteration completed, " << nodes_expanded_ida << " nodes, next threshold: " << t << endl;
        
        if (t == INT_MAX) break;
        threshold = t;
        
        // Memory management - clear TT if it gets too large
        if (transposition_table.size() > 1000000) {
            transposition_table.clear();
        }
    }
    
    throw Oops("IDA* search could not find solution");
}

// [這裡需要包含所有原始代碼的必要函數]
// 包括: loadstate, compress_to_bitstate, calculate_heuristic_bitstate_simple, 
//      is_solved, is_deadlock, generate_macro_moves 等

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <input_file>" << endl;
        return 1;
    }
    
    try {
        auto start_time = chrono::high_resolution_clock::now();
        string result = ida_star_solver(argv[1]);
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
