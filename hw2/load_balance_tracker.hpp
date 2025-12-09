#ifndef LOAD_BALANCE_TRACKER_HPP
#define LOAD_BALANCE_TRACKER_HPP

#include <vector>
#include <string>
#include <chrono>
#include <mpi.h>

class LoadBalanceTracker {
public:
    struct TaskRecord {
        int mpi_rank;
        int thread_id;
        std::string phase;
        double start_time;
        double end_time;
        double duration;
        int work_size;
    };

    LoadBalanceTracker(int rank, int size, bool enabled = true);
    ~LoadBalanceTracker();

    void task_start(const std::string& phase, int thread_id, int work_size);
    void task_end(const std::string& phase, int thread_id);
    
    void gather_all_records(int root = 0);
    void save_to_csv(const std::string& filename);

    // RAII helper for automatic tracking
    class ScopedTask {
    public:
        ScopedTask(LoadBalanceTracker* tracker, const std::string& phase, 
                   int thread_id, int work_size);
        ~ScopedTask();
    private:
        LoadBalanceTracker* tracker_;
        std::string phase_;
        int thread_id_;
    };

private:
    int rank_;
    int size_;
    bool enabled_;
    std::chrono::high_resolution_clock::time_point start_time_;
    std::vector<TaskRecord> local_records_;
    std::vector<TaskRecord> all_records_;
    
    struct ActiveTask {
        std::string phase;
        double start_time;
        int work_size;
    };
    std::vector<ActiveTask> active_tasks_;
    
    double get_current_time();
};

#endif // LOAD_BALANCE_TRACKER_HPP
