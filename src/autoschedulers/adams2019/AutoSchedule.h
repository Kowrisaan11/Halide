#ifndef AUTO_SCHEDULE_H
#define AUTO_SCHEDULE_H

#include <string>
#include <vector>
#include "CostModel.h"
#include "FunctionDAG.h"
#include "State.h"
#include <chrono>

namespace Halide {
namespace Internal {
namespace Autoscheduler {

struct SearchSpace {
    int beam_size{32};
    int max_children{1000};
    double exploration_factor{0.1};
};

class AutoScheduler {
public:
    AutoScheduler(CostModel* cost_model)
        : cost_model(cost_model), 
          session_start(std::chrono::system_clock::now()),
          user_login("Jathu03") {
        initialize_metrics();
    }

    void schedule(FunctionDAG& dag,
                 const std::vector<Function>& outputs,
                 const Target& target);

private:
    CostModel* cost_model;
    SearchSpace search_space;
    std::string user_login;
    std::chrono::system_clock::time_point session_start;
    
    // Tree representation handling
    TreeRepresentation create_initial_tree(const FunctionDAG& dag);
    void update_tree_with_schedule(TreeRepresentation& tree, const State& state);
    
    // Search methods
    IntrusivePtr<State> beam_search(FunctionDAG& dag, 
                                   const std::vector<Function>& outputs,
                                   const Target& target);
    
    // Schedule generation
    void apply_schedule(const State& state, FunctionDAG& dag);
    
    // Utility methods
    double evaluate_state(const State& state, const FunctionDAG& dag);
    bool is_valid_schedule(const State& state, const FunctionDAG& dag);
    void log_message(const std::string& message);
    std::string get_timestamp();
    
    // Performance tracking
    struct PerformanceMetrics {
        std::chrono::system_clock::time_point start_time;
        int states_evaluated;
        int valid_states;
        double best_cost;
    } metrics;

    void initialize_metrics() {
        metrics = {
            std::chrono::system_clock::now(),
            0,
            0,
            std::numeric_limits<double>::infinity()
        };
    }
};

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide

#endif  // AUTO_SCHEDULE_H
