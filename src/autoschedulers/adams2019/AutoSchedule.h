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
    int beam_size;
    int max_children;
    double exploration_factor;
};

class AutoScheduler {
public:
    AutoScheduler(CostModel* cost_model, const Adams2019Params& params);
    
    // Main entry point for autoscheduling
    void schedule(FunctionDAG& dag,
                 const std::vector<Function>& outputs,
                 const Target& target);

private:
    CostModel* cost_model;
    Adams2019Params params;
    SearchSpace search_space;
    
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
    
    // Performance tracking
    struct PerformanceMetrics {
        std::chrono::system_clock::time_point start_time;
        int states_evaluated;
        int valid_states;
        double best_cost;
    };
    PerformanceMetrics metrics;
};

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide

#endif  // AUTO_SCHEDULE_H
