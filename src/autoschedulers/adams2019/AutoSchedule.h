#ifndef AUTO_SCHEDULE_H
#define AUTO_SCHEDULE_H

#include <string>
#include <vector>
#include "CostModel.h"
#include "FunctionDAG.h"
#include "State.h"
#include "Adams2019Params.h"

namespace Halide {
namespace Internal {
namespace Autoscheduler {

struct SearchSpace {
    int beam_size{32};
    int max_children{1000};
    double exploration_factor{0.1};
};

class AutoScheduler {
private:
    CostModel* cost_model;
    SearchSpace search_space;
    std::string user_login{"Jathu03"};
    
    // Performance tracking
    struct PerformanceMetrics {
        int states_evaluated{0};
        int valid_states{0};
        double best_cost{std::numeric_limits<double>::infinity()};
    } metrics;

public:
    AutoScheduler(CostModel* cost_model) : cost_model(cost_model) {}

    void schedule(FunctionDAG& dag,
                 const std::vector<Function>& outputs,
                 const Target& target);

private:
    TreeRepresentation create_initial_tree(const FunctionDAG& dag);
    void update_tree_with_schedule(TreeRepresentation& tree, const State& state);
    IntrusivePtr<State> beam_search(FunctionDAG& dag, 
                                   const std::vector<Function>& outputs,
                                   const Target& target);
    void apply_schedule(const State& state, FunctionDAG& dag);
    double evaluate_state(const State& state, const FunctionDAG& dag);
    bool is_valid_schedule(const State& state, const FunctionDAG& dag);
    void log_message(const std::string& message);
};

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide

#endif  // AUTO_SCHEDULE_H
