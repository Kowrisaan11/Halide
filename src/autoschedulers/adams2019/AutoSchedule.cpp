/*
  AutoSchedule.cpp: Implementation of the adams2019 autoscheduler for Halide.
  Uses a cost model to evaluate schedules and search for an optimal one.
*/

#include "AutoSchedule.h"
#include "CostModel.h"
#include "SimpleLSTMModel.h"
#include "FunctionDAG.h"
#include "State.h"
#include "ASLog.h"
#include "Featurization.h"
#include "Timer.h"
#include <algorithm>
#include <memory>
#include <queue>
#include <set>
#include <unordered_map>
#include <vector>

namespace Halide {
namespace Internal {
namespace Autoscheduler {

struct AutoSchedule {
    const FunctionDAG &dag;
    const MachineParams ¶ms;
    std::unique_ptr<CostModel> cost_model;
    std::vector<IntrusivePtr<State>> best_states;

    AutoSchedule(const FunctionDAG &dag, const MachineParams ¶ms)
        : dag(dag), params(params) {
        // Initialize the cost model with paths to the .pt model and scaler parameters
        cost_model = std::make_unique<SimpleLSTMModel>(
            "/home/kowrisaan/fyp/Halide/src/autoschedulers/adams2019/model.pt",
            "/home/kowrisaan/fyp/Halide/src/autoschedulers/adams2019/scaler_params.json");
    }

    void generate_schedule() {
        // Initial state
        IntrusivePtr<State> root = new State(dag, params);
        root->compute_features(dag);

        // Priority queue for beam search
        std::priority_queue<std::pair<double, IntrusivePtr<State>>,
                            std::vector<std::pair<double, IntrusivePtr<State>>>,
                            std::greater<>> beam;
        beam.emplace(0.0, root);

        // Beam search parameters
        const int beam_size = params.beam_size;
        const int max_depth = 100; // Example limit
        std::set<uint64_t> visited;

        while (!beam.empty()) {
            // Get the top states
            std::vector<std::pair<double, IntrusivePtr<State>>> current_beam;
            for (int i = 0; i < beam_size && !beam.empty(); i++) {
                current_beam.push_back(beam.top());
                beam.pop();
            }

            // Expand each state
            for (const auto &p : current_beam) {
                IntrusivePtr<State> state = p.second;
                if (state->depth >= max_depth) {
                    continue;
                }

                // Generate child states
                std::vector<IntrusivePtr<State>> children = state->generate_children(dag, params);
                for (const auto &child : children) {
                    if (visited.find(child->hash()) != visited.end()) {
                        continue;
                    }
                    visited.insert(child->hash());

                    // Evaluate cost using SimpleLSTMModel
                    double cost = cost_model->evaluate_cost(child, dag);
                    child->cost = cost;

                    // Add to beam
                    beam.emplace(cost, child);

                    // Keep track of best states
                    if (best_states.size() < static_cast<size_t>(beam_size)) {
                        best_states.push_back(child);
                        std::push_heap(best_states.begin(), best_states.end(),
                                       [](const IntrusivePtr<State> &a, const IntrusivePtr<State> &b) {
                                           return a->cost > b->cost;
                                       });
                    } else if (cost < best_states.front()->cost) {
                        std::pop_heap(best_states.begin(), best_states.end(),
                                      [](const IntrusivePtr<State> &a, const IntrusivePtr<State> &b) {
                                          return a->cost > b->cost;
                                      });
                        best_states.back() = child;
                        std::push_heap(best_states.begin(), best_states.end(),
                                       [](const IntrusivePtr<State> &a, const IntrusivePtr<State> &b) {
                                           return a->cost > b->cost;
                                       });
                    }
                }
            }

            // Trim beam to beam_size
            while (beam.size() > static_cast<size_t>(beam_size)) {
                beam.pop();
            }
        }
    }

    IntrusivePtr<State> get_best_schedule() const {
        if (best_states.empty()) {
            return nullptr;
        }
        return best_states.front();
    }
};

void generate_function_schedule(FunctionDAG &dag, const MachineParams ¶ms) {
    AutoSchedule scheduler(dag, params);
    scheduler.generate_schedule();
    IntrusivePtr<State> best = scheduler.get_best_schedule();
    if (best != nullptr) {
        best->apply_schedule(dag);
    } else {
        ASLog::warning() << "No valid schedule found\n";
    }
}

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide
