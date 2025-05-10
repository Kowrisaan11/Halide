#include "AutoSchedule.h"
#include "Cache.h"
#include "ASLog.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace Halide {
namespace Internal {
namespace Autoscheduler {

std::string AutoScheduler::get_timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&now_time), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

void AutoScheduler::log_message(const std::string& message) {
    std::cout << "[" << get_timestamp() << " UTC] " << message << std::endl;
}

TreeRepresentation AutoScheduler::create_initial_tree(const FunctionDAG& dag) {
    return cost_model->convert_to_tree(dag);
}

void AutoScheduler::update_tree_with_schedule(TreeRepresentation& tree, const State& state) {
    json schedule_data;
    
    // Add metadata
    schedule_data["timestamp"] = "2025-05-10 18:01:07";
    schedule_data["user"] = user_login;
    
    // Extract schedule information
    schedule_data["stages"] = json::array();
    for (const auto& func : state.funcs) {
        json stage_info;
        stage_info["name"] = func.name;
        // Add other stage-specific information
        schedule_data["stages"].push_back(stage_info);
    }
    
    // Update tree data
    tree.tree_data["schedule"] = schedule_data;
}

double AutoScheduler::evaluate_state(const State& state, const FunctionDAG& dag) {
    TreeRepresentation tree = create_initial_tree(dag);
    update_tree_with_schedule(tree, state);
    
    auto prediction = cost_model->get_prediction(tree, false);  // Assuming CPU for now
    metrics.states_evaluated++;
    
    if (prediction.corrected_prediction < metrics.best_cost) {
        metrics.best_cost = prediction.corrected_prediction;
        log_message("New best cost found: " + std::to_string(metrics.best_cost));
    }
    
    return prediction.corrected_prediction;
}

bool AutoScheduler::is_valid_schedule(const State& state, const FunctionDAG& dag) {
    // Basic schedule validation
    metrics.valid_states++;
    return true;
}

IntrusivePtr<State> AutoScheduler::beam_search(FunctionDAG& dag,
                                             const vector<Function>& outputs,
                                             const Target& target) {
    log_message("Starting beam search...");
    
    vector<IntrusivePtr<State>> beam = {new State(dag)};
    vector<IntrusivePtr<State>> next_beam;
    
    while (!beam.empty()) {
        next_beam.clear();
        
        for (auto& state : beam) {
            vector<IntrusivePtr<State>> children;
            state->generate_children(dag, [&](IntrusivePtr<State>&& child) {
                if (is_valid_schedule(*child, dag)) {
                    children.push_back(std::move(child));
                }
            });
            
            // Evaluate and sort children
            for (auto& child : children) {
                child->cost = evaluate_state(*child, dag);
            }
            
            std::sort(children.begin(), children.end(),
                     [](const IntrusivePtr<State>& a, const IntrusivePtr<State>& b) {
                         return a->cost < b->cost;
                     });
            
            // Add top children to next beam
            for (size_t i = 0; i < std::min(static_cast<size_t>(search_space.beam_size), children.size()); i++) {
                next_beam.push_back(children[i]);
            }
        }
        
        beam = std::move(next_beam);
    }
    
    log_message("Beam search completed. States evaluated: " + 
                std::to_string(metrics.states_evaluated));
    
    return beam[0];
}

void AutoScheduler::apply_schedule(const State& state, FunctionDAG& dag) {
    log_message("Applying final schedule...");
    // Apply the schedule from the state to the DAG
    // This will be implemented based on your scheduling requirements
}

void AutoScheduler::schedule(FunctionDAG& dag,
                           const vector<Function>& outputs,
                           const Target& target) {
    log_message("Starting autoscheduling process...");
    
    // Initialize cost model with pipeline features
    cost_model->set_pipeline_features(dag);
    
    // Perform beam search to find best schedule
    auto best_state = beam_search(dag, outputs, target);
    
    // Apply the best schedule found
    apply_schedule(*best_state, dag);
    
    // Log performance metrics
    auto end_time = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - metrics.start_time);
    
    log_message("Autoscheduling completed:\n" 
                "  Total states evaluated: " + std::to_string(metrics.states_evaluated) + "\n" +
                "  Valid states found: " + std::to_string(metrics.valid_states) + "\n" +
                "  Best cost achieved: " + std::to_string(metrics.best_cost) + "\n" +
                "  Time taken: " + std::to_string(duration.count()) + " seconds");
}

// Register the autoscheduler
struct AutoSchedulerRegistry {
    void operator()(const Pipeline& pipeline,
                   const Target& target,
                   const AutoschedulerParams& params,
                   AutoSchedulerResults* results) {
        
        if (params.name != "DefaultAutoscheduler") {
            return;
        }
        
        // Create cost model
        auto cost_model = std::make_unique<DefaultCostModel>(
            "model.pt",
            "scaler_params.json",
            target.has_gpu_feature()
        );
        
        // Create autoscheduler instance
        AutoScheduler scheduler(cost_model.get());
        
        // Extract outputs
        vector<Function> outputs;
        for (const Func& f : pipeline.outputs()) {
            outputs.push_back(f.function());
        }
        
        // Create DAG representation
        FunctionDAG dag(outputs, target);
        
        // Generate schedule
        scheduler.schedule(dag, outputs, target);
        
        if (results) {
            // Store results if needed
        }
    }
};

// Register the autoscheduler
REGISTER_AUTOSCHEDULER(AutoSchedulerRegistry)

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide
