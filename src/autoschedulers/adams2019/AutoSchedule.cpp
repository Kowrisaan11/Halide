#include "AutoSchedule.h"
#include "Cache.h"
#include "ASLog.h"
#include <algorithm>
#include <chrono>
#include <iostream>

namespace Halide {
namespace Internal {
namespace Autoscheduler {

using std::vector;
using std::string;
using std::map;

AutoScheduler::AutoScheduler(CostModel* model, const Adams2019Params& p) 
    : cost_model(model), params(p) {
    search_space.beam_size = params.beam_size;
    search_space.max_children = 1000;
    search_space.exploration_factor = 0.1;
    
    metrics = {
        std::chrono::system_clock::now(),
        0,
        0,
        std::numeric_limits<double>::infinity()
    };
}

TreeRepresentation AutoScheduler::create_initial_tree(const FunctionDAG& dag) {
    // Create initial tree representation from DAG
    return cost_model->convert_to_tree(dag, params);
}

void AutoScheduler::update_tree_with_schedule(TreeRepresentation& tree, const State& state) {
    // Update tree representation with new schedule state
    json schedule_data;
    
    // Extract schedule information from state
    for (const auto& n : state.root->stages()) {
        json stage_info;
        stage_info["name"] = n.stage->func.name();
        stage_info["schedule_features"] = n.schedule_features;
        schedule_data["stages"].push_back(stage_info);
    }
    
    // Update tree data
    tree.tree_data["schedule"] = schedule_data;
    tree.extracted_features = cost_model->extract_features(tree.tree_data);
}

double AutoScheduler::evaluate_state(const State& state, const FunctionDAG& dag) {
    TreeRepresentation tree = create_initial_tree(dag);
    update_tree_with_schedule(tree, state);
    
    auto prediction = cost_model->get_prediction(tree, params.gpu_enabled);
    metrics.states_evaluated++;
    
    if (prediction.corrected_prediction < metrics.best_cost) {
        metrics.best_cost = prediction.corrected_prediction;
    }
    
    return prediction.corrected_prediction;
}

bool AutoScheduler::is_valid_schedule(const State& state, const FunctionDAG& dag) {
    // Implement schedule validation logic
    // Check for compute/store_at validity, buffer allocation, etc.
    return true; // Placeholder
}

IntrusivePtr<State> AutoScheduler::beam_search(FunctionDAG& dag,
                                             const vector<Function>& outputs,
                                             const Target& target) {
    vector<IntrusivePtr<State>> beam = {new State()};
    vector<IntrusivePtr<State>> next_beam;
    
    while (!beam.empty()) {
        next_beam.clear();
        
        // Generate children for each state in the beam
        for (auto& state : beam) {
            vector<IntrusivePtr<State>> children;
            state->generate_children(dag, params, cost_model, 
                [&](IntrusivePtr<State>&& child) {
                    if (is_valid_schedule(*child, dag)) {
                        children.push_back(std::move(child));
                        metrics.valid_states++;
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
            for (int i = 0; i < std::min(search_space.beam_size, (int)children.size()); i++) {
                next_beam.push_back(children[i]);
            }
        }
        
        // Sort and prune next beam
        std::sort(next_beam.begin(), next_beam.end(),
                 [](const IntrusivePtr<State>& a, const IntrusivePtr<State>& b) {
                     return a->cost < b->cost;
                 });
        
        if (next_beam.size() > search_space.beam_size) {
            next_beam.resize(search_space.beam_size);
        }
        
        beam = std::move(next_beam);
    }
    
    // Return best state found
    return beam[0];
}

void AutoScheduler::apply_schedule(const State& state, FunctionDAG& dag) {
    // Apply the final schedule to the pipeline
    state.apply_schedule(dag, params);
}

void AutoScheduler::schedule(FunctionDAG& dag,
                           const vector<Function>& outputs,
                           const Target& target) {
    aslog(1) << "Starting autoscheduling process...\n";
    
    // Initialize cost model with pipeline features
    cost_model->set_pipeline_features(dag, params);
    
    // Perform beam search to find best schedule
    auto best_state = beam_search(dag, outputs, target);
    
    // Apply the best schedule found
    apply_schedule(*best_state, dag);
    
    // Log performance metrics
    auto end_time = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - metrics.start_time);
    
    aslog(1) << "Autoscheduling completed:\n"
             << "  Total states evaluated: " << metrics.states_evaluated << "\n"
             << "  Valid states found: " << metrics.valid_states << "\n"
             << "  Best cost achieved: " << metrics.best_cost << "\n"
             << "  Time taken: " << duration.count() << " seconds\n";
}

// Register the autoscheduler
struct Adams2019Autoscheduler {
    void operator()(const Pipeline& pipeline,
                   const Target& target,
                   const AutoschedulerParams& autoscheduler_params,
                   AutoSchedulerResults* results) {
        if (autoscheduler_params.name != "Adams2019") {
            return;
        }
        
        // Parse parameters
        Adams2019Params params;
        {
            ParamParser parser(autoscheduler_params.extra);
            parser.parse("parallelism", &params.parallelism);
            parser.parse("beam_size", &params.beam_size);
            parser.parse("gpu_enabled", &params.gpu_enabled);
            // Add other parameters as needed
            parser.finish();
        }
        
        // Initialize cost model
        auto cost_model = std::make_unique<DefaultCostModel>(
            "model.pt",
            "scaler_params.json",
            "calibration_data.txt",
            params.gpu_enabled
        );
        
        // Create autoscheduler instance
        AutoScheduler scheduler(cost_model.get(), params);
        
        // Extract outputs
        vector<Function> outputs;
        for (const Func& f : pipeline.outputs()) {
            outputs.push_back(f.function());
        }
        
        // Create DAG representation
        FunctionDAG dag(outputs, target);
        
        // Generate schedule
        scheduler.schedule(dag, outputs, target);
        
        // Store results if needed
        if (results) {
            // Store schedule source and other results
            // results->schedule_source = ...;
        }
    }
};

// Register the autoscheduler
REGISTER_AUTOSCHEDULER(Adams2019Autoscheduler)

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide
