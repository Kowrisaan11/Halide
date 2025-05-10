#include "AutoSchedule.h"
#include <iostream>

namespace Halide {
namespace Internal {
namespace Autoscheduler {

void AutoScheduler::operator()(const Pipeline& pipeline,
                             const Target& target,
                             const AutoschedulerParams& params,
                             AutoSchedulerResults* results) {
    log_message("Starting autoscheduling process with " + metadata.device_type);

    // Create DAG representation
    json dag_data = create_dag_representation(pipeline);
    
    // Add metadata
    dag_data["metadata"] = {
        {"timestamp", metadata.timestamp},
        {"user", metadata.user},
        {"device_type", metadata.device_type},
        {"gpu_available", metadata.gpu_available}
    };

    // Get prediction
    double cost;
    cost_model->enqueue(dag_data, &cost);
    cost_model->evaluate_costs();

    log_message("Predicted cost: " + std::to_string(cost));

    // Apply the schedule if cost is acceptable
    if (cost >= 0) {
        apply_schedule(pipeline, dag_data);
        log_message("Schedule applied successfully");
    } else {
        log_message("Warning: Invalid cost prediction, schedule not applied");
    }

    if (results) {
        results->schedule_source = "// Autoscheduled by ML model\n";
        results->schedule_source += "// User: " + metadata.user + "\n";
        results->schedule_source += "// Timestamp: " + metadata.timestamp + "\n";
        results->schedule_source += "// Device: " + metadata.device_type + "\n";
        results->schedule_source += "// Predicted cost: " + std::to_string(cost) + "\n";
    }
}

json AutoScheduler::create_dag_representation(const Pipeline& pipeline) {
    json dag_data;
    
    // Add nodes
    dag_data["nodes"] = json::array();
    
    std::vector<Func> outputs = pipeline.outputs();
    for (const auto& func : outputs) {
        json node;
        node["name"] = func.name();
        node["type"] = "output";
        node["dimensions"] = func.dimensions();
        node["is_extern"] = func.is_extern();
        dag_data["nodes"].push_back(node);
    }

    // Add metadata
    dag_data["metadata"] = {
        {"timestamp", current_time},
        {"user", user_login},
        {"device_type", metadata.device_type}
    };

    log_message("Created DAG representation with " + 
                std::to_string(dag_data["nodes"].size()) + " nodes");
    
    return dag_data;
}

void AutoScheduler::apply_schedule(const Pipeline& pipeline, const json& schedule_data) {
    log_message("Applying schedule from ML model predictions");
    
    try {
        // Apply basic schedules for now
        for (auto& func : pipeline.outputs()) {
            func.compute_root();
        }
        
        log_message("Schedule applied successfully");
    } catch (const Error& e) {
        log_message("Error applying schedule: " + std::string(e.what()));
        throw;
    }
}

// Register the autoscheduler
REGISTER_AUTOSCHEDULER(AutoSchedulerRegistry)

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide
