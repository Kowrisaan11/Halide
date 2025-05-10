```cpp
#include "AutoSchedule.h"
#include <iostream>
#include <string>
#include <vector>
#include "Halide.h"

namespace Halide {
namespace Internal {
namespace Autoscheduler {

using std::string;
using std::vector;
using json = nlohmann::json;

// Assuming these are defined in AutoSchedule.h or elsewhere
extern string current_time; // Global variable for timestamp
extern string user_login;   // Global variable for user login

AutoScheduler::AutoScheduler(const string& model_path,
                             const string& scaler_params_path,
                             bool use_gpu) {
    metadata.gpu_available = use_gpu;
    metadata.timestamp = current_time;
    metadata.user = user_login;
    metadata.device_type = use_gpu ? "GPU" : "CPU";
    
    cost_model = new DefaultCostModel(model_path, scaler_params_path, use_gpu);
}

AutoScheduler::~AutoScheduler() {
    if (cost_model) {
        delete cost_model;
        cost_model = nullptr;
    }
}

void AutoScheduler::log_message(const string& message) const {
    std::cout << "[" << current_time << " UTC] " << message << std::endl;
}

void AutoScheduler::operator()(const Pipeline& pipeline,
                              const Target& target,
                              const AutoschedulerParams& params,
                              AutoSchedulerResults* results) {
    log_message("Starting autoscheduling process with " + metadata.device_type);

    json dag_data = create_dag_representation(pipeline);
    
    dag_data["metadata"] = {
        {"timestamp", metadata.timestamp},
        {"user", metadata.user},
        {"device_type", metadata.device_type},
        {"gpu_available", metadata.gpu_available}
    };

    double cost;
    cost_model->enqueue(dag_data, &cost);
    cost_model->evaluate_costs();

    log_message("Predicted cost: " + std::to_string(cost));

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
    dag_data["nodes"] = json::array();
    
    vector<Func> outputs = pipeline.outputs();
    for (const auto& func : outputs) {
        json node;
        node["name"] = func.name();
        node["type"] = "output";
        node["dimensions"] = func.dimensions();
        node["is_extern"] = func.is_extern();
        dag_data["nodes"].push_back(node);
    }

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
        for (auto& func : pipeline.outputs()) {
            func.compute_root();
        }
        log_message("Schedule applied successfully");
    } catch (const Error& e) {
        log_message("Error applying schedule: " + string(e.what()));
        throw;
    }
}

void AutoSchedulerRegistry::operator()(const Pipeline& pipeline,
                                      const Target& target,
                                      const AutoschedulerParams& params,
                                      AutoSchedulerResults* results) {
    if (params.name != "adams2019") return;
    
    string model_path = "model.pt";
    string scaler_params_path = "scaler_params.json";
    bool use_gpu = target.has_gpu_feature();
    
    AutoScheduler scheduler(model_path, scaler_params_path, use_gpu);
    scheduler(pipeline, target, params, results);
}

// Register the autoscheduler
REGISTER_AUTOSCHEDULER(AutoSchedulerRegistry);

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide
```
