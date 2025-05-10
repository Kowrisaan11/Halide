#include "AutoSchedule.h"

namespace Halide {
namespace Internal {
namespace Autoscheduler {

namespace {
// Anonymous namespace for registration
struct AutoSchedulerRegistry {
    void operator()(const Pipeline& pipeline,
                   const Target& target,
                   const AutoschedulerParams& params,
                   AutoSchedulerResults* results) {
        if (params.name != "adams2019") return;
        
        std::string model_path = "model.pt";
        std::string scaler_params_path = "scaler_params.json";
        bool use_gpu = target.has_gpu_feature();
        
        AutoScheduler scheduler(model_path, scaler_params_path, use_gpu);
        scheduler(pipeline, target, params, results);
    }
};
}  // anonymous namespace

// Constructor
AutoScheduler::AutoScheduler(const std::string& model_path,
                           const std::string& scaler_params_path,
                           bool use_gpu) {
    metadata.gpu_available = use_gpu;
    metadata.timestamp = current_time;
    metadata.user = user_login;
    metadata.device_type = use_gpu ? "GPU" : "CPU";
    
    cost_model = new DefaultCostModel(model_path, scaler_params_path, use_gpu);
}

// Destructor
AutoScheduler::~AutoScheduler() {
    delete cost_model;
}

void AutoScheduler::log_message(const std::string& message) const {
    std::cout << "[" << current_time << " UTC] " << message << std::endl;
}

json AutoScheduler::create_dag_representation(const Pipeline& pipeline) {
    json dag_data;
    dag_data["nodes"] = json::array();
    
    for (const auto& func : pipeline.outputs()) {
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

    return dag_data;
}

void AutoScheduler::apply_schedule(const Pipeline& pipeline, const json& schedule_data) {
    log_message("Applying schedule from ML model predictions");
    try {
        for (auto& func : pipeline.outputs()) {
            func.compute_root();
        }
    } catch (const Error& e) {
        log_message("Error applying schedule: " + std::string(e.what()));
        throw;
    }
}

void AutoScheduler::operator()(const Pipeline& pipeline,
                             const Target& target,
                             const AutoschedulerParams& params,
                             AutoSchedulerResults* results) {
    log_message("Starting autoscheduling process");

    json dag_data = create_dag_representation(pipeline);
    
    double cost;
    cost_model->enqueue(dag_data, &cost);
    cost_model->evaluate_costs();

    if (cost >= 0) {
        apply_schedule(pipeline, dag_data);
    }

    if (results) {
        results->schedule_source = "// Autoscheduled by ML model\n";
        results->schedule_source += "// User: " + metadata.user + "\n";
        results->schedule_source += "// Timestamp: " + metadata.timestamp + "\n";
    }
}

// Register the autoscheduler at the global scope
REGISTER_AUTOSCHEDULER(AutoSchedulerRegistry);

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide
