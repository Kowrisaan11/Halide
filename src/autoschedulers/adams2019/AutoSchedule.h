#ifndef AUTO_SCHEDULE_H
#define AUTO_SCHEDULE_H

#include <string>
#include <vector>
#include "CostModel.h"
#include "DefaultCostModel.h"
#include <nlohmann/json.hpp>
// Add these Halide includes
#include "Halide.h"
#include "AutoSchedule.h"
#include "Error.h"
#include "Func.h"
#include "Pipeline.h"
#include "Target.h"

using json = nlohmann::json;

namespace Halide {
namespace Internal {
namespace Autoscheduler {

class AutoScheduler {
private:
    CostModel* cost_model;
    const std::string current_time{"2025-05-10 18:41:09"};
    const std::string user_login{"Jathu03"};
    
    struct MetaData {
        bool gpu_available{false};
        std::string timestamp;
        std::string user;
        std::string device_type;
    } metadata;

    void log_message(const std::string& message) const {
        std::cout << "[" << current_time << " UTC] " << message << std::endl;
    }

    json create_dag_representation(const Pipeline& pipeline);
    void apply_schedule(const Pipeline& pipeline, const json& schedule_data);

public:
    AutoScheduler(const std::string& model_path,
                 const std::string& scaler_params_path,
                 bool use_gpu = false) {
        metadata.gpu_available = use_gpu;
        metadata.timestamp = current_time;
        metadata.user = user_login;
        metadata.device_type = use_gpu ? "GPU" : "CPU";
        
        cost_model = new DefaultCostModel(model_path, scaler_params_path, use_gpu);
    }

    ~AutoScheduler() {
        if (cost_model) {
            delete cost_model;
        }
    }

    void operator()(const Pipeline& pipeline,
                   const Target& target,
                   const AutoschedulerParams& params,
                   AutoSchedulerResults* results);
};

// Registration function
class AutoSchedulerRegistry {
public:
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

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide

#endif  // AUTO_SCHEDULE_H
