#ifndef AUTO_SCHEDULE_H
#define AUTO_SCHEDULE_H

#include <string>
#include <vector>
#include <iostream>
#include <Halide.h>
#include "CostModel.h"
#include "DefaultCostModel.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace Halide {
namespace Internal {
namespace Autoscheduler {

class AutoScheduler {
private:
    CostModel* cost_model;
    const std::string current_time{"2025-05-10 18:54:37"};
    const std::string user_login{"Jathu03"};
    
    struct MetaData {
        bool gpu_available{false};
        std::string timestamp;
        std::string user;
        std::string device_type;
    } metadata;

public:
    AutoScheduler(const std::string& model_path,
                 const std::string& scaler_params_path,
                 bool use_gpu = false);
    ~AutoScheduler();

    void operator()(const Pipeline& pipeline,
                   const Target& target,
                   const AutoschedulerParams& params,
                   AutoSchedulerResults* results);

private:
    void log_message(const std::string& message) const;
    json create_dag_representation(const Pipeline& pipeline);
    void apply_schedule(const Pipeline& pipeline, const json& schedule_data);
};

struct AutoSchedulerRegistry {
    void operator()(const Pipeline& pipeline,
                   const Target& target,
                   const AutoschedulerParams& params,
                   AutoSchedulerResults* results);
};

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide

#endif  // AUTO_SCHEDULE_H
