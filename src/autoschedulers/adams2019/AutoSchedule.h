#ifndef AUTO_SCHEDULE_H
#define AUTO_SCHEDULE_H

#include <string>
#include <vector>
#include <iostream>
#include "Halide.h"
#include "CostModel.h"
#include "DefaultCostModel.h"
#include <nlohmann/json.hpp>

namespace Halide {
namespace Internal {
namespace Autoscheduler {

using json = nlohmann::json;

class AutoScheduler {
private:
    CostModel* cost_model{nullptr};
    const std::string current_time{"2025-05-10 19:42:09"};
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

    json create_dag_representation(const Halide::Pipeline& pipeline);
    void apply_schedule(const Halide::Pipeline& pipeline, const json& schedule_data);

public:
    AutoScheduler(const std::string& model_path,
                 const std::string& scaler_params_path,
                 bool use_gpu = false);
    ~AutoScheduler();

    void operator()(const Halide::Pipeline& pipeline,
                   const Halide::Target& target,
                   const Halide::AutoschedulerParams& params,
                   Halide::AutoSchedulerResults* results);
};

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide

#endif  // AUTO_SCHEDULE_H
