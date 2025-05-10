#pragma once

#include "CostModel.h"
#include <torch/script.h>
#include <memory>
#include <vector>
#include <string>

namespace Halide {

class SimpleLSTMModel : public CostModel {
public:
    SimpleLSTMModel(const std::string &model_path,
                    const std::string &scaler_path,
                    bool use_gpu = false);
    ~SimpleLSTMModel() override = default;

    void set_pipeline_features(const Internal::Autoscheduler::FunctionDAG &dag,
                              const Internal::Autoscheduler::Adams2019Params &params) override;

    void enqueue(const Internal::Autoscheduler::FunctionDAG &dag,
                 const Halide::Internal::Autoscheduler::StageMapOfScheduleFeatures &schedule_feats,
                 double *cost_ptr) override;

    void evaluate_costs() override;
    void reset() override;

private:
    struct QueueItem {
        // Store enough info to run the model in evaluate_costs
        std::vector<double> features;
        double *cost_ptr;
        // Add more as needed (e.g. raw JSON if required)
    };

    std::vector<QueueItem> queue;

    torch::jit::script::Module model;
    bool use_gpu;
    torch::Device device;

    // Scaler and feature info
    std::vector<double> X_scalar_center, X_scalar_scale;
    double y_center, y_scale;
    std::vector<std::string> feature_columns;

    void load_scaler(const std::string &scaler_path);
    std::vector<double> extract_features(const Internal::Autoscheduler::FunctionDAG &dag,
                                         const Halide::Internal::Autoscheduler::StageMapOfScheduleFeatures &schedule_feats);

    double run_model(const std::vector<double> &features);
};

} // namespace Halide
