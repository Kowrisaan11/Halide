
/*
  SimpleLSTMModel.h: Header for LibTorch-based LSTM cost model for Halide autoscheduler.
*/

#ifndef HALIDE_AUTOSCHEDULER_SIMPLE_LSTM_MODEL_H
#define HALIDE_AUTOSCHEDULER_SIMPLE_LSTM_MODEL_H

#include "CostModel.h"
#include "TreeRepresentation.h"
#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <memory>
#include <vector>
#include <map>

namespace Halide {
namespace Internal {
namespace Autoscheduler {

class SimpleLSTMModel : public CostModel {
public:
    SimpleLSTMModel();
    ~SimpleLSTMModel() override = default;

    void set_pipeline_features(const FunctionDAG& dag, const Adams2019Params& params) override;
    void enqueue(const FunctionDAG& dag, const StageMapOfScheduleFeatures& schedule_feats, double* cost_ptr) override;
    void evaluate_costs() override;
    void reset() override;

private:
    std::map<std::string, double> extract_features(const nlohmann::json& json_data);
    double compute_complexity_score(const std::map<std::string, double>& features);
    double get_raw_prediction(torch::Tensor seq_input, torch::Tensor scalar_input);
    double correct_prediction(double raw_prediction, const std::map<std::string, double>& features);

    torch::Device device_;
    struct HardwareCorrectionFactors {
        double base_correction;
        double gpu_correction;
        double scaling_factor;
        double min_time_ms;
        double high_threshold_ms;
        double high_scaling;
    } factors_;
    nlohmann::json scaler_params_;
    std::vector<double> X_scalar_center_;
    std::vector<double> X_scalar_scale_;
    double y_center_;
    double y_scale_;
    std::vector<std::string> feature_columns_;
    torch::jit::script::Module model_;
    std::map<std::string, struct CategoryCorrection> category_calibration_;
    std::unique_ptr<TreeRepresentation> tree_;
    const FunctionDAG* dag_ = nullptr;
    Adams2019Params params_;
    std::vector<std::pair<StageMapOfScheduleFeatures, double*>> queue_;
    static constexpr int sequence_length_ = 3;
};

} // namespace Autoscheduler
} // namespace Internal
} // namespace Halide

#endif // HALIDE_AUTOSCHEDULER_SIMPLE_LSTM_MODEL_H
