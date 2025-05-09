/*
  SimpleLSTMModel.h: Header for SimpleLSTMModel, replacing DefaultCostModel.
  Integrates LibTorch-based LSTM model for cost prediction with feature extraction
  and calibration logic.
*/

#ifndef SIMPLE_LSTM_MODEL_H
#define SIMPLE_LSTM_MODEL_H

#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <map>
#include <unordered_set>
#include "Halide.h"
#include "FunctionDAG.h"
#include "TreeRepresentation.h"

using json = nlohmann::json;

namespace Halide {
namespace Internal {
namespace Autoscheduler {

class SimpleLSTMModel {
public:
    // Hardware correction factors
    struct HardwareCorrectionFactors {
        double base_correction;
        double gpu_correction;
        double scaling_factor;
        double min_time_ms;
        double high_threshold_ms;
        double high_scaling;
    };

    // Category correction
    struct CategoryCorrection {
        double scale_factor;
        double bias;
        double confidence;
        int sample_count;
    };

    // Constructor: Load model and scaler parameters
    SimpleLSTMModel(const std::string &model_path, const std::string &scaler_params_path);

    // Evaluate cost for a state
    double evaluate_cost(const IntrusivePtr<State> &state, const FunctionDAG &dag);

    // Extract features from TreeRepresentation
    std::map<std::string, double> extract_features(const TreeRepresentation &tree, const FunctionDAG &dag);

    // Correct prediction using calibration
    double correct_prediction(double raw_prediction, double actual_time, bool is_gpu,
                             const std::string &category, const std::map<std::string, double> &features);

private:
    torch::jit::script::Module model_;
    torch::Device device_;
    std::vector<std::string> feature_columns_;
    std::vector<double> X_scalar_center_;
    std::vector<double> X_scalar_scale_;
    double y_center_;
    double y_scale_;
    std::map<std::string, std::pair<double, double>> file_calibration_;
    std::map<std::string, CategoryCorrection> category_calibration_;
    const int sequence_length_ = 3;

    // Static constants
    static const std::vector<std::string> FIXED_FEATURES;
    static const HardwareCorrectionFactors GPU_CORRECTION_FACTORS;
    static const HardwareCorrectionFactors CPU_CORRECTION_FACTORS;

    // Helper functions
    double get_raw_prediction(torch::Tensor seq_input, torch::Tensor scalar_input);
    double compute_complexity_score(const std::map<std::string, double> &features);
    std::string get_file_category(const std::map<std::string, double> &features);
    void update_calibration_data(const std::string &file_path, double raw_prediction, double actual_time,
                                const std::string &category);
};

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide

#endif // SIMPLE_LSTM_MODEL_H
