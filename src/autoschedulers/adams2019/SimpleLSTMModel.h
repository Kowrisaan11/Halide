/*
  SimpleLSTMModel.h: Header for the SimpleLSTMModel class.
  Defines a cost model that preprocesses JSON input from TreeRepresentation
  and performs inference using a LibTorch LSTM model.
*/

#ifndef HALIDE_AUTOSCHEDULER_SIMPLE_LSTM_MODEL_H
#define HALIDE_AUTOSCHEDULER_SIMPLE_LSTM_MODEL_H

#include "CostModel.h"
#include <string>
#include <vector>
#include <map>
#include <torch/script.h>
#include <nlohmann/json.hpp>

namespace Halide {
namespace Internal {
namespace Autoscheduler {

class TreeRepresentation;

class SimpleLSTMModel : public CostModel {
public:
    // Constructor: Initializes the model with paths to the .pt model and scaler parameters
    SimpleLSTMModel(const std::string &model_path, const std::string &scaler_params_path);

    // Evaluates the cost of a schedule state using JSON input and LSTM inference
    double evaluate_cost(const IntrusivePtr<State> &state, const FunctionDAG &dag) override;

    // Extracts features from the TreeRepresentation JSON and FunctionDAG
    std::map<std::string, double> extract_features(const TreeRepresentation &tree, const FunctionDAG &dag);

    // Computes a complexity score based on extracted features
    double compute_complexity_score(const std::map<std::string, double> &features);

    // Determines the file category based on complexity
    std::string get_file_category(const std::map<std::string, double> &features);

    // Performs raw inference using the LSTM model
    double get_raw_prediction(torch::Tensor seq_input, torch::Tensor scalar_input);

    // Applies hardware and calibration corrections to the raw prediction
    double correct_prediction(double raw_prediction, double actual_time, bool is_gpu,
                             const std::string &category, const std::map<std::string, double> &features);

    // Updates calibration data based on prediction errors
    void update_calibration_data(const std::string &file_path, double raw_prediction,
                                double actual_time, const std::string &category);

private:
    torch::jit::script::Module model_; // LibTorch model
    torch::Device device_; // CPU or CUDA device
    std::vector<double> X_scalar_center_; // Scaler parameters for features
    std::vector<double> X_scalar_scale_;
    double y_center_; // Scaler parameters for output
    double y_scale_;
    std::vector<std::string> feature_columns_; // Feature names
    std::map<std::string, std::pair<double, double>> file_calibration_; // File-specific calibration
    struct CategoryCorrection {
        double scale_factor;
        double bias;
        double confidence;
        int sample_count;
    };
    std::map<std::string, CategoryCorrection> category_calibration_; // Category-specific calibration
    static const std::vector<std::string> FIXED_FEATURES; // List of expected features
    struct HardwareCorrectionFactors {
        double base_correction;
        double gpu_correction;
        double scaling_factor;
        double min_time_ms;
        double high_threshold_ms;
        double high_scaling;
    };
    static const HardwareCorrectionFactors GPU_CORRECTION_FACTORS;
    static const HardwareCorrectionFactors CPU_CORRECTION_FACTORS;
    static constexpr int sequence_length_ = 10; // Sequence length for LSTM input

    // Loads calibration data from files
    std::map<std::string, std::pair<double, double>> load_calibration_data(const std::string &filename);
    std::map<std::string, CategoryCorrection> load_category_calibration(const std::string &filename);
};

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide

#endif  // HALIDE_AUTOSCHEDULER_SIMPLE_LSTM_MODEL_H
