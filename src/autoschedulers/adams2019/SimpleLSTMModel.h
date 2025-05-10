#ifndef SIMPLE_LSTM_MODEL_H
#define SIMPLE_LSTM_MODEL_H

#include "CostModel.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <vector>
#include <string>
#include <map>
#include <mutex>
#include <memory>

namespace Halide {
namespace Internal {
namespace Autoscheduler {

// Structure to hold model parameters and scaling information
struct ModelParams {
    std::vector<double> X_scalar_center;
    std::vector<double> X_scalar_scale;
    double y_center;
    double y_scale;
    std::vector<std::string> feature_columns;
};

// Hardware correction factors for different platforms
struct HardwareCorrectionFactors {
    double base_correction;
    double gpu_correction;
    double scaling_factor;
    double min_time_ms;
    double high_threshold_ms;
    double high_scaling;
};

class SimpleLSTMModel : public Halide::CostModel {
public:
    SimpleLSTMModel(const std::string& model_path, const std::string& params_path);
    ~SimpleLSTMModel() override;

    // CostModel interface implementation
    void set_pipeline_features(const Internal::Autoscheduler::FunctionDAG &dag,
                              const Internal::Autoscheduler::Adams2019Params &params) override;

    void enqueue(const Internal::Autoscheduler::FunctionDAG &dag,
                const Internal::Autoscheduler::StageMapOfScheduleFeatures &schedule_feats,
                double *cost_ptr) override;

    void evaluate_costs() override;

    void reset() override;

private:
    // Model and device
    torch::jit::script::Module model_;
    torch::Device device_;
    bool is_gpu_available_;
    
    // Parameters
    ModelParams model_params_;
    HardwareCorrectionFactors correction_factors_;
    
    // Queue for batch processing
    std::mutex mtx_;
    std::vector<std::pair<std::map<std::string, double>, double*>> queue_;
    
    // Feature extraction and preprocessing
    std::map<std::string, double> extract_features_from_schedule(
        const Internal::Autoscheduler::FunctionDAG &dag,
        const Internal::Autoscheduler::StageMapOfScheduleFeatures &schedule_feats);
    
    // Model inference
    double get_raw_prediction(const std::map<std::string, double>& features);
    
    // Prediction correction
    double correct_prediction(double raw_prediction);
    
    // Load model parameters
    void load_model_params(const std::string& params_path);
};

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide

#endif  // SIMPLE_LSTM_MODEL_H
