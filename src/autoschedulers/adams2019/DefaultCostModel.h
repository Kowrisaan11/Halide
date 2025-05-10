#ifndef DEFAULT_COST_MODEL_H
#define DEFAULT_COST_MODEL_H

#include "CostModel.h"
#include <iostream>

namespace Halide {

struct HardwareCorrectionFactors {
    double base_correction;
    double gpu_correction;
    double scaling_factor;
    double min_time_ms;
};

struct CategoryCorrection {
    double scale_factor;
    double bias;
    double confidence;
    int sample_count;
};

class DefaultCostModel : public Internal::Autoscheduler::CostModel {
private:
    torch::jit::script::Module model;
    torch::Device device;
    json scaler_params;
    std::map<std::string, CategoryCorrection> category_calibration;
    const HardwareCorrectionFactors& correction_factors;
    const std::string timestamp{"2025-05-10 18:30:44"};
    const std::string user_login{"Jathu03"};
    
    std::vector<Internal::Autoscheduler::TreeRepresentation> queued_trees;
    std::vector<double*> queued_cost_ptrs;

public:
    DefaultCostModel(const std::string &model_path,
                    const std::string &scaler_params_path,
                    bool use_gpu);
    
    void enqueue(const json &dag_data, double *cost_ptr) override;
    Internal::Autoscheduler::PredictionResult get_prediction(
        const Internal::Autoscheduler::TreeRepresentation &tree_repr,
        bool is_gpu_available) override;
    void evaluate_costs() override;
    void reset() override;

protected:
    std::map<std::string, double> extract_features(const json &json_data) override;
    std::string get_file_category(const std::map<std::string, double> &features) override;
    double compute_complexity_score(const std::map<std::string, double> &features) override;

private:
    torch::Tensor prepare_input_tensor(const std::map<std::string, double>& features);
    double get_raw_prediction(const torch::Tensor &input_tensor);
    double correct_prediction(double raw_prediction,
                            bool is_gpu,
                            const std::string &category,
                            const std::map<std::string, double> &features);
    void initialize_default_calibrations();
    void log_message(const std::string& message) {
        std::cout << "[" << timestamp << " UTC] " << message << std::endl;
    }
};

}  // namespace Halide

#endif  // DEFAULT_COST_MODEL_H
