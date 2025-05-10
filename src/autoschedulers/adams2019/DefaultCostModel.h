#ifndef DEFAULT_COST_MODEL_H
#define DEFAULT_COST_MODEL_H

#include "CostModel.h"
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

namespace Halide {

struct HardwareCorrectionFactors {
    double base_correction;
    double gpu_correction;
    double scaling_factor;
    double min_time_ms;
    double high_threshold_ms;
    double high_scaling;
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
    
    std::vector<Internal::Autoscheduler::TreeRepresentation> queued_trees;
    std::vector<double*> queued_cost_ptrs;
    std::string user_login;
    std::chrono::system_clock::time_point session_start;

public:
    DefaultCostModel(const std::string &model_path,
                    const std::string &scaler_params_path,
                    bool use_gpu);
    
    void set_pipeline_features(const Internal::Autoscheduler::FunctionDAG &dag,
                             const Internal::Autoscheduler::Adams2019Params &params) override;
                             
    Internal::Autoscheduler::TreeRepresentation convert_to_tree(
        const Internal::Autoscheduler::FunctionDAG &dag,
        const Internal::Autoscheduler::Adams2019Params &params) override;
                                     
    void enqueue(const Internal::Autoscheduler::FunctionDAG &dag,
                const Internal::Autoscheduler::StageMapOfScheduleFeatures &schedule_feats,
                double *cost_ptr) override;
                
    Internal::Autoscheduler::PredictionResult get_prediction(
        const Internal::Autoscheduler::TreeRepresentation &tree_repr,
        bool is_gpu_available) override;
                                  
    void evaluate_costs() override;
    void reset() override;

protected:
    std::map<std::string, double> extract_features(const json &json_data) override;
    std::string get_file_category(const std::string &file_path, 
                                const std::map<std::string, double> &features) override;
    double compute_complexity_score(const std::map<std::string, double> &features) override;

private:
    torch::Tensor prepare_input_tensor(const std::map<std::string, double>& features);
    double get_raw_prediction(const torch::Tensor &input_tensor);
    double correct_prediction(double raw_prediction,
                            bool is_gpu,
                            const std::string &category,
                            const std::map<std::string, double> &features);
    void initialize_default_calibrations();
    void log_prediction_info(const std::string& category, 
                           double raw_prediction, 
                           double corrected_prediction);
};

}  // namespace Halide

#endif  // DEFAULT_COST_MODEL_H
