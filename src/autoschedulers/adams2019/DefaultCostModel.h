#ifndef DEFAULT_COST_MODEL_H
#define DEFAULT_COST_MODEL_H

#include "CostModel.h"
#include <torch/script.h>
#include <filesystem>

namespace fs = std::filesystem;

namespace Halide {

// Hardware-specific correction factors structure
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

class DefaultCostModel : public CostModel {
private:
    torch::jit::script::Module model;
    torch::Device device;
    json scaler_params;
    std::map<std::string, std::pair<double, double>> file_calibration;
    std::map<std::string, CategoryCorrection> category_calibration;
    const HardwareCorrectionFactors& correction_factors;
    
    // Queue for batch processing
    std::vector<TreeRepresentation> queued_trees;
    std::vector<double*> queued_cost_ptrs;

public:
    DefaultCostModel(const std::string &model_path,
                    const std::string &scaler_params_path,
                    const std::string &calibration_path,
                    bool use_gpu);
                    
    void set_pipeline_features(const Internal::Autoscheduler::FunctionDAG &dag,
                             const Internal::Autoscheduler::Adams2019Params &params) override;
                             
    TreeRepresentation convert_to_tree(const FunctionDAG &dag,
                                     const Adams2019Params &params) override;
                                     
    void enqueue(const Internal::Autoscheduler::FunctionDAG &dag,
                const StageMapOfScheduleFeatures &schedule_feats,
                double *cost_ptr) override;
                
    PredictionResult get_prediction(const TreeRepresentation &tree_repr,
                                  bool is_gpu_available) override;
                                  
    void evaluate_costs() override;
    void reset() override;

protected:
    std::map<std::string, double> extract_features(const json &json_data) override;
    std::string get_file_category(const std::string &file_path, 
                                const std::map<std::string, double> &features) override;
    double compute_complexity_score(const std::map<std::string, double> &features) override;

private:
    // Helper methods
    double get_raw_prediction(const torch::Tensor &seq_input, 
                            const torch::Tensor &scalar_input);
    double correct_prediction(double raw_prediction,
                            double actual_time,
                            bool is_gpu,
                            const std::string &category,
                            const std::map<std::string, double> &features);
    void update_calibration(const std::string &category,
                          double raw_prediction,
                          double actual_time);
};

}  // namespace Halide

#endif  // DEFAULT_COST_MODEL_H
