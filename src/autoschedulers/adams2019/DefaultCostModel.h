#ifndef DEFAULT_COST_MODEL_H
#define DEFAULT_COST_MODEL_H

#include "CostModel.h"
#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace Halide {

namespace Internal {
namespace Autoscheduler {
struct Adams2019Params;
}  // namespace Autoscheduler
}  // namespace Internal

struct ScalerParams {
    std::vector<std::string> feature_names;
    std::vector<float> means;
    std::vector<float> scales;
};

struct YScalerParams {
    float mean;
    float scale;
    bool is_log_transformed;
};

class DefaultCostModel : public CostModel {
private:
    std::shared_ptr<torch::jit::script::Module> pytorch_model;  // PyTorch model
    ScalerParams scaler_X;                                      // Feature scaler
    YScalerParams scaler_y;                                     // Output scaler
    Runtime::Buffer<float> input_tensors;                       // Batched input tensors
    Runtime::Buffer<float> costs;                               // Predicted costs
    Runtime::Buffer<double *> cost_ptrs;                        // Pointers to store costs
    int cursor = 0;                                             // Batch cursor
    int num_cores = 0;                                          // Number of CPU cores
    const std::string weights_in_path;                          // Path to model.pt and scaler_params.json
    const std::string weights_out_path;                         // Path to save model (optional)
    const bool randomize_weights;                               // Unused for PyTorch
    nlohmann::json json_data;                                   // JSON IR data

public:
    DefaultCostModel(const std::string &weights_in_path,
                     const std::string &weights_out_path,
                     bool randomize_weights);
    ~DefaultCostModel() override = default;

    // Configure the cost model with JSON IR
    void set_pipeline_features(const Internal::Autoscheduler::FunctionDAG &dag,
                              const Internal::Autoscheduler::Adams2019Params &params) override;
    void set_pipeline_features(const nlohmann::json &json_ir, int n);

    // Enqueue a schedule for evaluation
    void enqueue(const Internal::Autoscheduler::FunctionDAG &dag,
                 const Halide::Internal::Autoscheduler::StageMapOfScheduleFeatures &schedule_feats,
                 double *cost_ptr) override;
    void enqueue(int batch_idx, Runtime::Buffer<float> *input_tensor, double *cost_ptr);

    // Evaluate all schedules in the queue
    void evaluate_costs() override;

    // Discard all schedules in the queue
    void reset() override;

    // Save/Load the model and scalers
    void save_weights();
    void load_weights();
};

std::unique_ptr<DefaultCostModel> make_default_cost_model(const std::string &weights_in_dir = "",
                                                         const std::string &weights_out_dir = "",
                                                         bool randomize_weights = false);
}  // namespace Halide

#endif  // DEFAULT_COST_MODEL_H
