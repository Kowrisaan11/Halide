#ifndef DEFAULT_COST_MODEL_H
#define DEFAULT_COST_MODEL_H

#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include "HalideBuffer.h"
#include "CostModel.h"

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
    ScalerParams scaler_x;                                     // Input feature scaler
    YScalerParams scaler_y;                                    // Output scaler
    Runtime::Buffer<float> input_tensors;                      // Batched input tensors [batch_size, 1, feature_dim]
    Runtime::Buffer<float> costs;                              // Predicted costs [batch_size]
    Runtime::Buffer<double*> cost_ptrs;                        // Pointers to store costs [batch_size]
    std::vector<std::string> json_inputs;                      // JSON IR strings for each schedule
    int cursor = 0;                                            // Batch cursor
    int num_cores = 0;                                         // Number of CPU cores
    const std::string model_path;                              // Path to model.pt
    const std::string scaler_path;                             // Path to scaler_params.json
    const bool randomize_weights;                              // Unused for PyTorch model

public:
    DefaultCostModel(const std::string &model_path,
                     const std::string &scaler_path,
                     bool randomize_weights = false)
        : model_path(model_path),
          scaler_path(scaler_path),
          randomize_weights(randomize_weights) {
        load_weights();
    }
    ~DefaultCostModel() override = default;

    // Configure the cost model with JSON IR input
    void set_pipeline_features(const std::string &json_ir, int n);

    // Enqueue a schedule with JSON IR input
    void enqueue(const std::string &json_ir, double *cost_ptr);

    // Evaluate all schedules in the queue
    void evaluate_costs() override;

    // Discard all schedules in the queue
    void reset() override;

    // Load model and scaler parameters
    void load_weights();

    // Save model (optional)
    void save_weights();
};

std::unique_ptr<DefaultCostModel> make_default_cost_model(const std::string &model_path = "",
                                                         const std::string &scaler_path = "",
                                                         bool randomize_weights = false);

}  // namespace Halide

#endif  // DEFAULT_COST_MODEL_H
