#ifndef DEFAULT_COST_MODEL_H
#define DEFAULT_COST_MODEL_H

#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include "HalideBuffer.h"

namespace Halide {

namespace Internal {
namespace Autoscheduler {
struct Adams2019Params;
struct FunctionDAG;
struct StageMapOfScheduleFeatures;
}  // namespace Autoscheduler
}  // namespace Internal

// Scaler parameters for input features
struct ScalerParams {
    std::vector<std::string> feature_names;
    std::vector<float> means;
    std::vector<float> scales;
};

// Scaler parameters for output (execution time)
struct YScalerParams {
    float mean;
    float scale;
    bool is_log_transformed;
};

class DefaultCostModel {
private:
    std::shared_ptr<torch::jit::script::Module> pytorch_model;  // PyTorch model
    ScalerParams scaler_X;                                     // Input feature scaler
    YScalerParams scaler_y;                                    // Output scaler
    nlohmann::json json_ir;                                    // JSON IR from get_representation()
    Runtime::Buffer<float> feature_queue;                      // Batched input features
    Runtime::Buffer<float> costs;                              // Predicted costs
    Runtime::Buffer<double*> cost_ptrs;                        // Pointers to store costs
    int cursor = 0;                                            // Batch cursor
    int num_cores = 0;                                         // Number of CPU cores
    const std::string model_path;                              // Path to model.pt
    const std::string scaler_params_path;                      // Path to scaler_params.json
    const std::string weights_out_path;                        // Optional output path

public:
    DefaultCostModel(const std::string &model_path,
                     const std::string &scaler_params_path,
                     const std::string &weights_out_path)
        : model_path(model_path),
          scaler_params_path(scaler_params_path),
          weights_out_path(weights_out_path) {
        load_weights();
    }
    ~DefaultCostModel() override = default;

    // Configure the cost model with JSON IR from get_representation()
    void set_pipeline_features(const nlohmann::json &json_ir, int num_cores);
    void set_pipeline_features(const Internal::Autoscheduler::FunctionDAG &dag,
                              const Internal::Autoscheduler::Adams2019Params &params);

    // Enqueue a schedule for evaluation
    void enqueue(int ns, Runtime::Buffer<float> *schedule_feats, double *cost_ptr);
    void enqueue(const Internal::Autoscheduler::FunctionDAG &dag,
                 const Internal::Autoscheduler::StageMapOfScheduleFeatures &schedule_feats,
                 double *cost_ptr);

    // Evaluate all schedules in the queue
    void evaluate_costs();

    // Discard all schedules in the queue
    void reset();

    // Save/Load the model and scalers
    void save_weights();
    void load_weights();

private:
    // Helper methods for preprocessing
    std::map<std::string, float> extract_features(const nlohmann::json &data);
    torch::Tensor prepare_input_tensor(const std::map<std::string, float> &features);
    float inverse_transform_prediction(float scaled_prediction);
    ScalerParams load_scaler_x_params(const nlohmann::json &scaler_data);
    YScalerParams load_scaler_y_params(const nlohmann::json &scaler_data);
};

std::unique_ptr<DefaultCostModel> make_default_cost_model(
    const std::string &model_path = "/home/kowrisaan/fyp/Halide/model.pt",
    const std::string &scaler_params_path = "/home/kowrisaan/fyp/Halide/scaler_params.json",
    const std::string &weights_out_path = "");

}  // namespace Halide

#endif  // DEFAULT_COST_MODEL_H
