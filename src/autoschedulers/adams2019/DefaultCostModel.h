#ifndef DEFAULT_COST_MODEL_H
#define DEFAULT_COST_MODEL_H

#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <map>

namespace Halide {

class CostModel {
public:
    virtual ~CostModel() = default;
    virtual void set_pipeline_features(const std::string &json_path, int n) = 0;
    virtual void enqueue(int ns, double *cost_ptr) = 0;
    virtual void evaluate_costs() = 0;
    virtual void reset() = 0;
};

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
    std::shared_ptr<torch::jit::script::Module> pytorch_model;
    ScalerParams scaler_x;
    YScalerParams scaler_y;
    Runtime::Buffer<float> costs;
    Runtime::Buffer<double *> cost_ptrs;
    std::map<std::string, float> features; // Extracted features from JSON
    int cursor = 0;
    int num_cores = 0;
    const std::string model_path;
    const std::string scaler_x_path;
    const std::string scaler_y_path;

public:
    DefaultCostModel(const std::string &model_path,
                     const std::string &scaler_x_path,
                     const std::string &scaler_y_path)
        : model_path(model_path),
          scaler_x_path(scaler_x_path),
          scaler_y_path(scaler_y_path) {
        load_model_and_scalers();
    }
    ~DefaultCostModel() override = default;

    void set_pipeline_features(const std::string &json_path, int n) override;
    void enqueue(int ns, double *cost_ptr) override;
    void evaluate_costs() override;
    void reset() override;

private:
    void load_model_and_scalers();
    nlohmann::json load_json(const std::string &file_path);
    ScalerParams load_scaler_params(const std::string &scaler_path);
    YScalerParams load_y_scaler_params(const std::string &scaler_path);
    std::map<std::string, float> extract_features(const nlohmann::json &data);
    torch::Tensor prepare_input_tensor(const std::map<std::string, float> &features,
                                      const ScalerParams &scaler_x);
    float run_prediction(const torch::Tensor &input_tensor);
    float inverse_transform_prediction(float scaled_prediction, const YScalerParams &y_scaler);
};

std::unique_ptr<DefaultCostModel> make_default_cost_model(const std::string &model_path = "",
                                                         const std::string &scaler_x_path = "",
                                                         const std::string &scaler_y_path = "");

} // namespace Halide

#endif // DEFAULT_COST_MODEL_H
