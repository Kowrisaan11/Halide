#include "SimpleLSTMModel.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>
#include <cmath>

using json = nlohmann::json;

namespace Halide {

SimpleLSTMModel::SimpleLSTMModel(const std::string &model_path,
                                 const std::string &scaler_path,
                                 bool use_gpu_)
    : use_gpu(use_gpu_),
      device(use_gpu && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {
    // Load the .pt model
    model = torch::jit::load(model_path);
    model.to(device);
    model.eval();

    // Load scaler parameters
    load_scaler(scaler_path);
}

void SimpleLSTMModel::set_pipeline_features(const Internal::Autoscheduler::FunctionDAG &dag,
                                            const Internal::Autoscheduler::Adams2019Params &params) {
    // Optionally cache any pipeline-wide features/statistics if needed
    // for your feature extraction
}

void SimpleLSTMModel::enqueue(const Internal::Autoscheduler::FunctionDAG &dag,
                              const Halide::Internal::Autoscheduler::StageMapOfScheduleFeatures &schedule_feats,
                              double *cost_ptr) {
    // Extract features for this candidate (from DAG/schedule_feats)
    std::vector<double> features = extract_features(dag, schedule_feats);
    queue.push_back({features, cost_ptr});
}

void SimpleLSTMModel::evaluate_costs() {
    // For all queued candidates, run the model and fill in the costs
    for (auto &item : queue) {
        double cost = run_model(item.features);
        *(item.cost_ptr) = cost;
    }
    queue.clear();
}

void SimpleLSTMModel::reset() {
    queue.clear();
}

void SimpleLSTMModel::load_scaler(const std::string &scaler_path) {
    std::ifstream f(scaler_path);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open scaler params file: " + scaler_path);
    }
    json scaler_params;
    f >> scaler_params;
    X_scalar_center = scaler_params["X_scalar_center"].get<std::vector<double>>();
    X_scalar_scale = scaler_params["X_scalar_scale"].get<std::vector<double>>();
    y_center = scaler_params["y_center"][0].get<double>();
    y_scale = scaler_params["y_scale"][0].get<double>();
    feature_columns = scaler_params["feature_columns"].get<std::vector<std::string>>();
}

// You must implement this based on your JSON extraction logic adapted from your external code
std::vector<double> SimpleLSTMModel::extract_features(
    const Internal::Autoscheduler::FunctionDAG &dag,
    const Halide::Internal::Autoscheduler::StageMapOfScheduleFeatures &schedule_feats) {
    // TODO: Map DAG/schedule_feats to the feature vector
    // This is where you adapt your JSON-based logic to work with the in-memory Halide pipeline
    std::vector<double> features;
    // ... (implement feature extraction)
    return features;
}

double SimpleLSTMModel::run_model(const std::vector<double> &features) {
    // Prepare input tensor(s)
    const int sequence_length = 3; // As in your code
    torch::Tensor seq_input = torch::tensor(features, torch::kFloat32).repeat({sequence_length, 1}).unsqueeze(0);

    // Prepare scalar input
    std::vector<double> scalar_input;
    for (const auto &col : feature_columns) {
        // You may need to ensure mapping corresponds to your Python logic
        // (log1p, etc. See your code)
        scalar_input.push_back(/* ... */);
    }
    for (size_t i = 0; i < scalar_input.size(); ++i) {
        scalar_input[i] = (scalar_input[i] - X_scalar_center[i]) / X_scalar_scale[i];
    }
    torch::Tensor scalar_tensor = torch::tensor(scalar_input, torch::kFloat32).unsqueeze(0);

    // Run model
    std::vector<torch::jit::IValue> inputs = {seq_input.to(device), scalar_tensor.to(device)};
    torch::Tensor y_pred_scaled = model.forward(inputs).toTensor();
    torch::Tensor y_pred_transformed = y_pred_scaled * y_scale + y_center;
    torch::Tensor y_pred_actual = torch::expm1(y_pred_transformed);

    return y_pred_actual.item<double>();
}

} // namespace Halide
