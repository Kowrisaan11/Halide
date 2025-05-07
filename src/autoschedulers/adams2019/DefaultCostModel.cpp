// Wrapper around the PyTorch cost model that loads the model and scalers,
// preprocesses JSON IR, and maintains state for batch evaluation.

#include "DefaultCostModel.h"
#include "ASLog.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

namespace Halide {
namespace {

using Halide::Internal::aslog;
using nlohmann::json;

// Helper function to convert string to lowercase
std::string to_lowercase(const std::string &input) {
    std::string result = input;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

// Helper function to load JSON
json load_json(const std::string &file_path) {
    if (!std::filesystem::exists(file_path)) {
        throw std::runtime_error("File does not exist: " + file_path);
    }
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + file_path);
    }
    try {
        json data;
        file >> data;
        return data;
    } catch (const json::exception &e) {
        throw std::runtime_error("JSON parsing error in " + file_path + ": " + e.what());
    }
}

}  // namespace

// Load scaler parameters for input features
ScalerParams DefaultCostModel::load_scaler_params(const std::string &scaler_path) {
    try {
        json scaler_data = load_json(scaler_path);
        ScalerParams params;
        params.feature_names = scaler_data["feature_names"].get<std::vector<std::string>>();
        params.means = scaler_data["means"].get<std::vector<float>>();
        params.scales = scaler_data["scales"].get<std::vector<float>>();
        if (params.feature_names.size() != params.means.size() ||
            params.means.size() != params.scales.size()) {
            throw std::runtime_error("Scaler X dimensions mismatch in " + scaler_path);
        }
        aslog(1) << "Scaler X loaded with " << params.feature_names.size() << " features:\n";
        for (size_t i = 0; i < params.feature_names.size(); ++i) {
            aslog(1) << "  " << params.feature_names[i]
                     << " (mean=" << params.means[i]
                     << ", scale=" << params.scales[i] << ")\n";
        }
        return params;
    } catch (const std::exception &e) {
        throw std::runtime_error("Failed to load scaler params from " + scaler_path + ": " + e.what());
    }
}

// Load scaler parameters for output
YScalerParams DefaultCostModel::load_scaler_params(const std::string &scaler_path) {
    try {
        json scaler_data = load_json(scaler_path);
        YScalerParams params;
        params.mean = scaler_data["mean"].get<float>();
        params.scale = scaler_data["scale"].get<float>();
        params.is_log_transformed = scaler_data["is_log_transformed"].get<bool>();
        aslog(1) << "Scaler Y loaded: mean=" << params.mean
                 << ", scale=" << params.scale
                 << ", is_log_transformed=" << params.is_log_transformed << "\n";
        return params;
    } catch (const std::exception &e) {
        throw std::runtime_error("Failed to load Y scaler params from " + scaler_path + ": " + e.what());
    }
}

// Extract features from JSON IR (adapted from main.cpp)
std::map<std::string, float> DefaultCostModel::extract_features(const json &data) {
    std::map<std::string, float> features;

    auto prog_details = data.contains("programming_details") ? data["programming_details"] : json();
    std::vector<std::map<std::string, float>> nodes_features;
    std::vector<std::map<std::string, std::string>> edges_features;

    // Extract node features
    if (prog_details.contains("Nodes")) {
        for (const auto &node : prog_details["Nodes"]) {
            std::map<std::string, float> node_feature;
            if (node.contains("Details") && node["Details"].contains("Op histogram")) {
                for (const auto &op_line : node["Details"]["Op histogram"]) {
                    std::string line = op_line.get<std::string>();
                    size_t pos = line.find(':');
                    if (pos != std::string::npos) {
                        std::string op_name = "op_" + to_lowercase(line.substr(0, pos));
                        try {
                            float count = std::stof(line.substr(pos + 1));
                            node_feature[op_name] = count;
                        } catch (const std::exception &e) {
                            aslog(1) << "Warning: Invalid op count format in '" << line << "': " << e.what() << "\n";
                        }
                    }
                }
            }
            nodes_features.push_back(node_feature);
        }
    }

    // Extract edge features
    if (prog_details.contains("Edges")) {
        for (const auto &edge : prog_details["Edges"]) {
            std::map<std::string, std::string> edge_feature;
            edge_feature["From"] = edge.contains("From") ? edge["From"].get<std::string>() : "";
            edge_feature["To"] = edge.contains("To") ? edge["To"].get<std::string>() : "";
            edge_feature["Name"] = edge.contains("Name") ? edge["Name"].get<std::string>() : "";
            edges_features.push_back(edge_feature);
        }
    }

    // Extract scheduling features
    std::vector<std::map<std::string, float>> sched_features;
    auto scheduling_data = data.contains("scheduling_data") ? data["scheduling_data"].get<std::vector<json>>()
                                                           : (prog_details.contains("Schedules") ? prog_details["Schedules"].get<std::vector<json>>()
                                                                                                 : std::vector<json>());
    for (const auto &sched : scheduling_data) {
        std::map<std::string, float> sched_feature;
        if (sched.contains("Details") && sched["Details"].contains("scheduling_feature")) {
            auto sf = sched["Details"]["scheduling_feature"];
            for (const auto &[key, value] : sf.items()) {
                try {
                    sched_feature[key] = value.get<float>();
                } catch (const json::exception &e) {
                    aslog(1) << "Warning: Could not parse scheduling feature '" << key << "': " << e.what() << "\n";
                }
            }
        }
        sched_features.push_back(sched_feature);
    }

    // Basic features
    features["nodes_count"] = static_cast<float>(nodes_features.size());
    features["edges_count"] = static_cast<float>(edges_features.size());
    features["scheduling_count"] = static_cast<float>(sched_features.size());
    features["node_edge_ratio"] = edges_features.size() > 0 ? nodes_features.size() / static_cast<float>(edges_features.size()) : 0.0f;

    // Operation counts
    std::map<std::string, float> op_counts;
    for (const auto &node : nodes_features) {
        for (const auto &[key, value] : node) {
            if (key.find("op_") == 0) {
                op_counts[key] += value;
            }
        }
    }
    features.insert(op_counts.begin(), op_counts.end());

    // Scheduling features (use first entry, as in main.cpp)
    if (!sched_features.empty()) {
        std::vector<std::string> important_metrics = {
            "bytes_at_production", "bytes_at_realization", "bytes_at_root", "bytes_at_task",
            "inner_parallelism", "outer_parallelism", "num_productions", "num_realizations",
            "num_scalars", "num_vectors", "points_computed_total", "working_set"};
        for (const auto &metric : important_metrics) {
            features["sched_" + metric] = sched_features[0].count(metric) ? sched_features[0][metric] : 0.0f;
        }

        // Aggregated scheduling features
        float total_bytes_at_production = 0.0f;
        float total_vectors = 0.0f;
        float total_parallelism = 0.0f;
        for (const auto &sf : sched_features) {
            total_bytes_at_production += sf.count("bytes_at_production") ? sf.at("bytes_at_production") : 0.0f;
            total_vectors += sf.count("num_vectors") ? sf.at("num_vectors") : 0.0f;
            total_parallelism += (sf.count("inner_parallelism") ? sf.at("inner_parallelism") : 0.0f) *
                                 (sf.count("outer_parallelism") ? sf.at("outer_parallelism") : 1.0f);
        }
        features["total_bytes_at_production"] = total_bytes_at_production;
        features["total_vectors"] = total_vectors;
        features["total_parallelism"] = total_parallelism;
        features["bytes_per_vector"] = total_vectors > 0 ? total_bytes_at_production / total_vectors : 0.0f;
        if (sched_features[0].count("working_set") && sched_features[0].count("bytes_at_production")) {
            features["memory_pressure"] = sched_features[0]["bytes_at_production"] > 0
                                             ? sched_features[0]["working_set"] / sched_features[0]["bytes_at_production"]
                                             : 0.0f;
        } else {
            features["memory_pressure"] = 0.0f;
        }
    } else {
        std::vector<std::string> important_metrics = {
            "bytes_at_production", "bytes_at_realization", "bytes_at_root", "bytes_at_task",
            "inner_parallelism", "outer_parallelism", "num_productions", "num_realizations",
            "num_scalars", "num_vectors", "points_computed_total", "working_set"};
        for (const auto &metric : important_metrics) {
            features["sched_" + metric] = 0.0f;
        }
        features["total_bytes_at_production"] = 0.0f;
        features["total_vectors"] = 0.0f;
        features["total_parallelism"] = 0.0f;
        features["bytes_per_vector"] = 0.0f;
        features["memory_pressure"] = 0.0f;
    }

    // Node operation statistics
    if (!nodes_features.empty()) {
        float total_ops = 0.0f;
        for (const auto &[_, count] : op_counts) {
            total_ops += count;
        }
        features["avg_ops_per_node"] = total_ops / nodes_features.size();
        features["op_diversity"] = static_cast<float>(op_counts.size()) / nodes_features.size();
    } else {
        features["avg_ops_per_node"] = 0.0f;
        features["op_diversity"] = 0.0f;
    }

    aslog(1) << "Extracted features:\n";
    for (const auto &[key, value] : features) {
        aslog(1) << "  " << key << ": " << value << "\n";
    }

    return features;
}

// Prepare input tensor for PyTorch model
torch::Tensor DefaultCostModel::prepare_input_tensor(const std::map<std::string, float> &features) {
    std::vector<float> feature_vector(scaler_X.feature_names.size(), 0.0f);
    std::vector<std::string> missing_features;
    std::vector<std::string> unused_features;

    // Populate feature vector
    for (size_t i = 0; i < scaler_X.feature_names.size(); ++i) {
        const std::string &key = scaler_X.feature_names[i];
        if (features.count(key)) {
            feature_vector[i] = features.at(key);
        } else {
            feature_vector[i] = 0.0f; // Default to zero for missing features
            missing_features.push_back(key);
        }
    }

    // Log unused features
    for (const auto &feature : features) {
        const std::string &key = feature.first;
        if (key != "execution_time" &&
            std::find(scaler_X.feature_names.begin(), scaler_X.feature_names.end(), key) ==
                scaler_X.feature_names.end()) {
            unused_features.push_back(key);
        }
    }

    if (!missing_features.empty()) {
        aslog(1) << "Warning: " << missing_features.size() << " features missing, set to 0:\n";
        for (const auto &key : missing_features) {
            aslog(1) << "  " << key << "\n";
        }
    }
    if (!unused_features.empty()) {
        aslog(1) << "Warning: " << unused_features.size() << " features extracted but not used:\n";
        for (const auto &key : unused_features) {
            aslog(1) << "  " << key << "\n";
        }
    }

    // Scale the feature vector
    torch::Tensor x = torch::tensor(feature_vector, torch::kFloat32);
    torch::Tensor means = torch::tensor(scaler_X.means, torch::kFloat32);
    torch::Tensor scales = torch::tensor(scaler_X.scales, torch::kFloat32);
    scales = torch::where(scales == 0, torch::ones_like(scales), scales); // Avoid division by zero
    x = (x - means) / scales;

    // Reshape to [1, 1, feature_dim] for LSTM
    torch::Tensor input_tensor = x.reshape({1, 1, -1});
    aslog(1) << "Input tensor shape: " << input_tensor.sizes() << "\n";
    return input_tensor;
}

// Inverse transform the model’s output
float DefaultCostModel::inverse_transform_prediction(float scaled_prediction) {
    float unscaled = scaled_prediction * scaler_y.scale + scaler_y.mean;
    float result = scaler_y.is_log_transformed ? std::expm1(unscaled) : unscaled;
    aslog(1) << "Inverse transformed prediction: " << result << " ms\n";
    return result;
}

// Load model and scalers
void DefaultCostModel::load_weights() {
    internal_assert(!model_path.empty()) << "Model path not specified";
    try {
        pytorch_model = torch::jit::load(model_path, torch::kCPU);
        pytorch_model->eval();
        pytorch_model->to(torch::kCPU);
        aslog(1) << "Loaded PyTorch model from " << model_path << "\n";
    } catch (const std::exception &e) {
        throw std::runtime_error("Failed to load PyTorch model: " + std::string(e.what()));
    }

    scaler_X = load_scaler_params(scaler_x_path);
    scaler_y = load_scaler_params(scaler_y_path);
}

// Save model (optional)
void DefaultCostModel::save_weights() {
    internal_assert(!weights_out_path.empty());
    if (!pytorch_model) {
        aslog(1) << "No model to save\n";
        return;
    }
    try {
        torch::jit::save(*pytorch_model, weights_out_path);
        aslog(1) << "Saved PyTorch model to " << weights_out_path << "\n";
    } catch (const std::exception &e) {
        aslog(1) << "Failed to save PyTorch model: " << e.what() << "\n";
    }
}

// Set pipeline features from JSON IR
void DefaultCostModel::set_pipeline_features(const nlohmann::json &json_ir, int num_cores) {
    this->json_ir = json_ir;
    internal_assert(num_cores > 0);
    this->num_cores = num_cores;
    aslog(1) << "Set JSON IR and num_cores: " << num_cores << "\n";
}

// Fallback for Halide IR (not used)
void DefaultCostModel::set_pipeline_features(const Internal::Autoscheduler::FunctionDAG &dag,
                                            const Internal::Autoscheduler::Adams2019Params &params) {
    internal_assert(false) << "Halide IR not supported; use JSON IR from get_representation()";
}

// Enqueue schedule features
void DefaultCostModel::enqueue(int ns, Runtime::Buffer<float> *schedule_feats, double *cost_ptr) {
    const int batch_size = 1024;
    const int feature_dim = scaler_X.feature_names.size();

    if (!feature_queue.data()) {
        internal_assert(cursor == 0);
        feature_queue = Runtime::Buffer<float>(batch_size, feature_dim);
        if (!costs.data()) {
            costs = Runtime::Buffer<float>(batch_size);
            cost_ptrs = Runtime::Buffer<double *>(batch_size);
        }
    }

    if (cursor == batch_size) {
        evaluate_costs();
    }

    *schedule_feats = feature_queue.sliced(0, cursor);
    cost_ptrs(cursor) = cost_ptr;
    cursor++;
}

// Fallback for Halide schedules (not used)
void DefaultCostModel::enqueue(const Internal::Autoscheduler::FunctionDAG &dag,
                              const Internal::Autoscheduler::StageMapOfScheduleFeatures &schedule_feats,
                              double *cost_ptr) {
    internal_assert(false) << "Halide schedules not supported; use JSON IR";
}

// Evaluate batched schedules
void DefaultCostModel::evaluate_costs() {
    if (cursor == 0 || !feature_queue.data()) {
        return;
    }

    internal_assert(pytorch_model) << "PyTorch model not loaded";
    internal_assert(!json_ir.is_null()) << "JSON IR not set";

    // Extract features from JSON IR
    auto features_map = extract_features(json_ir);

    // Prepare batched input tensor
    torch::Tensor input_tensor = torch::zeros({cursor, 1, scaler_X.feature_names.size()}, torch::kFloat32);
    for (int i = 0; i < cursor; ++i) {
        // Copy features from feature_queue to tensor
        torch::Tensor feature_tensor = prepare_input_tensor(features_map);
        input_tensor.slice(0, i, i + 1) = feature_tensor;
    }

    // Run inference
    torch::NoGradGuard no_grad;
    input_tensor = input_tensor.to(torch::kCPU);
    std::vector<torch::jit::IValue> inputs = {input_tensor};
    auto output = pytorch_model->forward(inputs).toTensor();

    // Ensure output shape is [cursor]
    if (output.dim() != 1 || output.size(0) != cursor) {
        internal_assert(false) << "Unexpected output shape: " << output.sizes();
    }

    // Inverse transform predictions
    Runtime::Buffer<float> dst = costs.cropped(0, 0, cursor);
    for (int i = 0; i < cursor; ++i) {
        float scaled_prediction = output[i].item<float>();
        dst(i) = inverse_transform_prediction(scaled_prediction);
    }

    // Check for NaNs and assign to cost_ptrs
    bool any_nans = false;
    for (int i = 0; i < cursor; ++i) {
        internal_assert(cost_ptrs(i));
        *(cost_ptrs(i)) = dst(i);
        if (std::isnan(dst(i))) {
            any_nans = true;
            aslog(1) << "Prediction " << i << " is NaN\n";
        }
    }
    if (any_nans) {
        feature_queue.for_each_value([](float f) { if (std::isnan(f)) abort(); });
        abort();
    }

    cursor = 0;
}

// Reset batch queue
void DefaultCostModel::reset() {
    cursor = 0;
}

// Factory function
std::unique_ptr<DefaultCostModel> make_default_cost_model(const std::string &model_path,
                                                         const std::string &scaler_x_path,
                                                         const std::string &scaler_y_path,
                                                         const std::string &weights_out_path) {
    return std::unique_ptr<DefaultCostModel>(
        new DefaultCostModel(model_path, scaler_x_path, scaler_y_path, weights_out_path));
}

}  // namespace Halide
