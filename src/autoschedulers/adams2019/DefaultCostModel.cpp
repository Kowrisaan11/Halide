#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>
#include <sstream>
#include <vector>
#include "ASLog.h"
#include "DefaultCostModel.h"
#include "HalideBuffer.h"

namespace Halide {

namespace {

using Halide::Internal::aslog;
using nlohmann::json;

bool ends_with(const std::string &str, const std::string &suffix) {
    if (str.size() < suffix.size()) {
        return false;
    }
    size_t off = str.size() - suffix.size();
    for (size_t i = 0; i < suffix.size(); i++) {
        if (str[off + i] != suffix[i]) {
            return false;
        }
    }
    return true;
}

std::string to_lowercase(const std::string &input) {
    std::string result = input;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

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

std::map<std::string, float> extract_features(const json &data) {
    std::map<std::string, float> features;

    auto prog_details = data.contains("programming_details") ? data["programming_details"] : json();
    std::vector<std::map<std::string, float>> nodes_features;
    std::vector<std::map<std::string, std::string>> edges_features;

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
                            std::cerr << "Warning: Invalid op count format in '" << line << "': " << e.what() << "\n";
                        }
                    }
                }
            }
            nodes_features.push_back(node_feature);
        }
    }

    if (prog_details.contains("Edges")) {
        for (const auto &edge : prog_details["Edges"]) {
            std::map<std::string, std::string> edge_feature;
            edge_feature["From"] = edge.contains("From") ? edge["From"].get<std::string>() : "";
            edge_feature["To"] = edge.contains("To") ? edge["To"].get<std::string>() : "";
            edge_feature["Name"] = edge.contains("Name") ? edge["Name"].get<std::string>() : "";
            edges_features.push_back(edge_feature);
        }
    }

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
                    std::cerr << "Warning: Could not parse scheduling feature '" << key << "': " << e.what() << "\n";
                }
            }
        }
        sched_features.push_back(sched_feature);
    }

    features["nodes_count"] = static_cast<float>(nodes_features.size());
    features["edges_count"] = static_cast<float>(edges_features.size());
    features["scheduling_count"] = static_cast<float>(sched_features.size());
    features["node_edge_ratio"] = edges_features.size() > 0 ? nodes_features.size() / static_cast<float>(edges_features.size()) : 0.0f;

    std::map<std::string, float> op_counts;
    for (const auto &node : nodes_features) {
        for (const auto &[key, value] : node) {
            if (key.find("op_") == 0) {
                op_counts[key] += value;
            }
        }
    }
    features.insert(op_counts.begin(), op_counts.end());

    if (!sched_features.empty()) {
        std::vector<std::string> important_metrics = {
            "bytes_at_production", "bytes_at_realization", "bytes_at_root", "bytes_at_task",
            "inner_parallelism", "outer_parallelism", "num_productions", "num_realizations",
            "num_scalars", "num_vectors", "points_computed_total", "working_set"};
        for (const auto &metric : important_metrics) {
            features["sched_" + metric] = sched_features[0].count(metric) ? sched_features[0][metric] : 0.0f;
        }

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

torch::Tensor prepare_input_tensor(const std::map<std::string, float> &features,
                                  const ScalerParams &scaler_X) {
    std::vector<float> feature_vector(scaler_X.feature_names.size(), 0.0f);
    std::vector<std::string> missing_features;
    std::vector<std::string> unused_features;

    for (size_t i = 0; i < scaler_X.feature_names.size(); ++i) {
        const std::string &key = scaler_X.feature_names[i];
        if (features.count(key)) {
            feature_vector[i] = features.at(key);
        } else {
            feature_vector[i] = 0.0f;
            missing_features.push_back(key);
        }
    }

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

    torch::Tensor x = torch::tensor(feature_vector, torch::kFloat32);
    torch::Tensor means = torch::tensor(scaler_X.means, torch::kFloat32);
    torch::Tensor scales = torch::tensor(scaler_X.scales, torch::kFloat32);
    scales = torch::where(scales == 0, torch::ones_like(scales), scales);
    x = (x - means) / scales;

    torch::Tensor input_tensor = x.reshape({1, 1, -1});
    aslog(1) << "Input tensor shape: " << input_tensor.sizes() << "\n";
    return input_tensor;
}

float run_prediction(const torch::Tensor &input_tensor, const torch::jit::script::Module &model) {
    try {
        torch::NoGradGuard no_grad;
        torch::Tensor input = input_tensor.to(torch::kCPU);
        std::vector<torch::jit::IValue> inputs = {input};
        auto output = model.forward(inputs).toTensor();
        float prediction = output.item<float>();
        aslog(1) << "Model output: " << prediction << "\n";
        return prediction;
    } catch (const std::exception &e) {
        throw std::runtime_error("Model inference failed: " + std::string(e.what()));
    }
}

float inverse_transform_prediction(float scaled_prediction, const YScalerParams &y_scaler) {
    float unscaled = scaled_prediction * y_scaler.scale + y_scaler.mean;
    float result = y_scaler.is_log_transformed ? std::expm1(unscaled) : unscaled;
    aslog(1) << "Inverse transformed prediction: " << result << " ms\n";
    return result;
}

}  // namespace

DefaultCostModel::DefaultCostModel(const std::string &weights_in_path,
                                   const std::string &weights_out_path,
                                   bool randomize_weights)
    : weights_in_path(weights_in_path),
      weights_out_path(weights_out_path),
      randomize_weights(randomize_weights) {
    load_weights();
}

void DefaultCostModel::set_pipeline_features(const Internal::Autoscheduler::FunctionDAG &dag,
                                            const Internal::Autoscheduler::Adams2019Params &params) {
    // Load JSON IR from file (assumes get_representation() has been called)
    std::string json_path = "/home/kowrisaan/fyp/Halide/src/autoschedulers/adams2019/tree_representation.json";
    json_data = load_json(json_path);
    num_cores = params.parallelism;
    internal_assert(num_cores > 0);
}

void DefaultCostModel::set_pipeline_features(const nlohmann::json &json_ir, int n) {
    json_data = json_ir;
    num_cores = n;
    internal_assert(num_cores > 0);
}

void DefaultCostModel::enqueue(const Internal::Autoscheduler::FunctionDAG &dag,
                              const Halide::Internal::Autoscheduler::StageMapOfScheduleFeatures &schedule_feats,
                              double *cost_ptr) {
    // Use the stored JSON IR
    std::map<std::string, float> features = extract_features(json_data);
    torch::Tensor input_tensor = prepare_input_tensor(features, scaler_X);

    const int batch_size = 1024;
    if (!input_tensors.data()) {
        input_tensors = Runtime::Buffer<float>(batch_size, scaler_X.feature_names.size());
        costs = Runtime::Buffer<float>(batch_size);
        cost_ptrs = Runtime::Buffer<double *>(batch_size);
    }

    if (cursor == batch_size) {
        evaluate_costs();
    }

    Runtime::Buffer<float> tensor_slice = input_tensors.sliced(0, cursor);
    std::memcpy(tensor_slice.data(), input_tensor.data_ptr<float>(),
                scaler_X.feature_names.size() * sizeof(float));
    cost_ptrs(cursor) = cost_ptr;
    cursor++;
}

void DefaultCostModel::enqueue(int batch_idx, Runtime::Buffer<float> *input_tensor, double *cost_ptr) {
    internal_assert(batch_idx >= 0 && batch_idx < 1024);
    const int batch_size = 1024;
    if (!input_tensors.data()) {
        input_tensors = Runtime::Buffer<float>(batch_size, scaler_X.feature_names.size());
        costs = Runtime::Buffer<float>(batch_size);
        cost_ptrs = Runtime::Buffer<double *>(batch_size);
    }

    if (cursor == batch_size) {
        evaluate_costs();
    }

    *input_tensor = input_tensors.sliced(0, cursor);
    cost_ptrs(cursor) = cost_ptr;
    cursor++;
}

void DefaultCostModel::evaluate_costs() {
    if (cursor == 0 || !input_tensors.data()) {
        return;
    }

    internal_assert(pytorch_model);
    Runtime::Buffer<float> dst = costs.cropped(0, 0, cursor);

    for (int i = 0; i < cursor; ++i) {
        Runtime::Buffer<float> tensor_slice = input_tensors.sliced(0, i);
        torch::Tensor input = torch::from_blob(tensor_slice.data(),
                                              {1, 1, scaler_X.feature_names.size()},
                                              torch::kFloat32);
        float scaled_prediction = run_prediction(input, *pytorch_model);
        dst(i) = inverse_transform_prediction(scaled_prediction, scaler_y);
    }

    bool any_nans = false;
    for (int i = 0; i < cursor; i++) {
        internal_assert(cost_ptrs(i));
        *(cost_ptrs(i)) = dst(i);
        if (std::isnan(dst(i))) {
            any_nans = true;
            aslog(1) << "Prediction " << i << " is NaN\n";
        }
    }
    if (any_nans) {
        input_tensors.for_each_value([](float f) { if (std::isnan(f)) abort(); });
        abort();
    }

    cursor = 0;
}

void DefaultCostModel::load_weights() {
    // Load PyTorch model
    std::string model_path = "/home/kowrisaan/fyp/Halide/src/autoschedulers/adams2019/model.pt";
    aslog(1) << "Loading PyTorch model from " << model_path << " ...\n";
    try {
        pytorch_model = std::make_shared<torch::jit::script::Module>(
            torch::jit::load(model_path, torch::kCPU));
        pytorch_model->eval();
    } catch (const std::exception &e) {
        std::cerr << "Failed to load PyTorch model: " << e.what() << "\n";
        abort();
    }

    // Load scaler parameters
    std::string scaler_path = "/home/kowrisaan/fyp/Halide/src/autoschedulers/adams2019/scaler_params.json";
    aslog(1) << "Loading scaler params from " << scaler_path << " ...\n";
    try {
        json scaler_data = load_json(scaler_path);
        scaler_X.feature_names = scaler_data["feature_names"].get<std::vector<std::string>>();
        scaler_X.means = scaler_data["means"].get<std::vector<float>>();
        scaler_X.scales = scaler_data["scales"].get<std::vector<float>>();
        scaler_y.mean = scaler_data["y_mean"].get<float>();
        scaler_y.scale = scaler_data["y_scale"].get<float>();
        scaler_y.is_log_transformed = scaler_data["is_log_transformed"].get<bool>();

        if (scaler_X.feature_names.size() != scaler_X.means.size() ||
            scaler_X.means.size() != scaler_X.scales.size()) {
            throw std::runtime_error("Scaler X dimensions mismatch in " + scaler_path);
        }
        aslog(1) << "Scaler X loaded with " << scaler_X.feature_names.size() << " features\n";
        aslog(1) << "Scaler Y loaded: mean=" << scaler_y.mean
                 << ", scale=" << scaler_y.scale
                 << ", is_log_transformed=" << scaler_y.is_log_transformed << "\n";
    } catch (const std::exception &e) {
        std::cerr << "Failed to load scaler params: " << e.what() << "\n";
        abort();
    }
}

void DefaultCostModel::save_weights() {
    if (weights_out_path.empty() || !pytorch_model) {
        return;
    }
    if (ends_with(weights_out_path, ".pt")) {
        torch::jit::save(*pytorch_model, weights_out_path);
        aslog(1) << "Saved PyTorch model to " << weights_out_path << "\n";
    } else {
        std::cerr << "WARNING: Saving PyTorch model requires .pt extension\n";
    }
}

void DefaultCostModel::reset() {
    cursor = 0;
}

std::unique_ptr<DefaultCostModel> make_default_cost_model(const std::string &weights_in_path,
                                                         const std::string &weights_out_path,
                                                         bool randomize_weights) {
    return std::unique_ptr<DefaultCostModel>(
        new DefaultCostModel(weights_in_path, weights_out_path, randomize_weights));
}

}  // namespace Halide
