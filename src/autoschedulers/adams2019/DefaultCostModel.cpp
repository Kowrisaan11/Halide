#include "DefaultCostModel.h"
#include "ASLog.h"
#include <algorithm>
#include <filesystem>
#include <map>
#include <cmath>

namespace Halide {

namespace {

using Halide::Internal::aslog;
using Halide::Runtime::Buffer;

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

nlohmann::json parse_json(const std::string &json_str) {
    try {
        return nlohmann::json::parse(json_str);
    } catch (const nlohmann::json::exception &e) {
        throw std::runtime_error("JSON parsing error: " + std::string(e.what()));
    }
}

std::map<std::string, float> extract_features(const nlohmann::json &data) {
    std::map<std::string, float> features;
    float execution_time = -1.0f; // Not used for inference
    features["execution_time"] = execution_time;

    auto prog_details = data.contains("programming_details") ? data["programming_details"] : nlohmann::json();
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
                        std::string op_name = "op_" + std::string(line.substr(0, pos));
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
    auto scheduling_data = data.contains("scheduling_data") ? data["scheduling_data"].get<std::vector<nlohmann::json>>()
                                                           : (prog_details.contains("Schedules") ? prog_details["Schedules"].get<std::vector<nlohmann::json>>()
                                                                                                : std::vector<nlohmann::json>());
    for (const auto &sched : scheduling_data) {
        std::map<std::string, float> sched_feature;
        if (sched.contains("Details") && sched["Details"].contains("scheduling_feature")) {
            auto sf = sched["Details"]["scheduling_feature"];
            for (const auto &[key, value] : sf.items()) {
                try {
                    sched_feature[key] = value.get<float>();
                } catch (const nlohmann::json::exception &e) {
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

    // Scheduling features (use first entry)
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

torch::Tensor prepare_input_tensor(const std::map<std::string, float> &features,
                                  const ScalerParams &scaler_x) {
    std::vector<float> feature_vector(scaler_x.feature_names.size(), 0.0f);
    std::vector<std::string> missing_features;
    std::vector<std::string> unused_features;

    for (size_t i = 0; i < scaler_x.feature_names.size(); ++i) {
        const std::string &key = scaler_x.feature_names[i];
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
            std::find(scaler_x.feature_names.begin(), scaler_x.feature_names.end(), key) ==
                scaler_x.feature_names.end()) {
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
    torch::Tensor means = torch::tensor(scaler_x.means, torch::kFloat32);
    torch::Tensor scales = torch::tensor(scaler_x.scales, torch::kFloat32);
    scales = torch::where(scales == 0, torch::ones_like(scales), scales);
    x = (x - means) / scales;

    torch::Tensor input_tensor = x.reshape({1, 1, -1});
    aslog(1) << "Input tensor shape: " << input_tensor.sizes() << "\n";
    return input_tensor;
}

ScalerParams load_scaler_x_params(const std::string &scaler_path) {
    try {
        std::ifstream file(scaler_path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open scaler file: " + scaler_path);
        }
        nlohmann::json scaler_data;
        file >> scaler_data;
        ScalerParams params;
        params.feature_names = scaler_data["scaler_x"]["feature_names"].get<std::vector<std::string>>();
        params.means = scaler_data["scaler_x"]["means"].get<std::vector<float>>();
        params.scales = scaler_data["scaler_x"]["scales"].get<std::vector<float>>();
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
        throw std::runtime_error("Failed to load scaler X params: " + std::string(e.what()));
    }
}

YScalerParams load_y_scaler_params(const std::string &scaler_path) {
    try {
        std::ifstream file(scaler_path);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open scaler file: " + scaler_path);
        }
        nlohmann::json scaler_data;
        file >> scaler_data;
        YScalerParams params;
        params.mean = scaler_data["scaler_y"]["mean"].get<float>();
        params.scale = scaler_data["scaler_y"]["scale"].get<float>();
        params.is_log_transformed = scaler_data["scaler_y"]["is_log_transformed"].get<bool>();
        aslog(1) << "Scaler Y loaded: mean=" << params.mean
                 << ", scale=" << params.scale
                 << ", is_log_transformed=" << params.is_log_transformed << "\n";
        return params;
    } catch (const std::exception &e) {
        throw std::runtime_error("Failed to load Y scaler params: " + std::string(e.what()));
    }
}

float inverse_transform_prediction(float scaled_prediction, const YScalerParams &y_scaler) {
    float unscaled = scaled_prediction * y_scaler.scale + y_scaler.mean;
    float result = y_scaler.is_log_transformed ? std::expm1(unscaled) : unscaled;
    aslog(1) << "Inverse transformed prediction: " << result << " ms\n";
    return result;
}

}  // namespace

void DefaultCostModel::set_pipeline_features(const std::string &json_ir, int n) {
    json_inputs.clear(); // Clear previous inputs
    num_cores = n;
    internal_assert(n > 0) << "Number of cores must be positive";
    aslog(1) << "Set pipeline features with JSON IR and " << n << " cores\n";
}

void DefaultCostModel::enqueue(const std::string &json_ir, double *cost_ptr) {
    internal_assert(pytorch_model) << "PyTorch model not loaded";
    internal_assert(!scaler_x.feature_names.empty()) << "Scaler X not loaded";
    internal_assert(num_cores > 0) << "Pipeline features not set";

    const int batch_size = 1024;
    const int feature_dim = scaler_x.feature_names.size();

    // Initialize buffers if needed
    if (!input_tensors.data() || input_tensors.dim(0).extent() != batch_size ||
        input_tensors.dim(2).extent() != feature_dim) {
        internal_assert(cursor == 0) << "Buffer reallocation with non-empty queue";
        input_tensors = Runtime::Buffer<float>(batch_size, 1, feature_dim);
        if (!costs.data()) {
            costs = Runtime::Buffer<float>(batch_size);
            cost_ptrs = Runtime::Buffer<double*>(batch_size);
        }
    }

    // Process JSON IR
    nlohmann::json data = parse_json(json_ir);
    auto features = extract_features(data);
    torch::Tensor input_tensor = prepare_input_tensor(features, scaler_x);

    // Copy to Halide buffer
    auto slice = input_tensors.sliced(0, cursor);
    std::memcpy(slice.data(), input_tensor.data_ptr<float>(),
                feature_dim * sizeof(float));

    // Store cost pointer and JSON input
    cost_ptrs(cursor) = cost_ptr;
    json_inputs.push_back(json_ir);

    cursor++;
    if (cursor == batch_size) {
        evaluate_costs();
    }
}

void DefaultCostModel::evaluate_costs() {
    if (cursor == 0 || !input_tensors.data()) {
        return;
    }

    internal_assert(pytorch_model) << "PyTorch model not loaded";
    internal_assert(!scaler_x.feature_names.empty()) << "Scaler X not loaded";

    // Convert Halide buffer to PyTorch tensor
    torch::Tensor input = torch::from_blob(
        input_tensors.data(),
        {cursor, 1, input_tensors.dim(2).extent()},
        torch::kFloat32
    ).to(torch::kCPU);

    // Run inference
    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> inputs = {input};
    auto output = pytorch_model->forward(inputs).toTensor();

    // Ensure output shape is [batch_size]
    internal_assert(output.dim() == 1 && output.size(0) == cursor)
        << "Unexpected output shape: " << output.sizes();

    // Copy predictions to costs buffer and denormalize
    Runtime::Buffer<float> dst = costs.cropped(0, 0, cursor);
    auto output_data = output.data_ptr<float>();
    for (int i = 0; i < cursor; i++) {
        dst(i) = inverse_transform_prediction(output_data[i], scaler_y);
    }

    // Check for NaNs and assign to cost_ptrs
    bool any_nans = false;
    for (int i = 0; i < cursor; i++) {
        internal_assert(cost_ptrs(i)) << "Invalid cost pointer at index " << i;
        *(cost_ptrs(i)) = dst(i);
        if (std::isnan(dst(i))) {
            any_nans = true;
            aslog(1) << "Prediction " << i << " is NaN\n";
        }
    }
    if (any_nans) {
        input_tensors.for_each_value([](float f) {
            if (std::isnan(f)) {
                aslog(1) << "NaN found in input tensors\n";
                abort();
            }
        });
        abort();
    }

    cursor = 0;
    json_inputs.clear();
}

void DefaultCostModel::load_weights() {
    internal_assert(ends_with(model_path, ".pt")) << "Expected .pt file: " << model_path;
    aslog(1) << "Loading PyTorch model from " << model_path << " ...\n";
    try {
        pytorch_model = torch::jit::load(model_path, torch::kCPU);
        pytorch_model->eval();
        pytorch_model->to(torch::kCPU);
    } catch (const std::exception &e) {
        throw std::runtime_error("Failed to load PyTorch model: " + std::string(e.what()));
    }

    aslog(1) << "Loading scalers from " << scaler_path << " ...\n";
    scaler_x = load_scaler_x_params(scaler_path);
    scaler_y = load_y_scaler_params(scaler_path);
}

void DefaultCostModel::save_weights() {
    internal_assert(!model_path.empty()) << "No model path specified for saving";
    if (ends_with(model_path, ".pt")) {
        try {
            torch::jit::save(*pytorch_model, model_path);
            aslog(1) << "Saved PyTorch model to " << model_path << "\n";
        } catch (const std::exception &e) {
            aslog(1) << "Failed to save PyTorch model: " << e.what() << "\n";
        }
    } else {
        aslog(1) << "WARNING: Model path must have .pt extension\n";
    }
}

void DefaultCostModel::reset() {
    cursor = 0;
    json_inputs.clear();
}

std::unique_ptr<DefaultCostModel> make_default_cost_model(const std::string &model_path,
                                                         const std::string &scaler_path,
                                                         bool randomize_weights) {
    return std::unique_ptr<DefaultCostModel>(
        new DefaultCostModel(model_path, scaler_path, randomize_weights));
}

}  // namespace Halide
