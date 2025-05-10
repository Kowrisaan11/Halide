/*
  SimpleLSTMModel.cpp: Implementation of SimpleLSTMModel.
  Extracts features from TreeRepresentation JSON, performs LibTorch inference,
  and applies calibration corrections.
*/

#include "SimpleLSTMModel.h"
#include "TreeRepresentation.h"
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

namespace Halide {
namespace Internal {
namespace Autoscheduler {

namespace fs = std::filesystem;

const std::vector<std::string> SimpleLSTMModel::FIXED_FEATURES = {
    "cache_hits", "cache_misses", "execution_time_ms", "sched_num_realizations",
    "sched_num_productions", "sched_points_computed_total", "sched_innermost_loop_extent",
    "sched_inner_parallelism", "sched_outer_parallelism", "sched_bytes_at_realization",
    "sched_bytes_at_production", "sched_bytes_at_root", "sched_unique_bytes_read_per_realization",
    "sched_working_set", "sched_vector_size", "sched_num_vectors", "sched_num_scalars",
    "sched_bytes_at_task", "sched_working_set_at_task", "sched_working_set_at_production",
    "sched_working_set_at_realization", "sched_working_set_at_root", "total_parallelism",
    "scheduling_count", "total_bytes_at_production", "total_vectors", "computation_efficiency",
    "memory_pressure", "memory_utilization_ratio", "bytes_processing_rate", "bytes_per_parallelism",
    "bytes_per_vector", "nodes_count", "edges_count", "node_edge_ratio", "nodes_per_schedule",
    "op_diversity",
    "op_add", "op_sub", "op_mul", "op_div", "op_mod", "op_eq", "op_ne", "op_lt", "op_le",
    "op_or", "op_and", "op_not", "op_min", "op_max", "op_constant", "op_variable",
    "op_funccall", "op_imagecall", "op_externcall", "op_let", "op_param",
    "memory_transpose_0", "memory_transpose_1", "memory_transpose_2", "memory_transpose_3",
    "memory_slice_0", "memory_slice_1", "memory_slice_2", "memory_slice_3",
    "memory_broadcast_0", "memory_broadcast_1", "memory_broadcast_2", "memory_broadcast_3",
    "memory_pointwise_0", "memory_pointwise_1", "memory_pointwise_2", "memory_pointwise_3"
};

const SimpleLSTMModel::HardwareCorrectionFactors SimpleLSTMModel::GPU_CORRECTION_FACTORS = {
    0.28, 0.9, 0.95, 100.0, 500.0, 0.92
};

const SimpleLSTMModel::HardwareCorrectionFactors SimpleLSTMModel::CPU_CORRECTION_FACTORS = {
    0.35, 1.0, 0.97, 50.0, 300.0, 0.94
};

SimpleLSTMModel::SimpleLSTMModel(const std::string &model_path, const std::string &scaler_params_path) {
    // Check if CUDA is available
    bool is_gpu_available = torch::cuda::is_available();
    device_ = is_gpu_available ? torch::Device(torch::kCUDA, 0) : torch::kCPU;

    // Load scaler parameters
    std::ifstream scaler_file(scaler_params_path);
    if (!scaler_file.is_open()) {
        internal_error << "Failed to open " << scaler_params_path;
    }
    nlohmann::json scaler_params;
    scaler_file >> scaler_params;
    X_scalar_center_ = scaler_params["X_scalar_center"].get<std::vector<double>>();
    X_scalar_scale_ = scaler_params["X_scalar_scale"].get<std::vector<double>>();
    y_center_ = scaler_params["y_center"][0].get<double>();
    y_scale_ = scaler_params["y_scale"][0].get<double>();
    feature_columns_ = scaler_params["feature_columns"].get<std::vector<std::string>>();

    // Load model
    try {
        model_ = torch::jit::load(model_path);
        model_.to(device_);
        model_.eval();
    } catch (const c10::Error &e) {
        internal_error << "Error loading model: " << e.what();
    }

    // Load calibration data
    file_calibration_ = load_calibration_data("calibration_data.txt");
    category_calibration_ = load_category_calibration("category_calibration.txt");
}

std::map<std::string, double> SimpleLSTMModel::extract_features(const TreeRepresentation &tree, const FunctionDAG &dag) {
    // Convert TreeRepresentation to JSON
    nlohmann::json json_data = tree.to_json();
    std::map<std::string, double> features;

    // Extract global features
    auto global_node = std::find_if(json_data["children"].begin(), json_data["children"].end(),
        [](const nlohmann::json &child) { return child["name"] == "Global Features"; });
    if (global_node != json_data["children"].end()) {
        features["cache_hits"] = global_node->value("cache_hits", 0.0);
        features["cache_misses"] = global_node->value("cache_misses", 0.0);
        features["execution_time_ms"] = global_node->value("execution_time_ms", 0.0);
    }

    // Extract op_histogram
    std::map<std::string, int> op_histogram;
    for (const auto &node : json_data["children"]) {
        if (node.contains("op_histogram")) {
            for (const auto &[op, count] : node["op_histogram"].items()) {
                std::string op_lower = op;
                std::transform(op_lower.begin(), op_lower.end(), op_lower.begin(), ::tolower);
                op_histogram[op_lower] += count.get<int>();
            }
        }
    }
    for (const auto &[op, count] : op_histogram) {
        features["op_" + op] = static_cast<double>(count);
    }

    // Extract memory patterns
    std::map<std::string, std::vector<double>> memory_patterns;
    for (const auto &node : json_data["children"]) {
        if (node.contains("memory_patterns")) {
            for (const auto &[pattern, values] : node["memory_patterns"].items()) {
                std::string pattern_lower = pattern;
                std::transform(pattern_lower.begin(), pattern_lower.end(), pattern_lower.begin(), ::tolower);
                if (memory_patterns.find(pattern_lower) == memory_patterns.end()) {
                    memory_patterns[pattern_lower] = {0.0, 0.0, 0.0, 0.0};
                }
                auto curr_values = memory_patterns[pattern_lower];
                auto json_values = values.get<std::vector<double>>();
                for (size_t i = 0; i < json_values.size() && i < 4; ++i) {
                    curr_values[i] += json_values[i];
                }
                memory_patterns[pattern_lower] = curr_values;
            }
        }
    }
    for (const auto &[pattern, values] : memory_patterns) {
        for (size_t i = 0; i < 4; ++i) {
            features["memory_" + pattern + "_" + std::to_string(i)] = values[i];
        }
    }

    // Extract scheduling features
    std::vector<std::string> scheduling_keys = {
        "num_realizations", "num_productions", "points_computed_total", "innermost_loop_extent",
        "inner_parallelism", "outer_parallelism", "bytes_at_realization", "bytes_at_production",
        "bytes_at_root", "unique_bytes_read_per_realization", "working_set", "vector_size",
        "num_vectors", "num_scalars", "bytes_at_task", "working_set_at_task", "working_set_at_production",
        "working_set_at_realization", "working_set_at_root"
    };
    std::map<std::string, double> scheduling_sums;
    int node_count = 0;
    for (const auto &node : json_data["children"]) {
        if (node.contains("scheduling")) {
            node_count++;
            for (const auto &key : scheduling_keys) {
                scheduling_sums[key] += node["scheduling"].value(key, 0.0);
            }
        }
    }
    for (const auto &key : scheduling_keys) {
        if (key == "inner_parallelism" || key == "outer_parallelism") {
            features["sched_" + key] = node_count > 0 ? scheduling_sums[key] / node_count : 0.0;
        } else {
            features["sched_" + key] = scheduling_sums[key];
        }
    }

    // Derived features
    features["total_parallelism"] = features["sched_inner_parallelism"] + features["sched_outer_parallelism"];
    features["scheduling_count"] = features["sched_num_realizations"] + features["sched_num_productions"];
    features["total_bytes_at_production"] = features["sched_bytes_at_production"];
    features["total_vectors"] = features["sched_num_vectors"];
    double bytes_at_realization = features["sched_bytes_at_realization"];
    features["computation_efficiency"] = (bytes_at_realization > 0) ? features["sched_points_computed_total"] / bytes_at_realization : 0.0;
    double bytes_at_root = features["sched_bytes_at_root"];
    features["memory_pressure"] = (bytes_at_root > 0) ? features["sched_working_set"] / bytes_at_root : 0.0;
    double bytes_at_task = features["sched_bytes_at_task"];
    features["memory_utilization_ratio"] = (bytes_at_task > 0) ? features["sched_unique_bytes_read_per_realization"] / bytes_at_task : 0.0;
    double execution_time_ms = features["execution_time_ms"];
    features["bytes_processing_rate"] = (execution_time_ms > 0) ? features["sched_bytes_at_realization"] / execution_time_ms : 0.0;
    double total_parallelism = features["total_parallelism"];
    features["bytes_per_parallelism"] = (total_parallelism > 0) ? features["sched_bytes_at_task"] / total_parallelism : 0.0;
    double num_vectors = features["sched_num_vectors"];
    features["bytes_per_vector"] = (num_vectors > 0) ? features["sched_bytes_at_realization"] / num_vectors : 0.0;
    int nodes_count = json_data["children"].size();
    int edges_count = 0;
    for (const auto &node : json_data["children"]) {
        edges_count += node.value("children", nlohmann::json::array()).size();
    }
    features["nodes_count"] = nodes_count;
    features["edges_count"] = edges_count;
    features["node_edge_ratio"] = (edges_count > 0) ? static_cast<double>(nodes_count) / edges_count : static_cast<double>(nodes_count);
    double scheduling_count = features["scheduling_count"];
    features["nodes_per_schedule"] = (scheduling_count > 0) ? nodes_count / scheduling_count : 0.0;
    int op_diversity = 0;
    for (const auto &[key, value] : features) {
        if (key.find("op_") == 0 && value > 0) {
            op_diversity++;
        }
    }
    features["op_diversity"] = op_diversity;

    return features;
}

double SimpleLSTMModel::compute_complexity_score(const std::map<std::string, double> &features) {
    double complexity = 0.0;
    complexity += features.at("nodes_count") * 0.01;
    complexity += features.at("edges_count") * 0.005;
    complexity += features.at("sched_points_computed_total") * 0.00001;
    complexity += features.at("sched_num_vectors") * 0.01;
    complexity += features.at("sched_working_set") * 0.0001;
    complexity += features.at("sched_bytes_at_production") * 0.00005;
    complexity += features.at("op_diversity") * 0.1;
    return complexity;
}

std::string SimpleLSTMModel::get_file_category(const std::map<std::string, double> &features) {
    double complexity = compute_complexity_score(features);
    if (complexity > 100.0) return "unknown_complex";
    else if (complexity > 50.0) return "unknown_medium";
    else return "unknown_simple";
}

double SimpleLSTMModel::get_raw_prediction(torch::Tensor seq_input, torch::Tensor scalar_input) {
    seq_input = seq_input.to(device_);
    scalar_input = scalar_input.to(device_);
    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> inputs = {seq_input, scalar_input};
    torch::Tensor y_pred_scaled;
    try {
        y_pred_scaled = model_.forward(inputs).toTensor();
    } catch (const c10::Error &e) {
        if (device_.is_cuda()) {
            torch::Device cpu_device = torch::kCPU;
            model_.to(cpu_device);
            seq_input = seq_input.to(cpu_device);
            scalar_input = scalar_input.to(cpu_device);
            inputs = {seq_input, scalar_input};
            y_pred_scaled = model_.forward(inputs).toTensor();
        } else {
            internal_error << "Error during model inference: " << e.what();
        }
    }
    torch::Tensor y_pred_transformed = y_pred_scaled * y_scale_ + y_center_;
    torch::Tensor y_pred_actual = torch::expm1(y_pred_transformed);
    return y_pred_actual.item<float>();
}

double SimpleLSTMModel::correct_prediction(double raw_prediction, double actual_time, bool is_gpu,
                                           const std::string &category, const std::map<std::string, double> &features) {
    const HardwareCorrectionFactors &factors = is_gpu ? GPU_CORRECTION_FACTORS : CPU_CORRECTION_FACTORS;
    auto cat_it = category_calibration_.find(category);
    if (cat_it != category_calibration_.end() && cat_it->second.confidence > 0.6) {
        const auto &correction = cat_it->second;
        return std::max(correction.scale_factor * raw_prediction + correction.bias, 0.0);
    }
    double hw_correction = factors.base_correction * (is_gpu ? factors.gpu_correction : 1.0);
    if (category.find("unknown") != std::string::npos) {
        double complexity = compute_complexity_score(features);
        if (category == "unknown_complex") hw_correction *= 0.92;
        else if (category == "unknown_simple") hw_correction *= 1.05;
        if (complexity > 150) hw_correction *= 0.95;
        else if (complexity < 20) hw_correction *= 1.03;
    }
    double corrected;
    if (raw_prediction <= factors.min_time_ms) {
        corrected = raw_prediction * hw_correction;
    } else if (raw_prediction <= factors.high_threshold_ms) {
        double base = factors.min_time_ms * hw_correction;
        double excess = raw_prediction - factors.min_time_ms;
        corrected = base + (excess * hw_correction * factors.scaling_factor);
    } else {
        double base = factors.min_time_ms * hw_correction;
        double mid_excess = factors.high_threshold_ms - factors.min_time_ms;
        double high_excess = raw_prediction - factors.high_threshold_ms;
        corrected = base + (mid_excess * hw_correction * factors.scaling_factor) +
                    (high_excess * hw_correction * factors.scaling_factor * factors.high_scaling);
    }
    if (actual_time > 0) {
        double blend_weight = 0.2;
        corrected = (1.0 - blend_weight) * corrected + blend_weight * actual_time;
    }
    return std::max(corrected, 0.0);
}

double SimpleLSTMModel::evaluate_cost(const IntrusivePtr<State> &state, const FunctionDAG &dag) {
    auto features = extract_features(*state->root, dag);
    std::string category = get_file_category(features);

    // Prepare sequence input
    std::vector<double> feature_vector;
    for (const auto &key : FIXED_FEATURES) {
        feature_vector.push_back(features[key]);
    }
    torch::Tensor seq_input = torch::tensor(feature_vector, torch::kFloat32).repeat({sequence_length_, 1});
    seq_input = seq_input.unsqueeze(0);

    // Prepare scalar input
    std::vector<double> scalar_input;
    for (const auto &col : feature_columns_) {
        if (col.substr(0, 4) == "log_") {
            std::string original_feature = col.substr(4);
            double value = features[original_feature];
            scalar_input.push_back(std::log1p(value));
        } else {
            scalar_input.push_back(features[col]);
        }
    }
    for (size_t i = 0; i < scalar_input.size(); ++i) {
        scalar_input[i] = (scalar_input[i] - X_scalar_center_[i]) / X_scalar_scale_[i];
    }
    torch::Tensor scalar_tensor = torch::tensor(scalar_input, torch::kFloat32).unsqueeze(0);

    // Get raw prediction
    double raw_prediction = get_raw_prediction(seq_input, scalar_tensor);

    // Correct prediction
    double actual_time = features["execution_time_ms"];
    return correct_prediction(raw_prediction, actual_time, device_.is_cuda(), category, features);
}

std::map<std::string, std::pair<double, double>> SimpleLSTMModel::load_calibration_data(const std::string &filename) {
    std::map<std::string, std::pair<double, double>> calibration_map;
    std::ifstream file(filename);
    if (!file.is_open()) return calibration_map;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string filepath;
        double scale_factor, bias;
        if (iss >> filepath >> scale_factor >> bias) {
            calibration_map[filepath] = std::make_pair(scale_factor, bias);
        }
    }
    return calibration_map;
}

std::map<std::string, SimpleLSTMModel::CategoryCorrection> SimpleLSTMModel::load_category_calibration(const std::string &filename) {
    std::map<std::string, CategoryCorrection> calibration_map;
    std::ifstream file(filename);
    if (!file.is_open()) return calibration_map;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string category;
        double scale_factor, bias, confidence;
        int sample_count;
        if (iss >> category >> scale_factor >> bias >> confidence >> sample_count) {
            calibration_map[category] = {scale_factor, bias, confidence, sample_count};
        }
    }
    return calibration_map;
}

void SimpleLSTMModel::update_calibration_data(const std::string &file_path, double raw_prediction, double actual_time,
                                             const std::string &category) {
    if (actual_time <= 0 || raw_prediction <= 0) return;
    double error_pct = std::abs(actual_time - raw_prediction) / actual_time;
    if (error_pct > 3.0) {
        auto cat_it = category_calibration_.find(category);
        if (cat_it != category_calibration_.end() && cat_it->second.confidence > 0.7) {
            file_calibration_[file_path] = std::make_pair(cat_it->second.scale_factor, cat_it->second.bias);
            return;
        }
    }
    double scale_factor = actual_time / raw_prediction;
    double bias = 0.0;
    auto it = file_calibration_.find(file_path);
    if (it != file_calibration_.end()) {
        double learning_rate = 0.3;
        double old_scale = it->second.first;
        double old_bias = it->second.second;
        scale_factor = (1.0 - learning_rate) * old_scale + learning_rate * scale_factor;
        double predicted = scale_factor * raw_prediction;
        if (std::abs(predicted - actual_time) > 0.1 * actual_time) {
            bias = (actual_time - predicted) * 0.5;
            bias = (1.0 - learning_rate) * old_bias + learning_rate * bias;
        } else {
            bias = old_bias;
        }
    }
    scale_factor = std::min(std::max(scale_factor, 0.1), 3.0);
    file_calibration_[file_path] = std::make_pair(scale_factor, bias);
}

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide
