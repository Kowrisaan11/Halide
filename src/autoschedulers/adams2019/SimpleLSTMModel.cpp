#include "SimpleLSTMModel.h"
#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <iostream>

using json = nlohmann::json;

namespace Halide {
namespace Internal {
namespace Autoscheduler {

namespace {

const std::vector<std::string> FIXED_FEATURES = {
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

const std::vector<std::string> LOW_IMPORTANCE_FEATURES = {
    "op_cast", "memory_pointwise_1", "memory_pointwise_2", "memory_pointwise_3"
};

std::map<std::string, double> extract_features(const TreeRepresentation& tree_repr) {
    std::map<std::string, double> features;

    // Convert TreeRepresentation to JSON for compatibility with inference code
    json json_data = {{"name", "Root"}, {"children", json::array()}};
    json global_features_node = {
        {"name", "Global Features"},
        {"cache_hits", tree_repr.global_features.cache_hits},
        {"cache_misses", tree_repr.global_features.cache_misses},
        {"execution_time_ms", tree_repr.global_features.execution_time_ms},
        {"children", json::array()}
    };
    json_data["children"].push_back(global_features_node);

    for (const auto& node : tree_repr.nodes) {
        json node_entry = {
            {"name", node.name},
            {"execution_order", node.execution_order},
            {"op_histogram", node.stages[0].op_histogram},
            {"memory_patterns", node.stages[0].memory_patterns},
            {"scheduling", node.scheduling_features},
            {"children", json::array()}
        };
        for (const auto& edge : tree_repr.edges) {
            if (edge.source_name == node.name) {
                json edge_child = {
                    {"target_name", edge.target_name},
                    {"footprint", edge.footprint},
                    {"load_jacobian", edge.load_jacobian}
                };
                node_entry["children"].push_back(edge_child);
            }
        }
        json_data["children"].push_back(node_entry);
    }

    // Extract global features
    auto global_node = std::find_if(json_data["children"].begin(), json_data["children"].end(),
        [](const json& child) { return child["name"] == "Global Features"; });
    if (global_node != json_data["children"].end()) {
        features["cache_hits"] = global_node->value("cache_hits", 0.0);
        features["cache_misses"] = global_node->value("cache_misses", 0.0);
        features["execution_time_ms"] = global_node->value("execution_time_ms", 0.0);
    }

    // Extract op_histogram features
    std::map<std::string, int> op_histogram;
    for (const auto& node : json_data["children"]) {
        if (node.contains("op_histogram")) {
            for (const auto& [op, count] : node["op_histogram"].items()) {
                std::string op_lower = op;
                std::transform(op_lower.begin(), op_lower.end(), op_lower.begin(), ::tolower);
                op_histogram[op_lower] += count.get<int>();
            }
        }
    }
    for (const auto& [op, count] : op_histogram) {
        features["op_" + op] = static_cast<double>(count);
    }

    // Extract memory patterns
    std::map<std::string, std::vector<double>> memory_patterns;
    for (const auto& node : json_data["children"]) {
        if (node.contains("memory_patterns")) {
            for (const auto& [pattern, values] : node["memory_patterns"].items()) {
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
    for (const auto& [pattern, values] : memory_patterns) {
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
    for (const auto& node : json_data["children"]) {
        if (node.contains("scheduling")) {
            node_count++;
            for (const auto& key : scheduling_keys) {
                scheduling_sums[key] += node["scheduling"].value(key, 0.0);
            }
        }
    }
    for (const auto& key : scheduling_keys) {
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
    for (const auto& node : json_data["children"]) {
        edges_count += node.value("children", json::array()).size();
    }
    features["nodes_count"] = nodes_count;
    features["edges_count"] = edges_count;
    features["node_edge_ratio"] = (edges_count > 0) ? static_cast<double>(nodes_count) / edges_count : static_cast<double>(nodes_count);
    double scheduling_count = features["scheduling_count"];
    features["nodes_per_schedule"] = (scheduling_count > 0) ? nodes_count / scheduling_count : 0.0;
    int op_diversity = 0;
    for (const auto& [key, value] : features) {
        if (key.find("op_") == 0 && value > 0) {
            op_diversity++;
        }
    }
    features["op_diversity"] = op_diversity;

    return features;
}

} // anonymous namespace

SimpleLSTMModel::SimpleLSTMModel(const std::string& weights_path)
    : weights_path_(weights_path),
      device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {
    // Load model
    try {
        model_ = torch::jit::load(weights_path_);
        model_.to(device_);
        model_.eval();
    } catch (const c10::Error& e) {
        internal_error << "Error loading model: " << e.what();
    }

    // Load scaler parameters
    json scaler_params;
    std::ifstream scaler_file("scaler_params.json");
    if (!scaler_file.is_open()) {
        internal_error << "Failed to open scaler_params.json";
    }
    scaler_params << scaler_file;
    feature_columns_ = scaler_params["feature_columns"].get<std::vector<std::string>>();
    X_scalar_center_ = scaler_params["X_scalar_center"].get<std::vector<double>>();
    X_scalar_scale_ = scaler_params["X_scalar_scale"].get<std::vector<double>>();
    y_center_ = scaler_params["y_center"][0].get<double>();
    y_scale_ = scaler_params["y_scale"][0].get<double>();
}

void SimpleLSTMModel::set_pipeline_features(const FunctionDAG& dag, const Adams2019Params& params) {
    // Initialize model for the pipeline (e.g., reset internal state)
    reset();
}

void SimpleLSTMModel::enqueue(const FunctionDAG& dag,
                             const StageMapOfScheduleFeatures& schedule_feats,
                             const TreeRepresentation& tree_repr,
                             double* cost_ptr) {
    evaluation_queue_.emplace_back(tree_repr, cost_ptr);
}

void SimpleLSTMModel::evaluate_costs() {
    constexpr int sequence_length = 3;

    for (const auto& [tree_repr, cost_ptr] : evaluation_queue_) {
        // Extract features
        auto features = extract_features(tree_repr);

        // Prepare sequence input
        std::vector<double> feature_vector;
        for (const auto& key : FIXED_FEATURES) {
            feature_vector.push_back(features[key]);
        }
        torch::Tensor seq_input = torch::tensor(feature_vector, torch::kFloat32).repeat({sequence_length, 1});
        seq_input = seq_input.unsqueeze(0).to(device_);

        // Prepare scalar input
        std::vector<double> scalar_input;
        for (const auto& col : feature_columns_) {
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
        torch::Tensor scalar_tensor = torch::tensor(scalar_input, torch::kFloat32).unsqueeze(0).to(device_);

        // Run inference
        torch::NoGradGuard no_grad;
        std::vector<torch::jit::IValue> inputs = {seq_input, scalar_tensor};
        torch::Tensor y_pred_scaled;
        try {
            y_pred_scaled = model_.forward(inputs).toTensor();
        } catch (const c10::Error& e) {
            internal_error << "Error during model inference: " << e.what();
        }

        // Inverse transform prediction
        torch::Tensor y_pred_transformed = y_pred_scaled * y_scale_ + y_center_;
        torch::Tensor y_pred_actual = torch::expm1(y_pred_transformed);
        *cost_ptr = y_pred_actual.item<float>();
    }
}

void SimpleLSTMModel::reset() {
    evaluation_queue_.clear();
}

std::unique_ptr<CostModel> make_simple_lstm_model(const std::string& weights_path) {
    return std::make_unique<SimpleLSTMModel>(weights_path);
}

} // namespace Autoscheduler
} // namespace Internal
} // namespace Halide
