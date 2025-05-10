/*
  SimpleLSTMModel.cpp: Implementation of CostModel using LibTorch LSTM model.
  Predicts execution times for Halide schedules using features from TreeRepresentation.
*/

#include "CostModel.h"
#include "TreeRepresentation.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <unordered_set>

namespace Halide {
namespace Internal {
namespace Autoscheduler {

namespace fs = std::filesystem;
using json = nlohmann::json;

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

struct HardwareCorrectionFactors {
    double base_correction;
    double gpu_correction;
    double scaling_factor;
    double min_time_ms;
    double high_threshold_ms;
    double high_scaling;
};

const HardwareCorrectionFactors GPU_CORRECTION_FACTORS = {
    0.28, 0.9, 0.95, 100.0, 500.0, 0.92
};

const HardwareCorrectionFactors CPU_CORRECTION_FACTORS = {
    0.35, 1.0, 0.97, 50.0, 300.0, 0.94
};

struct CategoryCorrection {
    double scale_factor;
    double bias;
    double confidence;
    int sample_count;
};

class SimpleLSTMModel : public CostModel {
public:
    SimpleLSTMModel() : device_(torch::kCPU) {
        // Check for CUDA availability
        if (torch::cuda::is_available()) {
            std::cout << "CUDA is available, using GPU\n";
            device_ = torch::Device(torch::kCUDA, 0);
            factors_ = GPU_CORRECTION_FACTORS;
        } else {
            std::cout << "Using CPU\n";
            factors_ = CPU_CORRECTION_FACTORS;
        }

        // Load scaler parameters
        std::ifstream scaler_file("/home/kowrisaan/fyp/Halide/src/autoschedulers/adams2019/scaler_params.json");
        if (!scaler_file.is_open()) {
            throw std::runtime_error("Failed to open scaler_params.json");
        }
        scaler_file >> scaler_params_;
        X_scalar_center_ = scaler_params_["X_scalar_center"].get<std::vector<double>>();
        X_scalar_scale_ = scaler_params_["X_scalar_scale"].get<std::vector<double>>();
        y_center_ = scaler_params_["y_center"][0].get<double>();
        y_scale_ = scaler_params_["y_scale"][0].get<double>();
        feature_columns_ = scaler_params_["feature_columns"].get<std::vector<std::string>>();

        // Load model
        try {
            model_ = torch::jit::load("/home/kowrisaan/fyp/Halide/src/autoschedulers/adams2019/model.pt");
            model_.to(device_);
            model_.eval();
            std::cout << "Model loaded successfully\n";
        } catch (const c10::Error& e) {
            throw std::runtime_error("Error loading model.pt: " + std::string(e.what()));
        }

        // Initialize category calibration
        category_calibration_["default"] = {0.35, 0.0, 0.7, 1};
    }

    void set_pipeline_features(const FunctionDAG& dag, const Adams2019Params& params) override {
        // Store DAG and params for later use
        dag_ = &dag;
        params_ = params;

        // Initialize TreeRepresentation for the pipeline
        std::vector<Function> outputs;
        for (const auto& node : dag.nodes) {
            if (node.is_output) {
                outputs.push_back(node.func);
            }
        }
        tree_ = std::make_unique<TreeRepresentation>(outputs);
    }

    void enqueue(const FunctionDAG& dag, const StageMapOfScheduleFeatures& schedule_feats, double* cost_ptr) override {
        if (&dag != dag_) {
            throw std::runtime_error("DAG mismatch in enqueue");
        }

        // Store schedule features and cost pointer
        queue_.emplace_back(schedule_feats, cost_ptr);
    }

    void evaluate_costs() override {
        for (auto& [schedule_feats, cost_ptr] : queue_) {
            // Generate JSON representation
            json json_data = tree_->to_json();

            // Extract features
            auto features = extract_features(json_data);

            // Prepare inputs
            std::vector<double> feature_vector;
            for (const auto& key : FIXED_FEATURES) {
                feature_vector.push_back(features[key]);
            }
            torch::Tensor seq_input = torch::tensor(feature_vector, torch::kFloat32).repeat({sequence_length_, 1});
            seq_input = seq_input.unsqueeze(0);

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
            torch::Tensor scalar_tensor = torch::tensor(scalar_input, torch::kFloat32).unsqueeze(0);

            // Get raw prediction
            double raw_prediction = get_raw_prediction(seq_input, scalar_tensor);
            if (raw_prediction < 0) {
                std::cerr << "Failed to get prediction\n";
                *cost_ptr = std::numeric_limits<double>::infinity();
                continue;
            }

            // Correct prediction
            double corrected_prediction = correct_prediction(raw_prediction, features);
            *cost_ptr = corrected_prediction;
        }
    }

    void reset() override {
        queue_.clear();
    }

private:
    std::map<std::string, double> extract_features(const json& json_data) {
        std::map<std::string, double> features;

        // Extract global features
        auto global_node = std::find_if(json_data["children"].begin(), json_data["children"].end(),
            [](const json& child) { return child["name"] == "Global Features"; });
        if (global_node != json_data["children"].end()) {
            features["cache_hits"] = global_node->value("cache_hits", 0.0);
            features["cache_misses"] = global_node->value("cache_misses", 0.0);
            features["execution_time_ms"] = global_node->value("execution_time_ms", 0.0);
        }

        // Extract op_histogram
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
        features["computation_efficiency"] = bytes_at_realization > 0 ? features["sched_points_computed_total"] / bytes_at_realization : 0.0;
        double bytes_at_root = features["sched_bytes_at_root"];
        features["memory_pressure"] = bytes_at_root > 0 ? features["sched_working_set"] / bytes_at_root : 0.0;
        double bytes_at_task = features["sched_bytes_at_task"];
        features["memory_utilization_ratio"] = bytes_at_task > 0 ? features["sched_unique_bytes_read_per_realization"] / bytes_at_task : 0.0;
        double execution_time_ms = features["execution_time_ms"];
        features["bytes_processing_rate"] = execution_time_ms > 0 ? features["sched_bytes_at_realization"] / execution_time_ms : 0.0;
        double total_parallelism = features["total_parallelism"];
        features["bytes_per_parallelism"] = total_parallelism > 0 ? features["sched_bytes_at_task"] / total_parallelism : 0.0;
        double num_vectors = features["sched_num_vectors"];
        features["bytes_per_vector"] = num_vectors > 0 ? features["sched_bytes_at_realization"] / num_vectors : 0.0;
        int nodes_count = json_data["children"].size();
        int edges_count = 0;
        for (const auto& node : json_data["children"]) {
            edges_count += node.value("children", json::array()).size();
        }
        features["nodes_count"] = nodes_count;
        features["edges_count"] = edges_count;
        features["node_edge_ratio"] = edges_count > 0 ? static_cast<double>(nodes_count) / edges_count : static_cast<double>(nodes_count);
        double scheduling_count = features["scheduling_count"];
        features["nodes_per_schedule"] = scheduling_count > 0 ? nodes_count / scheduling_count : 0.0;
        int op_diversity = 0;
        for (const auto& [key, value] : features) {
            if (key.find("op_") == 0 && value > 0) {
                op_diversity++;
            }
        }
        features["op_diversity"] = op_diversity;

        return features;
    }

    double compute_complexity_score(const std::map<std::string, double>& features) {
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

    double get_raw_prediction(torch::Tensor seq_input, torch::Tensor scalar_input) {
        seq_input = seq_input.to(device_);
        scalar_input = scalar_input.to(device_);

        torch::NoGradGuard no_grad;
        std::vector<torch::jit::IValue> inputs = {seq_input, scalar_input};
        torch::Tensor y_pred_scaled;

        try {
            auto start = std::chrono::high_resolution_clock::now();
            y_pred_scaled = model_.forward(inputs).toTensor();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            if (duration > 1000) {
                std::cout << "Model inference took " << duration << "ms\n";
            }
        } catch (const c10::Error& e) {
            if (device_.is_cuda()) {
                std::cout << "GPU inference failed, falling back to CPU\n";
                torch::Device cpu_device = torch::kCPU;
                torch::jit::script::Module cpu_model = model_.clone();
                cpu_model.to(cpu_device);
                seq_input = seq_input.to(cpu_device);
                scalar_input = scalar_input.to(cpu_device);
                inputs = {seq_input, scalar_input};
                try {
                    y_pred_scaled = cpu_model.forward(inputs).toTensor();
                } catch (const c10::Error& e) {
                    std::cerr << "CPU inference failed: " << e.what() << "\n";
                    return -1.0;
                }
            } else {
                std::cerr << "Inference failed: " << e.what() << "\n";
                return -1.0;
            }
        }

        torch::Tensor y_pred_transformed = y_pred_scaled * y_scale_ + y_center_;
        torch::Tensor y_pred_actual = torch::expm1(y_pred_transformed);
        return y_pred_actual.item<float>();
    }

    double correct_prediction(double raw_prediction, const std::map<std::string, double>& features) {
        // Simplified correction: use hardware factors and default category
        double complexity = compute_complexity_score(features);
        double hw_correction = factors_.base_correction;
        if (device_.is_cuda()) {
            hw_correction *= factors_.gpu_correction;
        }

        if (complexity > 150) {
            hw_correction *= 0.95;
        } else if (complexity < 20) {
            hw_correction *= 1.03;
        }

        double corrected;
        if (raw_prediction <= factors_.min_time_ms) {
            corrected = raw_prediction * hw_correction;
        } else if (raw_prediction <= factors_.high_threshold_ms) {
            double base = factors_.min_time_ms * hw_correction;
            double excess = raw_prediction - factors_.min_time_ms;
            corrected = base + (excess * hw_correction * factors_.scaling_factor);
        } else {
            double base = factors_.min_time_ms * hw_correction;
            double mid_excess = factors_.high_threshold_ms - factors_.min_time_ms;
            double high_excess = raw_prediction - factors_.high_threshold_ms;
            corrected = base +
                       (mid_excess * hw_correction * factors_.scaling_factor) +
                       (high_excess * hw_correction * factors_.scaling_factor * factors_.high_scaling);
        }

        auto it = category_calibration_.find("default");
        if (it != category_calibration_.end() && it->second.confidence > 0.6) {
            corrected = corrected * it->second.scale_factor + it->second.bias;
        }

        return std::max(corrected, 0.0);
    }

    torch::Device device_;
    HardwareCorrectionFactors factors_;
    json scaler_params_;
    std::vector<double> X_scalar_center_;
    std::vector<double> X_scalar_scale_;
    double y_center_;
    double y_scale_;
    std::vector<std::string> feature_columns_;
    torch::jit::script::Module model_;
    std::map<std::string, CategoryCorrection> category_calibration_;
    std::unique_ptr<TreeRepresentation> tree_;
    const FunctionDAG* dag_ = nullptr;
    Adams2019Params params_;
    std::vector<std::pair<StageMapOfScheduleFeatures, double*>> queue_;
    static constexpr int sequence_length_ = 3;
};

} // namespace Autoscheduler
} // namespace Internal
} // namespace Halide
