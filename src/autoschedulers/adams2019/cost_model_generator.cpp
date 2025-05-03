#include <torch/script.h>
#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <mutex>

#include "Halide.h"

using namespace Halide;
using json = nlohmann::json;

// Define FIXED_FEATURES as in the Python code
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

// Enhanced hardware-specific correction factors
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

// Category-specific correction factors
struct CategoryCorrection {
    double scale_factor;
    double bias;
    double confidence;
    int sample_count;
};

// Simplified cost model using the pre-trained SimpleLSTMModel for inference
class CostModel : public Generator<CostModel> {
protected:
    bool allow_out_of_order_inputs_and_outputs() const override {
        return true;
    }

public:
    // Inputs
    GeneratorInput<Buffer<float>> input_buffer{"input_buffer", 3};
    GeneratorInput<int> batch_size{"batch_size", 1};
    // Remove the std::string GeneratorParam and use environment variable instead
    GeneratorInput<float> actual_runtime{"actual_runtime", -1.0f};

    // Output
    GeneratorOutput<Buffer<float>> prediction_output{"prediction_output", 1};

    // PyTorch model and device
    std::shared_ptr<torch::jit::Module> pytorch_model;
    torch::Device device;

    // Scaler parameters
    std::vector<double> X_scalar_center;
    std::vector<double> X_scalar_scale;
    double y_center;
    double y_scale;
    std::vector<std::string> feature_columns;

    // Calibration data
    std::map<std::string, std::pair<double, double>> file_calibration;
    std::map<std::string, CategoryCorrection> category_calibration;

    // Mutex for file writing
    std::mutex file_mutex;

    // Constructor to initialize the PyTorch model and load scaler parameters
    CostModel() : device(torch::kCPU) {
        // Initialize device
        bool is_gpu_available = torch::cuda::is_available();
        if (is_gpu_available) {
            device = torch::Device(torch::kCUDA);
            std::cout << "Using GPU" << std::endl;
        } else {
            std::cout << "Using CPU" << std::endl;
        }

        // Load the PyTorch model
        try {
            pytorch_model = std::make_shared<torch::jit::Module>(torch::jit::load("model.pt"));
            pytorch_model->to(device);
            pytorch_model->eval();
        } catch (const c10::Error& e) {
            std::cerr << "Error loading the PyTorch model: " << e.what() << std::endl;
            throw;
        }

        // Load scaler parameters
        json scaler_params;
        std::ifstream scaler_file("scaler_params.json");
        if (!scaler_file.is_open()) {
            std::cerr << "Failed to open scaler_params.json" << std::endl;
            throw;
        }
        scaler_file >> scaler_params;
        scaler_file.close();
        
        X_scalar_center = scaler_params["X_scalar_center"].get<std::vector<double>>();
        X_scalar_scale = scaler_params["X_scalar_scale"].get<std::vector<double>>();
        y_center = scaler_params["y_center"][0].get<double>();
        y_scale = scaler_params["y_scale"][0].get<double>();
        feature_columns = scaler_params["feature_columns"].get<std::vector<std::string>>();

        // Validate scaler parameters
        if (X_scalar_center.size() != X_scalar_scale.size() || 
            X_scalar_center.size() != feature_columns.size()) {
            std::cerr << "Scaler parameter size mismatch: "
                      << "X_scalar_center(" << X_scalar_center.size() << "), "
                      << "X_scalar_scale(" << X_scalar_scale.size() << "), "
                      << "feature_columns(" << feature_columns.size() << ")" << std::endl;
            throw std::runtime_error("Invalid scaler parameters");
        }
    }

    // Function to extract features from JSON data
    std::map<std::string, double> extract_features(const json& json_data) {
        std::map<std::string, double> features;

        // Initialize features to 0
        for (const auto& key : FIXED_FEATURES) {
            features[key] = 0.0;
        }

        // Check if required keys exist
        if (!json_data.contains("global_features") || !json_data.contains("nodes") || !json_data.contains("edges")) {
            std::cerr << "JSON data missing required keys: global_features, nodes, or edges" << std::endl;
            return features; // Return default features
        }

        // Extract global features
        auto global_features = json_data["global_features"];
        features["cache_hits"] = global_features.value("cache_hits", 0.0);
        features["cache_misses"] = global_features.value("cache_misses", 0.0);
        features["execution_time_ms"] = global_features.value("execution_time_ms", 0.0);

        // Aggregate node features
        std::vector<json> nodes = json_data["nodes"];
        std::map<std::string, double> op_histogram;
        std::map<std::string, std::vector<double>> memory_patterns;
        std::map<std::string, double> scheduling;

        // Initialize operation histogram
        for (const auto& op_key : {"Const", "Variable", "Param", "Add", "Sub", "Mod", "Mul", "Div",
                                   "Min", "Max", "EQ", "NE", "LT", "LE", "And", "Or", "Not", "Select",
                                   "ImageCall", "FuncCall", "SelfCall", "ExternCall", "Let"}) {
            std::string op_lower = op_key;
            std::transform(op_lower.begin(), op_lower.end(), op_lower.begin(), ::tolower);
            op_histogram[op_lower] = 0.0;
        }

        // Initialize memory patterns
        for (const auto& memory_key : {"Pointwise", "Transpose", "Broadcast", "Slice"}) {
            std::string memory_lower = memory_key;
            std::transform(memory_lower.begin(), memory_lower.end(), memory_lower.begin(), ::tolower);
            memory_patterns[memory_lower] = std::vector<double>(4, 0.0);
        }

        // Initialize scheduling features
        std::vector<std::string> scheduling_keys = {
            "num_realizations", "num_productions", "points_computed_total", "innermost_loop_extent",
            "inner_parallelism", "outer_parallelism", "bytes_at_realization", "bytes_at_production",
            "bytes_at_root", "unique_bytes_read_per_realization", "working_set", "vector_size",
            "num_vectors", "num_scalars", "bytes_at_task", "working_set_at_task", "working_set_at_production",
            "working_set_at_realization", "working_set_at_root"
        };
        for (const auto& key : scheduling_keys) {
            scheduling[key] = 0.0;
        }

        // Aggregate features across nodes
        int node_count = 0;
        for (const auto& node : nodes) {
            if (!node.contains("features")) continue;
            json node_features = node["features"];
            node_count++;

            // Operation histogram
            if (node_features.contains("op_histogram")) {
                for (const auto& [op, count] : node_features["op_histogram"].items()) {
                    std::string op_lower = op;
                    std::transform(op_lower.begin(), op_lower.end(), op_lower.begin(), ::tolower);
                    if (op_histogram.count(op_lower)) {
                        op_histogram[op_lower] += count.get<double>();
                    }
                }
            }

            // Memory patterns
            if (node_features.contains("memory_patterns")) {
                for (const auto& [pattern, values] : node_features["memory_patterns"].items()) {
                    std::string pattern_lower = pattern;
                    std::transform(pattern_lower.begin(), pattern_lower.end(), pattern_lower.begin(), ::tolower);
                    if (memory_patterns.count(pattern_lower)) {
                        auto json_values = values.get<std::vector<double>>();
                        for (size_t i = 0; i < json_values.size() && i < 4; ++i) {
                            memory_patterns[pattern_lower][i] += json_values[i];
                        }
                    }
                }
            }

            // Scheduling features
            if (node_features.contains("scheduling")) {
                for (const auto& key : scheduling_keys) {
                    scheduling[key] += node_features["scheduling"].value(key, 0.0);
                }
            }
        }

        // Map features to FIXED_FEATURES
        for (const auto& [op, count] : op_histogram) {
            features["op_" + op] = count;
        }
        for (const auto& [pattern, values] : memory_patterns) {
            for (size_t i = 0; i < 4; ++i) {
                features["memory_" + pattern + "_" + std::to_string(i)] = values[i];
            }
        }
        for (const auto& key : scheduling_keys) {
            if (key == "inner_parallelism" || key == "outer_parallelism") {
                features["sched_" + key] = node_count > 0 ? scheduling[key] / node_count : 0.0;
            } else {
                features["sched_" + key] = scheduling[key];
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
        int nodes_count = nodes.size();
        int edges_count = json_data["edges"].size();
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

    // Function to compute complexity score from features
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

    // Function to determine file category
    std::string get_file_category(const std::string& file_path, const std::map<std::string, double>& features) {
        std::filesystem::path path(file_path);
        std::string base_category;

        if (path.has_parent_path()) {
            base_category = path.parent_path().filename().string();
        } else {
            base_category = "unknown";
        }

        if (base_category == "unknown") {
            double complexity = compute_complexity_score(features);
            if (complexity > 100.0) {
                return "unknown_complex";
            } else if (complexity > 50.0) {
                return "unknown_medium";
            } else {
                return "unknown_simple";
            }
        }

        return base_category;
    }

    // Load category-specific calibration data
    std::map<std::string, CategoryCorrection> load_category_calibration(const std::string& filename) {
        std::map<std::string, CategoryCorrection> calibration_map;
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cout << "No category calibration file found. Using default correction factors." << std::endl;
            calibration_map["default"] = {1.0, 0.0, 0.7, 1}; // Default calibration
            return calibration_map;
        }

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

    // Save category-specific calibration data
    void save_category_calibration(const std::string& filename, 
                                   const std::map<std::string, CategoryCorrection>& calibration_map) {
        std::lock_guard<std::mutex> lock(file_mutex);
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open category calibration file for writing." << std::endl;
            return;
        }

        for (const auto& [category, correction] : calibration_map) {
            file << category << " " << correction.scale_factor << " " << correction.bias 
                 << " " << correction.confidence << " " << correction.sample_count << std::endl;
        }
    }

    // Load file-specific calibration data
    std::map<std::string, std::pair<double, double>> load_calibration_data(const std::string& filename) {
        std::map<std::string, std::pair<double, double>> calibration_map;
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cout << "No file-specific calibration file found. Using category-based corrections." << std::endl;
            calibration_map["default"] = std::make_pair(1.0, 0.0); // Default calibration
            return calibration_map;
        }

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

    // Save file-specific calibration data
    void save_calibration_data(const std::string& filename, 
                               const std::map<std::string, std::pair<double, double>>& calibration_map) {
        std::lock_guard<std::mutex> lock(file_mutex);
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file-specific calibration file for writing." << std::endl;
            return;
        }

        for (const auto& [filepath, factors] : calibration_map) {
            file << filepath << " " << factors.first << " " << factors.second << std::endl;
        }
    }

    // Update category-specific calibration data
    void update_category_calibration(std::map<std::string, CategoryCorrection>& category_map,
                                     const std::string& category, double raw_prediction, double actual_time) {
        if (actual_time <= 0 || raw_prediction <= 0) return;

        double error_pct = std::abs(actual_time - raw_prediction) / actual_time;
        if (error_pct > 5.0) return;  // Skip outliers

        double scale_factor = actual_time / raw_prediction;
        double bias = 0.0;

        auto it = category_map.find(category);
        if (it != category_map.end()) {
            double base_lr = 0.2;
            double confidence = it->second.confidence;
            int sample_count = it->second.sample_count;
            double learning_rate = base_lr / (1.0 + 0.1 * std::log1p(sample_count));
            if (confidence > 0.8) learning_rate *= 0.8;

            double old_scale = it->second.scale_factor;
            double old_bias = it->second.bias;
            scale_factor = (1.0 - learning_rate) * old_scale + learning_rate * scale_factor;

            double predicted = scale_factor * raw_prediction;
            if (std::abs(predicted - actual_time) > 0.05 * actual_time) {
                bias = (actual_time - predicted) * 0.5;
                bias = (1.0 - learning_rate) * old_bias + learning_rate * bias;
            } else {
                bias = old_bias;
            }

            double accuracy = 1.0 - std::min(error_pct, 1.0);
            double new_confidence = (confidence * sample_count + accuracy) / (sample_count + 1);
            category_map[category] = {scale_factor, bias, new_confidence, sample_count + 1};
        } else {
            category_map[category] = {scale_factor, bias, 0.7, 1};
        }

        category_map[category].scale_factor = std::min(std::max(category_map[category].scale_factor, 0.1), 3.0);
    }

    // Update file-specific calibration data
    void update_calibration_data(std::map<std::string, std::pair<double, double>>& calibration_map,
                                 const std::string& file_path, double raw_prediction, double actual_time,
                                 const std::map<std::string, CategoryCorrection>& category_map,
                                 const std::string& category) {
        if (actual_time <= 0 || raw_prediction <= 0) return;

        double error_pct = std::abs(actual_time - raw_prediction) / actual_time;
        if (error_pct > 3.0) {
            auto cat_it = category_map.find(category);
            if (cat_it != category_map.end() && cat_it->second.confidence > 0.7) {
                calibration_map[file_path] = std::make_pair(cat_it->second.scale_factor, cat_it->second.bias);
                return;
            }
        }

        double scale_factor = actual_time / raw_prediction;
        double bias = 0.0;
        auto it = calibration_map.find(file_path);
        if (it != calibration_map.end()) {
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
        calibration_map[file_path] = std::make_pair(scale_factor, bias);
    }

    // Correct prediction with hardware, category, and file-specific factors
    double correct_prediction(double raw_prediction, double actual_time, bool is_gpu,
                              const HardwareCorrectionFactors& factors,
                              const std::map<std::string, std::pair<double, double>>& file_calibration,
                              const std::map<std::string, CategoryCorrection>& category_calibration,
                              const std::string& file_path,
                              const std::string& category,
                              const std::map<std::string, double>& features) {
        auto file_it = file_calibration.find(file_path);
        if (file_it != file_calibration.end()) {
            const auto& [scale_factor, bias] = file_it->second;
            return std::max(scale_factor * raw_prediction + bias, 0.0);
        }

        auto cat_it = category_calibration.find(category);
        if (cat_it != category_calibration.end() && cat_it->second.confidence > 0.6) {
            const auto& correction = cat_it->second;
            return std::max(correction.scale_factor * raw_prediction + correction.bias, 0.0);
        }

        double hw_correction = factors.base_correction;
        if (is_gpu) hw_correction *= factors.gpu_correction;

        if (category.find("unknown") != std::string::npos) {
            double complexity = compute_complexity_score(features);
            if (category == "unknown_complex") {
                hw_correction *= 0.92;
            } else if (category == "unknown_simple") {
                hw_correction *= 1.05;
            }
            if (complexity > 150) {
                hw_correction *= 0.95;
            } else if (complexity < 20) {
                hw_correction *= 1.03;
            }
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
            corrected = base + 
                        (mid_excess * hw_correction * factors.scaling_factor) +
                        (high_excess * hw_correction * factors.scaling_factor * factors.high_scaling);
        }

        if (actual_time > 0) {
            double blend_weight = 0.2;
            corrected = (1.0 - blend_weight) * corrected + blend_weight * actual_time;
        }

        return std::max(corrected, 0.0);
    }

    void generate() {
        Var n("n");

        // Step 1: Load calibration data
        file_calibration = load_calibration_data("calibration_data.txt");
        category_calibration = load_category_calibration("category_calibration.txt");

        // Step 2: Get the JSON file path from environment
        const char* json_file_path = getenv("HALIDE_JSON_FILE_PATH");
        if (!json_file_path) {
            std::cerr << "Error: HALIDE_JSON_FILE_PATH environment variable not set" << std::endl;
            prediction_output(n) = 0.0f; // Default output
            return;
        }

        // Step 3: Load and parse the JSON file
        std::ifstream ifs(json_file_path);
        if (!ifs.is_open()) {
            std::cerr << "Error: Could not open JSON file " << json_file_path << std::endl;
            prediction_output(n) = 0.0f; // Default output
            return;
        }
        json graph_data;
        try {
            ifs >> graph_data;
        } catch (const json::exception& e) {
            std::cerr << "Error parsing JSON file " << json_file_path << ": " << e.what() << std::endl;
            prediction_output(n) = 0.0f; // Default output
            return;
        }
        ifs.close();

        // Step 4: Extract features
        auto features = extract_features(graph_data);

        // Step 5: Determine category
        std::string category = get_file_category(json_file_path, features);

        // Step 6: Prepare inputs for the PyTorch model
        int batch_size_val = evaluate<int>(batch_size);
        if (batch_size_val <= 0) {
            std::cerr << "Invalid batch_size: " << batch_size_val << std::endl;
            prediction_output(n) = 0.0f; // Default output
            return;
        }
        const int sequence_length = 3;
        const int seq_input_size = FIXED_FEATURES.size();
        const int scalar_input_size = feature_columns.size();

        std::vector<float> seq_input_data(batch_size_val * sequence_length * seq_input_size, 0.0f);
        std::vector<float> scalar_input_data(batch_size_val * scalar_input_size, 0.0f);

        // Prepare seq_input
        for (int b = 0; b < batch_size_val; b++) {
            for (int t = 0; t < sequence_length; t++) {
                int offset = (b * sequence_length + t) * seq_input_size;
                for (size_t i = 0; i < FIXED_FEATURES.size(); i++) {
                    seq_input_data[offset + i] = static_cast<float>(features[FIXED_FEATURES[i]]);
                }
            }
        }

        // Prepare scalar_input
        for (int b = 0; b < batch_size_val; b++) {
            int offset = b * scalar_input_size;
            for (size_t i = 0; i < feature_columns.size(); i++) {
                const auto& col = feature_columns[i];
                double value = features[col];
                if (col.substr(0, 4) == "log_") {
                    std::string original_feature = col.substr(4);
                    value = features[original_feature];
                    scalar_input_data[offset + i] = static_cast<float>(std::log1p(value));
                } else {
                    scalar_input_data[offset + i] = static_cast<float>(value);
                }
                // Apply RobustScaler
                scalar_input_data[offset + i] = (scalar_input_data[offset + i] - X_scalar_center[i]) / X_scalar_scale[i];
            }
        }

        // Step 7: Create PyTorch tensors
        torch::Tensor seq_input_tensor = torch::from_blob(seq_input_data.data(),
                                                         {batch_size_val, sequence_length, seq_input_size},
                                                         torch::kFloat32);
        torch::Tensor scalar_input_tensor = torch::from_blob(scalar_input_data.data(),
                                                            {batch_size_val, scalar_input_size},
                                                            torch::kFloat32);

        // Move tensors to device
        seq_input_tensor = seq_input_tensor.to(device);
        scalar_input_tensor = scalar_input_tensor.to(device);

        // Step 8: Run the PyTorch model
        std::vector<torch::jit::IValue> inputs = {seq_input_tensor, scalar_input_tensor};
        torch::Tensor output_tensor;
        try {
            auto start = std::chrono::high_resolution_clock::now();
            torch::NoGradGuard no_grad;
            output_tensor = pytorch_model->forward(inputs).toTensor();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            if (duration > 1000) {
                std::cout << "Model inference took " << duration << "ms" << std::endl;
            }
        } catch (const c10::Error& e) {
            if (device.is_cuda()) {
                std::cout << "GPU inference failed: " << e.what() << ". Falling back to CPU" << std::endl;
                device = torch::Device(torch::kCPU);
                pytorch_model->to(device);
                seq_input_tensor = seq_input_tensor.to(device);
                scalar_input_tensor = scalar_input_tensor.to(device);
                inputs = {seq_input_tensor, scalar_input_tensor};
                try {
                    torch::NoGradGuard no_grad;
                    output_tensor = pytorch_model->forward(inputs).toTensor();
                } catch (const c10::Error& e) {
                    std::cerr << "Error during CPU fallback inference: " << e.what() << std::endl;
                    prediction_output(n) = 0.0f; // Default output
                    return;
                }
            } else {
                std::cerr << "Error during model inference: " << e.what() << std::endl;
                prediction_output(n) = 0.0f; // Default output
                return;
            }
        }

        // Step 9: Convert PyTorch output to Halide
        auto output_accessor = output_tensor.accessor<float, 2>();
        Func predicted_scaled_runtime("predicted_scaled_runtime");
        predicted_scaled_runtime(n) = 0.0f;
        for (int b = 0; b < batch_size_val; b++) {
            predicted_scaled_runtime(b) = output_accessor[b][0];
        }

        // Step 10: Inverse transform the output with clamping
        Func predicted_transformed("predicted_transformed");
        predicted_transformed(n) = predicted_scaled_runtime(n) * static_cast<float>(y_scale) + static_cast<float>(y_center);
        Func raw_prediction("raw_prediction");
        Expr clamped_input = clamp(predicted_transformed(n), -50.0f, 50.0f); // Prevent exp overflow
        raw_prediction(n) = exp(clamped_input) - 1.0f;

        // Step 11: Apply correction
        Func corrected_prediction("corrected_prediction");
        corrected_prediction(n) = 0.0f;
        const HardwareCorrectionFactors& factors = device.is_cuda() ? GPU_CORRECTION_FACTORS : CPU_CORRECTION_FACTORS;
        float actual_time = evaluate<float>(actual_runtime);
        for (int b = 0; b < batch_size_val; b++) {
            float raw_pred = evaluate<float>(raw_prediction(b));
            double corrected = correct_prediction(raw_pred, actual_time, device.is_cuda(),
                                                 factors, file_calibration, category_calibration,
                                                 json_file_path, category, features);
            corrected_prediction(b) = static_cast<float>(corrected);
        }

        // Step 12: Update calibration if actual runtime is provided
        if (actual_time > 0) {
            for (int b = 0; b < batch_size_val; b++) {
                float raw_pred = evaluate<float>(raw_prediction(b));
                update_category_calibration(category_calibration, category, raw_pred, actual_time);
                update_calibration_data(file_calibration, json_file_path, raw_pred, actual_time,
                                       category_calibration, category);
            }
            save_calibration_data("calibration_data.txt", file_calibration);
            save_category_calibration("category_calibration.txt", category_calibration);
        }

        // Step 13: Set the final output
        prediction_output(n) = corrected_prediction(n);

        // Step 14: Set estimates for autoscheduling
        batch_size.set_estimate(1);
        prediction_output.set_estimates({{0, batch_size_val}});

        // Step 15: Simplified scheduling for inference
        Var no;
        prediction_output.compute_root().split(n, no, n, 8).parallel(no);
        prediction_output.bound(n, 0, batch_size);
    }
};

HALIDE_REGISTER_GENERATOR(CostModel, cost_model)
