#include "SimpleLSTMModel.h"
#include <fstream>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>

namespace Halide {
namespace Internal {
namespace Autoscheduler {

using json = nlohmann::json;

SimpleLSTMModel::SimpleLSTMModel(const std::string& model_path, const std::string& scaler_params_path) {
    // Initialize fixed features list
    fixed_features_ = {
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

    // Load scaler parameters
    std::ifstream scaler_file(scaler_params_path);
    if (!scaler_file.is_open()) {
        std::cerr << "Failed to open scaler_params.json at " << scaler_params_path << std::endl;
        return;
    }
    
    json scaler_params;
    try {
        scaler_file >> scaler_params;
        X_scalar_center_ = scaler_params["X_scalar_center"].get<std::vector<double>>();
        X_scalar_scale_ = scaler_params["X_scalar_scale"].get<std::vector<double>>();
        y_center_ = scaler_params["y_center"][0].get<double>();
        y_scale_ = scaler_params["y_scale"][0].get<double>();
        feature_columns_ = scaler_params["feature_columns"].get<std::vector<std::string>>();
    } catch (const json::exception& e) {
        std::cerr << "Error parsing scaler parameters: " << e.what() << std::endl;
        return;
    }

    // Check if CUDA is available
    bool is_gpu_available = torch::cuda::is_available();
    device_ = is_gpu_available ? torch::Device(torch::kCUDA, 0) : torch::kCPU;
    
    // Load the model
    try {
        model_ = torch::jit::load(model_path);
        model_.to(device_);
        model_.eval();
        model_loaded_ = true;
        std::cout << "LSTM model loaded successfully from " << model_path 
                  << " (using " << (is_gpu_available ? "GPU" : "CPU") << ")" << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        model_loaded_ = false;
    }
}

SimpleLSTMModel::~SimpleLSTMModel() {
    // Clean up resources if needed
}

void SimpleLSTMModel::set_pipeline_features(const FunctionDAG &dag,
                                           const Adams2019Params &params) {
    // This method is called once per pipeline
    // We could extract pipeline-level features here if needed
    // For now, we'll just verify the model is loaded
    if (!model_loaded_) {
        std::cerr << "Warning: LSTM model not loaded correctly. Cost predictions may be inaccurate." << std::endl;
    }
}

void SimpleLSTMModel::enqueue(const FunctionDAG &dag,
                             const StageMapOfScheduleFeatures &schedule_feats,
                             double *cost_ptr) {
    // Extract features from the schedule
    auto features = extract_features_from_schedule(dag, schedule_feats);
    
    // Add to queue for batch evaluation
    std::lock_guard<std::mutex> lock(mtx_);
    queue_.push_back({features, cost_ptr});
}

void SimpleLSTMModel::evaluate_costs() {
    std::lock_guard<std::mutex> lock(mtx_);
    
    if (queue_.empty()) {
        return;
    }
    
    if (!model_loaded_) {
        // If model failed to load, assign a default high cost
        for (auto& item : queue_) {
            *(item.cost_ptr) = 1000.0;  // High default cost
        }
        queue_.clear();
        return;
    }
    
    // Process each item in the queue
    for (auto& item : queue_) {
        double cost = run_inference(item.features);
        *(item.cost_ptr) = cost;
    }
    
    // Clear the queue
    queue_.clear();
}

void SimpleLSTMModel::reset() {
    std::lock_guard<std::mutex> lock(mtx_);
    queue_.clear();
}

std::map<std::string, double> SimpleLSTMModel::extract_features_from_schedule(
    const FunctionDAG &dag,
    const StageMapOfScheduleFeatures &schedule_feats) {
    
    // This is where you would extract features from Halide's internal structures
    // For now, we'll create a placeholder with default values
    std::map<std::string, double> features;
    
    // Initialize with zeros
    for (const auto& feature : fixed_features_) {
        features[feature] = 0.0;
    }
    
    // Extract what we can from the schedule features
    // This is a simplified version - you would need to adapt this to extract
    // the actual features from Halide's data structures
    
    int nodes_count = 0;
    int edges_count = 0;
    
    // Count stages as nodes
    for (const auto& pair : schedule_feats.table) {
        nodes_count++;
        
        // Extract features from ScheduleFeatures
        const auto& feat = pair.second;
        
        // These are just examples - you would need to map Halide's ScheduleFeatures
        // to your model's expected features
        features["sched_num_realizations"] += feat.num_realizations;
        features["sched_points_computed_total"] += feat.points_computed_per_realization;
        features["sched_bytes_at_realization"] += feat.bytes_at_realization;
        // ... map other features as needed
    }
    
    // Count function dependencies as edges
    for (size_t i = 0; i < dag.nodes.size(); i++) {
        edges_count += dag.nodes[i].outgoing_edges.size();
    }
    
    features["nodes_count"] = nodes_count;
    features["edges_count"] = edges_count;
    features["node_edge_ratio"] = (edges_count > 0) ? 
        static_cast<double>(nodes_count) / edges_count : static_cast<double>(nodes_count);
    
    // Compute derived features
    features["total_parallelism"] = features["sched_inner_parallelism"] + features["sched_outer_parallelism"];
    features["scheduling_count"] = features["sched_num_realizations"] + features["sched_num_productions"];
    features["total_bytes_at_production"] = features["sched_bytes_at_production"];
    features["total_vectors"] = features["sched_num_vectors"];
    
    double bytes_at_realization = features["sched_bytes_at_realization"];
    if (bytes_at_realization > 0) {
        features["computation_efficiency"] = features["sched_points_computed_total"] / bytes_at_realization;
    }
    
    double bytes_at_root = features["sched_bytes_at_root"];
    if (bytes_at_root > 0) {
        features["memory_pressure"] = features["sched_working_set"] / bytes_at_root;
    }
    
    // Count op diversity
    int op_diversity = 0;
    for (const auto& [key, value] : features) {
        if (key.find("op_") == 0 && value > 0) {
            op_diversity++;
        }
    }
    features["op_diversity"] = op_diversity;
    
    return features;
}

double SimpleLSTMModel::run_inference(const std::map<std::string, double>& features) {
    // Prepare sequence input
    const int sequence_length = 3;  // As in your original code
    std::vector<double> feature_vector;
    
    for (const auto& key : fixed_features_) {
        auto it = features.find(key);
        if (it != features.end()) {
            feature_vector.push_back(it->second);
        } else {
            feature_vector.push_back(0.0);  // Default value if feature not found
        }
    }
    
    torch::Tensor seq_input = torch::tensor(feature_vector, torch::kFloat32).repeat({sequence_length, 1});
    seq_input = seq_input.unsqueeze(0);  // Add batch dimension
    
    // Prepare scalar input
    std::vector<double> scalar_input;
    for (const auto& col : feature_columns_) {
        if (col.substr(0, 4) == "log_") {
            std::string original_feature = col.substr(4);
            auto it = features.find(original_feature);
            double value = (it != features.end()) ? it->second : 0.0;
            scalar_input.push_back(std::log1p(value));
        } else {
            auto it = features.find(col);
            scalar_input.push_back((it != features.end()) ? it->second : 0.0);
        }
    }
    
    // Scale scalar input
    for (size_t i = 0; i < scalar_input.size(); ++i) {
        scalar_input[i] = (scalar_input[i] - X_scalar_center_[i]) / X_scalar_scale_[i];
    }
    
    torch::Tensor scalar_tensor = torch::tensor(scalar_input, torch::kFloat32).unsqueeze(0);
    
    // Move inputs to device
    seq_input = seq_input.to(device_);
    scalar_tensor = scalar_tensor.to(device_);
    
    // Run inference
    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> inputs = {seq_input, scalar_tensor};
    torch::Tensor y_pred_scaled;
    
    try {
        auto start = std::chrono::high_resolution_clock::now();
        y_pred_scaled = model_.forward(inputs).toTensor();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        if (duration > 100) {  // Only log slow inferences
            std::cout << "Model inference took " << duration << "ms" << std::endl;
        }
    } catch (const c10::Error& e) {
        if (device_.is_cuda()) {
            std::cout << "GPU inference failed, falling back to CPU" << std::endl;
            // Try CPU fallback
            torch::Device cpu_device = torch::kCPU;
            
            seq_input = seq_input.to(cpu_device);
            scalar_tensor = scalar_tensor.to(cpu_device);
            
            inputs = {seq_input, scalar_tensor};
            try {
                // Create a CPU copy of the model
                torch::jit::script::Module cpu_model = model_.clone();
                cpu_model.to(cpu_device);
                y_pred_scaled = cpu_model.forward(inputs).toTensor();
            } catch (const c10::Error& e) {
                std::cerr << "Error during CPU fallback inference: " << e.what() << std::endl;
                return 1000.0;  // Return a high cost on error
            }
        } else {
            std::cerr << "Error during model inference: " << e.what() << std::endl;
            return 1000.0;  // Return a high cost on error
        }
    }
    
    // Inverse transform prediction
    torch::Tensor y_pred_transformed = y_pred_scaled * y_scale_ + y_center_;
    torch::Tensor y_pred_actual = torch::expm1(y_pred_transformed);
    
    // Get the prediction as a double
    double prediction = y_pred_actual.item<float>();
    
    // Apply simple correction factor (you could make this more sophisticated)
    bool is_gpu = torch::cuda::is_available();
    double correction_factor = is_gpu ? 0.28 : 0.35;  // From your GPU/CPU correction factors
    
    return std::max(prediction * correction_factor, 0.0);
}

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide
