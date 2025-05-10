#include "TorchCostModel.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>

namespace Halide {

TorchCostModel::TorchCostModel(const std::string& model_path, 
                             const std::string& scaler_params_path,
                             bool use_gpu) {
    // Load the PyTorch model
    try {
        model_ = torch::jit::load(model_path);
        
        // Set device
        device_ = (use_gpu && torch::cuda::is_available()) 
                ? torch::Device(torch::kCUDA, 0) 
                : torch::Device(torch::kCPU);
        
        model_.to(device_);
        model_.eval();
        
        std::cout << "TorchCostModel: Model loaded successfully on " 
                 << (device_.is_cuda() ? "GPU" : "CPU") << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        throw;
    }
    
    // Load scaler parameters
    std::ifstream scaler_file(scaler_params_path);
    if (!scaler_file.is_open()) {
        std::cerr << "Failed to open scaler_params.json" << std::endl;
        throw std::runtime_error("Failed to open scaler parameters file");
    }
    
    nlohmann::json scaler_params;
    scaler_file >> scaler_params;
    
    X_scalar_center_ = scaler_params["X_scalar_center"].get<std::vector<double>>();
    X_scalar_scale_ = scaler_params["X_scalar_scale"].get<std::vector<double>>();
    y_center_ = scaler_params["y_center"][0].get<double>();
    y_scale_ = scaler_params["y_scale"][0].get<double>();
    feature_columns_ = scaler_params["feature_columns"].get<std::vector<std::string>>();
    
    // Set hardware correction factors based on device
    if (device_.is_cuda()) {
        correction_factors_ = {
            0.28,   // Base correction factor
            0.9,    // GPU-specific additional correction
            0.95,   // Scaling factor for non-linear correction
            100.0,  // Minimum time threshold in ms
            500.0,  // High execution time threshold
            0.92    // Special scaling for high execution times
        };
    } else {
        correction_factors_ = {
            0.35,   // Base correction factor
            1.0,    // No additional GPU correction
            0.97,   // Scaling factor for non-linear correction
            50.0,   // Minimum time threshold in ms
            300.0,  // High execution time threshold
            0.94    // Special scaling for high execution times
        };
    }
}

void TorchCostModel::set_pipeline_features(
    const Internal::Autoscheduler::FunctionDAG &dag,
    const Internal::Autoscheduler::Adams2019Params &params) {
    // This method is called once per pipeline
    // We don't need to do anything special here for our model
}

void TorchCostModel::enqueue(
    const Internal::Autoscheduler::FunctionDAG &dag,
    const Internal::Autoscheduler::StageMapOfScheduleFeatures &schedule_feats,
    double *cost_ptr) {
    
    // Extract features from the DAG and schedule features
    std::map<std::string, double> features = extract_features_from_dag(dag, schedule_feats);
    
    // Add to queue for batch evaluation
    std::lock_guard<std::mutex> lock(queue_mutex_);
    queue_.emplace_back(std::move(features), cost_ptr);
}

void TorchCostModel::evaluate_costs() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    if (queue_.empty()) {
        return;
    }
    
    // Process all items in the queue
    const int sequence_length = 3;  // Same as in your original code
    
    for (auto& [features, cost_ptr] : queue_) {
        // Prepare sequence input
        std::vector<double> feature_vector;
        for (const auto& key : FIXED_FEATURES) {
            feature_vector.push_back(features[key]);
        }
        
        torch::Tensor seq_input = torch::tensor(feature_vector, torch::kFloat32)
                                 .repeat({sequence_length, 1});
        seq_input = seq_input.unsqueeze(0);  // Add batch dimension
        
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
        
        // Scale scalar input
        for (size_t i = 0; i < scalar_input.size(); ++i) {
            scalar_input[i] = (scalar_input[i] - X_scalar_center_[i]) / X_scalar_scale_[i];
        }
        
        torch::Tensor scalar_tensor = torch::tensor(scalar_input, torch::kFloat32).unsqueeze(0);
        
        // Get raw prediction
        double raw_prediction = get_raw_prediction(seq_input, scalar_tensor);
        
        // Apply correction
        double corrected_prediction = correct_prediction(raw_prediction);
        
        // Set the cost
        *cost_ptr = corrected_prediction;
    }
    
    // Clear the queue
    queue_.clear();
}

void TorchCostModel::reset() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    queue_.clear();
}

double TorchCostModel::get_raw_prediction(
    torch::Tensor seq_input, 
    torch::Tensor scalar_input) {
    
    // Move inputs to device
    seq_input = seq_input.to(device_);
    scalar_input = scalar_input.to(device_);
    
    // Run inference
    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> inputs = {seq_input, scalar_input};
    torch::Tensor y_pred_scaled;
    
    try {
        y_pred_scaled = model_.forward(inputs).toTensor();
    } catch (const c10::Error& e) {
        if (device_.is_cuda()) {
            // Try CPU fallback
            torch::Device cpu_device = torch::kCPU;
            torch::jit::script::Module cpu_model = model_.clone();
            cpu_model.to(cpu_device);
            
            seq_input = seq_input.to(cpu_device);
            scalar_input = scalar_input.to(cpu_device);
            
            inputs = {seq_input, scalar_input};
            try {
                y_pred_scaled = cpu_model.forward(inputs).toTensor();
            } catch (const c10::Error& e) {
                std::cerr << "Error during CPU fallback inference: " << e.what() << std::endl;
                return -1.0;
            }
        } else {
            std::cerr << "Error during model inference: " << e.what() << std::endl;
            return -1.0;
        }
    }
    
    // Inverse transform prediction
    torch::Tensor y_pred_transformed = y_pred_scaled * y_scale_ + y_center_;
    torch::Tensor y_pred_actual = torch::expm1(y_pred_transformed);
    
    // Return the raw prediction
    return y_pred_actual.item<float>();
}

double TorchCostModel::correct_prediction(double raw_prediction) {
    // Apply hardware-specific correction
    double hw_correction = correction_factors_.base_correction;
    if (device_.is_cuda()) {
        hw_correction *= correction_factors_.gpu_correction;
    }
    
    // Apply multi-stage non-linear correction
    double corrected;
    
    if (raw_prediction <= correction_factors_.min_time_ms) {
        // Basic correction for small predictions
        corrected = raw_prediction * hw_correction;
    } else if (raw_prediction <= correction_factors_.high_threshold_ms) {
        // Medium-range predictions: apply basic non-linear correction
        double base = correction_factors_.min_time_ms * hw_correction;
        double excess = raw_prediction - correction_factors_.min_time_ms;
        corrected = base + (excess * hw_correction * correction_factors_.scaling_factor);
    } else {
        // High-range predictions: apply additional scaling for very large values
        double base = correction_factors_.min_time_ms * hw_correction;
        double mid_excess = correction_factors_.high_threshold_ms - correction_factors_.min_time_ms;
        double high_excess = raw_prediction - correction_factors_.high_threshold_ms;
        
        corrected = base + 
                   (mid_excess * hw_correction * correction_factors_.scaling_factor) +
                   (high_excess * hw_correction * correction_factors_.scaling_factor * 
                    correction_factors_.high_scaling);
    }
    
    return std::max(corrected, 0.0);
}

std::map<std::string, double> TorchCostModel::extract_features_from_dag(
    const Internal::Autoscheduler::FunctionDAG &dag,
    const Internal::Autoscheduler::StageMapOfScheduleFeatures &schedule_feats) {
    
    // This is a simplified version - you'll need to adapt this to extract the features
    // your model expects from the Halide DAG and schedule features
    std::map<std::string, double> features;
    
    // Initialize with default values
    for (const auto& key : FIXED_FEATURES) {
        features[key] = 0.0;
    }
    
    // Extract features from the DAG and schedule_feats
    // This is where you'll need to map from Halide's internal representation
    // to the features your model expects
    
    // Example of extracting some basic features:
    int nodes_count = 0;
    int edges_count = 0;
    double total_compute = 0.0;
    double total_memory = 0.0;
    
    // Count nodes and extract compute/memory requirements
    for (size_t i = 0; i < dag.nodes.size(); i++) {
        const auto& node = dag.nodes[i];
        nodes_count++;
        
        // Count edges
        for (const auto& edge : node.outgoing_edges) {
            edges_count++;
        }
        
        // Extract compute requirements for each stage
        for (size_t j = 0; j < node.stages.size(); j++) {
            const auto& stage = node.stages[j];
            // Accumulate compute and memory metrics
            total_compute += stage.vector_size * stage.loop_extent;
            total_memory += stage.bytes_at_production;
        }
    }
    
    // Set extracted features
    features["nodes_count"] = nodes_count;
    features["edges_count"] = edges_count;
    features["sched_points_computed_total"] = total_compute;
    features["sched_bytes_at_production"] = total_memory;
    
    // Extract schedule-specific features from schedule_feats
    // This would depend on your specific model's requirements
    
    // Compute derived features
    features["node_edge_ratio"] = (edges_count > 0) ? 
        static_cast<double>(nodes_count) / edges_count : static_cast<double>(nodes_count);
    
    // Count operation types (simplified)
    int op_diversity = 0;
    features["op_diversity"] = op_diversity;
    
    return features;
}

double TorchCostModel::compute_complexity_score(const std::map<std::string, double>& features) {
    // Combine key metrics into a single complexity score
    double complexity = 0.0;
    
    // Node and edge complexity
    complexity += features.at("nodes_count") * 0.01;
    complexity += features.at("edges_count") * 0.005;
    
    // Computational complexity
    complexity += features.at("sched_points_computed_total") * 0.00001;
    
    // Memory complexity
    complexity += features.at("sched_bytes_at_production") * 0.00005;
    
    // Operation complexity
    complexity += features.at("op_diversity") * 0.1;
    
    return complexity;
}

} // namespace Halide
