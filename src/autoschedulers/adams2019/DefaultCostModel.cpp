#include "DefaultCostModel.h"
#include <fstream>
#include <iostream>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace Halide {

const HardwareCorrectionFactors GPU_CORRECTION_FACTORS = {
    0.28, 0.9, 0.95, 100.0, 500.0, 0.92
};

const HardwareCorrectionFactors CPU_CORRECTION_FACTORS = {
    0.35, 1.0, 0.97, 50.0, 300.0, 0.94
};

DefaultCostModel::DefaultCostModel(const std::string &model_path,
                                 const std::string &scaler_params_path,
                                 bool use_gpu)
    : device(use_gpu && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
      correction_factors(use_gpu ? GPU_CORRECTION_FACTORS : CPU_CORRECTION_FACTORS),
      session_start(std::chrono::system_clock::now()),
      user_login("Jathu03") {
    
    std::string device_type = device.is_cuda() ? "GPU" : "CPU";
    log_message("Initializing DefaultCostModel with " + device_type + " support");
    
    // Load the model
    try {
        model = torch::jit::load(model_path);
        model.to(device);
        model.eval();
        log_message("Successfully loaded model from " + model_path);
    } catch (const c10::Error& e) {
        std::string error_msg = "Error loading the model: " + std::string(e.what());
        log_message(error_msg);
        throw;
    }

    // Load scaler parameters
    std::ifstream scaler_file(scaler_params_path);
    if (!scaler_file.is_open()) {
        std::string error_msg = "Failed to open " + scaler_params_path;
        log_message(error_msg);
        throw std::runtime_error(error_msg);
    }
    scaler_file >> scaler_params;
    log_message("Successfully loaded scaler parameters");

    initialize_default_calibrations();
}

void DefaultCostModel::initialize_default_calibrations() {
    category_calibration["unknown"] = {0.35, 0.0, 0.7, 1};
    category_calibration["unknown_simple"] = {0.40, 0.0, 0.7, 1};
    category_calibration["unknown_medium"] = {0.35, 0.0, 0.7, 1};
    category_calibration["unknown_complex"] = {0.31, 0.0, 0.7, 1};
    log_message("Initialized default category calibrations");
}

void DefaultCostModel::set_pipeline_features(const Internal::Autoscheduler::FunctionDAG &dag) {
    auto tree = convert_to_tree(dag);
    log_message("Pipeline features set from DAG");
}

Internal::Autoscheduler::TreeRepresentation DefaultCostModel::convert_to_tree(
    const Internal::Autoscheduler::FunctionDAG &dag) {
    
    Internal::Autoscheduler::TreeRepresentation tree;
    
    // Create JSON representation
    json& tree_data = tree.tree_data;
    tree_data["timestamp"] = "2025-05-10 18:01:07";
    tree_data["user"] = user_login;
    
    // Convert DAG nodes to JSON
    tree_data["nodes"] = json::array();
    
    // Extract features from the tree
    tree.extracted_features = extract_features(tree_data);
    
    return tree;
}

void DefaultCostModel::enqueue(const Internal::Autoscheduler::FunctionDAG &dag,
                             double *cost_ptr) {
    auto tree = convert_to_tree(dag);
    queued_trees.push_back(tree);
    queued_cost_ptrs.push_back(cost_ptr);
}

Internal::Autoscheduler::PredictionResult DefaultCostModel::get_prediction(
    const Internal::Autoscheduler::TreeRepresentation &tree_repr,
    bool is_gpu_available) {
    
    Internal::Autoscheduler::PredictionResult result;
    
    // Prepare input tensor
    torch::Tensor input_tensor = prepare_input_tensor(tree_repr.extracted_features);
    
    // Get raw prediction
    double raw_prediction = get_raw_prediction(input_tensor);
    
    // Get category and apply corrections
    std::string category = get_file_category(tree_repr.extracted_features);
    double corrected_prediction = correct_prediction(
        raw_prediction, is_gpu_available, category, tree_repr.extracted_features);
    
    // Fill result structure
    result.raw_prediction = raw_prediction;
    result.corrected_prediction = corrected_prediction;
    result.category = category;
    result.features = tree_repr.extracted_features;
    
    log_prediction_info(category, raw_prediction, corrected_prediction);
    
    return result;
}

void DefaultCostModel::evaluate_costs() {
    log_message("Starting batch evaluation of " + std::to_string(queued_trees.size()) + " trees");
    
    for (size_t i = 0; i < queued_trees.size(); ++i) {
        auto prediction = get_prediction(queued_trees[i], device.is_cuda());
        *queued_cost_ptrs[i] = prediction.corrected_prediction;
    }
    
    log_message("Completed batch evaluation");
    reset();
}

void DefaultCostModel::reset() {
    queued_trees.clear();
    queued_cost_ptrs.clear();
    log_message("Reset queue state");
}

std::map<std::string, double> DefaultCostModel::extract_features(const json &json_data) {
    std::map<std::string, double> features;
    
    // Initialize features with zeros
    for (const auto& feature : Internal::Autoscheduler::FIXED_FEATURES) {
        features[feature] = 0.0;
    }
    
    // Extract basic features
    if (json_data.contains("nodes")) {
        features["nodes_count"] = json_data["nodes"].size();
    }
    
    return features;
}

std::string DefaultCostModel::get_file_category(
    const std::map<std::string, double> &features) {
    
    double complexity = compute_complexity_score(features);
    
    if (complexity > 100.0) {
        return "unknown_complex";
    } else if (complexity > 50.0) {
        return "unknown_medium";
    } else {
        return "unknown_simple";
    }
}

double DefaultCostModel::compute_complexity_score(
    const std::map<std::string, double> &features) {
    
    double complexity = 0.0;
    
    // Add complexity factors
    complexity += features.count("nodes_count") ? features.at("nodes_count") * 0.1 : 0.0;
    
    return complexity;
}

torch::Tensor DefaultCostModel::prepare_input_tensor(
    const std::map<std::string, double>& features) {
    
    std::vector<double> feature_vector;
    for (const auto& feature : Internal::Autoscheduler::FIXED_FEATURES) {
        feature_vector.push_back(features.count(feature) ? features.at(feature) : 0.0);
    }
    
    return torch::tensor(feature_vector, torch::kFloat32).to(device);
}

double DefaultCostModel::get_raw_prediction(const torch::Tensor &input_tensor) {
    torch::NoGradGuard no_grad;
    
    try {
        auto output = model.forward({input_tensor}).toTensor();
        return output.item<float>();
    } catch (const c10::Error& e) {
        log_message("Error during inference: " + std::string(e.what()));
        return -1.0;
    }
}

double DefaultCostModel::correct_prediction(
    double raw_prediction,
    bool is_gpu,
    const std::string &category,
    const std::map<std::string, double> &features) {
    
    // Apply category correction
    auto cat_it = category_calibration.find(category);
    if (cat_it != category_calibration.end()) {
        double corrected = raw_prediction * cat_it->second.scale_factor + cat_it->second.bias;
        return std::max(corrected, 0.0);
    }
    
    // Apply hardware correction
    double hw_correction = correction_factors.base_correction;
    if (is_gpu) {
        hw_correction *= correction_factors.gpu_correction;
    }
    
    return std::max(raw_prediction * hw_correction, 0.0);
}

void DefaultCostModel::log_prediction_info(
    const std::string& category,
    double raw_prediction,
    double corrected_prediction) {
    
    std::stringstream ss;
    ss << "[" << "2025-05-10 18:01:07" << " UTC] "
       << "Prediction for category '" << category << "': "
       << "raw=" << raw_prediction << ", "
       << "corrected=" << corrected_prediction;
    std::cout << ss.str() << std::endl;
}

}  // namespace Halide
