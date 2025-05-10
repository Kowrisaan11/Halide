DefaultCostModel::DefaultCostModel(const std::string &model_path,
                                 const std::string &scaler_params_path,
                                 bool use_gpu)
    : device(use_gpu && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
      correction_factors(use_gpu ? GPU_CORRECTION_FACTORS : CPU_CORRECTION_FACTORS) {
    
    // Load the model
    try {
        model = torch::jit::load(model_path);
        model.to(device);
        model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        throw;
    }

    // Load scaler parameters
    std::ifstream scaler_file(scaler_params_path);
    if (!scaler_file.is_open()) {
        throw std::runtime_error("Failed to open scaler_params.json");
    }
    scaler_file >> scaler_params;

    // Initialize default correction factors for unknown categories
    category_calibration["unknown"] = {0.35, 0.0, 0.7, 1};
    category_calibration["unknown_simple"] = {0.40, 0.0, 0.7, 1};
    category_calibration["unknown_medium"] = {0.35, 0.0, 0.7, 1};
    category_calibration["unknown_complex"] = {0.31, 0.0, 0.7, 1};
}
