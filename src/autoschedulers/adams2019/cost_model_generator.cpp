#include <torch/script.h>
#include <torch/torch.h>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>
#include <chrono>
#include <cmath>

#include "Halide.h"

using namespace Halide;

// Define FIXED_FEATURES as in the Python code
const std::vector<std::string> FIXED_FEATURES = {
    "cache_hits", "cache_misses", "execution_time_ms", "sched_num_realizations",
    // ...many more features (shortened for clarity)
};


// Hardware correction factors struct
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

// Cost model generator class
class CostModel : public Generator<CostModel> {
protected:
    bool allow_out_of_order_inputs_and_outputs() const override {
        return true;
    }

public:
    // Inputs - defined separately with correct types
    Input<int> batch_size{"batch_size", 1};
    Input<Buffer<float>> features_input{"features_input", 2};  // [batch_size x num_features]
    Input<float> actual_runtime{"actual_runtime", -1.0f};

    // Output
    Output<Buffer<float>> prediction_output{"prediction_output", 1};

    // PyTorch model and device
    std::shared_ptr<torch::jit::Module> pytorch_model;
    torch::Device device = torch::Device(torch::kCPU); // Initialize with valid value

    // Define variables for dimensions
    Var n{"n"}, f{"f"};

    CostModel() {
        // Check for GPU availability
        bool is_gpu_available = torch::cuda::is_available();
        device = is_gpu_available ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);
        std::cout << (is_gpu_available ? "Using GPU" : "Using CPU") << std::endl;

        // Initialize PyTorch model (try-catch is retained)
        try {
            pytorch_model = std::make_shared<torch::jit::Module>(torch::jit::load("model.pt"));
            pytorch_model->to(device);
            pytorch_model->eval();
            std::cout << "Model loaded successfully" << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
        }
    }

    void generate() {
        // Access batch_size using its internal value
        Expr batch_size_val = batch_size;
        
        // Create a simple implementation for demonstration
        Func model_output("model_output");
        // Simple placeholder computation (replace with actual model integration)
        model_output(n) = sum(features_input(n, f)) / 100.0f;
        
        // Apply any post-processing
        Func processed_prediction("processed_prediction");
        processed_prediction(n) = model_output(n);
        
        // Set the final output
        prediction_output(n) = processed_prediction(n);
        
        // Set estimates for autoscheduling
        // Note: using Vars for dimensions, not integers
        features_input.set_estimates({{0, batch_size}, {0, cast<int>(FIXED_FEATURES.size())}});
        actual_runtime.set_estimate(0.0f);
        prediction_output.set_estimates({{0, batch_size}});
        
        // Simple schedule
        prediction_output.compute_root().parallel(n);
    }
};

HALIDE_REGISTER_GENERATOR(CostModel, cost_model)
