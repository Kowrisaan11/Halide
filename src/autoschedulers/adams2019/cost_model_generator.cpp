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
#include "NetworkSize.h"

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

// Cost model generator that matches the function signature expected by DefaultCostModel.cpp
class CostModel : public Generator<CostModel> {
protected:
    bool allow_out_of_order_inputs_and_outputs() const override {
        return true;
    }

public:
    // Inputs - match the signature in DefaultCostModel.cpp:evaluate_costs()
    GeneratorInput<int> num_stages{"num_stages", 1};
    GeneratorInput<int> batch_size{"batch_size", 1};
    GeneratorInput<int> num_cores{"num_cores", 1};
    GeneratorInput<Buffer<float>> pipeline_feat_queue{"pipeline_feat_queue", 3};
    GeneratorInput<Buffer<float>> schedule_feat_queue{"schedule_feat_queue", 3};
    GeneratorInput<Buffer<float>> head1_filter{"head1_filter", 2};
    GeneratorInput<Buffer<float>> head1_bias{"head1_bias", 1};
    GeneratorInput<Buffer<float>> head2_filter{"head2_filter", 2};
    GeneratorInput<Buffer<float>> head2_bias{"head2_bias", 1};
    GeneratorInput<Buffer<float>> conv1_filter{"conv1_filter", 4};
    GeneratorInput<Buffer<float>> conv1_bias{"conv1_bias", 1};
    GeneratorInput<float> learning_rate{"learning_rate", 0.0f};
    GeneratorInput<int> timestep{"timestep", 0};
    GeneratorInput<int> fastest_idx{"fastest_idx", 0};
    GeneratorInput<Buffer<float>> true_runtimes{"true_runtimes", 1};

    // Outputs
    GeneratorOutput<Buffer<float>> costs{"costs", 1};
    GeneratorOutput<Buffer<float>> loss{"loss", 0};

    // PyTorch model and device
    std::shared_ptr<torch::jit::Module> pytorch_model;
    torch::Device device;

    // Constructor
    CostModel() : device(torch::kCPU) {
        // Force CPU usage regardless of CUDA availability
        std::cout << "Using CPU for model inference" << std::endl;

        // Set environment variable to disable CUDA
        setenv("CUDA_VISIBLE_DEVICES", "", 1);

        // Load the PyTorch model
        try {
            pytorch_model = std::make_shared<torch::jit::Module>(torch::jit::load("/home/kowrisaan/fyp/Halide/src/autoschedulers/adams2019/model.pt", device));
            pytorch_model->eval();
        } catch (const c10::Error& e) {
            std::cerr << "Error loading the PyTorch model: " << e.what() << std::endl;
            // Don't throw, just continue with a warning
            std::cerr << "Continuing without model - will use default cost model" << std::endl;
        }
    }

    void generate() {
        Var n("n");
        
        // Implement the cost model logic here
        // This should match what DefaultCostModel.cpp expects in evaluate_costs()
        
        // For now, just provide a simple implementation
        costs(n) = 1.0f;  // Default cost
        loss() = 0.0f;    // Default loss
        
        // Set estimates
        num_stages.set_estimate(1);
        batch_size.set_estimate(1);
        num_cores.set_estimate(8);
        costs.set_estimates({{0, batch_size}});
        
        // Simple scheduling
        costs.compute_root();
    }
};

// Train cost model generator that matches the function signature expected by DefaultCostModel.cpp
class TrainCostModel : public Generator<TrainCostModel> {
protected:
    bool allow_out_of_order_inputs_and_outputs() const override {
        return true;
    }

public:
    // Inputs - match the signature in DefaultCostModel.cpp:backprop()
    GeneratorInput<int> num_stages{"num_stages", 1};
    GeneratorInput<int> batch_size{"batch_size", 1};
    GeneratorInput<int> num_cores{"num_cores", 1};
    GeneratorInput<Buffer<float>> pipeline_feat_queue{"pipeline_feat_queue", 3};
    GeneratorInput<Buffer<float>> schedule_feat_queue{"schedule_feat_queue", 3};
    GeneratorInput<Buffer<float>> head1_filter{"head1_filter", 2};
    GeneratorInput<Buffer<float>> head1_bias{"head1_bias", 1};
    GeneratorInput<Buffer<float>> head2_filter{"head2_filter", 2};
    GeneratorInput<Buffer<float>> head2_bias{"head2_bias", 1};
    GeneratorInput<Buffer<float>> conv1_filter{"conv1_filter", 4};
    GeneratorInput<Buffer<float>> conv1_bias{"conv1_bias", 1};
    GeneratorInput<float> learning_rate{"learning_rate", 0.001f};
    GeneratorInput<int> timestep{"timestep", 0};
    GeneratorInput<int> fastest_idx{"fastest_idx", 0};
    GeneratorInput<Buffer<float>> true_runtimes{"true_runtimes", 1};
    
    // Outputs
    GeneratorOutput<Buffer<float>> head1_filter_update{"head1_filter_update", 3};
    GeneratorOutput<Buffer<float>> head1_bias_update{"head1_bias_update", 2};
    GeneratorOutput<Buffer<float>> head2_filter_update{"head2_filter_update", 3};
    GeneratorOutput<Buffer<float>> head2_bias_update{"head2_bias_update", 2};
    GeneratorOutput<Buffer<float>> conv1_filter_update{"conv1_filter_update", 5};
    GeneratorOutput<Buffer<float>> conv1_bias_update{"conv1_bias_update", 2};
    GeneratorOutput<Buffer<float>> costs{"costs", 1};
    GeneratorOutput<Buffer<float>> loss{"loss", 0};

    // PyTorch model and device
    std::shared_ptr<torch::jit::Module> pytorch_model;
    torch::Device device;

    // Constructor
    TrainCostModel() : device(torch::kCPU) {
        // Force CPU usage regardless of CUDA availability
        std::cout << "Using CPU for model training" << std::endl;

        // Set environment variable to disable CUDA
        setenv("CUDA_VISIBLE_DEVICES", "", 1);

        // Load the PyTorch model
        try {
            pytorch_model = std::make_shared<torch::jit::Module>(torch::jit::load("/home/kowrisaan/fyp/Halide/src/autoschedulers/adams2019/model.pt", device));
            // Set to training mode
            pytorch_model->train();
        } catch (const c10::Error& e) {
            std::cerr << "Error loading the PyTorch model for training: " << e.what() << std::endl;
            // Don't throw, just continue with a warning
            std::cerr << "Continuing without model - will use default training" << std::endl;
        }
    }

    void generate() {
        Var n("n");
        
        // Simple implementation for the training generator
        // This should match what DefaultCostModel.cpp expects in backprop()
        
        // For now, just provide dummy outputs
        head1_filter_update(n, m, b) = 0.0f;
        head1_bias_update(n, b) = 0.0f;
        head2_filter_update(n, m, b) = 0.0f;
        head2_bias_update(n, b) = 0.0f;
        conv1_filter_update(n, m, r, s, b) = 0.0f;
        conv1_bias_update(n, b) = 0.0f;
        costs(n) = 0.0f;
        loss() = 0.0f;
        
        // Set estimates
        num_stages.set_estimate(1);
        batch_size.set_estimate(32);
        num_cores.set_estimate(8);
        costs.set_estimates({{0, batch_size}});
        
        std::cout << "TrainCostModel generator called - this is a placeholder implementation" << std::endl;
    }

private:
    Var n{"n"}, m{"m"}, r{"r"}, s{"s"}, b{"b"};
};

// Register both generators
HALIDE_REGISTER_GENERATOR(CostModel, cost_model)
HALIDE_REGISTER_GENERATOR(TrainCostModel, train_cost_model)
