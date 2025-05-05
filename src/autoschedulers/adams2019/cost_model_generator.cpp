#include <utility>
#include <fstream>
#include <torch/script.h>
#include <nlohmann/json.hpp>

#include "Halide.h"
#include "NetworkSize.h"

using namespace Halide;
using json = nlohmann::json;

// Struct to hold the scaler parameters
struct ScalerParams {
    std::vector<std::string> feature_names;
    std::vector<float> means;
    std::vector<float> scales;
};

// Struct to hold the Y scaler parameters
struct YScalerParams {
    float mean;
    float scale;
    bool is_log_transformed;
};

// A model weight is either just an input, or an input and an output
template<bool training>
struct ModelWeight;

template<>
struct ModelWeight<false> : public GeneratorInput<Buffer<float>> {
    ModelWeight(const std::string &name, int dim)
        : GeneratorInput<Buffer<float>>(name, dim) {
    }
    void backprop(const Derivative &d, const Expr &learning_rate, const Expr &step) {
    }
    void set_shape(int s0 = 0, int s1 = 0, int s2 = 0) {
        if (s0) {
            dim(0).set_bounds(0, s0);
        }
        if (s1) {
            dim(1).set_bounds(0, s1);
        }
        if (s2) {
            dim(2).set_bounds(0, s2);
        }
    }
};

template<>
struct ModelWeight<true> : public GeneratorInput<Buffer<float>> {
    GeneratorOutput<Buffer<float>> grad;

    ModelWeight(const std::string &name, int dim)
        : GeneratorInput<Buffer<float>>(name, dim), grad("updated_" + name, dim + 1) {
    }
    void backprop(const Derivative &d, const Expr &learning_rate, const Expr &step) {
        // This is only used in training mode, which we're not supporting
        // with the custom model
    }

    void set_shape(int s0 = 0, int s1 = 0, int s2 = 0) {
        if (s0) {
            dim(0).set_bounds(0, s0);
            grad.dim(0).set_bounds(0, s0);
        }
        if (s1) {
            dim(1).set_bounds(0, s1);
            grad.dim(1).set_bounds(0, s1);
        }
        if (s2) {
            dim(2).set_bounds(0, s2);
            grad.dim(2).set_bounds(0, s2);
        }
        grad.dim(dimensions()).set_bounds(0, 4);
    }
};

// Helper function to load JSON data
json load_json(const std::string &file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + file_path);
    }
    try {
        json data;
        file >> data;
        return data;
    } catch (const json::exception &e) {
        throw std::runtime_error("JSON parsing error in " + file_path + ": " + e.what());
    }
}

// Load scaler parameters from JSON file
ScalerParams load_scaler_params(const std::string &scaler_path) {
    try {
        json scaler_data = load_json(scaler_path);
        ScalerParams params;
        params.feature_names = scaler_data["feature_names"].get<std::vector<std::string>>();
        params.means = scaler_data["means"].get<std::vector<float>>();
        params.scales = scaler_data["scales"].get<std::vector<float>>();
        return params;
    } catch (const std::exception &e) {
        throw std::runtime_error("Failed to load scaler params from " + scaler_path + ": " + e.what());
    }
}

// Load Y scaler parameters from JSON file
YScalerParams load_y_scaler_params(const std::string &scaler_path) {
    try {
        json scaler_data = load_json(scaler_path);
        YScalerParams params;
        params.mean = scaler_data["mean"].get<float>();
        params.scale = scaler_data["scale"].get<float>();
        params.is_log_transformed = scaler_data["is_log_transformed"].get<bool>();
        return params;
    } catch (const std::exception &e) {
        throw std::runtime_error("Failed to load Y scaler params from " + scaler_path + ": " + e.what());
    }
}

template<bool training>
class CostModel : public Generator<CostModel<training>> {
protected:
    bool allow_out_of_order_inputs_and_outputs() const override {
        return true;
    }

public:
    template<typename T>
    using Input = GeneratorInput<T>;
    template<typename T>
    using Output = GeneratorOutput<T>;
    using Generator<CostModel<training>>::using_autoscheduler;
    using Generator<CostModel<training>>::get_pipeline;

    // Number of pipeline stages
    Input<int> num_stages{"num_stages", 1};

    // Batch size. Every item in the batch is a different schedule for
    // the same algorithm.
    Input<int> batch_size{"batch_size", 1};

    // Number of cores on the target machine. Used to reason about idle cores.
    Input<int> num_cores{"num_cores", 1};

    // Algorithm-specific features
    Input<Buffer<float>> pipeline_features{"pipeline_features", 3};

    // Schedule-specific features
    Input<Buffer<float>> schedule_features{"schedule_features", 3};

    // Network weights (kept for compatibility)
    using Weight = ModelWeight<training>;
    Weight head1_filter{"head1_filter", 3};
    Weight head1_bias{"head1_bias", 1};
    Weight head2_filter{"head2_filter", 2};
    Weight head2_bias{"head2_bias", 1};
    Weight filter1{"filter1", 2};
    Weight bias1{"bias1", 1};

    // Some extra inputs for training mode.
    Input<float> learning_rate{"learning_rate", 1.0f};
    Input<int> timestep{"timestep", 0};  // Needed by ADAM

    // The index of the fastest schedule in the batch. Used as a
    // reference point for computing relative throughput.
    Input<int> reference{"reference", 0};

    // The true runtimes obtained by benchmarking.
    Input<Buffer<float>> true_runtime{"true_runtime", 1};

    // The predicted runtimes
    Output<Buffer<float>> prediction_output{"prediction_output", 1};

    // The loss. L2 on relative throughput.
    Output<Buffer<float>> loss_output{"loss_output", 0};

    // Paths to model and scaler parameters
    std::string model_path = "/home/kowrisaan/fyp/Halide/src/autoschedulers/adams2019/model.pt";
    std::string scaler_params_path = "/home/kowrisaan/fyp/Halide/src/autoschedulers/adams2019/scaler_params.json";

    // Loaded model and scaler parameters
    torch::jit::script::Module pytorch_model;
    ScalerParams scaler_params;
    YScalerParams y_scaler_params;
    
    void load_pytorch_model() {
        try {
            // Load the PyTorch model with explicit CPU mapping
            pytorch_model = torch::jit::load(model_path, torch::kCPU);
            pytorch_model.eval();
            
            // Load the scaler parameters
            scaler_params = load_scaler_params(scaler_params_path);
            
            // Extract y_scaler parameters from the same file or load from a separate file
            y_scaler_params = load_y_scaler_params(scaler_params_path);
        } catch (const std::exception &e) {
            std::cerr << "Error loading model or parameters: " << e.what() << std::endl;
            exit(1);
        }
    }

    // Helper function to extract features from schedule data
    std::map<std::string, float> extract_features(const Func &runtime_per_stage, int n) {
        std::map<std::string, float> features;
        
        // Unpack all of the schedule features as in the original code
        int idx = 0;
        Expr num_realizations = schedule_features(n, idx++, 0);
        Expr num_productions = schedule_features(n, idx++, 0);
        Expr points_computed_per_realization = schedule_features(n, idx++, 0);
        Expr points_computed_per_production = schedule_features(n, idx++, 0);
        Expr points_computed_total = schedule_features(n, idx++, 0);
        Expr points_computed_minimum = schedule_features(n, idx++, 0);
        Expr innermost_loop_extent = schedule_features(n, idx++, 0);
        Expr innermost_pure_loop_extent = schedule_features(n, idx++, 0);
        Expr unrolled_loop_extent = schedule_features(n, idx++, 0);
        Expr inner_parallelism = schedule_features(n, idx++, 0);
        Expr outer_parallelism = schedule_features(n, idx++, 0);
        Expr bytes_at_realization = schedule_features(n, idx++, 0);
        Expr bytes_at_production = schedule_features(n, idx++, 0);
        Expr bytes_at_root = schedule_features(n, idx++, 0);
        Expr innermost_bytes_at_realization = schedule_features(n, idx++, 0);
        Expr innermost_bytes_at_production = schedule_features(n, idx++, 0);
        Expr innermost_bytes_at_root = schedule_features(n, idx++, 0);
        Expr inlined_calls = schedule_features(n, idx++, 0);
        Expr unique_bytes_read_per_realization = schedule_features(n, idx++, 0);
        Expr unique_lines_read_per_realization = schedule_features(n, idx++, 0);
        Expr allocation_bytes_read_per_realization = schedule_features(n, idx++, 0);
        Expr working_set = schedule_features(n, idx++, 0);
        Expr vector_size = schedule_features(n, idx++, 0);
        Expr native_vector_size = schedule_features(n, idx++, 0);
        Expr num_vectors = schedule_features(n, idx++, 0);
        Expr num_scalars = schedule_features(n, idx++, 0);
        Expr scalar_loads_per_vector = schedule_features(n, idx++, 0);
        Expr vector_loads_per_vector = schedule_features(n, idx++, 0);
        Expr scalar_loads_per_scalar = schedule_features(n, idx++, 0);
        Expr bytes_at_task = schedule_features(n, idx++, 0);
        Expr innermost_bytes_at_task = schedule_features(n, idx++, 0);
        Expr unique_bytes_read_per_vector = schedule_features(n, idx++, 0);
        Expr unique_lines_read_per_vector = schedule_features(n, idx++, 0);
        Expr unique_bytes_read_per_task = schedule_features(n, idx++, 0);
        Expr unique_lines_read_per_task = schedule_features(n, idx++, 0);
        Expr working_set_at_task = schedule_features(n, idx++, 0);
        Expr working_set_at_production = schedule_features(n, idx++, 0);
        Expr working_set_at_realization = schedule_features(n, idx++, 0);
        Expr working_set_at_root = schedule_features(n, idx++, 0);
        
        // Map features to the expected format for the custom model
        features["num_realizations"] = num_realizations;
        features["num_productions"] = num_productions;
        features["points_computed_per_realization"] = points_computed_per_realization;
        features["points_computed_per_production"] = points_computed_per_production;
        features["points_computed_total"] = points_computed_total;
        features["inner_parallelism"] = inner_parallelism;
        features["outer_parallelism"] = outer_parallelism;
        features["bytes_at_realization"] = bytes_at_realization;
        features["bytes_at_production"] = bytes_at_production;
        features["bytes_at_root"] = bytes_at_root;
        features["working_set"] = working_set;
        features["num_vectors"] = num_vectors;
        features["num_scalars"] = num_scalars;
        features["bytes_at_task"] = bytes_at_task;
        
        // Add derived features that might be needed by your model
        features["memory_pressure"] = select(bytes_at_production > 0, 
                                    working_set / bytes_at_production, 
                                    0.0f);
        features["bytes_per_vector"] = select(num_vectors > 0,
                                     bytes_at_production / num_vectors,
                                     0.0f);
        features["total_parallelism"] = inner_parallelism * outer_parallelism;
        
        return features;
    }

    // Prepare input tensor using scaler parameters
    torch::Tensor prepare_input_tensor(const std::map<std::string, float> &features) {
        std::vector<float> feature_vector(scaler_params.feature_names.size(), 0.0f);
        
        // Populate feature vector based on feature names in the scaler params
        for (size_t i = 0; i < scaler_params.feature_names.size(); ++i) {
            const std::string &key = scaler_params.feature_names[i];
            if (features.count(key)) {
                feature_vector[i] = features.at(key);
            } else {
                feature_vector[i] = 0.0f; // Default to zero for missing features
            }
        }
        
        // Scale the feature vector
        torch::Tensor x = torch::tensor(feature_vector, torch::kFloat32);
        torch::Tensor means = torch::tensor(scaler_params.means, torch::kFloat32);
        torch::Tensor scales = torch::tensor(scaler_params.scales, torch::kFloat32);
        scales = torch::where(scales == 0, torch::ones_like(scales), scales); // Avoid division by zero
        x = (x - means) / scales;
        
        // Reshape according to your model's expected input shape (e.g., [1, 1, feature_dim] for LSTM)
        torch::Tensor input_tensor = x.reshape({1, 1, -1});
        
        return input_tensor;
    }

    // Run prediction with the PyTorch model
    float run_prediction(const torch::Tensor &input_tensor) {
        torch::NoGradGuard no_grad;
        
        // Ensure model is in evaluation mode
        pytorch_model.eval();
        
        // Move input to CPU (ensure we're not using CUDA)
        torch::Tensor input = input_tensor.to(torch::kCPU);
        
        std::vector<torch::jit::IValue> inputs = {input};
        auto output = pytorch_model.forward(inputs).toTensor();
        
        float prediction = output.item<float>();
        return prediction;
    }

    // Apply inverse transformation to the model output
    float inverse_transform_prediction(float scaled_prediction) {
        float unscaled = scaled_prediction * y_scaler_params.scale + y_scaler_params.mean;
        float result = y_scaler_params.is_log_transformed ? std::exp(unscaled) - 1 : unscaled;
        return result;
    }

    void generate() {
        // Load the PyTorch model and scaler parameters at generation time
        if (!training) {
            load_pytorch_model();
        }
        
        Var n("n"), w("w");
        
        // Define a function to compute the runtime prediction for each stage
        Func runtime_per_stage("runtime_per_stage");
        
        if (!training) {
            // In inference mode, use our custom PyTorch model
            runtime_per_stage(n, w) = 0.0f;  // Initialize
            
            // Since we can't directly use the PyTorch model in Halide code,
            // we'll use a placeholder here and implement the actual prediction logic
            // in a separate step that executes during the Halide runtime
            
            // For now, we'll use a simple placeholder that captures the necessary information
            // for the actual prediction to happen at runtime
            runtime_per_stage(n, w) = schedule_features(n, 0, w) * 0.0f;  // Dummy calculation
            
            // Sum across the stages to get the total runtime
            Func prediction;
            RDom r_reduce(0, num_stages);
            prediction(n) += runtime_per_stage(n, r_reduce);
            
            prediction_output(n) = prediction(n);
            loss_output() = 0.0f;
        } else {
            // In training mode, we don't support the custom model
            // This is just a placeholder implementation
            runtime_per_stage(n, w) = 0.0f;
            Func prediction;
            prediction(n) = 0.0f;
            prediction_output(n) = prediction(n);
            loss_output() = 0.0f;
        }

        // Set up estimates for autoscheduling this pipeline
        num_cores.set_estimate(32);
        reference.set_estimate(0);
        batch_size.set_estimate(80);
        num_stages.set_estimate(13);
        prediction_output.set_estimates({{0, 80}});
        learning_rate.set_estimate(0.001f);
        timestep.set_estimate(37);
        pipeline_features.set_estimates({{0, head1_w}, {0, head1_h}, {0, 13}});
        // Fix the dimension mismatch in schedule_features estimates
        schedule_features.set_estimates({{0, head2_w}, {0, head2_channels}, {0, 13}});
        true_runtime.set_estimates({{0, 80}});

        // All the model weight shapes are statically known
        head1_filter.set_shape(head1_channels, head1_w, head1_h);
        head1_bias.set_shape(head1_channels);
        head2_filter.set_shape(head2_channels, head2_w);
        head2_bias.set_shape(head2_channels);
        filter1.set_shape(conv1_channels, head1_channels + head2_channels);
        bias1.set_shape(conv1_channels);

        // SCHEDULE
        if (using_autoscheduler()) {
            // Do nothing
        } else {
            // Simple schedule for inference
            Var no;
            prediction_output.specialize(batch_size < 8).split(n, no, n, 1);
            prediction_output.compute_root().split(n, no, n, 8).parallel(no);
            prediction_output.bound(n, 0, batch_size);
            
            // In the actual implementation, we would implement the model prediction
            // as an ExternalCode node that calls into the PyTorch runtime
        }
    }
};

// Maintain compatibility with existing code
using CostModelInference = CostModel<false>;
using CostModelTraining = CostModel<true>;

HALIDE_REGISTER_GENERATOR(CostModelInference, cost_model);
HALIDE_REGISTER_GENERATOR(CostModelTraining, train_cost_model);
