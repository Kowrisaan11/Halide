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
#include "cost_model_schedule.h"

using namespace Halide;
using json = nlohmann::json;

// Template-based model weight handling, just like in the original code
template<bool training>
struct ModelWeight;

template<>
struct ModelWeight<false> : public GeneratorInput<Buffer<float>> {
    ModelWeight(const std::string &name, int dim)
        : GeneratorInput<Buffer<float>>(name, dim) {
    }
    void backprop(const Derivative &d, const Expr &learning_rate, const Expr &timestep) {
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
    void backprop(const Derivative &d, const Expr &learning_rate, const Expr &timestep) {
        // Implementation from original code
        std::vector<Expr> args(dimensions() + 1);
        for (size_t i = 0; i < args.size(); i++) {
            args[i] = Var();
        }
        grad(args) = undef<float>();

        args.back() = 0;
        FuncRef new_weight = grad(args);
        args.back() = 1;
        FuncRef smoothed_deriv = grad(args);
        args.back() = 2;
        FuncRef smoothed_second_moment = grad(args);
        args.back() = 3;
        FuncRef loss_gradient = grad(args);

        args.pop_back();
        Expr current_weight = (*this)(args);

        loss_gradient = d(*this)(args);

        smoothed_deriv = 0.9f * smoothed_deriv + 0.1f * loss_gradient;
        smoothed_second_moment = 0.999f * smoothed_second_moment + 0.001f * pow(loss_gradient, 2);

        Expr smoothed_deriv_correction = 1 / (1 - pow(0.9f, timestep + 1));
        Expr smoothed_second_moment_correction = 1 / (1 - pow(0.999f, timestep + 1));

        Expr step = learning_rate * smoothed_deriv * smoothed_deriv_correction;
        step /= sqrt(smoothed_second_moment * smoothed_second_moment_correction) + 1e-5f;

        new_weight = current_weight - step;
    }

    void set_shape(int s0 = 0, int s1 = 0, int s2 = 0) {
        // Implementation from original code
        if (s0) {
            dim(0).set_bounds(0, s0);
            dim(0).set_estimate(0, s0);
            grad.dim(0).set_bounds(0, s0);
            grad.dim(0).set_estimate(0, s0);
            grad.bound(grad.args()[0], 0, s0);
            grad.set_estimate(grad.args()[0], 0, s0);
        }
        if (s1) {
            dim(1).set_bounds(0, s1);
            dim(1).set_estimate(0, s1);
            grad.dim(1).set_bounds(0, s1);
            grad.dim(1).set_estimate(0, s1);
            grad.bound(grad.args()[1], 0, s1);
            grad.set_estimate(grad.args()[1], 0, s1);
        }
        if (s2) {
            dim(2).set_bounds(0, s2);
            dim(2).set_estimate(0, s2);
            grad.dim(2).set_bounds(0, s2);
            grad.dim(2).set_estimate(0, s2);
            grad.bound(grad.args()[2], 0, s2);
            grad.set_estimate(grad.args()[2], 0, s2);
        }
        grad.dim(dimensions()).set_bounds(0, 4);
        grad.dim(dimensions()).set_estimate(0, 4);
    }
};

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

    // Batch size
    Input<int> batch_size{"batch_size", 1};

    // Number of cores on the target machine
    Input<int> num_cores{"num_cores", 1};

    // Algorithm-specific features
    Input<Buffer<float>> pipeline_features{"pipeline_features", 3};

    // Schedule-specific features
    Input<Buffer<float>> schedule_features{"schedule_features", 3};

    // Network weights
    using Weight = ModelWeight<training>;
    Weight head1_filter{"head1_filter", 3};
    Weight head1_bias{"head1_bias", 1};
    Weight head2_filter{"head2_filter", 2};
    Weight head2_bias{"head2_bias", 1};
    Weight filter1{"filter1", 2};
    Weight bias1{"bias1", 1};

    // Some extra inputs for training mode
    Input<float> learning_rate{"learning_rate", 1.0f};
    Input<int> timestep{"timestep", 0};

    // The index of the fastest schedule in the batch
    Input<int> reference{"reference", 0};

    // The true runtimes obtained by benchmarking
    Input<Buffer<float>> true_runtime{"true_runtime", 1};

    // The predicted runtimes
    Output<Buffer<float>> prediction_output{"prediction_output", 1};

    // The loss - 0-dimensional buffer output
    Output<Buffer<float>> loss_output{"loss_output", 0};

    // PyTorch model and device
    std::shared_ptr<torch::jit::Module> pytorch_model;
    torch::Device device;

    // Network dimensions (should match NetworkSize.h)
    const int head1_channels = 8;
    const int head1_w = 3;
    const int head1_h = 3;
    const int head2_channels = 8;
    const int head2_w = 32;
    const int conv1_channels = 1;

    // Constructor
    CostModel() : device(torch::kCPU) {
        // Force CPU usage regardless of CUDA availability
        std::cout << "Using CPU for model " << (training ? "training" : "inference") << std::endl;

        // Set environment variable to disable CUDA
        setenv("CUDA_VISIBLE_DEVICES", "", 1);

        // Load the PyTorch model
        try {
            pytorch_model = std::make_shared<torch::jit::Module>(torch::jit::load("/home/kowrisaan/fyp/Halide/src/autoschedulers/adams2019/model.pt", device));
            if (training) {
                pytorch_model->train();
            } else {
                pytorch_model->eval();
            }
        } catch (const c10::Error& e) {
            std::cerr << "Error loading the PyTorch model: " << e.what() << std::endl;
            // Don't throw, just continue with a warning
            std::cerr << "Continuing without model - will use default cost model" << std::endl;
        }
    }

    void generate() {
        Var c("c"), w("w"), n("n"), j("j"), s("s");

        // Initialize all inputs with proper estimates FIRST
        num_cores.set_estimate(32);
        reference.set_estimate(0);
        batch_size.set_estimate(80);
        num_stages.set_estimate(13);
        learning_rate.set_estimate(0.001f);
        timestep.set_estimate(37);
        
        // Initialize network dimensions
        head1_filter.set_shape(head1_channels, head1_w, head1_h);
        head1_bias.set_shape(head1_channels);
        head2_filter.set_shape(head2_channels, head2_w);
        head2_bias.set_shape(head2_channels);
        filter1.set_shape(conv1_channels, head1_channels + head2_channels);
        bias1.set_shape(conv1_channels);
        
        // Set estimates for buffer inputs using proper Halide Region syntax
        // Fix: Convert std::vector<std::pair<int, int>> to Region (std::vector<Range>)
        {
            Region estimates;
            estimates.push_back(Range(0, head1_w));
            estimates.push_back(Range(0, head1_h));
            estimates.push_back(Range(0, 13)); // num_stages value
            pipeline_features.set_estimates(estimates);
        }
        
        {
            Region estimates;
            estimates.push_back(Range(0, 80)); // batch_size value
            estimates.push_back(Range(0, head2_w));
            estimates.push_back(Range(0, 13)); // num_stages value
            schedule_features.set_estimates(estimates);
        }
        
        {
            Region estimates;
            estimates.push_back(Range(0, 80)); // batch_size value
            true_runtime.set_estimates(estimates);
        }
        
        // Set estimates for buffer outputs
        {
            Region estimates;
            estimates.push_back(Range(0, 80)); // batch_size value
            prediction_output.set_estimates(estimates);
        }
        
        // 0-dimensional buffer needs empty estimates
        {
            Region estimates;
            loss_output.set_estimates(estimates);
        }

        // Implement the original cost model logic
        Func normalized_schedule_features("normalized_schedule_features");
        normalized_schedule_features(n, c, s) = log(max(1e-20f, schedule_features(n, c, s) + 1));

        // Force the weights of the algorithm embedding layer to be positive and bounded
        Func squashed_head1_filter("squashed_head1_filter");
        squashed_head1_filter(c, s, n) = sigmoid(head1_filter(c, s, n));

        // The conv layer that embeds the algorithm-specific features
        Func head1_conv("head1_conv");
        RDom r_head1(0, head1_w, 0, head1_h);
        head1_conv(c, w) = head1_bias(c);
        head1_conv(c, w) += (squashed_head1_filter(c, r_head1.x, r_head1.y) *
                            pipeline_features(r_head1.x, r_head1.y, w));

        // The conv layer that embeds the schedule-specific features
        Func head2_conv("head2_conv");
        RDom r_head2(0, head2_w);
        head2_conv(c, w, n) = head2_bias(c);
        head2_conv(c, w, n) += head2_filter(c, r_head2) * normalized_schedule_features(n, r_head2, w);

        Func head2_relu("head2_relu");
        head2_relu(c, w, n) = max(head2_conv(c, w, n), 0.0f);

        // The conv layer that computes coefficients
        Func conv1_stage1("conv1_stage1");
        RDom r1_stage1(0, head1_channels);
        conv1_stage1(c, w) = bias1(c);
        conv1_stage1(c, w) += filter1(c, r1_stage1.x) * head1_conv(r1_stage1.x, w);

        Func conv1_stage2("conv1_stage2");
        RDom r1_stage2(0, head2_channels);
        conv1_stage2(c, w, n) = conv1_stage1(c, w);
        conv1_stage2(c, w, n) += filter1(c, head1_channels + r1_stage2.x) * head2_relu(r1_stage2.x, w, n);

        Func relu1("relu1");
        relu1(c, w, n) = max(conv1_stage2(c, w, n), 0.0f);

        // Calculate the cost using the coefficients
        Func runtime_per_stage("runtime_per_stage");
        runtime_per_stage(n, w) = 1.0f + relu1(0, w, n); // Basic cost model

        // Sum across the stages
        Func prediction("prediction");
        RDom r_reduce(0, num_stages);
        prediction(n) = sum(runtime_per_stage(n, r_reduce));

        prediction_output(n) = prediction(n);

        // Define loss function
        Func loss_func("loss_func");
        if (!training) {
            // For inference mode, just set loss to zero
            loss_func() = 0.0f;
        } else {
            // Training mode logic - compute mean squared error
            Func squared_error("squared_error");
            squared_error(n) = pow(prediction_output(n) - true_runtime(n), 2);
            
            // Sum over the batch using a reduction domain
            RDom r_batch(0, batch_size);
            loss_func() = sum(squared_error(r_batch)) / cast<float>(batch_size);
            
            // If using backpropagation in a real implementation, you would add that here
        }
        
        loss_output() = loss_func();

        // SCHEDULE
        if (training && !using_autoscheduler()) {
            do_cost_model_schedule(get_pipeline());
        } else if (using_autoscheduler()) {
            // Do nothing
        } else {
            // Inference schedule
            Var no;
            prediction_output.specialize(batch_size < 8).split(n, no, n, 1);
            prediction_output.compute_root().split(n, no, n, 8).parallel(no);
            prediction_output.bound(n, 0, batch_size);
        }
    }

private:
    Expr sigmoid(const Expr &e) {
        return 1.0f / (1.0f + exp(-e));
    }
};

using CostModelInference = CostModel<false>;
using CostModelTraining = CostModel<true>;

HALIDE_REGISTER_GENERATOR(CostModelInference, cost_model);
HALIDE_REGISTER_GENERATOR(CostModelTraining, train_cost_model);
