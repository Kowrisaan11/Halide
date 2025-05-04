// GNNCostModel.cpp

#include <torch/script.h>
#include <Halide.h>
#include <vector>
#include <iostream>

using namespace Halide;

// Helpers: Convert Halide Buffers to torch::Tensor
static torch::Tensor buffer_to_tensor(const Buffer<float> &buf) {
    // buf: [dim0, dim1] -> tensor of shape {dim0, dim1}
    int64_t D0 = buf.dim(0).extent();
    int64_t D1 = buf.dim(1).extent();
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor t = torch::empty({D0, D1}, options);
    for (int64_t i = 0; i < D0; i++) {
        for (int64_t j = 0; j < D1; j++) {
            t[i][j] = buf(i, j);
        }
    }
    return t;
}

static torch::Tensor buffer_to_long_tensor(const Buffer<int> &buf) {
    // buf: [2, num_edges] -> tensor of shape {2, num_edges}
    int64_t D0 = buf.dim(0).extent();
    int64_t D1 = buf.dim(1).extent();
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    torch::Tensor t = torch::empty({D0, D1}, options);
    for (int64_t i = 0; i < D0; i++) {
        for (int64_t j = 0; j < D1; j++) {
            t[i][j] = buf(i, j);
        }
    }
    return t;
}


// Halide Generator

template<bool training>
class GNNCostModel : public Generator<GNNCostModel<training>> {
public:
    // Inputs
    Input<int>           batch_size{     "batch_size",      1};
    Input<int>           num_nodes{      "num_nodes",       1};
    Input<Buffer<float>> node_features{  "node_features",   2};
    Input<Buffer<int>>   edge_index{     "edge_index",      2};
    Input<Buffer<float>> true_runtime{   "true_runtime",    1};  // optional

    // Outputs
    Output<Buffer<float>> prediction_output{ "prediction_output", 1 };
    Output<Buffer<float>> loss_output{       "loss_output",        0 };

    // Path to your scripted GNN model
    std::string model_path = "scripted_model_new.pt";
    torch::jit::script::Module pytorch_model;

    // Load the scripted GNN
    void load_pytorch_model() {
        try {
            pytorch_model = torch::jit::load(model_path);
            pytorch_model.eval();
        } catch (const std::exception &e) {
            std::cerr << "Error loading GNN model: " << e.what() << std::endl;
            exit(1);
        }
    }

    void generate() {
        // On AOT‐compile, load model for inference mode
        if (!training) {
            load_pytorch_model();
        }

        Var g("g");  // graph index in batch (we assume batch_size==1 for simplicity)

        Func pred("pred");
        pred(g) = 0.0f;               // initialize

        if (!training) {
            // 1) Pull Halide Buffers
            Buffer<float>  x_buf = node_features.get();
            Buffer<int>    e_buf = edge_index.get();

            // 2) Convert to torch::Tensor
            torch::Tensor x_t = buffer_to_tensor(x_buf);   // [num_nodes, num_features]
            torch::Tensor e_t = buffer_to_long_tensor(e_buf); // [2, num_edges]

            // 3) Forward through GNN
            std::vector<torch::jit::IValue> inputs = { x_t, e_t };
            torch::Tensor out = pytorch_model.forward(inputs).toTensor();  // [num_nodes, 1]

            // 4) Pool node‐predictions to a scalar per graph
            float graph_pred = out.mean().item<float>();

            // 5) Emit to Halide
            pred(g) = graph_pred;
            prediction_output(g) = pred(g);
            loss_output() = 0.0f;
        } else {
            // training‐mode stub (no real backprop here)
            pred(g) = 0.0f;
            prediction_output(g) = pred(g);
            loss_output() = pow(prediction_output(0) - true_runtime(0), 2);
        }

        
        // Estimates for Halide autoscheduler
        batch_size.set_estimate(1);
        num_nodes.set_estimate(100);
        prediction_output.set_estimates({{0, batch_size}});
        true_runtime.set_estimates({{0, batch_size}});
    }
};

// Register generator variants
using GNNCostInference = GNNCostModel<false>;
using GNNCostTraining  = GNNCostModel<true>;

HALIDE_REGISTER_GENERATOR(GNNCostInference, gnn_cost_model)
HALIDE_REGISTER_GENERATOR(GNNCostTraining,  train_gnn_cost_model)
