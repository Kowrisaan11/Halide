#include <torch/script.h>
#include <torch/torch.h>
#include <HalideRuntime.h>

// Global model instance (loaded once)
static std::shared_ptr<torch::jit::Module> g_pytorch_model;
static torch::Device g_device(torch::kCPU);
static bool g_model_loaded = false;

// Helper to load model
void ensure_model_loaded() {
    if (g_model_loaded) return;
    
    try {
        if (torch::cuda::is_available()) {
            g_device = torch::Device(torch::kCUDA, 0);
        }
        
        g_pytorch_model = std::make_shared<torch::jit::Module>(
            torch::jit::load("model.pt"));
        g_pytorch_model->to(g_device);
        g_pytorch_model->eval();
        g_model_loaded = true;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
    }
}

// Function called by Halide
extern "C" int halide_torch_inference(halide_buffer_t *features_buf,
                                     int32_t batch_size,
                                     int32_t num_features,
                                     halide_buffer_t *output_buf) {
    ensure_model_loaded();
    
    if (!g_model_loaded || features_buf == NULL || output_buf == NULL) {
        // Fill with zeros on error
        for (int i = 0; i < batch_size; i++) {
            ((float*)output_buf->host)[i] = 0.0f;
        }
        return 0;
    }
    
    try {
        // Create tensor from buffer
        torch::Tensor features_tensor = torch::from_blob(
            features_buf->host,
            {batch_size, num_features},
            torch::kFloat32);
        
        // Move to device and run inference
        features_tensor = features_tensor.to(g_device);
        torch::NoGradGuard no_grad;
        torch::Tensor output_tensor = g_pytorch_model->forward({features_tensor}).toTensor();
        
        // Copy results back
        output_tensor = output_tensor.to(torch::kCPU);
        auto accessor = output_tensor.accessor<float, 1>();
        for (int i = 0; i < batch_size; i++) {
            ((float*)output_buf->host)[i] = accessor[i];
        }
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Inference error: " << e.what() << std::endl;
        return -1;
    }
}
