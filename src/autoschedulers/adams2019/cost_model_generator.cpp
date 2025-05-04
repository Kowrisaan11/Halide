#include "Halide.h"
#include "cost_model_schedule.h"
#include "NetworkSize.h"

using namespace Halide;

// Template for inference/training
template <bool training>
// In your CostModel generator class:
class CostModel : public Generator<CostModel> {
public:
    // Input buffers
    Input<Buffer<float>> pipeline_features{"pipeline_features", 3};
    Input<Buffer<float>> schedule_features{"schedule_features", 3};
    Input<Buffer<float>> true_runtime{"true_runtime", 1};
    
    // Weight buffers
    Input<Buffer<float>> head1_filter{"head1_filter", 2};
    Input<Buffer<float>> head1_bias{"head1_bias", 1};
    Input<Buffer<float>> head2_filter{"head2_filter", 2};
    Input<Buffer<float>> head2_bias{"head2_bias", 1};
    Input<Buffer<float>> conv1_filter{"conv1_filter", 2};
    Input<Buffer<float>> conv1_bias{"conv1_bias", 1};

    // Scalar parameters
    Input<int32_t> batch_size{"batch_size", 1};
    Input<int32_t> num_stages{"num_stages", 1};
    Input<int32_t> num_cores{"num_cores", 1};
    Input<float> learning_rate{"learning_rate", 1.0f};
    Input<int32_t> timestep{"timestep", 0};

    // Output buffers
    Output<Buffer<float>> prediction_output{"prediction_output", 1};
    Output<Buffer<float>> loss_output{"loss_output", 1};

    void generate() {
        // Your implementation here
        // ... (maintain existing pipeline construction code)

        // Set estimates for ALL inputs
        pipeline_features.set_estimates({{0, 256}, {0, 256}, {0, 3}});
        schedule_features.set_estimates({{0, 256}, {0, 256}, {0, 3}});
        true_runtime.set_estimates({{0, 256}});
        head1_filter.set_estimates({{0, 64}, {0, 3}});
        head1_bias.set_estimates({{0, 64}});
        // ... set estimates for other weights
        batch_size.set_estimate(256);
        num_stages.set_estimate(3);
        num_cores.set_estimate(8);
    }
};

using CostModelInference = CostModel<false>;
using CostModelTraining = CostModel<true>;

HALIDE_REGISTER_GENERATOR(CostModelInference, cost_model)
HALIDE_REGISTER_GENERATOR(CostModelTraining, train_cost_model)
