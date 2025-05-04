#include "Halide.h"
#include "cost_model_schedule.h"
#include "NetworkSize.h"

using namespace Halide;

// Template for inference/training
template <bool training>
class CostModel : public Generator<CostModel<training>> {
public:
    // Inputs
    GeneratorInput<Buffer<float>> pipeline_features{"pipeline_features", 3};
    GeneratorInput<Buffer<float>> schedule_features{"schedule_features", 3};
    GeneratorInput<Buffer<float>> true_runtime{"true_runtime", 1};
    GeneratorInput<int> batch_size{"batch_size", 1};
    GeneratorInput<int> num_stages{"num_stages", 1};

    // Outputs
    GeneratorOutput<Buffer<float>> prediction_output{"prediction_output", 1};
    GeneratorOutput<Buffer<float>> loss_output{"loss_output", 0};

    void generate() {
        Var n("n");

        // Dummy prediction: just sum the first feature dimension for each batch
        Func pred("pred");
        RDom r(0, pipeline_features.dim(0).extent());
        pred(n) = sum(pipeline_features(r, 0, 0)); // Just a dummy op

        prediction_output(n) = pred(n);

        // Dummy loss: squared error over batch
        Func loss("loss");
        RDom rb(0, 80); // Use the same constant as set_estimate below
        loss() = 0.0f;
        loss() += pow(prediction_output(rb) - true_runtime(rb), 2);

        loss_output = loss;

        // Set estimates (must use constants!)
        pipeline_features.set_estimates({{0, 3}, {0, 3}, {0, 13}});
        schedule_features.set_estimates({{0, 80}, {0, 3}, {0, 13}});
        true_runtime.set_estimates({{0, 80}});
        batch_size.set_estimate(80);
        num_stages.set_estimate(13);
        prediction_output.set_estimates({{0, 80}});
        loss_output.set_estimates({}); // 0-dimensional
    }
};

using CostModelInference = CostModel<false>;
using CostModelTraining = CostModel<true>;

HALIDE_REGISTER_GENERATOR(CostModelInference, cost_model)
HALIDE_REGISTER_GENERATOR(CostModelTraining, train_cost_model)
