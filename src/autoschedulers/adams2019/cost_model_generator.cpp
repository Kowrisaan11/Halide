#include "DefaultCostModel.h"
#include "ASLog.h"
#include "CostModel.h"
#include "FunctionDAG.h"
#include "LoopNest.h"
#include "State.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "adams2019_cost_model.h"
#include "adams2019_train_cost_model.h"

namespace Halide {

using namespace Internal;

DefaultCostModel::DefaultCostModel(const std::vector<float> &weights, bool use_updated_model)
    : weights(weights), use_updated_model(use_updated_model), timestep(0) {
}

float DefaultCostModel::evaluate_cost(const FunctionDAG &dag, const MachineParams &params, const LoopNest &loop_nest, const State &state, int cursor) {
    // Placeholder implementation for cost evaluation
    // This method is not directly related to the error
    return 0.0f;
}

float DefaultCostModel::backprop(const Runtime::Buffer<const float, -1, 4> &pipeline_features, float learning_rate) {
    // Ensure buffers are initialized
    if (!schedule_features.is_defined() || !predictions.is_defined() || !loss.is_defined()) {
        aslog(1) << "Buffers not initialized in DefaultCostModel::backprop\n";
        return 0.0f;
    }

    const int num_stages = pipeline_features.dim(3).extent();
    const int batch_size = schedule_features.dim(0).extent();
    const int num_cores = params.num_cores;

    // Find the index of the fastest schedule
    int fastest_idx = 0;
    float min_runtime = std::numeric_limits<float>::infinity();
    for (int i = 0; i < batch_size; i++) {
        if (true_runtimes(i) < min_runtime) {
            min_runtime = true_runtimes(i);
            fastest_idx = i;
        }
    }

    // Call train_cost_model without weight update buffers
    int result = train_cost_model(
        num_stages,
        batch_size,
        num_cores,
        pipeline_features.raw_buffer(),
        schedule_features.raw_buffer(),
        weights.head1_filter.raw_buffer(),
        weights.head1_bias.raw_buffer(),
        weights.head2_filter.raw_buffer(),
        weights.head2_bias.raw_buffer(),
        weights.conv1_filter.raw_buffer(),
        weights.conv1_bias.raw_buffer(),
        learning_rate,
        timestep++,
        fastest_idx,
        true_runtimes.raw_buffer(),
        predictions.raw_buffer(),
        loss.raw_buffer());

    if (result != 0) {
        aslog(1) << "train_cost_model failed with error code: " << result << "\n";
        return 0.0f;
    }

    // Return the loss
    return loss(0);
}

void DefaultCostModel::set_pipeline_features(const FunctionDAG &dag, const MachineParams &params) {
    // Placeholder implementation
}

void DefaultCostModel::enqueue(const LoopNest &loop_nest, const State &state, int idx, Runtime::Buffer<float, -1, 4> *sched_feats) {
    // Placeholder implementation
}

void DefaultCostModel::set_weights(const std::vector<float> &new_weights) {
    // Placeholder implementation
}

}  // namespace Halide
