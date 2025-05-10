#ifndef COST_MODEL_H
#define COST_MODEL_H

#include <string>
#include <memory>
#include "Featurization.h"
#include "FunctionDAG.h"
#include "HalideBuffer.h"
#include "PerfectHashMap.h"

// This is the abstract base class for all cost models (including LSTM/ML-based ones).
// To add a new cost model (e.g., using a PyTorch .pt model), inherit from this interface.

namespace Halide {

namespace Internal {
namespace Autoscheduler {

// Type alias for mapping DAG stages to their features.
typedef PerfectHashMap<FunctionDAG::Node::Stage, ScheduleFeatures> StageMapOfScheduleFeatures;

// Parameters for the Adams2019 autoscheduler.
struct Adams2019Params {
    int parallelism = 16;
    int beam_size = 32;
    int random_dropout = 100;
    int random_dropout_seed = 0;
    std::string weights_path;
    int disable_subtiling = 0;
    int disable_memoized_features = 0;
    int disable_memoized_blocks = 0;
    int64_t memory_limit = -1;
};

}  // namespace Autoscheduler
}  // namespace Internal

// Abstract interface for any cost model (NN-based, hand-written, etc).
class CostModel {
public:
    virtual ~CostModel() = default;

    // Called at the start of scheduling for a new pipeline.
    virtual void set_pipeline_features(const Internal::Autoscheduler::FunctionDAG &dag,
                                       const Internal::Autoscheduler::Adams2019Params &params) = 0;

    // Enqueue a candidate schedule for cost evaluation.
    // Will write the estimated cost to *cost_ptr during evaluate_costs().
    virtual void enqueue(const Internal::Autoscheduler::FunctionDAG &dag,
                         const Halide::Internal::Autoscheduler::StageMapOfScheduleFeatures &schedule_feats,
                         double *cost_ptr) = 0;

    // Evaluate all queued schedules (batched for performance).
    virtual void evaluate_costs() = 0;

    // Discard all queued but unevaluated schedules.
    virtual void reset() = 0;
};

}  // namespace Halide

#endif  // COST_MODEL_H
