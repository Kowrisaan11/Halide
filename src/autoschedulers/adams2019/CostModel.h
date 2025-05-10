/*
  CostModel.h: Abstract base class for cost models in Halide autoscheduler.
*/

#ifndef HALIDE_AUTOSCHEDULER_COST_MODEL_H
#define HALIDE_AUTOSCHEDULER_COST_MODEL_H

#include <string>
#include <cstdint>

#include "Featurization.h"
#include "FunctionDAG.h"
#include "HalideBuffer.h"
#include "PerfectHashMap.h"

namespace Halide {

namespace Internal {
namespace Autoscheduler {

typedef PerfectHashMap<FunctionDAG::Node::Stage, ScheduleFeatures> StageMapOfScheduleFeatures;

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

class CostModel {
public:
    virtual ~CostModel() = default;

    // Configure the cost model for the algorithm to be scheduled.
    virtual void set_pipeline_features(const FunctionDAG& dag,
                                      const Adams2019Params& params) = 0;

    // Enqueue a schedule to be evaluated. Will annotate the value at cost_ptr when evaluation occurs.
    virtual void enqueue(const FunctionDAG& dag,
                        const StageMapOfScheduleFeatures& schedule_feats,
                        double* cost_ptr) = 0;

    // Evaluate all schedules in the queue.
    virtual void evaluate_costs() = 0;

    // Discard all schedules in the queue.
    virtual void reset() = 0;
};

} // namespace Autoscheduler
} // namespace Internal
} // namespace Halide

#endif // HALIDE_AUTOSCHEDULER_COST_MODEL_H
