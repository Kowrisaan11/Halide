#pragma once

#include "FunctionDAG.h"
#include "CostModel.h"
#include <vector>
#include <random>

namespace Halide {
namespace Internal {
namespace Autoscheduler {

// Forward declaration
struct Adams2019Params;
struct AutoSchedulerResults;
class Cache;
struct State;
struct LoopNest;
struct CachingOptions;

// Top-level schedule generation
void generate_schedule(const std::vector<Function> &outputs,
                       const Target &target,
                       const Adams2019Params &params,
                       AutoSchedulerResults *auto_scheduler_results);

// Run search and apply a schedule; optionally return featurization
void find_and_apply_schedule(FunctionDAG &dag,
                             const std::vector<Function> &outputs,
                             const Adams2019Params &params,
                             CostModel *cost_model,
                             StageMapOfScheduleFeatures *schedule_features);

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide
