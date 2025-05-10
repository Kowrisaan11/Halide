/*
  AutoSchedule.h: Interface for Adams2019 autoscheduler.
*/

#ifndef HALIDE_AUTOSCHEDULER_AUTO_SCHEDULE_H
#define HALIDE_AUTOSCHEDULER_AUTO_SCHEDULE_H

#include "CostModel.h"
#include "FunctionDAG.h"
#include "SimpleLSTMModel.h"
#include <random>
#include <string>
#include <vector>

namespace Halide {
namespace Internal {
namespace Autoscheduler {

struct AutoSchedulerResults {
    std::string schedule_source;
    std::vector<char> featurization;
    AutoschedulerParams autoscheduler_params;
};

struct CachingOptions {
    bool cache_blocks = false;
    bool cache_features = false;

    static CachingOptions MakeOptionsFromParams(const Adams2019Params &params) {
        CachingOptions options;
        options.cache_features = !params.disable_memoized_features;
        options.cache_blocks = !params.disable_memoized_blocks;
        return options;
    }
};

void generate_schedule(const std::vector<Function> &outputs,
                       const Target &target,
                       const Adams2019Params &params,
                       AutoSchedulerResults *auto_scheduler_results);

void find_and_apply_schedule(FunctionDAG &dag,
                             const std::vector<Function> &outputs,
                             const Adams2019Params &params,
                             CostModel *cost_model,
                             std::map<std::string, ScheduleFeatures> *schedule_features);

} // namespace Autoscheduler
} // namespace Internal
} // namespace Halide

#endif // HALIDE_AUTOSCHEDULER_AUTO_SCHEDULE_H
