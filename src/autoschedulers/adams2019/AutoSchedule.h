/*
  AutoSchedule.h: Header for Adams2019 autoscheduler with SimpleLSTMModel integration.
  Defines parameters and interface for generating schedules using beam search
  and a PyTorch-based LSTM cost model.
*/

#ifndef AUTOSCHEDULE_H
#define AUTOSCHEDULE_H

#include "Halide.h"
#include <string>
#include <vector>

namespace Halide {
namespace Internal {
namespace Autoscheduler {

struct Adams2019Params {
    int beam_size = 32;
    int random_dropout = 80;
    int random_dropout_seed = 0;
    bool disable_subtiling = false;
    int parallelism = 4;
    int64_t memory_limit = 1ULL << 30;

    Adams2019Params() = default;
};

void generate_schedule(const std::vector<Function> &outputs,
                       const Target &target,
                       const Adams2019Params &params,
                       AutoSchedulerResults *auto_scheduler_results);

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide

#endif // AUTOSCHEDULE_H
