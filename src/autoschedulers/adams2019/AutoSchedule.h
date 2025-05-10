/*
  AutoSchedule.h: Header for the Adams2019 autoscheduler.
  Declares functions and structures for generating schedules using SimpleLSTMModel.
*/

#ifndef HALIDE_AUTOSCHEDULER_AUTO_SCHEDULE_H
#define HALIDE_AUTOSCHEDULER_AUTO_SCHEDULE_H

#include "Halide.h"
#include "SimpleLSTMModel.h"
#include <vector>
#include <string>

namespace Halide {
namespace Internal {
namespace Autoscheduler {

struct Adams2019Params {
    int beam_size{10};
    int max_samples{1000};
    int random_seed{0};
    double learning_rate{0.01};
    int verbosity{0};
};

struct AutoSchedulerResults {
    std::string schedule_source;
    std::vector<uint8_t> featurization;
};

void generate_schedule(const std::vector<Function> &outputs,
                       const Target &target,
                       const Adams2019Params &params,
                       AutoSchedulerResults *auto_scheduler_results);

} // namespace Autoscheduler
} // namespace Internal
} // namespace Halide

#endif // HALIDE_AUTOSCHEDULER_AUTO_SCHEDULE_H
