cat > /home/kowrisaan/fyp/Halide/src/autoschedulers/adams2019/AutoSchedule.h << 'EOF'
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
    int beam_size = 32;              // Number of states to track in beam search
    int random_dropout = 80;         // Percentage chance to keep states (0-100)
    int random_dropout_seed = 0;     // Seed for random dropout
    bool disable_subtiling = false;  // Disable subtiling in scheduling
    int parallelism = 4;             // Number of threads for parallel execution
    int64_t memory_limit = 1ULL << 30; // Memory limit in bytes (default 1GB)

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
EOF
