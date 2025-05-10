/*
  AutoSchedule.h: Header for the adams2019 autoscheduler.
  Declares the function to generate schedules for a FunctionDAG.
*/

#ifndef HALIDE_AUTOSCHEDULER_AUTO_SCHEDULE_H
#define HALIDE_AUTOSCHEDULER_AUTO_SCHEDULE_H

#include "FunctionDAG.h"
#include "MachineParams.h"

namespace Halide {
namespace Internal {
namespace Autoscheduler {

void generate_function_schedule(FunctionDAG &dag, const MachineParams &params);

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide

#endif  // HALIDE_AUTOSCHEDULER_AUTO_SCHEDULE_H
