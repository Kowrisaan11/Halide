#ifndef AUTOSCHEDULE_H
#define AUTOSCHEDULE_H

#include "CostModel.h"
#include "GraphRepresentation.h"
#include "Halide.h"
#include "PerfectHashMap.h"
#include <vector>

namespace Halide {
namespace Internal {
namespace Autoscheduler {

typedef PerfectHashMap<GraphRepresentation::Node::Stage, ScheduleFeatures> StageMapOfScheduleFeatures;

void find_and_apply_schedule(GraphRepresentation& graph, const std::vector<Function>& outputs,
                             const Adams2019Params& params, CostModel* cost_model,
                             int beam_size, StageMapOfScheduleFeatures* schedule_features);

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide

#endif  // AUTOSCHEDULE_H
