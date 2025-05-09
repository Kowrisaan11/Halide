#ifndef HALIDE_COST_MODEL_H
#define HALIDE_COST_MODEL_H

#include "FunctionDAG.h"
#include "Featurization.h"
#include "TreeRepresentation.h"
#include <memory>
#include <string>

namespace Halide {
namespace Internal {
namespace Autoscheduler {

struct Adams2019Params {
    std::string weights_path;
    std::string model_type = "SimpleLSTM";
};

class CostModel {
public:
    virtual ~CostModel() = default;

    virtual void set_pipeline_features(const FunctionDAG &dag,
                                      const Adams2019Params &params) = 0;

    virtual void enqueue(const FunctionDAG &dag,
                         const StageMapOfScheduleFeatures &schedule_feats,
                         const TreeRepresentation &tree_repr,
                         double *cost_ptr) = 0;

    virtual void evaluate_costs() = 0;

    virtual void reset() = 0;
};

} // namespace Autoscheduler
} // namespace Internal
} // namespace Halide

#endif // HALIDE_COST_MODEL_H
