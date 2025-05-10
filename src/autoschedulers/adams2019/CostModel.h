#ifndef HALIDE_AUTOSCHEDULER_COST_MODEL_H
#define HALIDE_AUTOSCHEDULER_COST_MODEL_H

#include "Halide.h"
#include "TreeRepresentation.h"

namespace Halide {
namespace Internal {
namespace Autoscheduler {

class State;

class CostModel {
public:
    virtual ~CostModel() = default;

    // Evaluate the cost of a given state
    virtual double evaluate_cost(const IntrusivePtr<State> &state) = 0;

    // Update calibration data (optional, for models that support online learning)
    virtual void update_calibration_data(const std::map<std::string, double> &features,
                                        double predicted_cost, double actual_cost) {
        // Default implementation: do nothing
    }
};

} // namespace Autoscheduler
} // namespace Internal
} // namespace Halide

#endif // HALIDE_AUTOSCHEDULER_COST_MODEL_H
