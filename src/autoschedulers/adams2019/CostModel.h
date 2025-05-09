/*
  CostModel.h: Abstract base class for cost models in the Halide autoscheduler.
  Defines the interface for evaluating the cost of a schedule state.
*/

#ifndef HALIDE_AUTOSCHEDULER_COST_MODEL_H
#define HALIDE_AUTOSCHEDULER_COST_MODEL_H

#include "FunctionDAG.h"
#include "Halide.h"

namespace Halide {
namespace Internal {
namespace Autoscheduler {

// Forward declaration of TreeRepresentation
class TreeRepresentation;

class CostModel {
public:
    virtual ~CostModel() = default;

    // Evaluate the cost of a given schedule state for a function DAG.
    // Returns the estimated cost as a double.
    virtual double evaluate_cost(const IntrusivePtr<State> &state, const FunctionDAG &dag) = 0;
};

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide

#endif  // HALIDE_AUTOSCHEDULER_COST_MODEL_H
