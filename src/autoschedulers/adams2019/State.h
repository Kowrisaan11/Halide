/*
  State.h: Header for State class in Adams2019 autoscheduler.
  Represents a state in the beam search with a TreeRepresentation root.
*/

#ifndef STATE_H
#define STATE_H

#include "Halide.h"
#include "TreeRepresentation.h"
#include "SimpleLSTMModel.h"
#include "FunctionDAG.h"
#include <string>
#include <vector>
#include <functional>

namespace Halide {
namespace Internal {
namespace Autoscheduler {

struct State {
    IntrusivePtr<TreeRepresentation> root;
    IntrusivePtr<State> parent;
    double cost = 0.0;
    int num_decisions_made = 0;
    bool penalized = false;
    std::string schedule_source;
    mutable RefCount ref_count;

    State() = default;

    void generate_children(const FunctionDAG &dag,
                          const Adams2019Params &params,
                          SimpleLSTMModel *cost_model,
                          const std::function<void(IntrusivePtr<State> &&)> &enqueue);

    void apply_schedule(const FunctionDAG &dag, const Adams2019Params &params);

    void save_featurization(const FunctionDAG &dag,
                            const Adams2019Params &params,
                            std::ostream &out);

    uint64_t structural_hash(int depth) const;

    void dump(std::ostream &os) const;
};

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide

#endif // STATE_H
