#ifndef STATE_H
#define STATE_H

#include <map>
#include <string>
#include <vector>

#include "Cache.h"
#include "CostModel.h"
#include "GraphRepresentation.h"
#include "Halide.h"
#include "LoopNest.h"
#include "NetworkSize.h"

namespace Halide {
namespace Internal {
namespace Autoscheduler {

struct Adams2019Params;

struct State {
    mutable RefCount ref_count;
    IntrusivePtr<State> parent;
    IntrusivePtr<LoopNest> root;
    IntrusivePtr<LoopNest> current;

    std::map<std::string, std::string> schedule_source;

    double cost = 0;
    int num_decisions_made = 0;
    bool penalized = false;

    State() = default;
    State(const State& other) = delete;
    State& operator=(const State& other) = delete;
    State(State&& other) = delete;
    State& operator=(State&& other) = delete;

    // Calls `compute_featurization` and prints those features to `out`.
    void save_featurization(const GraphRepresentation& graph,
                            const Adams2019Params& params,
                            const CachingOptions& cache_options,
                            std::ostream& out) const;

    // Compute the featurization of this state (based on `root`),
    // and store features in `features`. Defers to `root->compute_features()`.
    void compute_featurization(const GraphRepresentation& graph,
                              const Adams2019Params& params,
                              StageMap<ScheduleFeatures>* features,
                              const CachingOptions& cache_options) const;

    // Performs some pruning to decide if this state is worth queuing in
    // the cost_model. If it is, calls `cost_model->enqueue` to compute the cost.
    void calculate_cost(const GraphRepresentation& graph,
                        const Adams2019Params& params,
                        CostModel* cost_model,
                        const CachingOptions& cache_options,
                        int verbosity = 0);

    // Make a child copy of this state. The loop nest is const (we
    // make mutated copies of it, rather than mutating it), so we can
    // continue to point to the same one and so this is a cheap
    // operation.
    IntrusivePtr<State> make_child() const;

    // Generate the successor states to this state.
    // If they are not pruned by `calculate_cost()`,
    // then calls `accept_child()` on them.
    void generate_children(const GraphRepresentation& graph,
                          const Adams2019Params& params,
                          CostModel* cost_model,
                          std::function<void(IntrusivePtr<State> &&)> &accept_child,
                          Cache* cache) const;

    // Apply the schedule represented by this state to a Halide
    // Pipeline. Also generate source code for the schedule for the
    // user to copy-paste to freeze this schedule as permanent artifact.
    // Also fills `schedule_source`.
    void apply_schedule(const GraphRepresentation& graph,
                        const Adams2019Params& params);

    // Computes a hash of the state structure for caching purposes.
    uint64_t structural_hash(int depth) const;

    // Dumps cost, the `root` LoopNest, and then `schedule_source` to `os`.
    void dump(std::ostream& os) const;
};

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide

#endif  // STATE_H