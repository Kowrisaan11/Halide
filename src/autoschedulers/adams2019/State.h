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

    void save_featurization(const GraphRepresentation& graph,
                            const Adams2019Params& params,
                            const CachingOptions& cache_options,
                            std::ostream& out) const;

    void compute_featurization(const GraphRepresentation& graph,
                              const Adams2019Params& params,
                              StageMap<ScheduleFeatures>* features,
                              const CachingOptions& cache_options) const;

    void calculate_cost(const GraphRepresentation& graph,
                        const Adams2019Params& params,
                        CostModel* cost_model,
                        const CachingOptions& cache_options,
                        int verbosity = 0);

    void generate_children(const GraphRepresentation& graph,
                          const Adams2019Params& params,
                          CostModel* cost_model,
                          const std::function<void(IntrusivePtr<State>&&)>& enqueue,
                          Cache* cache);

    void apply_schedule(const GraphRepresentation& graph,
                        const Adams2019Params& params);

    uint64_t structural_hash(int depth) const;

    void dump(std::ostream& os) const;
};

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide

#endif  // STATE_H
