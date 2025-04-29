#include "State.h"
#include "ASLog.h"
#include "CostModel.h"
#include "GraphRepresentation.h"
#include "LoopNest.h"
#include <sstream>

namespace Halide {
namespace Internal {
namespace Autoscheduler {

namespace {

void dump_stages(const std::vector<std::pair<std::string, int>>& stages, std::ostream& os) {
    bool first = true;
    for (const auto& s : stages) {
        if (!first) os << ", ";
        os << s.first;
        if (s.second != 0) os << ".update(" << s.second << ")";
        first = false;
    }
}

}  // namespace

void State::dump(std::ostream& os) const {
    os << "State cost: " << cost << "\n";
    root->dump(os);
    os << "\n";
}

uint64_t State::structural_hash(int depth) const {
    uint64_t h = root->structural_hash(depth);
    if (parent.defined() && depth > 0) {
        h += parent->structural_hash(depth - 1);
    }
    return h;
}

void State::save_featurization(const GraphRepresentation& graph,
                              const Adams2019Params& params,
                              const CachingOptions& cache_options,
                              std::ostream& out) const {
    StageMap<ScheduleFeatures> features;
    compute_featurization(graph, params, &features, cache_options);
    for (const auto& n : graph.nodes) {
        for (size_t s = 0; s < n.stages.size(); s++) {
            std::string name = n.stages[s].name;
            const auto& f = features.get({n.stages[s].node, (int)s});
            out << name << "\n";
            for (int i = 0; i < ScheduleFeatures::num_features(); i++) {
                out << f.features[i] << " ";
            }
            out << "\n";
        }
    }
}

void State::compute_featurization(const GraphRepresentation& graph,
                                 const Adams2019Params& params,
                                 StageMap<ScheduleFeatures>* features,
                                 const CachingOptions& cache_options) const {
    std::vector<std::pair<std::string, int>> stage_ids;
    root->get_stages(&stage_ids);
    FeatureIntermediates intermediates;
    root->compute_features(graph, stage_ids, params, features, cache_options, &intermediates);
}

void State::calculate_cost(const GraphRepresentation& graph,
                          const Adams2019Params& params,
                          CostModel* cost_model,
                          const CachingOptions& cache_options,
                          int verbosity) {
    static int64_t cost_calculations = 0;
    cost_calculations++;
    StageMap<ScheduleFeatures> features;
    compute_featurization(graph, params, &features, cache_options);
    if (cost_model) {
        cost_model->enqueue(graph, features, &cost);
        if (verbosity > 0) {
            aslog(verbosity) << "Predicted cost: " << cost << "\n";
            for (const auto& n : graph.nodes) {
                for (size_t s = 0; s < n.stages.size(); s++) {
                    const auto& f = features.get({n.stages[s].node, (int)s});
                    aslog(verbosity) << n.stages[s].name << ": ";
                    for (int i = 0; i < ScheduleFeatures::num_features(); i++) {
                        aslog(verbosity) << f.features[i] << " ";
                    }
                    aslog(verbosity) << "\n";
                }
            }
        }
    } else {
        cost = 0;
        for (const auto& n : graph.nodes) {
            for (size_t s = 0; s < n.stages.size(); s++) {
                const auto& f = features.get({n.stages[s].node, (int)s});
                cost += f.features[0];
            }
        }
    }
}

void State::generate_children(const GraphRepresentation& graph,
                             const Adams2019Params& params,
                             CostModel* cost_model,
                             const std::function<void(IntrusivePtr<State>&&)>& enqueue,
                             Cache* cache) {
    std::vector<std::pair<std::string, int>> stage_ids;
    root->get_stages(&stage_ids);

    for (const auto& n : graph.nodes) {
        bool scheduled = false;
        for (const auto& s : stage_ids) {
            if (s.first == n.func.name()) {
                scheduled = true;
                break;
            }
        }
        if (scheduled) continue;

        if (root->should_be_inlined(graph, &n, params)) {
            IntrusivePtr<State> new_state(new State);
            new_state->parent = this;
            new_state->root = root->make_inlined(graph, &n, params);
            new_state->current = new_state->root;
            new_state->num_decisions_made = num_decisions_made + 1;
            new_state->calculate_cost(graph, params, cost_model, cache->options);
            if (cache->should_cache(new_state->structural_hash(0))) {
                enqueue(std::move(new_state));
            }
            continue;
        }

        std::vector<int64_t> tile_sizes = {1, 4, 16, 64, 256};
        for (int64_t tile_size : tile_sizes) {
            IntrusivePtr<State> new_state(new State);
            new_state->parent = this;
            new_state->root = root->make_tiled(graph, &n, tile_size, params);
            new_state->current = new_state->root;
            new_state->num_decisions_made = num_decisions_made + 1;
            new_state->calculate_cost(graph, params, cost_model, cache->options);
            if (cache->should_cache(new_state->structural_hash(0))) {
                enqueue(std::move(new_state));
            }
        }

        IntrusivePtr<State> parallel_state(new State);
        parallel_state->parent = this;
        parallel_state->root = root->make_parallelized(graph, &n, 16, params);
        parallel_state->current = parallel_state->root;
        parallel_state->num_decisions_made = num_decisions_made + 1;
        parallel_state->calculate_cost(graph, params, cost_model, cache->options);
        if (cache->should_cache(parallel_state->structural_hash(0))) {
            enqueue(std::move(parallel_state));
        }

        for (int v = 0; v < n.dimensions; v++) {
            IntrusivePtr<State> vectorized_state(new State);
            vectorized_state->parent = this;
            vectorized_state->root = root->make_vectorized(graph, &n, v, params);
            vectorized_state->current = vectorized_state->root;
            vectorized_state->num_decisions_made = num_decisions_made + 1;
            vectorized_state->calculate_cost(graph, params, cost_model, cache->options);
            if (cache->should_cache(vectorized_state->structural_hash(0))) {
                enqueue(std::move(vectorized_state));
            }
        }
    }
}

void State::apply_schedule(const GraphRepresentation& graph,
                          const Adams2019Params& params) {
    std::vector<std::string> schedule_source;
    std::vector<std::pair<std::string, int>> stage_ids;
    root->get_stages(&stage_ids);
    root->apply(graph, stage_ids, params, &schedule_source);
    for (const auto& s : schedule_source) {
        this->schedule_source[s] = s;
    }
}

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide
