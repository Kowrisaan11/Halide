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

uint64_t State::structural_hash(int depth) const {
    uint64_t h = num_decisions_made;
    internal_assert(root.defined());
    root->structural_hash(h, depth);
    return h;
}

void State::compute_featurization(const FunctionDAG &dag, const Adams2019Params &params,
                                  StageMap<ScheduleFeatures> *features, const CachingOptions &cache_options) {
    StageMap<LoopNest::Sites> sites;
    sites.make_large(dag.nodes[0].stages[0].max_id);
    features->make_large(dag.nodes[0].stages[0].max_id);
    internal_assert(root.defined());
    root->get_sites(sites);

    // For the input nodes and unscheduled outputs, the compute
    // and store sites are root, and the produce and innermost
    // sites are unset (nullptr)
    for (const auto &n : dag.nodes) {
        if (n.is_input || n.is_output) {
            for (const auto &stage : n.stages) {
                auto &s = sites.get_or_create(&stage);
                if (s.compute == nullptr) {
                    s.compute = root.get();
                    s.store = root.get();
                }
            }
        }
    }

    // For the unscheduled nodes, give them sites as deep as they
    // could possibly be. We'll ignore the possibility of inlining
    // them for now.
    map<const LoopNest *, pair<const LoopNest *, int>> parent;
    compute_loop_nest_parents(parent, root.get(), 0);
    for (const auto &n : dag.nodes) {
        if (sites.contains(&(n.stages[0]))) {
            continue;
        }
        const LoopNest *loop = nullptr;
        for (const auto *e : n.outgoing_edges) {
            const auto &consumer_site = sites.get(e->consumer);
            const LoopNest *l = consumer_site.innermost;
            if (!l) {
                l = consumer_site.compute;
            }
            if (!l) {
                std::ostringstream err;
                dump(err);
                err << e->producer->func.name() << " -> " << e->consumer->name << "\n";
                internal_error << err.str();
            }
            if (loop) {
                loop = deepest_common_ancestor(parent, l, loop);
            } else {
                loop = l;
            }
        }
        internal_assert(loop)
            << "Could not compute plausible site for unscheduled Func: "
            << n.func.name() << "\n";
        for (const auto &stage : n.stages) {
            auto &site = sites.get_or_create(&stage);
            site.compute = loop;
            site.store = loop;
        }
    }

    if (cache_options.cache_features) {
        // Store unique hashes for each Site, to be used as keys into cache
        for (const auto &c : root->children) {
            sites.get(c->stage).hash_of_producers_stored_at_root = c->compute_hash_of_producers_stored_at_root(sites);
        }
    }

    root->compute_features(dag, params, sites, 1, 1, nullptr, nullptr, *root, nullptr, features, cache_options.cache_features);

    for (const auto &n : dag.nodes) {
        if (sites.get(&(n.stages[0])).produce == nullptr) {
            internal_assert(!features->contains(&(n.stages[0])))
                << "Somehow an input or unscheduled node ended up in the featurization: "
                << n.func.name() << "\n";
        }
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

void State::generate_children(const FunctionDAG &dag,
                              const Adams2019Params &params,
                              CostModel *cost_model,
                              std::function<void(IntrusivePtr<State> &&)> &accept_child,
                              Cache *cache) const {

    internal_assert(root.defined() && root->is_root()) << "generate_children needs defined root\n";

    if (num_decisions_made == 2 * (int)dag.nodes.size()) {
        return;
    }

    int next_node = num_decisions_made / 2;
    int phase = num_decisions_made % 2;

    if (params.disable_subtiling) {
        // When emulating the older search space, we do all
        // parallelizing last, so that it is independent of the
        // tiling decisions.
        next_node = num_decisions_made % dag.nodes.size();
        phase = num_decisions_made / dag.nodes.size();
    }

    // Enumerate all legal ways to schedule the next Func
    const FunctionDAG::Node *node = &dag.nodes[next_node];
    for (const auto *e : node->outgoing_edges) {
        internal_assert(root->computes(e->consumer->node))
            << "Partially scheduled code doesn't compute " << e->consumer->name
            << ", which is one of the consumers of " << node->func.name();
    }

    if (node->is_input) {
        // We don't need to schedule nodes that represent inputs,
        // and there are no other decisions to be made about them
        // at this time.
        // aslog(1) << "Skipping over scheduling input node: " << node->func.name() << "\n";
        auto child = make_child();
        child->num_decisions_made++;
        accept_child(std::move(child));
        return;
    }

    if (!node->outgoing_edges.empty() && !root->calls(node)) {
        std::ostringstream err;
        err << "In state:\n";
        dump(err);
        err << node->func.name() << " is consumed by:\n";
        for (const auto *e : node->outgoing_edges) {
            err << e->consumer->name << "\n";
            err << "Which in turn consumes:\n";
            for (const auto *e2 : e->consumer->incoming_edges) {
                err << "  " << e2->producer->func.name() << "\n";
            }
        }
        err << "Pipeline so far doesn't use next Func: " << node->func.name() << "\n";
        internal_error << err.str();
    }

    int num_children = 0;

    if (phase == 0) {
        // Injecting realizations
        {
            // 1) Inline it
            if (node->stages.size() == 1 && !node->is_output) {
                auto child = make_child();
                LoopNest *new_root = new LoopNest;
                new_root->copy_from(*root);
                new_root->inline_func(node);
                child->root = new_root;
                child->num_decisions_made++;
                if (child->calculate_cost(dag, params, cost_model, cache->options)) {
                    num_children++;
                    accept_child(std::move(child));
                }
            }
        }

        // Some search-space pruning. If a node is pointwise, and
        // so are all its inputs and so is its sole output, and
        // inlining it is legal, just inline it. This saves time
        // on long chains of pointwise things.
        bool must_inline = (node->is_pointwise &&
                            (num_children > 0) &&
                            (node->outgoing_edges.size() == 1));
        if (must_inline) {
            for (const auto *e : node->stages[0].incoming_edges) {
                must_inline &= e->producer->is_pointwise;
            }
            for (const auto *e : node->outgoing_edges) {
                must_inline &= (e->consumer->node->is_pointwise ||
                                e->consumer->node->is_boundary_condition);
            }
            if (must_inline) {
                return;
            }
        }

        // Construct a list of plausible dimensions to vectorize
        // over. Currently all of them. TODO: Pre-prune the list
        // of sane dimensions to vectorize a Func over to reduce
        // branching factor.
        vector<int> vector_dims;
        if (!node->is_input && !node->is_output) {
            for (int v = 0; v < node->dimensions; v++) {
                const auto &p = root->get_bounds(node)->region_computed(v);
                if (p.extent() >= node->vector_size) {
                    vector_dims.push_back(v);
                }
            }
        }

        // Outputs must be vectorized over their innermost
        // dimension, because we don't have control of the
        // storage. Infer which dimension(s) is(are) the innermost one(s) by
        // looking at the stride. Note that there can be more than one in
        // case some dimensions have an extent of 1.
        if (node->is_output && !node->func.output_buffers().empty()) {
            const Parameter &output = node->func.output_buffers()[0];
            int num_dims = output.dimensions();
            for (int i = 0; i < num_dims; ++i) {
                const Expr stride = output.stride_constraint(i);
                const int64_t *s = as_const_int(stride);
                if (s && *s == 1) {
                    vector_dims.push_back(i);
                }
            }
        }

        if (vector_dims.empty()) {
            // This can happen if the output strides aren't known, or if all
            // the dimensions are smaller than the vector size.
            // TBD: consider extending compute_in_tiles to support -1 as a
            // vector dim to indicate no vectorization.
            for (int v = 0; v < node->dimensions; v++) {
                vector_dims.push_back(v);
            }
            // Handle the case of full reductions that generate a scalar.
            // We need at least one vector dimension to call compute_in_tiles
            // below.
            // TBD: figure out a better fallback strategy.
            if (vector_dims.empty()) {
                vector_dims.push_back(0);
            }
        }

        // 2) Realize it somewhere
        for (int vector_dim : vector_dims) {
            auto tile_options = root->compute_in_tiles(node, nullptr, params, vector_dim, false);
            for (IntrusivePtr<const LoopNest> &n : tile_options) {

                if (root->max_inlined_calls() >= 7) {
                    continue;
                }
                
                auto child = make_child();
                child->root = std::move(n);
                child->num_decisions_made++;
                if (child->calculate_cost(dag, params, cost_model, cache->options)) {
                    num_children++;
                    accept_child(std::move(child));
                }
            }
        }
    } else {
        // We are parallelizing the loops of the func we just injected a realization for.

        bool should_parallelize = false;
        const vector<int64_t> *pure_size = nullptr;
        if (params.parallelism > 1) {
            for (const auto &c : root->children) {
                if (c->node == node && node->dimensions > 0) {
                    if (c->stage->index == 0) {
                        pure_size = &(c->size);
                    }
                }
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