#include "LoopNest.h"
#include "ASLog.h"
#include "CostModel.h"
#include "GraphRepresentation.h"
#include <algorithm>
#include <sstream>

namespace Halide {
namespace Internal {
namespace Autoscheduler {

namespace {

std::string to_string(const GraphRepresentation::Span& s) {
    std::ostringstream os;
    os << "[" << s.min() << ", " << s.max() << "]";
    return os.str();
}

}  // namespace

void LoopNest::copy_from(const LoopNest& other, bool copy_children) {
    nodes = other.nodes;
    stage_ids = other.stage_ids;
    size = other.size;
    stage_to_stages_idx = other.stage_to_stages_idx;
    store_at = other.store_at;
    parallel_var = other.parallel_var;
    parallel_size = other.parallel_size;
    vectorized_stage = other.vectorized_stage;
    vector_dim = other.vector_dim;
    vectorized_size = other.vectorized_size;
    is_gpu_loop = other.is_gpu_loop;
    vectorized = other.vectorized;
    parallel = other.parallel;
    inlined = other.inlined;
    all_stages_scheduled = other.all_stages_scheduled;
    hash_of_producers_stored_at_root = other.hash_of_producers_stored_at_root;
    if (copy_children) {
        children.clear();
        for (const auto& c : other.children) {
            IntrusivePtr<LoopNest> child = new LoopNest;
            child->copy_from(*c, true);
            children.push_back(child);
        }
    }
}

bool LoopNest::is_root() const {
    return nodes.empty() && stage_ids.empty() && !inlined && !vectorized && !parallel;
}

void LoopNest::dump(std::ostream& os, const std::string& indent) const {
    os << indent << "LoopNest:\n";
    if (!nodes.empty()) {
        os << indent << "  Nodes: ";
        for (const auto* n : nodes) {
            os << n->func.name() << " ";
        }
        os << "\n";
    }
    if (!stage_ids.empty()) {
        os << indent << "  Stages: ";
        for (const auto& s : stage_ids) {
            os << s.first;
            if (s.second != 0) os << ".update(" << s.second << ")";
            os << " ";
        }
        os << "\n";
    }
    if (!size.empty()) {
        os << indent << "  Size: ";
        for (int64_t s : size) {
            os << s << " ";
        }
        os << "\n";
    }
    if (!store_at.empty()) {
        os << indent << "  Store at: " << store_at << "\n";
    }
    if (parallel) {
        os << indent << "  Parallel: " << parallel_var << " (" << parallel_size << ")\n";
    }
    if (vectorized) {
        os << indent << "  Vectorized: stage " << vectorized_stage << ", dim " << vector_dim << " (" << vectorized_size << ")\n";
    }
    if (is_gpu_loop) {
        os << indent << "  GPU loop\n";
    }
    if (inlined) {
        os << indent << "  Inlined\n";
    }
    for (const auto& c : children) {
        c->dump(os, indent + "  ");
    }
}

void LoopNest::compute_features(const GraphRepresentation& graph,
                               const std::vector<std::pair<std::string, int>>& stage_ids,
                               const Adams2019Params& params,
                               StageMap<ScheduleFeatures>* features,
                               const CachingOptions& cache_options,
                               FeatureIntermediates* intermediates) const {
    for (const auto& s : stage_ids) {
        ScheduleFeatures& f = features->get_or_create({nullptr, 0});
        for (const auto& n : graph.nodes) {
            for (size_t i = 0; i < n.stages.size(); i++) {
                if (n.stages[i].name == s.first && (int)i == s.second) {
                    f = n.stages[i].features;
                    break;
                }
            }
        }
    }
    for (const auto& c : children) {
        c->compute_features(graph, stage_ids, params, features, cache_options, intermediates);
    }
}

bool LoopNest::get_bounds(const GraphRepresentation& graph,
                         const GraphRepresentation::Node* node,
                         const std::vector<std::pair<std::string, int>>& stage_ids,
                         GraphRepresentation::Bound* bound,
                         int64_t* bytes_at_root,
                         int64_t* inner_bytes_at_root) const {
    *bound = GraphRepresentation::Bound();
    bound->resize(node->func.dimensions());
    for (int i = 0; i < node->func.dimensions(); i++) {
        (*bound)[i] = GraphRepresentation::Span::empty_span();
    }
    for (const auto& c : children) {
        GraphRepresentation::Bound child_bound;
        if (c->get_bounds(graph, node, stage_ids, &child_bound, bytes_at_root, inner_bytes_at_root)) {
            for (size_t i = 0; i < bound->size(); i++) {
                (*bound)[i].union_with(child_bound[i]);
            }
        }
    }
    return true;
}

void LoopNest::set_bounds(const GraphRepresentation& graph,
                         const GraphRepresentation::Node* node,
                         const GraphRepresentation::Bound& bound) {
    // No-op for now, as bounds are computed dynamically
}

bool LoopNest::should_be_inlined(const GraphRepresentation& graph,
                                const GraphRepresentation::Node* node,
                                const Adams2019Params& params) const {
    return node->is_pointwise || node->is_boundary_condition;
}

bool LoopNest::apply(const GraphRepresentation& graph,
                    const std::vector<std::pair<std::string, int>>& stage_ids,
                    const Adams2019Params& params,
                    std::vector<std::string>* schedule_source) const {
    for (const auto& n : nodes) {
        std::ostringstream ss;
        ss << n->func.name() << ".compute_at(" << store_at << ")\n";
        schedule_source->push_back(ss.str());
    }
    if (parallel) {
        std::ostringstream ss;
        ss << stage_ids[0].first << ".parallel(" << parallel_var << ")\n";
        schedule_source->push_back(ss.str());
    }
    if (vectorized) {
        std::ostringstream ss;
        ss << stage_ids[vectorized_stage].first << ".vectorize(" << vector_dim << ")\n";
        schedule_source->push_back(ss.str());
    }
    for (const auto& c : children) {
        c->apply(graph, stage_ids, params, schedule_source);
    }
    return true;
}

uint64_t LoopNest::structural_hash(int depth) const {
    uint64_t h = 0;
    for (const auto* n : nodes) {
        h += n->id;
    }
    for (const auto& s : stage_ids) {
        h += std::hash<std::string>{}(s.first) + s.second;
    }
    if (parallel) h += 1;
    if (vectorized) h += 2;
    if (inlined) h += 3;
    for (const auto& c : children) {
        if (depth > 0) {
            h += c->structural_hash(depth - 1);
        }
    }
    return h;
}

int LoopNest::max_stages(const GraphRepresentation& graph) const {
    int max = 0;
    for (const auto* n : nodes) {
        max += (int)n->stages.size();
    }
    for (const auto& c : children) {
        max = std::max(max, c->max_stages(graph));
    }
    return max;
}

IntrusivePtr<LoopNest> LoopNest::make_tiled(const GraphRepresentation& graph,
                                           const GraphRepresentation::Node* node,
                                           int64_t tile_size,
                                           const Adams2019Params& params,
                                           bool at_root) const {
    IntrusivePtr<LoopNest> new_loop = new LoopNest;
    new_loop->copy_from(*this, true);
    new_loop->nodes.push_back(node);
    new_loop->stage_ids.emplace_back(node->func.name(), 0);
    new_loop->size.push_back(tile_size);
    new_loop->store_at = at_root ? "root" : node->func.name();
    return new_loop;
}

IntrusivePtr<LoopNest> LoopNest::make_inlined(const GraphRepresentation& graph,
                                             const GraphRepresentation::Node* node,
                                             const Adams2019Params& params) const {
    IntrusivePtr<LoopNest> new_loop = new LoopNest;
    new_loop->copy_from(*this, true);
    new_loop->nodes.push_back(node);
    new_loop->stage_ids.emplace_back(node->func.name(), 0);
    new_loop->inlined = true;
    new_loop->store_at = "inline";
    return new_loop;
}

IntrusivePtr<LoopNest> LoopNest::make_serialized(const GraphRepresentation& graph,
                                                const GraphRepresentation::Node* node,
                                                int64_t tile_size,
                                                const Adams2019Params& params) const {
    IntrusivePtr<LoopNest> new_loop = new LoopNest;
    new_loop->copy_from(*this, true);
    new_loop->nodes.push_back(node);
    new_loop->stage_ids.emplace_back(node->func.name(), 0);
    new_loop->size.push_back(tile_size);
    return new_loop;
}

IntrusivePtr<LoopNest> LoopNest::make_parallelized(const GraphRepresentation& graph,
                                                  const GraphRepresentation::Node* node,
                                                  int64_t tile_size,
                                                  const Adams2019Params& params) const {
    IntrusivePtr<LoopNest> new_loop = new LoopNest;
    new_loop->copy_from(*this, true);
    new_loop->nodes.push_back(node);
    new_loop->stage_ids.emplace_back(node->func.name(), 0);
    new_loop->size.push_back(tile_size);
    new_loop->parallel = true;
    new_loop->parallel_var = "p";
    new_loop->parallel_size = tile_size;
    return new_loop;
}

IntrusivePtr<LoopNest> LoopNest::make_vectorized(const GraphRepresentation& graph,
                                                const GraphRepresentation::Node* node,
                                                int vector_dim,
                                                const Adams2019Params& params) const {
    IntrusivePtr<LoopNest> new_loop = new LoopNest;
    new_loop->copy_from(*this, true);
    new_loop->nodes.push_back(node);
    new_loop->stage_ids.emplace_back(node->func.name(), 0);
    new_loop->vectorized = true;
    new_loop->vectorized_stage = 0;
    new_loop->vector_dim = vector_dim;
    new_loop->vectorized_size = node->vector_size;
    return new_loop;
}

IntrusivePtr<LoopNest> LoopNest::make_gpu_threads(const GraphRepresentation& graph,
                                                 const GraphRepresentation::Node* node,
                                                 int block_factor,
                                                 int thread_factor,
                                                 const Adams2019Params& params) const {
    IntrusivePtr<LoopNest> new_loop = new LoopNest;
    new_loop->copy_from(*this, true);
    new_loop->nodes.push_back(node);
    new_loop->stage_ids.emplace_back(node->func.name(), 0);
    new_loop->is_gpu_loop = true;
    return new_loop;
}

void LoopNest::get_stages(std::vector<std::pair<std::string, int>>* stages) const {
    stages->insert(stages->end(), stage_ids.begin(), stage_ids.end());
    for (const auto& c : children) {
        c->get_stages(stages);
    }
}

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide
