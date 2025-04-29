#ifndef LOOP_NEST_H
#define LOOP_NEST_H

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "Cache.h"
#include "CostModel.h"
#include "Errors.h"
#include "GraphRepresentation.h"
#include "Halide.h"
#include "NetworkSize.h"

namespace Halide {
namespace Internal {
namespace Autoscheduler {

struct Adams2019Params;

struct LoopNest {
    mutable RefCount ref_count;
    std::vector<IntrusivePtr<const LoopNest>> children;
    std::vector<const GraphRepresentation::Node*> nodes;
    std::vector<std::pair<std::string, int>> stage_ids;
    std::vector<int64_t> size;
    std::map<int, int> stage_to_stages_idx;
    std::string store_at;
    std::string parallel_var;
    int64_t parallel_size = 0;
    int vectorized_stage = -1;
    int vector_dim = -1;
    int64_t vectorized_size = 0;
    bool is_gpu_loop = false;
    bool vectorized = false;
    bool parallel = false;
    bool inlined = false;
    bool all_stages_scheduled = false;

    uint64_t hash_of_producers_stored_at_root = 0;

    LoopNest() = default;
    LoopNest(const LoopNest& other) = delete;
    LoopNest& operator=(const LoopNest& other) = delete;
    LoopNest(LoopNest&& other) = delete;
    LoopNest& operator=(LoopNest&& other) = delete;

    void copy_from(const LoopNest& other, bool copy_children);

    bool is_root() const;
    void dump(std::ostream& os, const std::string& indent = "") const;

    void compute_features(const GraphRepresentation& graph,
                         const std::vector<std::pair<std::string, int>>& stage_ids,
                         const Adams2019Params& params,
                         StageMap<ScheduleFeatures>* features,
                         const CachingOptions& cache_options,
                         FeatureIntermediates* intermediates) const;

    bool get_bounds(const GraphRepresentation& graph,
                   const GraphRepresentation::Node* node,
                   const std::vector<std::pair<std::string, int>>& stage_ids,
                   GraphRepresentation::Bound* bound,
                   int64_t* bytes_at_root,
                   int64_t* inner_bytes_at_root) const;

    void set_bounds(const GraphRepresentation& graph,
                    const GraphRepresentation::Node* node,
                    const GraphRepresentation::Bound& bound);

    bool should_be_inlined(const GraphRepresentation& graph,
                          const GraphRepresentation::Node* node,
                          const Adams2019Params& params) const;

    bool apply(const GraphRepresentation& graph,
               const std::vector<std::pair<std::string, int>>& stage_ids,
               const Adams2019Params& params,
               std::vector<std::string>* schedule_source) const;

    uint64_t structural_hash(int depth) const;

    int max_stages(const GraphRepresentation& graph) const;

    IntrusivePtr<LoopNest> make_tiled(const GraphRepresentation& graph,
                                      const GraphRepresentation::Node* node,
                                      int64_t tile_size,
                                      const Adams2019Params& params,
                                      bool at_root = false) const;

    IntrusivePtr<LoopNest> make_inlined(const GraphRepresentation& graph,
                                        const GraphRepresentation::Node* node,
                                        const Adams2019Params& params) const;

    IntrusivePtr<LoopNest> make_serialized(const GraphRepresentation& graph,
                                           const GraphRepresentation::Node* node,
                                           int64_t tile_size,
                                           const Adams2019Params& params) const;

    IntrusivePtr<LoopNest> make_parallelized(const GraphRepresentation& graph,
                                             const GraphRepresentation::Node* node,
                                             int64_t tile_size,
                                             const Adams2019Params& params) const;

    IntrusivePtr<LoopNest> make_vectorized(const GraphRepresentation& graph,
                                           const GraphRepresentation::Node* node,
                                           int vector_dim,
                                           const Adams2019Params& params) const;

    IntrusivePtr<LoopNest> make_gpu_threads(const GraphRepresentation& graph,
                                            const GraphRepresentation::Node* node,
                                            int block_factor,
                                            int thread_factor,
                                            const Adams2019Params& params) const;

    void get_stages(std::vector<std::pair<std::string, int>>* stages) const;
};

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide

#endif  // LOOP_NEST_H
