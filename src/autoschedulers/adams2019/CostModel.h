#ifndef COST_MODEL_H
#define COST_MODEL_H

#include <string>
#include <vector>
#include <map>
#include <nlohmann/json.hpp>
#include "FunctionDAG.h"
#include <torch/torch.h>
#include <torch/script.h>

using json = nlohmann::json;

namespace Halide {
namespace Internal {
namespace Autoscheduler {

// Constants
const std::vector<std::string> FIXED_FEATURES = {
    "cache_hits", "cache_misses", "execution_time_ms", "sched_num_realizations",
    "sched_num_productions", "sched_points_computed_total", "sched_innermost_loop_extent",
    "sched_inner_parallelism", "sched_outer_parallelism", "sched_bytes_at_realization",
    "sched_bytes_at_production", "sched_bytes_at_root", "sched_unique_bytes_read_per_realization",
    "sched_working_set", "sched_vector_size", "sched_num_vectors", "sched_num_scalars",
    "sched_bytes_at_task", "sched_working_set_at_task", "sched_working_set_at_production",
    "sched_working_set_at_realization", "sched_working_set_at_root", "total_parallelism",
    "scheduling_count", "total_bytes_at_production", "total_vectors", "computation_efficiency",
    "memory_pressure", "memory_utilization_ratio", "bytes_processing_rate", "bytes_per_parallelism",
    "bytes_per_vector", "nodes_count", "edges_count", "node_edge_ratio", "nodes_per_schedule",
    "op_diversity"
};

struct TreeRepresentation {
    json tree_data;
    std::map<std::string, double> extracted_features;
    
    TreeRepresentation() = default;
    TreeRepresentation(const FunctionDAG &dag, const Adams2019Params &params);
    void initialize_from_dag(const FunctionDAG &dag, const Adams2019Params &params);
};

struct PredictionResult {
    double raw_prediction;
    double corrected_prediction;
    std::string category;
    std::map<std::string, double> features;
};

class CostModel {
public:
    virtual ~CostModel() = default;

    virtual void set_pipeline_features(const FunctionDAG &dag,
                                     const Adams2019Params &params) = 0;

    virtual TreeRepresentation convert_to_tree(const FunctionDAG &dag,
                                             const Adams2019Params &params) = 0;

    virtual void enqueue(const FunctionDAG &dag,
                        const StageMapOfScheduleFeatures &schedule_feats,
                        double *cost_ptr) = 0;

    virtual PredictionResult get_prediction(const TreeRepresentation &tree_repr,
                                          bool is_gpu_available) = 0;

    virtual void evaluate_costs() = 0;
    virtual void reset() = 0;

protected:
    virtual std::map<std::string, double> extract_features(const json &json_data) = 0;
    virtual std::string get_file_category(const std::string &file_path, 
                                        const std::map<std::string, double> &features) = 0;
    virtual double compute_complexity_score(const std::map<std::string, double> &features) = 0;

    // Utility function to log messages with timestamp
    void log_message(const std::string& message) const {
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::cout << "[" << std::put_time(std::gmtime(&now_time), "%Y-%m-%d %H:%M:%S UTC") 
                  << "] " << message << std::endl;
    }
};

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide

#endif  // COST_MODEL_H
