#ifndef TORCH_COST_MODEL_H
#define TORCH_COST_MODEL_H

#include "CostModel.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <vector>
#include <map>
#include <mutex>
#include <string>
#include <memory>

namespace Halide {

class TorchCostModel : public CostModel {
public:
    TorchCostModel(const std::string& model_path, 
                  const std::string& scaler_params_path,
                  bool use_gpu = true);
    
    ~TorchCostModel() override = default;

    // Configure the cost model for the algorithm to be scheduled.
    void set_pipeline_features(const Internal::Autoscheduler::FunctionDAG &dag,
                              const Internal::Autoscheduler::Adams2019Params &params) override;

    // Enqueue a schedule to be evaluated
    void enqueue(const Internal::Autoscheduler::FunctionDAG &dag,
                const Internal::Autoscheduler::StageMapOfScheduleFeatures &schedule_feats,
                double *cost_ptr) override;

    // Evaluate all schedules in the queue
    void evaluate_costs() override;

    // Discard all schedules in the queue
    void reset() override;

private:
    // PyTorch model
    torch::jit::script::Module model_;
    torch::Device device_;
    
    // Scaler parameters
    std::vector<double> X_scalar_center_;
    std::vector<double> X_scalar_scale_;
    double y_center_;
    double y_scale_;
    std::vector<std::string> feature_columns_;
    
    // Hardware correction factors
    struct HardwareCorrectionFactors {
        double base_correction;
        double gpu_correction;
        double scaling_factor;
        double min_time_ms;
        double high_threshold_ms;
        double high_scaling;
    };
    
    HardwareCorrectionFactors correction_factors_;
    
    // Queue for batched evaluation
    std::mutex queue_mutex_;
    std::vector<std::pair<std::map<std::string, double>, double*>> queue_;
    
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
        // Additional operation features omitted for brevity
    };
    
    // Helper methods
    std::map<std::string, double> extract_features_from_dag(
        const Internal::Autoscheduler::FunctionDAG &dag,
        const Internal::Autoscheduler::StageMapOfScheduleFeatures &schedule_feats);
    
    double get_raw_prediction(torch::Tensor seq_input, torch::Tensor scalar_input);
    
    double correct_prediction(double raw_prediction);
    
    double compute_complexity_score(const std::map<std::string, double>& features);
};

} // namespace Halide

#endif // TORCH_COST_MODEL_H
