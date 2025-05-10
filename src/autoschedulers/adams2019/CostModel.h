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

// Define FIXED_FEATURES for your model
const std::vector<std::string> FIXED_FEATURES = {
    "cache_hits", "cache_misses", "execution_time_ms", "sched_num_realizations",
    // ... (your FIXED_FEATURES list)
};

struct TreeRepresentation {
    json tree_data;
    std::map<std::string, double> extracted_features;
    
    TreeRepresentation() = default;
    TreeRepresentation(const FunctionDAG &dag);
    void initialize_from_dag(const FunctionDAG &dag);
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

    // Configure the cost model for the algorithm to be scheduled
    virtual void set_pipeline_features(const FunctionDAG &dag) = 0;

    // Convert FunctionDAG to tree representation
    virtual TreeRepresentation convert_to_tree(const FunctionDAG &dag) = 0;

    // Enqueue a DAG for evaluation
    virtual void enqueue(const FunctionDAG &dag, double *cost_ptr) = 0;

    // Get prediction for a specific tree representation
    virtual PredictionResult get_prediction(const TreeRepresentation &tree_repr,
                                          bool is_gpu_available) = 0;

    // Evaluate all DAGs in the queue
    virtual void evaluate_costs() = 0;

    // Discard all DAGs in the queue
    virtual void reset() = 0;

protected:
    virtual std::map<std::string, double> extract_features(const json &json_data) = 0;
    virtual std::string get_file_category(const std::map<std::string, double> &features) = 0;
    virtual double compute_complexity_score(const std::map<std::string, double> &features) = 0;
};

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide

#endif  // COST_MODEL_H
