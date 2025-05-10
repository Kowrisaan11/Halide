#ifndef COST_MODEL_H
#define COST_MODEL_H

#include <string>
#include <vector>
#include <map>
#include <nlohmann/json.hpp>
#include <torch/torch.h>
#include <torch/script.h>

using json = nlohmann::json;

namespace Halide {
namespace Internal {
namespace Autoscheduler {

struct TreeRepresentation {
    json tree_data;
    std::map<std::string, double> extracted_features;
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
    virtual void enqueue(const json &dag_data, double *cost_ptr) = 0;
    virtual PredictionResult get_prediction(const TreeRepresentation &tree_repr,
                                          bool is_gpu_available) = 0;
    virtual void evaluate_costs() = 0;
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
