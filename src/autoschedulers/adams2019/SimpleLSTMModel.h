#ifndef SIMPLE_LSTM_MODEL_H
#define SIMPLE_LSTM_MODEL_H

#include "CostModel.h"
#include "TreeRepresentation.h"
#include <torch/script.h>
#include <vector>
#include <string>
#include <memory>

namespace Halide {
namespace Internal {
namespace Autoscheduler {

class SimpleLSTMModel : public CostModel {
public:
    explicit SimpleLSTMModel(const std::string& weights_path);
    ~SimpleLSTMModel() override = default;

    void set_pipeline_features(const FunctionDAG& dag,
                              const Adams2019Params& params) override;

    void enqueue(const FunctionDAG& dag,
                 const StageMapOfScheduleFeatures& schedule_feats,
                 const TreeRepresentation& tree_repr,
                 double* cost_ptr) override;

    void evaluate_costs() override;

    void reset() override;

private:
    std::string weights_path_;
    torch::jit::script::Module model_;
    torch::Device device_;
    std::vector<std::pair<TreeRepresentation, double*>> evaluation_queue_;
    std::vector<std::string> feature_columns_;
    std::vector<double> X_scalar_center_;
    std::vector<double> X_scalar_scale_;
    double y_center_;
    double y_scale_;
};

std::unique_ptr<CostModel> make_simple_lstm_model(const std::string& weights_path);

} // namespace Autoscheduler
} // namespace Internal
} // namespace Halide

#endif // SIMPLE_LSTM_MODEL_H
