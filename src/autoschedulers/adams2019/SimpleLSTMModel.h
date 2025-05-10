/*
  SimpleLSTMModel.h: Header for SimpleLSTMModel class.
  Implements a cost model using an LSTM-based PyTorch model for the Halide autoscheduler.
*/

#ifndef HALIDE_AUTOSCHEDULER_SIMPLE_LSTM_MODEL_H
#define HALIDE_AUTOSCHEDULER_SIMPLE_LSTM_MODEL_H

#include "CostModel.h"
#include "TreeRepresentation.h"
#include "State.h" // Include State.h for State type
#include <torch/script.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <map>

namespace Halide {
namespace Internal {
namespace Autoscheduler {

class SimpleLSTMModel : public CostModel {
public:
    struct CategoryCorrection {
        double slope{1.0};
        double offset{0.0};
    };

    SimpleLSTMModel(const std::string &model_path, const std::string &scaler_params_path);
    std::map<std::string, double> extract_features(const TreeRepresentation &tree, const FunctionDAG &dag);
    double evaluate_cost(const IntrusivePtr<State> &state, const FunctionDAG &dag) override;
    void update_calibration_data(const std::map<std::string, double> &features, double predicted_cost, double actual_cost);

private:
    torch::jit::script::Module model_;
    std::map<std::string, std::pair<double, double>> scaler_params_;
    std::map<std::string, std::pair<double, double>> file_calibration_;
    std::map<std::string, CategoryCorrection> category_calibration_;
    std::map<std::string, double> extract_node_features(const TreeRepresentation::Node &node);
    torch::Tensor prepare_input(const std::map<std::string, double> &features);
    double normalize_feature(const std::string &key, double value);
    double predict_cost(const std::map<std::string, double> &features);
    double apply_calibration(double predicted_cost, const std::map<std::string, double> &features);
};

} // namespace Autoscheduler
} // namespace Internal
} // namespace Halide

#endif // HALIDE_AUTOSCHEDULER_SIMPLE_LSTM_MODEL_H
