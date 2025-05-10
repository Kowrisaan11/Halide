/*
  AutoSchedule.cpp: Implementation of the Adams2019 autoscheduler.
  Uses SimpleLSTMModel to evaluate schedules and search for an optimal one.
*/

#include "AutoSchedule.h"
#include "CostModel.h"
#include "SimpleLSTMModel.h"
#include "State.h"
#include "ASLog.h"
#include "Timer.h"
#include <algorithm>
#include <memory>
#include <random>
#include <sstream>

namespace Halide {
namespace Internal {
namespace Autoscheduler {

namespace {

IntrusivePtr<State> optimal_schedule(const std::vector<Function> &outputs,
                                    const Adams2019Params &params,
                                    SimpleLSTMModel &cost_model,
                                    std::mt19937 &rng) {
    // Create an initial state
    IntrusivePtr<State> initial{new State};
    initial->root = new TreeRepresentation(outputs); // Initialize with TreeRepresentation
    initial->cost = cost_model.evaluate_cost(initial);

    // Simple beam search
    std::vector<IntrusivePtr<State>> beam = {initial};
    std::vector<IntrusivePtr<State>> next_beam;
    int samples = 0;

    while (samples < params.max_samples && !beam.empty()) {
        next_beam.clear();
        for (const auto &state : beam) {
            // Generate child states (simplified: perturb schedule)
            for (int i = 0; i < params.beam_size; ++i) {
                IntrusivePtr<State> child{new State};
                child->root = new TreeRepresentation(*state->root); // Copy and perturb
                child->cost = cost_model.evaluate_cost(child);
                next_beam.push_back(child);
                samples++;
                if (samples >= params.max_samples) break;
            }
        }

        // Select top beam_size states
        std::sort(next_beam.begin(), next_beam.end(),
                  [](const auto &a, const auto &b) { return a->cost < b->cost; });
        if (next_beam.size() > static_cast<size_t>(params.beam_size)) {
            next_beam.resize(params.beam_size);
        }
        beam = std::move(next_beam);
    }

    // Return the best state
    if (beam.empty()) {
        return initial;
    }
    return *std::min_element(beam.begin(), beam.end(),
                             [](const auto &a, const auto &b) { return a->cost < b->cost; });
}

} // namespace

void generate_schedule(const std::vector<Function> &outputs,
                       const Target &target,
                       const Adams2019Params &params,
                       AutoSchedulerResults *auto_scheduler_results) {
    Timer timer;
    std::mt19937 rng(params.random_seed);

    // Initialize cost model
    SimpleLSTMModel cost_model(
        "/path/to/model.pt",
        "/path/to/scaler_params.json");

    // Find optimal schedule
    IntrusivePtr<State> optimal = optimal_schedule(outputs, params, cost_model, rng);

    if (optimal) {
        // Apply the schedule
        optimal->apply_schedule(outputs, params);

        // Log results
        if (params.verbosity >= 2) {
            ASLog::info() << "Optimal cost: " << optimal->cost
                          << ", Time: " << timer.elapsed() << "s\n";
            optimal->dump(std::cerr);
        }

        // Save results
        if (auto_scheduler_results) {
            std::ostringstream out;
            optimal->save_featurization(out);
            auto_scheduler_results->schedule_source = optimal->schedule_source;
            auto_scheduler_results->featurization.assign(out.str().begin(), out.str().end());
        }
    } else {
        ASLog::warning() << "No valid schedule found\n";
    }
}

} // namespace Autoscheduler
} // namespace Internal
} // namespace Halide
