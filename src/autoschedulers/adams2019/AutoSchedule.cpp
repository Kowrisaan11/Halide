/*
  Modified AutoSchedule.cpp for Adams2019 autoscheduler with SimpleLSTMModel.
  Integrates LibTorch-based LSTM cost model and custom TreeRepresentation.
  Core functionality:
  - Navigates search space using beam search.
  - Computes featurization using FunctionDAG and Featurization.
  - Evaluates costs using SimpleLSTMModel with model.pt and scaler_params.json.
  - Applies schedules to Halide pipeline.

  Environment variables:
  - HL_DEBUG_AUTOSCHEDULE: Debug log level (0-2).
  - HL_PERMIT_FAILED_UNROLL: Set to 1 to allow non-constant unrolling.
*/

#include "HalidePlugin.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <torch/script.h>
#include <nlohmann/json.hpp>
#include "ASLog.h"
#include "AutoSchedule.h"
#include "FunctionDAG.h"
#include "Featurization.h"
#include "SimpleLSTMModel.h"
#include "TreeRepresentation.h"
#include "Halide.h"
#include "Timer.h"

#ifdef _WIN32
#include <io.h>
#define _isatty isatty
#else
#include <unistd.h>
#endif

namespace Halide {
namespace Internal {
namespace Autoscheduler {

using std::string;
using std::vector;
using json = nlohmann::json;

struct ProgressBar {
    void set(double progress) {
        if (!draw_progress_bar) return;
        auto &os = aslog(1).get_ostream();
        counter++;
        const int bits = 11;
        if (counter & ((1 << bits) - 1)) return;
        const int pos = (int)(progress * 78);
        os << "[";
        for (int j = 0; j < 78; j++) {
            if (j < pos) os << ".";
            else if (j - 1 < pos) os << "/-\\|"[(counter >> bits) % 4];
            else os << " ";
        }
        os << "]";
        for (int j = 0; j < 80; j++) os << "\b";
    }

    void clear() {
        if (counter) {
            auto &os = aslog(1).get_ostream();
            for (int j = 0; j < 80; j++) os << " ";
            for (int j = 0; j < 80; j++) os << "\b";
        }
    }

private:
    uint32_t counter = 0;
    const bool draw_progress_bar = isatty(2) && aslog::aslog_level() >= 1;
};

bool random_dropout(const Adams2019Params &params, std::mt19937 &rng, size_t num_decisions) {
    if (params.random_dropout >= 100) return false;
    double t = params.random_dropout / 100.0;
    t = std::pow(t, 1.0f / num_decisions);
    t *= 100;
    uint32_t r = rng();
    return (r % 100) >= t;
}

class StateQueue {
private:
    struct CompareStates {
        bool operator()(const IntrusivePtr<State> &a, const IntrusivePtr<State> &b) const {
            return a->cost > b->cost;
        }
    };
    std::vector<IntrusivePtr<State>> storage;
    size_t sz = 0;

public:
    void emplace(IntrusivePtr<State> &&s) {
        if (sz >= storage.size()) storage.resize(std::max(sz * 2, (size_t)64));
        storage[sz] = std::move(s);
        sz++;
        std::push_heap(storage.begin(), storage.begin() + sz, CompareStates{});
    }

    IntrusivePtr<State> pop() {
        std::pop_heap(storage.begin(), storage.begin() + sz, CompareStates{});
        sz--;
        return std::move(storage[sz]);
    }

    bool empty() const { return sz == 0; }
    size_t size() const { return sz; }
    void clear() {
        for (size_t i = 0; i < sz; i++) storage[i] = IntrusivePtr<State>{};
        sz = 0;
    }
    void resort() {
        std::make_heap(storage.begin(), storage.begin() + sz, CompareStates{});
    }
};

IntrusivePtr<State> optimal_schedule_pass(FunctionDAG &dag,
                                         const vector<Function> &outputs,
                                         const Adams2019Params &params,
                                         SimpleLSTMModel &cost_model,
                                         std::mt19937 &rng,
                                         int pass_idx,
                                         int num_passes,
                                         ProgressBar &tick) {
    StateQueue q, pending;
    IntrusivePtr<State> initial{new State};
    initial->root = new TreeRepresentation;
    q.emplace(std::move(initial));

    int expanded = 0;
    std::unordered_map<uint64_t, int> hashes;

    std::function<void(IntrusivePtr<State> &&)> enqueue_new_children =
        [&](IntrusivePtr<State> &&s) {
            internal_assert(s->num_decisions_made == s->parent->num_decisions_made + 1);
            int progress = s->num_decisions_made * params.beam_size + expanded;
            size_t max_progress = dag.nodes.size() * params.beam_size * 2;
            tick.set(double(progress) / max_progress);
            s->penalized = false;
            q.emplace(std::move(s));
        };

    for (;;) {
        q.swap(pending);
        if (pending.empty()) internal_error << "Ran out of legal states with beam size " << params.beam_size << "\n";
        expanded = 0;
        while (expanded < params.beam_size && !pending.empty()) {
            IntrusivePtr<State> state{pending.pop()};

            if (params.beam_size > 1 && num_passes > 1) {
                if (!state->penalized) {
                    uint64_t h = state->structural_hash(pass_idx);
                    int penalty = ++hashes[h];
                    if (penalty > 1) {
                        state->penalized = true;
                        state->cost *= penalty;
                        if (!pending.empty() && state->cost > pending.top()->cost) {
                            pending.emplace(std::move(state));
                            continue;
                        }
                    }
                }
            }

            if (pending.size() > 1 && random_dropout(params, rng, dag.nodes.size() * 2)) continue;

            if (state->num_decisions_made == 2 * (int)dag.nodes.size()) {
                return state;
            }

            state->generate_children(dag, params, &cost_model, enqueue_new_children);
            expanded++;
        }

        pending.clear();
        cost_model.evaluate_costs();
        q.resort();
    }
}

IntrusivePtr<State> optimal_schedule(FunctionDAG &dag,
                                    const vector<Function> &outputs,
                                    const Adams2019Params &params,
                                    SimpleLSTMModel &cost_model,
                                    std::mt19937 &rng) {
    int num_passes = (params.beam_size == 1) ? 1 : 5;
    IntrusivePtr<State> best;
    for (int i = 0; i < num_passes; i++) {
        ProgressBar tick;
        Timer timer;
        auto pass = optimal_schedule_pass(dag, outputs, params, cost_model, rng, i, num_passes, tick);
        auto milli = std::chrono::duration_cast<std::chrono::milliseconds>(timer.elapsed()).count();
        tick.clear();
        aslog(1) << "Pass " << i << " of " << num_passes << ", cost: " << pass->cost << ", time (ms): " << milli << "\n";
        if (i == 0 || pass->cost < best->cost) best = pass;
    }
    aslog(1) << "Best cost: " << best->cost << "\n";
    return best;
}

void generate_schedule(const std::vector<Function> &outputs,
                       const Target &target,
                       const Adams2019Params &params,
                       AutoSchedulerResults *auto_scheduler_results) {
    aslog(1) << "generate_schedule for target=" << target.to_string() << "\n";
    HALIDE_TIC;

    std::mt19937 rng((uint32_t)params.random_dropout_seed);
    FunctionDAG dag(outputs, target);
    if (aslog::aslog_level() >= 2) dag.dump(aslog(2).get_ostream());

    SimpleLSTMModel cost_model("/home/kowrisaan/fyp/Halide/src/autoschedulers/adams2019/model.pt",
                               "/home/kowrisaan/fyp/Halide/src/autoschedulers/adams2019/scaler_params.json");

    IntrusivePtr<State> optimal = optimal_schedule(dag, outputs, params, cost_model, rng);

    HALIDE_TOC;

    optimal->apply_schedule(dag, params);
    aslog(1) << "** Optimal schedule:\n";
    if (aslog::aslog_level() >= 2) optimal->dump(aslog(2).get_ostream());

    if (auto_scheduler_results) {
        auto_scheduler_results->schedule_source = optimal->schedule_source;
        std::ostringstream out;
        optimal->save_featurization(dag, params, out);
        auto_scheduler_results->featurization = out.str();
    }
}

struct Adams2019 {
    void operator()(const Pipeline &p, const Target &target, const AutoschedulerParams &params_in, AutoSchedulerResults *results) {
        internal_assert(params_in.name == "Adams2019");
        std::vector<Function> outputs;
        for (const Func &f : p.outputs()) outputs.push_back(f.function());
        Adams2019Params params;
        ParamParser parser(params_in.extra);
        parser.parse("parallelism", &params.parallelism);
        parser.parse("beam_size", &params.beam_size);
        parser.parse("random_dropout", &params.random_dropout);
        parser.parse("random_dropout_seed", &params.random_dropout_seed);
        parser.parse("disable_subtiling", &params.disable_subtiling);
        parser.parse("memory_limit", &params.memory_limit);
        parser.finish();
        generate_schedule(outputs, target, params, results);
        results->autoscheduler_params = params_in;
    }
};

REGISTER_AUTOSCHEDULER(Adams2019)

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide
