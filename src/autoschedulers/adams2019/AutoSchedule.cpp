/*
  This file is the core of the autoscheduler. Most of the code here is
  about navigating the search space and computing the
  featurization. This also contains the top-level interface into the
  autoscheduler.

  The most interesting classes to look at are:

  LoopNest               Represents one node in our tree representation of loop nests. (Now in LoopNest.(h | cpp)).
  State                  A state in the beam search. Holds a root loop nest. (Now in State.(h | cpp)).

  Interesting functions below are:

  generate_schedule            The top-level entrypoint, which computes and applies a schedule to a Halide pipeline
  optimal_schedule             Runs the passes of the coarse-to-fine beam search
  optimal_schedule_pass        Runs a single pass of beam search
  LoopNest::compute_features   Recursively walks over a loop nest tree, computing our featurization using Halide's analysis tools.
  LoopNest::apply              Actually apply a computed schedule to a Halide pipeline
  State::generate_children     Generates successor states to a state in the beam search
*/

#include "AutoSchedule.h"
#include "ASLog.h"
#include "Cache.h"
#include "CostModel.h"
#include "SimpleLSTMModel.h" // <-- Use your new cost model here
#include "Errors.h"
#include "Featurization.h"
#include "FunctionDAG.h"
#include "Halide.h"
#include "LoopNest.h"
#include "ParamParser.h"
#include "PerfectHashMap.h"
#include "State.h"
#include "Timer.h"

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
#include <memory>
#include <cstring>

#ifdef _WIN32
#include <io.h>
#define _isatty isatty;
#else
#include <unistd.h>
#endif

namespace Halide {
namespace Internal {
namespace Autoscheduler {

namespace {

using std::string;
using std::vector;

// ProgressBar logic as in original code...
struct ProgressBar {
    void set(double progress) {
        if (!draw_progress_bar) {
            return;
        }
        auto &os = aslog(ProgressBarLogLevel).get_ostream();
        counter++;
        const int bits = 11;
        if (counter & ((1 << bits) - 1)) {
            return;
        }
        const int pos = (int)(progress * 78);
        os << "[";
        for (int j = 0; j < 78; j++) {
            if (j < pos) {
                os << ".";
            } else if (j - 1 < pos) {
                os << "/-\\|"[(counter >> bits) % 4];
            } else {
                os << " ";
            }
        }
        os << "]";
        for (int j = 0; j < 80; j++) {
            os << "\b";
        }
    }

    void clear() {
        if (counter) {
            auto &os = aslog(ProgressBarLogLevel).get_ostream();
            for (int j = 0; j < 80; j++) {
                os << " ";
            }
            for (int j = 0; j < 80; j++) {
                os << "\b";
            }
        }
    }

private:
    uint32_t counter = 0;
    static constexpr int ProgressBarLogLevel = 1;
    const bool draw_progress_bar = isatty(2) && aslog::aslog_level() >= ProgressBarLogLevel;
};

// Decide whether or not to drop a beam search state (for randomness/training)
bool random_dropout(const Adams2019Params &params, std::mt19937 &rng, size_t num_decisions) {
    if (params.random_dropout >= 100) {
        return false;
    }
    double t = params.random_dropout;
    t /= 100;
    t = std::pow(t, 1.0f / num_decisions);
    t *= 100;
    uint32_t r = rng();
    bool drop_it = (r % 100) >= t;
    return drop_it;
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
        if (sz >= storage.size()) {
            storage.resize(std::max(sz * 2, (size_t)64));
        }
        storage[sz] = std::move(s);
        sz++;
        std::push_heap(storage.begin(), storage.begin() + sz, CompareStates{});
    }
    IntrusivePtr<State> pop() {
        std::pop_heap(storage.begin(), storage.begin() + sz, CompareStates{});
        sz--;
        return std::move(storage[sz]);
    }
    const IntrusivePtr<State> &top() { return storage[0]; }
    bool empty() const { return sz == 0; }
    size_t size() const { return sz; }
    void swap(StateQueue &other) noexcept {
        storage.swap(other.storage);
        std::swap(sz, other.sz);
    }
    IntrusivePtr<State> operator[](int idx) const { return storage[idx]; }
    void resort() { std::make_heap(storage.begin(), storage.begin() + sz, CompareStates{}); }
    void clear() {
        for (size_t i = 0; i < sz; i++) storage[i] = IntrusivePtr<State>{};
        sz = 0;
    }
};

void configure_pipeline_features(const FunctionDAG &dag,
                                 const Adams2019Params &params,
                                 CostModel *cost_model) {
    cost_model->reset();
    cost_model->set_pipeline_features(dag, params);
}

IntrusivePtr<State> optimal_schedule_pass(FunctionDAG &dag,
                                          const vector<Function> &outputs,
                                          const Adams2019Params &params,
                                          CostModel *cost_model,
                                          std::mt19937 &rng,
                                          int pass_idx,
                                          int num_passes,
                                          ProgressBar &tick,
                                          std::unordered_set<uint64_t> &permitted_hashes,
                                          Cache *cache) {
    if (cost_model) {
        configure_pipeline_features(dag, params, cost_model);
    }
    StateQueue q, pending;
    {
        IntrusivePtr<State> initial{new State};
        initial->root = new LoopNest;
        q.emplace(std::move(initial));
    }
    int expanded = 0;
    std::function<void(IntrusivePtr<State> &&)> enqueue_new_children =
        [&](IntrusivePtr<State> &&s) {
            internal_assert(s->num_decisions_made == s->parent->num_decisions_made + 1);
            int progress = s->num_decisions_made * params.beam_size + expanded;
            size_t max_progress = dag.nodes.size() * params.beam_size * 2;
            tick.set(double(progress) / max_progress);
            s->penalized = false;
            q.emplace(std::move(s));
        };

    // Beam search logic
    for (;;) {
        std::unordered_map<uint64_t, int> hashes;
        q.swap(pending);

        if (pending.empty()) {
            internal_error << "Ran out of legal states with beam size " << params.beam_size << "\n";
        }
        if ((int)pending.size() > params.beam_size * 10000) {
            aslog(1) << "*** Warning: Huge number of states generated (" << pending.size() << ").\n";
        }
        expanded = 0;
        while (expanded < params.beam_size && !pending.empty()) {
            IntrusivePtr<State> state{pending.pop()};
            if (params.beam_size > 1 && num_passes > 1) {
                if (!state->penalized) {
                    uint64_t h1 = state->structural_hash(pass_idx + 1);
                    uint64_t h0 = state->structural_hash(pass_idx - 1);
                    int penalty = ++hashes[h1];
                    if (pass_idx > 0 && !permitted_hashes.count(h0)) {
                        penalty += 10;
                    }
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
            if (pending.size() > 1 && random_dropout(params, rng, dag.nodes.size() * 2)) {
                continue;
            }
            if (state->num_decisions_made == 2 * (int)dag.nodes.size()) {
                auto best = state;
                if (pass_idx + 1 < num_passes) {
                    int blessed = 0;
                    while (state->cost <= 1.2 * best->cost && blessed < params.beam_size) {
                        const State *s = state.get();
                        while (s) {
                            uint64_t h1 = s->structural_hash(pass_idx);
                            permitted_hashes.insert(h1);
                            s = s->parent.get();
                        }
                        if (pending.empty()) break;
                        state = pending.pop();
                        blessed++;
                    }
                }
                return best;
            }
            state->generate_children(dag, params, cost_model, enqueue_new_children, cache);
            expanded++;
        }
        pending.clear();
        if (cost_model) {
            cost_model->evaluate_costs();
            q.resort();
        }
    }
}

IntrusivePtr<State> optimal_schedule(FunctionDAG &dag,
                                     const vector<Function> &outputs,
                                     const Adams2019Params &params,
                                     CostModel *cost_model,
                                     std::mt19937 &rng,
                                     const CachingOptions &options) {
    IntrusivePtr<State> best;
    std::unordered_set<uint64_t> permitted_hashes;
    Cache cache(options, dag.nodes.size());
    int num_passes = (params.beam_size == 1) ? 1 : 5;
    for (int i = 0; i < num_passes; i++) {
        ProgressBar tick;
        Timer timer;
        auto pass = optimal_schedule_pass(dag, outputs, params, cost_model,
                                          rng, i, num_passes, tick, permitted_hashes, &cache);
        std::chrono::duration<double> total_time = timer.elapsed();
        auto milli = std::chrono::duration_cast<std::chrono::milliseconds>(total_time).count();
        tick.clear();
        switch (aslog::aslog_level()) {
        case 0:
            break;
        case 1:
            aslog(1) << "Pass " << i << " of " << num_passes << ", cost: " << pass->cost << ", time (ms): " << milli << "\n";
            break;
        default:
            aslog(2) << "Pass " << i << " result: ";
            pass->dump(aslog(2).get_ostream());
        }
        if (i == 0 || pass->cost < best->cost) {
            best = pass;
        }
    }
    aslog(1) << "Best cost: " << best->cost << "\n";
    if (options.cache_blocks) {
        aslog(1) << "Cache (block) hits: " << cache.cache_hits << "\n";
        aslog(1) << "Cache (block) misses: " << cache.cache_misses << "\n";
    }
    return best;
}

void generate_schedule(const std::vector<Function> &outputs,
                       const Target &target,
                       const Adams2019Params &params,
                       AutoSchedulerResults *auto_scheduler_results) {
    aslog(1) << "generate_schedule for target=" << target.to_string() << "\n";
    aslog(1) << "Adams2019.parallelism:" << params.parallelism << "\n";
    aslog(1) << "Adams2019.beam_size:" << params.beam_size << "\n";
    aslog(1) << "Adams2019.random_dropout:" << params.random_dropout << "\n";
    aslog(1) << "Adams2019.random_dropout_seed:" << params.random_dropout_seed << "\n";
    aslog(1) << "Adams2019.weights_path:" << params.weights_path << "\n";
    aslog(1) << "Adams2019.disable_subtiling:" << params.disable_subtiling << "\n";
    aslog(1) << "Adams2019.disable_memoized_features:" << params.disable_memoized_features << "\n";
    aslog(1) << "Adams2019.disable_memoized_blocks:" << params.disable_memoized_blocks << "\n";
    aslog(1) << "Adams2019.memory_limit:" << params.memory_limit << "\n";

    HALIDE_TIC;
    State::cost_calculations = 0;
    std::mt19937 rng((uint32_t)params.random_dropout_seed);

    // Setup your LSTM cost model here
    // You may want to pass the model path and scaler path via params/weights_path or hardcode here.
    std::string model_path = "model.pt";
    std::string scaler_path = "scaler_params.json";
    bool use_gpu = true;
    std::unique_ptr<CostModel> cost_model =
        std::make_unique<SimpleLSTMModel>(model_path, scaler_path, use_gpu);

    internal_assert(cost_model != nullptr);

    IntrusivePtr<State> optimal;
    CachingOptions cache_options = CachingOptions::MakeOptionsFromParams(params);
    optimal = optimal_schedule(FunctionDAG(outputs, target), outputs, params, cost_model.get(), rng, cache_options);

    HALIDE_TOC;
    aslog(1) << "Cost evaluated this many times: " << State::cost_calculations << "\n";
    aslog(1) << "** Optimal schedule:\n";
    optimal->calculate_cost(FunctionDAG(outputs, target), params, cost_model.get(), cache_options, 1);
    optimal->apply_schedule(FunctionDAG(outputs, target), params);
    if (aslog::aslog_level() >= 2) {
        optimal->dump(aslog(2).get_ostream());
    }
    if (auto_scheduler_results) {
        auto_scheduler_results->schedule_source = optimal->schedule_source;
        std::ostringstream out;
        optimal->save_featurization(FunctionDAG(outputs, target), params, cache_options, out);
        auto_scheduler_results->featurization.resize(out.str().size());
        memcpy(auto_scheduler_results->featurization.data(), out.str().data(), out.str().size());
    }
}

void find_and_apply_schedule(FunctionDAG &dag,
                             const std::vector<Function> &outputs,
                             const Adams2019Params &params,
                             CostModel *cost_model,
                             StageMapOfScheduleFeatures *schedule_features) {
    std::mt19937 rng(12345);
    CachingOptions cache_options = CachingOptions::MakeOptionsFromParams(params);
    IntrusivePtr<State> optimal = optimal_schedule(dag, outputs, params, cost_model, rng, cache_options);

    optimal->apply_schedule(dag, params);

    if (schedule_features) {
        optimal->compute_featurization(dag, params, schedule_features, cache_options);
    }
}

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide
