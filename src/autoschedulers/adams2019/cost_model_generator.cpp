#include <utility>
#include <fstream>
#include <vector>
#include <nlohmann/json.hpp>

#include "Halide.h"
#include "NetworkSize.h"
#include "cost_model_schedule.h"

using namespace Halide;
using namespace nlohmann;
using Halide::Derivative;

template<bool training>
struct ModelWeight;

template<>
struct ModelWeight<false> : public GeneratorInput<Buffer<float>> {
    ModelWeight(const std::string &name, int dim)
        : GeneratorInput<Buffer<float>>(name, dim) {
    }
    void backprop(const Derivative &d, const Expr &learning_rate, const Expr &timestep) {
    }
    void set_shape(int s0 = 0, int s1 = 0, int s2 = 0) {
        if (s0) {
            dim(0).set_bounds(0, s0);
        }
        if (s1) {
            dim(1).set_bounds(0, s1);
        }
        if (s2) {
            dim(2).set_bounds(0, s2);
        }
    }
};

template<>
struct ModelWeight<true> : public GeneratorInput<Buffer<float>> {
    GeneratorOutput<Buffer<float>> grad;

    ModelWeight(const std::string &name, int dim)
        : GeneratorInput<Buffer<float>>(name, dim), grad("updated_" + name, dim + 1) {
    }
    void backprop(const Derivative &d, const Expr &learning_rate, const Expr &timestep) {
        std::vector<Expr> args(dimensions() + 1);
        for (auto &e : args) {
            e = Var();
        }
        grad(args) = undef<float>();

        args.back() = 0;
        FuncRef new_weight = grad(args);
        args.back() = 1;
        FuncRef smoothed_deriv = grad(args);
        args.back() = 2;
        FuncRef smoothed_second_moment = grad(args);
        args.back() = 3;
        FuncRef loss_gradient = grad(args);

        args.pop_back();
        Expr current_weight = (*this)(args);

        loss_gradient = d(*this)(args);

        smoothed_deriv = 0.9f * smoothed_deriv + 0.1f * loss_gradient;
        smoothed_second_moment = 0.999f * smoothed_second_moment + 0.001f * pow(loss_gradient, 2);

        Expr smoothed_deriv_correction = 1 / (1 - pow(0.9f, timestep + 1));
        Expr smoothed_second_moment_correction = 1 / (1 - pow(0.999f, timestep + 1));

        Expr step = learning_rate * smoothed_deriv * smoothed_deriv_correction;
        step /= sqrt(smoothed_second_moment * smoothed_second_moment_correction) + 1e-5f;

        new_weight = current_weight - step;
    }

    void set_shape(int s0 = 0, int s1 = 0, int s2 = 0) {
        if (s0) {
            dim(0).set_bounds(0, s0);
            dim(0).set_estimate(0, s0);
            grad.dim(0).set_bounds(0, s0);
            grad.dim(0).set_estimate(0, s0);
            grad.bound(grad.args()[0], 0, s0);
            grad.set_estimate(grad.args()[0], 0, s0);
        }
        if (s1) {
            dim(1).set_bounds(0, s1);
            dim(1).set_estimate(0, s1);
            grad.dim(1).set_bounds(0, s1);
            grad.dim(1).set_estimate(0, s1);
            grad.bound(grad.args()[1], 0, s1);
            grad.set_estimate(grad.args()[1], 0, s1);
        }
        if (s2) {
            dim(2).set_bounds(0, s2);
            dim(2).set_estimate(0, s2);
            grad.dim(2).set_bounds(0, s2);
            grad.dim(2).set_estimate(0, s2);
            grad.bound(grad.args()[2], 0, s2);
            grad.set_estimate(grad.args()[2], 0, s2);
        }
        grad.dim(dimensions()).set_bounds(0, 4);
        grad.dim(dimensions()).set_estimate(0, 4);
    }
};

template<bool training>
class CostModel : public Generator<CostModel<training>> {
protected:
    bool allow_out_of_order_inputs_and_outputs() const override {
        return true;
    }

public:
    template<typename T>
    using Input = GeneratorInput<T>;
    template<typename T>
    using Output = GeneratorOutput<T>;
    using Generator<CostModel<training>>::using_autoscheduler;
    using Generator<CostModel<training>>::get_pipeline;

    Input<int> num_stages{"num_stages", 1};
    Input<int> batch_size{"batch_size", 1};
    Input<int> num_cores{"num_cores", 1};
    Input<std::string> json_input_path{"json_input_path", ""}; // New input for JSON file
    Input<float> learning_rate{"learning_rate", 1.0f};
    Input<int> timestep{"timestep", 0};
    Input<int> reference{"reference", 0};
    Input<Buffer<float>> true_runtime{"true_runtime", 1};

    using Weight = ModelWeight<training>;
    Weight head1_filter{"head1_filter", 3};
    Weight head1_bias{"head1_bias", 1};
    Weight head2_filter{"head2_filter", 2};
    Weight head2_bias{"head2_bias", 1};
    Weight filter1{"filter1", 2};
    Weight bias1{"bias1", 1};

    Output<Buffer<float>> prediction_output{"prediction_output", 1};
    Output<Buffer<float>> loss_output{"loss_output", 0};

    Func pad_stages(const Funceminent &f, const Expr &stages) {
        Halide::Region bounds(f.dimensions());
        bounds[1].min = 0;
        bounds[1].extent = stages;
        return BoundaryConditions::constant_exterior(f, cast(f.value().type(), 0), bounds);
    }

    Expr activation(const Expr &e) {
        return max(e, 0);
    }

    Expr sigmoid(const Expr &e) {
        return 1 / (1 + exp(-e));
    }

    void generate() {
        Var c("c"), w("w"), n("n"), j("j"), s("s");

        // Define buffers for features
        Func pipeline_features("pipeline_features");
        Func schedule_features("normalized_schedule_features");

        // Parse JSON input
        json j;
        bool json_valid = false;
        if (!json_input_path.get().empty()) {
            std::ifstream json_file(json_input_path.get());
            if (json_file.is_open()) {
                try {
                    json_file >> j;
                    json_valid = true;
                } catch (const std::exception &e) {
                    // Fallback to default initialization
                    json_valid = false;
                }
                json_file.close();
            }
        }

        // Initialize feature buffers
        pipeline_features(c, w, s) = 0.0f;
        schedule_features(n, c, s) = 0.0f;

        if (json_valid) {
            // Extract scheduling features
            std::vector<float> sched_features(head2_w, 0.0f);
            int node_count = 0;
            for (const auto &child : j["children"]) {
                if (child.contains("scheduling")) {
                    auto sched = child["scheduling"];
                    int idx = 0;
                    for (const auto &key : {
                        "num_realizations", "num_productions", "points_computed_per_realization",
                        "points_computed_per_production", "points_computed_total", "points_computed_minimum",
                        "innermost_loop_extent", "innermost_pure_loop_extent", "unrolled_loop_extent",
                        "inner_parallelism", "outer_parallelism", "bytes_at_realization", "bytes_at_production",
                        "bytes_at_root", "innermost_bytes_at_realization", "innermost_bytes_at_production",
                        "innermost_bytes_at_root", "inlined_calls", "unique_bytes_read_per_realization",
                        "unique_lines_read_per_realization", "allocation_bytes_read_per_realization",
                        "working_set", "vector_size", "native_vector_size", "num_vectors", "num_scalars",
                        "scalar_loads_per_vector", "vector_loads_per_vector", "scalar_loads_per_scalar",
                        "bytes_at_task", "innermost_bytes_at_task", "unique_bytes_read_per_vector",
                        "unique_lines_read_per_vector", "unique_bytes_read_per_task", "unique_lines_read_per_task",
                        "working_set_at_task", "working_set_at_production", "working_set_at_realization",
                        "working_set_at_root"
                    }) {
                        if (sched.contains(key)) {
                            sched_features[idx] = sched[key].get<float>();
                        }
                        idx++;
                        if (idx >= head2_w) break;
                    }
                    // Assign to schedule_features buffer
                    for (int c_idx = 0; c_idx < head2_w; ++c_idx) {
                        schedule_features(node_count, c_idx, s) = fast_log(sched_features[c_idx] + 1);
                    }
                    node_count++;
                    if (node_count >= batch_size) break;
                }
            }

            // Extract pipeline features (e.g., from global features and op histograms)
            int stage_idx = 0;
            for (const auto &child : j["children"]) {
                if (child.contains("cache_hits")) {
                    // Global features
                    pipeline_features(0, 0, stage_idx) = child["cache_hits"].get<float>();
                    pipeline_features(0, 1, stage_idx) = child["cache_misses"].get<float>();
                    pipeline_features(0, 2, stage_idx) = child["execution_time_ms"].get<float>();
                } else if (child.contains("op_histogram")) {
                    // Op histogram features
                    auto op_hist = child["op_histogram"];
                    int c_idx = 0;
                    for (const auto &op : op_hist.items()) {
                        if (c_idx < head1_w) {
                            pipeline_features(c_idx, 0, stage_idx) = op.value().get<float>();
                            c_idx++;
                        }
                    }
                    stage_idx++;
                    if (stage_idx >= num_stages) break;
                }
            }
        } else {
            // Fallback: Initialize with zeros or default values
            schedule_features(n, c, s) = 0.0f;
            pipeline_features(c, w, s) = 0.0f;
        }

        // Rest of the neural network remains the same
        Func squashed_head1_filter("squashed_head1_filter");
        squashed_head1_filter(c, s, n) = sigmoid(head1_filter(c, s, n));

        Func squashed_head1_filter_broadcast("squashed_head1_filter_broadcast");
        squashed_head1_filter_broadcast(c, w, s, n) = squashed_head1_filter(c, s, n);

        Func head1_conv("head1_conv");
        RDom r_head1(0, head1_w, 0, head1_h);
        head1_conv(c, w) = head1_bias(c);
        head1_conv(c, w) += (squashed_head1_filter_broadcast(c, w, r_head1.x, r_head1.y) *
                             pipeline_features(r_head1.x, r_head1.y, w));

        Func head2_conv("head2_conv");
        RDom r_head2(0, head2_w);
        head2_conv(c, w, n) = head2_bias(c);
        head2_conv(c, w, n) += head2_filter(c, r_head2) * schedule_features(n, r_head2, w);

        Func head2_relu("head2_relu");
        head2_relu(c, w, n) = activation(head2_conv(c, w, n));

        Func conv1_stage1("conv1_stage1");
        RDom r1_stage1(0, head1_channels);
        conv1_stage1(c, w) = bias1(c);
        conv1_stage1(c, w) += filter1(c, r1_stage1.x) * head1_conv(r1_stage1.x, w);

        Func conv1_stage2("conv1_stage2");
        RDom r1_stage2(0, head2_channels);
        conv1_stage2(c, w, n) = conv1_stage1(c, w);
        conv1_stage2(c, w, n) += filter1(c, head1_filter.dim(0).extent() + r1_stage2.x) * head2_relu(r1_stage2.x, w, n);

        Func relu1("relu1");
        relu1(c, w, n) = activation(conv1_stage2(c, w, n));

        // Unpack scheduling features (for cost computation)
        int idx = 0;
        Expr num_realizations = schedule_features(n, idx++, w);
        Expr num_productions = schedule_features(n, idx++, w);
        Expr points_computed_per_realization = schedule_features(n, idx++, w);
        Expr points_computed_per_production = schedule_features(n, idx++, w);
        Expr points_computed_total = schedule_features(n, idx++, w);
        Expr points_computed_minimum = schedule_features(n, idx++, w);
        Expr innermost_loop_extent = schedule_features(n, idx++, w);
        Expr innermost_pure_loop_extent = schedule_features(n, idx++, w);
        Expr unrolled_loop_extent = schedule_features(n, idx++, w);
        Expr inner_parallelism = schedule_features(n, idx++, w);
        Expr outer_parallelism = schedule_features(n, idx++, w);
        Expr bytes_at_realization = schedule_features(n, idx++, w);
        Expr bytes_at_production = schedule_features(n, idx++, w);
        Expr bytes_at_root = schedule_features(n, idx++, w);
        Expr innermost_bytes_at_realization = schedule_features(n, idx++, w);
        Expr innermost_bytes_at_production = schedule_features(n, idx++, w);
        Expr innermost_bytes_at_root = schedule_features(n, idx++, w);
        Expr inlined_calls = schedule_features(n, idx++, w);
        Expr unique_bytes_read_per_realization = schedule_features(n, idx++, w);
        Expr unique_lines_read_per_realization = schedule_features(n, idx++, w);
        Expr allocation_bytes_read_per_realization = schedule_features(n, idx++, w);
        Expr working_set = schedule_features(n, idx++, w);
        Expr vector_size = schedule_features(n, idx++, w);
        Expr native_vector_size = schedule_features(n, idx++, w);
        Expr num_vectors = schedule_features(n, idx++, w);
        Expr num_scalars = schedule_features(n, idx++, w);
        Expr scalar_loads_per_vector = schedule_features(n, idx++, w);
        Expr vector_loads_per_vector = schedule_features(n, idx++, w);
        Expr scalar_loads_per_scalar = schedule_features(n, idx++, w);
        Expr bytes_at_task = schedule_features(n, idx++, w);
        Expr innermost_bytes_at_task = schedule_features(n, idx++, w);
        Expr unique_bytes_read_per_vector = schedule_features(n, idx++, w);
        Expr unique_lines_read_per_vector = schedule_features(n, idx++, w);
        Expr unique_bytes_read_per_task = schedule_features(n, idx++, w);
        Expr unique_lines_read_per_task = schedule_features(n, idx++, w);
        Expr working_set_at_task = schedule_features(n, idx++, w);
        Expr working_set_at_production = schedule_features(n, idx++, w);
        Expr working_set_at_realization = schedule_features(n, idx++, w);
        Expr working_set_at_root = schedule_features(n, idx++, w);
        assert(idx == head2_w);

        Expr compute_cost = select(inlined_calls == 0,
                                   (vector_size * num_vectors * relu1(0, w, n) +
                                    num_scalars * relu1(1, w, n)),
                                   (vector_size * num_vectors * relu1(2, w, n) +
                                    num_scalars * relu1(3, w, n)));

        Expr num_tasks = max(1, inner_parallelism * outer_parallelism);
        Expr tasks_per_core = num_tasks / num_cores;
        Expr idle_core_wastage = ceil(tasks_per_core) / max(1, tasks_per_core);
        compute_cost *= idle_core_wastage;

        Expr load_cost = (num_realizations * unique_lines_read_per_realization * relu1(5, w, n) +
                          num_realizations * unique_bytes_read_per_realization * relu1(6, w, n) +
                          num_vectors * scalar_loads_per_vector * relu1(7, w, n) +
                          num_scalars * scalar_loads_per_scalar * relu1(8, w, n) +
                          num_vectors * vector_loads_per_vector * relu1(9, w, n) +
                          num_scalars * unique_bytes_read_per_vector * relu1(10, w, n) +
                          num_vectors * unique_bytes_read_per_vector * relu1(11, w, n) +
                          num_scalars * unique_lines_read_per_vector * relu1(12, w, n) +
                          num_vectors * unique_lines_read_per_vector * relu1(13, w, n) +
                          num_tasks * unique_bytes_read_per_task * relu1(14, w, n) +
                          num_tasks * unique_lines_read_per_task * relu1(15, w, n));

        Expr lines_written_per_realization = inner_parallelism * (bytes_at_task / max(1, innermost_bytes_at_task));

        Expr alpha = select(inner_parallelism > 1, relu1(16, w, n),
                            w == 0, relu1(17, w, n),
                            relu1(18, w, n));
        Expr beta = select(inner_parallelism > 1, relu1(19, w, n),
                           w == 0, relu1(20, w, n),
                           relu1(21, w, n));

        Expr store_cost = num_realizations * (lines_written_per_realization * alpha +
                                              bytes_at_realization * beta);

        Expr cost_of_false_sharing =
            select(inner_parallelism > 1,
                   relu1(22, w, n) * (num_vectors + num_scalars) / max(1, innermost_bytes_at_task),
                   0.0f);

        store_cost += cost_of_false_sharing;

        Expr max_threads_hitting_same_page_fault = min(inner_parallelism, 4096 / max(1, innermost_bytes_at_task));
        const Expr &num_page_faults = bytes_at_production;
        Expr cost_of_page_faults = (num_page_faults * max_threads_hitting_same_page_fault *
                                    inner_parallelism * outer_parallelism * relu1(23, w, n));

        store_cost += cost_of_page_faults;

        Expr cost_of_malloc = relu1(24, w, n) * num_realizations;

        Expr cost_of_parallel_launches = num_productions * select(inner_parallelism > 1, relu1(25, w, n), 0.0f);
        Expr cost_of_parallel_tasks = num_productions * (inner_parallelism - 1) * relu1(26, w, n);
        Expr cost_of_parallelism = cost_of_parallel_tasks + cost_of_parallel_launches;

        Expr cost_of_working_set = working_set * relu1(27, w, n);

        store_cost *= 2;

        Expr cost = (compute_cost +
                     store_cost +
                     load_cost +
                     cost_of_malloc +
                     cost_of_parallelism +
                     cost_of_working_set);

        for (int i = 0; i < 32; i++) {
            cost += 0.0f * relu1(i, w, n);
        }

        Func runtime_per_stage;
        runtime_per_stage(n, w) = cost * 1e-9f;

        Func prediction;
        RDom r_reduce(0, num_stages);
        prediction(n) += runtime_per_stage(n, r_reduce);

        prediction_output(n) = prediction(n);

        Func err;

        if (!training) {
            loss_output() = 0.0f;
        } else {
            RDom r_batch(0, batch_size);
            RDom r_conv1_output(0, conv1_channels, 0, num_stages);
            Expr regularize = sum(-min(conv1_stage2(r_conv1_output.x, r_conv1_output.y, n), 0));

            Expr n2 = clamp(reference, 0, batch_size - 1);
            Expr scale = 1.0f / true_runtime(n2);

            Expr p1 = prediction(n) * scale;
            Expr r1 = true_runtime(n) * scale;

            Expr delta = pow(1.0f / max(p1, 1e-10f) - 1.0f / r1, 2);

            err(n) = delta + 1e-5f * regularize;

            Expr loss = sum(err(r_batch));

            loss_output() = loss;

            Derivative d_loss_d = propagate_adjoints(loss_output);

            Weight *weights[] = {&head1_filter, &head1_bias,
                                 &head2_filter, &head2_bias,
                                 &filter1, &bias1};

            for (Weight *w : weights) {
                w->backprop(d_loss_d, learning_rate, timestep);
            }
        }

        head1_filter.set_shape(head1_channels, head1_w, head1_h);
        head1_bias.set_shape(head1_channels);
        head2_filter.set_shape(head2_channels, head2_w);
        head2_bias.set_shape(head2_channels);
        filter1.set_shape(conv1_channels, head1_channels + head2_channels);
        bias1.set_shape(conv1_channels);

        num_cores.set_estimate(32);
        reference.set_estimate(0);
        batch_size.set_estimate(80);
        num_stages.set_estimate(13);
        prediction_output.set_estimates({{0, 80}});
        learning_rate.set_estimate(0.001f);
        timestep.set_estimate(37);

        // SCHEDULE
        if (training && !using_autoscheduler()) {
            do_cost_model_schedule(get_pipeline());
        } else if (using_autoscheduler()) {
            // Do nothing.
        } else {
            Var no;
            prediction_output.specialize(batch_size < 8).split(n, no, n, 1);
            prediction_output.compute_root().split(n, no, n, 8).parallel(no);
            prediction_output.bound(n, 0, batch_size);

            const int vec = 8;

            auto schedule_conv = [&](Func conv, Func relu, const RVar &r_channels) {
                Var ci, wi;
                if (!training) {
                    relu
                        .compute_at(prediction_output, n)
                        .store_at(prediction_output, no)
                        .tile(c, w, ci, wi, vec, 4, TailStrategy::RoundUp)
                        .vectorize(ci);
                    conv.compute_at(relu, c);
                } else {
                    conv.in()
                        .compute_root()
                        .tile(c, w, ci, wi, vec, 1, TailStrategy::RoundUp)
                        .vectorize(ci)
                        .unroll(wi)
                        .parallel(n, 8);
                    conv.compute_at(conv.in(), c);
                    relu
                        .compute_root()
                        .reorder_storage(c, w, n)
                        .reorder(c, w, n)
                        .vectorize(c, vec)
                        .parallel(n, 8);
                }
                conv
                    .vectorize(c)
                    .unroll(w)
                    .update()
                    .vectorize(c)
                    .unroll(w)
                    .reorder(c, w, r_channels);
            };

            conv1_stage1.compute_root().vectorize(c).update().vectorize(c);
            squashed_head1_filter.compute_root().vectorize(c);

            if (!training) {
                schedule_features
                    .compute_at(prediction_output, no)
                    .vectorize(n);
            } else {
                schedule_features
                    .compute_root()
                    .vectorize(n, 8);
            }

            schedule_conv(head2_conv, head2_relu, r_head2.x);
            schedule_conv(conv1_stage2, relu1, r1_stage2.x);
        }
    }
};

using CostModelInference = CostModel<false>;
using CostModelTraining = CostModel<true>;

HALIDE_REGISTER_GENERATOR(CostModelInference, cost_model);
HALIDE_REGISTER_GENERATOR(CostModelTraining, train_cost_model);
