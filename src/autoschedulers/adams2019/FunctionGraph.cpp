#include "FunctionGraph.h"
#include "ASLog.h"
#include <memory>
#include <sstream>

namespace Halide {
namespace Internal {
namespace Autoscheduler {

namespace {

class Featurizer : public IRVisitor {
    using IRVisitor::visit;

    Function &func;
    FunctionGraph::Node::Stage &stage;

    int &op_bucket(PipelineFeatures::OpType op_type, Type scalar_type) {
        int type_bucket = (int)classify_type(scalar_type);
        stage.features.types_in_use[type_bucket] = true;
        return stage.features.op_histogram[(int)op_type][type_bucket];
    }

    PipelineFeatures::ScalarType classify_type(Type t) {
        if (t.is_float() && t.bits() > 32) {
            return PipelineFeatures::ScalarType::Double;
        } else if (t.is_float()) {
            return PipelineFeatures::ScalarType::Float;
        } else if (t.bits() == 1) {
            return PipelineFeatures::ScalarType::Bool;
        } else if (t.bits() <= 8) {
            return PipelineFeatures::ScalarType::UInt8;
        } else if (t.bits() <= 16) {
            return PipelineFeatures::ScalarType::UInt16;
        } else if (t.bits() <= 32) {
            return PipelineFeatures::ScalarType::UInt32;
        } else {
            return PipelineFeatures::ScalarType::UInt64;
        }
    }

    void visit(const Variable *op) override {
        if (op->param.defined()) {
            op_bucket(PipelineFeatures::OpType::Param, op->type)++;
        } else {
            op_bucket(PipelineFeatures::OpType::Variable, op->type)++;
        }
    }

    void visit(const IntImm *op) override {
        op_bucket(PipelineFeatures::OpType::Const, op->type)++;
    }

    void visit(const UIntImm *op) override {
        op_bucket(PipelineFeatures::OpType::Const, op->type)++;
    }

    void visit(const FloatImm *op) override {
        op_bucket(PipelineFeatures::OpType::Const, op->type)++;
    }

    void visit(const Add *op) override {
        op_bucket(PipelineFeatures::OpType::Add, op->type)++;
        IRVisitor::visit(op);
    }

    void visit(const Sub *op) override {
        op_bucket(PipelineFeatures::OpType::Sub, op->type)++;
        IRVisitor::visit(op);
    }

    void visit(const Mul *op) override {
        op_bucket(PipelineFeatures::OpType::Mul, op->type)++;
        IRVisitor::visit(op);
    }

    void visit(const Mod *op) override {
        op_bucket(PipelineFeatures::OpType::Mod, op->type)++;
        IRVisitor::visit(op);
    }

    void visit(const Div *op) override {
        op_bucket(PipelineFeatures::OpType::Div, op->type)++;
        IRVisitor::visit(op);
    }

    void visit(const Min *op) override {
        op_bucket(PipelineFeatures::OpType::Min, op->type)++;
        IRVisitor::visit(op);
    }

    void visit(const Max *op) override {
        op_bucket(PipelineFeatures::OpType::Max, op->type)++;
        IRVisitor::visit(op);
    }

    void visit(const EQ *op) override {
        op_bucket(PipelineFeatures::OpType::EQ, op->type)++;
        IRVisitor::visit(op);
    }

    void visit(const NE *op) override {
        op_bucket(PipelineFeatures::OpType::NE, op->type)++;
        IRVisitor::visit(op);
    }

    void visit(const LT *op) override {
        op_bucket(PipelineFeatures::OpType::LT, op->type)++;
        IRVisitor::visit(op);
    }

    void visit(const LE *op) override {
        op_bucket(PipelineFeatures::OpType::LE, op->type)++;
        IRVisitor::visit(op);
    }

    void visit(const GT *op) override {
        op_bucket(PipelineFeatures::OpType::LT, op->type)++;
        IRVisitor::visit(op);
    }

    void visit(const GE *op) override {
        op_bucket(PipelineFeatures::OpType::LE, op->type)++;
        IRVisitor::visit(op);
    }

    void visit(const And *op) override {
        op_bucket(PipelineFeatures::OpType::And, op->type)++;
        IRVisitor::visit(op);
    }

    void visit(const Or *op) override {
        op_bucket(PipelineFeatures::OpType::Or, op->type)++;
        IRVisitor::visit(op);
    }

    void visit(const Not *op) override {
        op_bucket(PipelineFeatures::OpType::Not, op->type)++;
        IRVisitor::visit(op);
    }

    void visit(const Select *op) override {
        op_bucket(PipelineFeatures::OpType::Select, op->type)++;
        IRVisitor::visit(op);
    }

    Scope<Expr> lets;

    void visit(const Let *op) override {
        ScopedBinding<Expr> bind(lets, op->name, op->value);
        op_bucket(PipelineFeatures::OpType::Let, op->type)++;
        IRVisitor::visit(op);
    }

    void visit(const Call *op) override {
        IRVisitor::visit(op);
        if (op->call_type == Call::Halide) {
            if (op->name == func.name()) {
                visit_memory_access(op->name, op->type, op->args, PipelineFeatures::AccessType::LoadSelf);
                op_bucket(PipelineFeatures::OpType::SelfCall, op->type)++;
            } else {
                visit_memory_access(op->name, op->type, op->args, PipelineFeatures::AccessType::LoadFunc);
                op_bucket(PipelineFeatures::OpType::FuncCall, op->type)++;
            }
        } else if (op->call_type == Call::Extern || op->call_type == Call::PureExtern ||
                   op->call_type == Call::Intrinsic || op->call_type == Call::PureIntrinsic) {
            op_bucket(PipelineFeatures::OpType::ExternCall, op->type)++;
        } else if (op->call_type == Call::Image) {
            visit_memory_access(op->name, op->type, op->args, PipelineFeatures::AccessType::LoadImage);
            op_bucket(PipelineFeatures::OpType::ImageCall, op->type)++;
        }
    }

    Scope<OptionalRational> dlets;

    OptionalRational differentiate(const Expr &e, const string &v) {
        if (!expr_uses_var(e, v, lets)) {
            return {true, 0, 1};
        } else if (const Variable *var = e.as<Variable>()) {
            if (var->name == v) {
                return {true, 1, 1};
            }
            for (const auto &l : stage.loop) {
                if (var->name == l.var) {
                    return {true, 0, 1};
                }
            }
            if (var->param.defined()) {
                return {true, 0, 1};
            } else if (lets.contains(var->name)) {
                string key = v + " " + var->name;
                if (dlets.contains(key)) {
                    return dlets.get(key);
                }
                auto a = differentiate(lets.get(var->name), v);
                dlets.push(key, a);
                return a;
            }
            internal_error << "Encountered unbound variable in call args: " << var->name << "\n";
            return {false, 0, 0};
        } else if (const Add *op = e.as<Add>()) {
            auto a = differentiate(op->a, v);
            a += differentiate(op->b, v);
            return a;
        } else if (const Sub *op = e.as<Sub>()) {
            auto a = differentiate(op->a, v);
            auto b = differentiate(op->b, v);
            b.numerator = -b.numerator;
            a += b;
            return a;
        } else if (const Mul *op = e.as<Mul>()) {
            auto a = differentiate(op->a, v);
            if (const int64_t *ib = as_const_int(op->b)) {
                a.numerator *= *ib;
                return a;
            } else {
                return {false, 0, 0};
            }
        } else if (const Div *op = e.as<Div>()) {
            auto a = differentiate(op->a, v);
            if (const int64_t *ib = as_const_int(op->b)) {
                if (a.numerator != 0) {
                    a.denominator *= *ib;
                }
                return a;
            } else {
                return {false, 0, 0};
            }
        } else if (const Call *op = e.as<Call>()) {
            if (op->is_intrinsic(Call::likely)) {
                return differentiate(op->args[0], v);
            }
        }
        return {false, 0, 0};
    }

    void visit_memory_access(const std::string &name, Type t, const vector<Expr> &args, PipelineFeatures::AccessType type) {
        vector<vector<OptionalRational>> matrix;
        vector<size_t> ones_per_row(args.size(), 0), zeros_per_row(args.size(), 0),
            ones_per_col(stage.loop.size(), 0), zeros_per_col(stage.loop.size(), 0);
        matrix.resize(args.size());
        bool is_pointwise = args.size() == stage.loop.size();
        for (size_t i = 0; i < args.size(); i++) {
            matrix[i].resize(stage.loop.size());
            for (size_t j = 0; j < stage.loop.size(); j++) {
                auto deriv = differentiate(args[i], stage.loop[j].var);
                zeros_per_row[i] += deriv == 0;
                ones_per_row[i] += deriv == 1;
                zeros_per_col[j] += deriv == 0;
                ones_per_col[j] += deriv == 1;
                is_pointwise &= (i == j ? deriv == 1 : deriv == 0);
                matrix[i][j] = deriv;
            }
        }
        bool is_transpose = (args.size() == stage.loop.size());
        bool is_broadcast = true, is_slice = true;
        for (size_t i = 0; i < args.size(); i++) {
            bool single_one = (ones_per_row[i] == 1) && (zeros_per_row[i] == stage.loop.size() - 1);
            bool all_zero = (zeros_per_row[i] == stage.loop.size());
            is_transpose &= single_one;
            is_broadcast &= single_one;
            is_slice &= single_one || all_zero;
        }
        for (size_t j = 0; j < stage.loop.size(); j++) {
            bool single_one = (ones_per_col[j] == 1) && (zeros_per_col[j] == args.size() - 1);
            bool all_zero = (zeros_per_col[j] == args.size());
            is_transpose &= single_one || all_zero;
            is_broadcast &= single_one;
            is_slice &= single_one;
        }

        auto type_class = classify_type(t);
        stage.features.pointwise_accesses[(int)type][(int)type_class] += is_pointwise;
        stage.features.transpose_accesses[(int)type][(int)type_class] += is_transpose;
        stage.features.broadcast_accesses[(int)type][(int)type_class] += is_broadcast;
        stage.features.slice_accesses[(int)type][(int)type_class] += is_slice;

        for (auto *e : stage.incoming_edges) {
            if (e->producer->name == name) {
                vector<vector<OptionalRational>> copy = matrix;
                e->add_load_jacobian(LoadJacobian(std::move(copy)));
            }
        }
    }

public:
    Featurizer(Function &func, FunctionGraph::Node::Stage &stage)
        : func(func), stage(stage) {}

    void visit_store_args(const std::string &name, Type t, vector<Expr> args) {
        for (auto &e : args) {
            e = common_subexpression_elimination(simplify(e));
        }
        visit_memory_access(name, t, args, PipelineFeatures::AccessType::Store);
    }
};

class DependsOnEstimate : public IRVisitor {
public:
    bool found_estimate = false;

private:
    using IRVisitor::visit;

    void visit(const Variable *op) override {
        found_estimate |= op->param.defined();
    }
};

bool depends_on_estimate(const Expr &expr) {
    DependsOnEstimate dependency_checker;
    expr.accept(&dependency_checker);
    return dependency_checker.found_estimate;
}

}  // namespace

void LoadJacobian::dump(std::ostream &os, const char *prefix) const {
    if (count() > 1) {
        os << prefix << count() << " x\n";
    }
    for (size_t i = 0; i < producer_storage_dims(); i++) {
        os << prefix << "  [";
        for (size_t j = 0; j < consumer_loop_dims(); j++) {
            const auto &c = (*this)(i, j);
            if (!c.exists) {
                os << " _  ";
            } else if (c.denominator == 1) {
                os << " " << c.numerator << "  ";
            } else {
                os << c.numerator << "/" << c.denominator << " ";
            }
        }
        os << "]\n";
    }
    os << "\n";
}

FunctionGraph::Edge::BoundInfo::BoundInfo(const Expr &e, const Node::Stage &consumer, bool dependent)
    : expr(e), depends_on_estimate(dependent) {
    const Add *add = expr.as<Add>();
    const Mul *mul = add ? add->a.as<Mul>() : expr.as<Mul>();
    const IntImm *coeff_imm = mul ? mul->b.as<IntImm>() : nullptr;
    const IntImm *constant_imm = add ? add->b.as<IntImm>() : nullptr;
    Expr v = (mul ? mul->a : add ? add->a : expr);
    const Variable *var = v.as<Variable>();

    if (const IntImm *c = e.as<IntImm>()) {
        affine = true;
        coeff = 0;
        constant = c->value;
    } else if (var && (!mul || coeff_imm) && (!add || constant_imm)) {
        affine = true;
        coeff = mul ? coeff_imm->value : 1;
        constant = add ? constant_imm->value : 0;
        consumer_dim = -1;
        for (int i = 0; i < (int)consumer.loop.size(); i++) {
            const auto &in = consumer.loop[i];
            if (var->name == consumer.node->func.name() + "." + in.var + ".min") {
                consumer_dim = i;
                uses_max = false;
                break;
            } else if (var->name == consumer.node->func.name() + "." + in.var + ".max") {
                consumer_dim = i;
                uses_max = true;
                break;
            }
        }
        internal_assert(consumer_dim >= 0) << "Could not find consumer loop variable: " << var->name << "\n";
        aslog(2) << "Bound is affine: " << e << " == " << var->name << " * " << coeff << " + " << constant << "\n";
    } else {
        affine = false;
        aslog(2) << "Bound is non-affine: " << e << "\n";
    }
}

void FunctionGraph::Edge::add_load_jacobian(LoadJacobian j1) {
    for (auto &j2 : load_jacobians) {
        if (j2.merge(j1)) {
            return;
        }
    }
    load_jacobians.emplace_back(std::move(j1));
}

void FunctionGraph::Edge::expand_footprint(const Span *consumer_loop, Span *producer_required) const {
    const auto &symbolic_loop = consumer->loop;
    map<string, Expr> s;
    if (!all_bounds_affine) {
        for (size_t i = 0; i < symbolic_loop.size(); i++) {
            auto p = consumer_loop[i];
            const string &var = symbolic_loop[i].var;
            s[consumer->node->func.name() + "." + var + ".min"] = (int)p.min();
            s[consumer->node->func.name() + "." + var + ".max"] = (int)p.max();
        }
    }
    for (int i = 0; i < producer->func.dimensions(); i++) {
        bool bounds_are_constant = true;
        auto eval_bound = [&](const BoundInfo &b) {
            bounds_are_constant &= !b.depends_on_estimate;
            if (b.affine) {
                if (b.coeff == 0) {
                    return b.constant;
                } else {
                    const auto &src_pair = consumer_loop[b.consumer_dim];
                    int64_t src = b.uses_max ? src_pair.max() : src_pair.min();
                    bounds_are_constant &= src_pair.constant_extent();
                    return src * b.coeff + b.constant;
                }
            } else {
                Expr substituted = substitute(s, b.expr);
                Expr e = simplify(substituted);
                const int64_t *i = as_const_int(e);
                internal_assert(i) << "Should be constant: " << b.expr << " -> " << substituted << " -> " << e << "\n";
                bounds_are_constant = false;
                return *i;
            }
        };
        int64_t a = eval_bound(bounds[i].first);
        int64_t b = eval_bound(bounds[i].second);
        producer_required[i].union_with(Span(a, b, bounds_are_constant));
    }
}

void FunctionGraph::Node::required_to_computed(const Span *required, Span *computed) const {
    map<string, Expr> required_map;
    if (!region_computed_all_common_cases) {
        for (int i = 0; i < func.dimensions(); i++) {
            required_map[region_required[i].min.name()] = (int)required[i].min();
            required_map[region_required[i].max.name()] = (int)required[i].max();
        }
    }
    for (int i = 0; i < func.dimensions(); i++) {
        computed[i] = required[i];
    }
}

void FunctionGraph::Node::loop_nest_for_region(int stage_idx, const Span *computed, Span *loop) const {
    const auto &s = stages[stage_idx];
    map<string, Expr> computed_map;
    if (!s.loop_nest_all_common_cases) {
        for (int i = 0; i < func.dimensions(); i++) {
            computed_map[region_required[i].min.name()] = (int)computed[i].min();
            computed_map[region_required[i].max.name()] = (int)computed[i].max();
        }
    }
    for (size_t i = 0; i < s.loop.size(); i++) {
        const auto &l = s.loop[i];
        if (l.equals_region_computed) {
            loop[i] = computed[l.region_computed_dim];
        } else if (l.bounds_are_constant) {
            loop[i] = Span(l.c_min, l.c_max, true);
        } else {
            Expr min = simplify(substitute(computed_map, l.min));
            Expr max = simplify(substitute(computed_map, l.max));
            const int64_t *imin = as_const_int(min);
            const int64_t *imax = as_const_int(max);
            internal_assert(imin && imax) << min << ", " << max << "\n";
            loop[i] = Span(*imin, *imax, false);
        }
    }
}

FunctionGraph::FunctionGraph(const vector<Function> &outputs, const Target &target) {
    map<string, Function> env = build_environment(outputs);
    class ApplyParamEstimates : public IRMutator {
        using IRMutator::visit;
        Expr visit(const Variable *op) override {
            Expr expr;
            if (op->param.defined()) {
                if (!op->param.is_buffer()) {
                    expr = op->param.estimate();
                } else {
                    for (int i = 0; i < op->param.dimensions(); i++) {
                        if (op->name == op->param.name() + ".min." + std::to_string(i)) {
                            expr = op->param.min_constraint_estimate(i);
                        } else if (op->name == op->param.name() + ".extent." + std::to_string(i)) {
                            expr = op->param.extent_constraint_estimate(i);
                        }
                    }
                }
                internal_assert(expr.defined()) << "Missing estimate for " << op->name << "\n";
                return expr;
            } else {
                return op;
            }
        }
    } apply_param_estimates;

    vector<string> order = topological_order(outputs, env);
    nodes.resize(order.size());
    map<Function, Node *, Function::Compare> node_map;
    for (size_t i = 0; i < order.size(); i++) {
        Function f = env[order[order.size() - i - 1]];
        nodes[i].func = f;
        nodes[i].name = f.name();
        nodes[i].id = (int)i;
        nodes[i].max_id = (int)order.size();
        node_map[f] = &nodes[i];
    }

    int stage_count = 0;
    for (size_t i = order.size(); i > 0; i--) {
        Node &node = nodes[order.size() - i];
        Function consumer = node.func;
        Scope<Interval> scope;
        for (int j = 0; j < consumer.dimensions(); j++) {
            Halide::Var min_var(consumer.name() + "." + consumer.args()[j] + ".min");
            Halide::Var max_var(consumer.name() + "." + consumer.args()[j] + ".max");
            Interval interval(min_var, max_var);
            scope.push(consumer.args()[j], interval);
            node.region_required.emplace_back(SymbolicInterval{min_var, max_var});
        }

        auto pure_args = node.func.args();
        for (int s = 0; s <= (int)consumer.updates().size(); s++) {
            stage_count++;
            if (s == 0) {
                node.stages.emplace_back(Stage(consumer, consumer.definition(), 0));
            } else {
                node.stages.emplace_back(Stage(consumer, consumer.update(s - 1), s));
            }
        }

        for (int s = 0; s <= (int)consumer.updates().size(); s++) {
            auto &stage = node.stages[s];
            stage.node = &node;
            stage.name = consumer.name();
            if (s > 0) {
                stage.name += ".update(" + std::to_string(s - 1) + ")";
            }

            const Definition &def = (s == 0) ? consumer.definition() : consumer.update(s - 1);
            const StageSchedule &sched = def.schedule();

            Scope<Interval> stage_scope_with_concrete_rvar_bounds, stage_scope_with_symbolic_rvar_bounds;
            stage_scope_with_concrete_rvar_bounds.set_containing_scope(&scope);
            stage_scope_with_symbolic_rvar_bounds.set_containing_scope(&scope);
            for (const auto &rv : sched.rvars()) {
                Expr min = simplify(apply_param_estimates.mutate(rv.min));
                Expr max = simplify(apply_param_estimates.mutate(rv.min + rv.extent - 1));
                stage_scope_with_concrete_rvar_bounds.push(rv.var, Interval(min, max));
                min = Variable::make(Int(32), consumer.name() + "." + rv.var + ".min");
                max = Variable::make(Int(32), consumer.name() + "." + rv.var + ".max");
                stage_scope_with_symbolic_rvar_bounds.push(rv.var, Interval(min, max));
            }

            if (s == 0) {
                node.region_computed.resize(consumer.dimensions());
            }

            FuncValueBounds func_value_bounds = compute_function_value_bounds(order, env);
            for (int j = 0; j < consumer.dimensions(); j++) {
                Interval in = bounds_of_expr_in_scope(def.args()[j], stage_scope_with_concrete_rvar_bounds, func_value_bounds);
                internal_assert(in.is_bounded()) << "Region computed of " << consumer.name() << " is unbounded: [" << in.min << " " << in.max << "]\n";
                if (s == 0) {
                    node.region_computed[j] = Span(in.min.as<IntImm>()->value, in.max.as<IntImm>()->value, false);
                } else {
                    node.region_computed[j].union_with(Span(in.min.as<IntImm>()->value, in.max.as<IntImm>()->value, false));
                }
            }

            user_assert(sched.splits().empty()) << "The Func \"" << consumer.name() << "\" has scheduling directive(s) applied to it; you must remove these, or conditionalize them using `if (!auto_schedule)`, to use the autoscheduler on this pipeline.";
            stage.loop_nest_all_common_cases = true;
            for (size_t i = 0; i < sched.dims().size(); i++) {
                const auto &d = sched.dims()[i];
                if (!stage_scope_with_symbolic_rvar_bounds.contains(d.var)) {
                    continue;
                }

                Node::Loop l;
                l.var = d.var;
                l.accessor = stage.name + ".get_schedule().dims()[" + std::to_string(i) + "].var";
                Interval in = stage_scope_with_concrete_rvar_bounds.get(l.var);
                l.min = in.min;
                l.max = in.max;
                l.pure = d.is_pure();
                l.rvar = d.is_rvar();
                l.pure_dim = -1;

                l.equals_region_computed = false;
                for (int j = 0; j < consumer.dimensions(); j++) {
                    if (l.var == pure_args[j]) {
                        l.pure_dim = j;
                    }
                    if (equal(l.min, node.region_computed[j].min()) && equal(l.max, node.region_computed[j].max())) {
                        l.equals_region_computed = true;
                        l.region_computed_dim = j;
                        break;
                    }
                }

                if (!l.equals_region_computed) {
                    const int64_t *c_min = as_const_int(l.min), *c_max = as_const_int(l.max);
                    if (c_min && c_max) {
                        l.bounds_are_constant = true;
                        l.c_min = *c_min;
                        l.c_max = *c_max;
                    } else {
                        l.bounds_are_constant = false;
                    }
                }

                stage.loop_nest_all_common_cases &= (l.bounds_are_constant || l.equals_region_computed);
                stage.loop.emplace_back(std::move(l));
            }

            class CheckTypes : public IRVisitor {
                using IRVisitor::visit;

                void visit(const IntImm *op) override { check_type(op->type); }
                void visit(const UIntImm *op) override { check_type(op->type); }
                void visit(const FloatImm *op) override { check_type(op->type); }
                void visit(const Variable *op) override { check_type(op->type); }
                void visit(const Call *op) override {
                    calls[op->name]++;
                    IRVisitor::visit(op);
                    check_type(op->type);
                    if (op->call_type == Call::Halide || op->call_type == Call::Image) {
                        is_pointwise &= op->args.size() == func.args().size();
                        if (is_pointwise) {
                            for (size_t i = 0; i < op->args.size(); i++) {
                                const Variable *v = op->args[i].as<Variable>();
                                is_pointwise &= (v != nullptr) && (v->name == func.args()[i]);
                            }
                        }
                    }
                }
                void visit(const Cast *op) override {
                    IRVisitor::visit(op);
                    check_type(op->type);
                }
                void visit(const Reinterpret *op) override {
                    IRVisitor::visit(op);
                    check_type(op->type);
                }
                void check_type(Type t) {
                    if (t.bits() > 1 && (!narrowest_type.bits() || t.bits() < narrowest_type.bits())) {
                        narrowest_type = t;
                    }
                }
                Function func;

            public:
                bool is_pointwise = true;
                int leaves = 0;
                Type narrowest_type;
                map<string, int> calls;
                explicit CheckTypes(const Function &f) : func(f) {}
            };

            vector<Expr> exprs_vector = def.args();
            exprs_vector.insert(exprs_vector.end(), def.values().begin(), def.values().end());
            if (def.predicate().defined()) {
                exprs_vector.push_back(def.predicate());
            }
            Expr exprs = Call::make(Int(32), "dummy", exprs_vector, Call::Extern);

            CheckTypes checker(consumer);
            exprs.accept(&checker);

            Type widest_output_type = def.values()[0].type();
            int bytes_per_point = 0;
            for (const auto &e : def.values()) {
                bytes_per_point += e.type().bytes();
                if (e.type().bytes() > widest_output_type.bytes()) {
                    widest_output_type = e.type();
                }
            }
            if (s == 0) {
                node.bytes_per_point = bytes_per_point;
            }

            stage.vector_size = target.natural_vector_size(checker.narrowest_type);
            if (s == 0) {
                node.vector_size = stage.vector_size;
            } else {
                node.vector_size = std::max(node.vector_size, stage.vector_size);
            }

            node.is_output = false;
            for (const auto &o : outputs) {
                node.is_output |= o.same_as(node.func);
            }

            if (node.is_output) {
                map<string, Span> estimates;
                for (const auto &b : consumer.schedule().estimates()) {
                    const int64_t *i_min = as_const_int(b.min);
                    const int64_t *i_extent = as_const_int(b.extent);
                    user_assert(i_min && i_extent) << "Min/extent of estimate or bound is not constant in \"" << consumer.name() << "\", var:" << b.var << ", min:" << b.min << ", extent:" << b.extent;
                    estimates[b.var] = Span(*i_min, *i_min + *i_extent - 1, false);
                }
                for (const auto &b : consumer.schedule().bounds()) {
                    const int64_t *i_min = as_const_int(b.min);
                    const int64_t *i_extent = as_const_int(b.extent);
                    if (i_min && i_extent) {
                        estimates[b.var] = Span(*i_min, *i_min + *i_extent - 1, true);
                    }
                }
                for (int i = 0; i < consumer.dimensions(); i++) {
                    auto it = estimates.find(consumer.args()[i]);
                    user_assert(it != estimates.end()) << "Need an estimate on dimension " << i << " of \"" << consumer.name() << "\"";
                    node.estimated_region_required.push_back(it->second);
                }
            }

            stage.index = s;
            exprs = apply_param_estimates.mutate(exprs);

            bool any_incoming_edges = false;
            node.is_pointwise = !node.func.has_update_definition();
            node.is_boundary_condition = node.is_pointwise && starts_with(node.func.name(), "repeat_edge");

            auto boxes = boxes_required(exprs, stage_scope_with_symbolic_rvar_bounds, func_value_bounds);
            for (auto &p : boxes) {
                auto it = env.find(p.first);
                if (it != env.end() && p.first != consumer.name()) {
                    Edge edge;
                    edge.consumer = &stage;
                    edge.producer = node_map.at(env[p.first]);
                    edge.all_bounds_affine = true;

                    for (Interval &in : p.second.bounds) {
                        internal_assert(in.is_bounded()) << "Unbounded producer->consumer relationship: " << edge.producer->func.name() << " -> " << edge.consumer->name << "\n";
                        bool min_dependent = depends_on_estimate(in.min);
                        bool max_dependent = depends_on_estimate(in.max);
                        Expr min_value = simplify(apply_param_estimates.mutate(in.min));
                        Expr max_value = simplify(apply_param_estimates.mutate(in.max));
                        Edge::BoundInfo min(min_value, *edge.consumer, min_dependent);
                        Edge::BoundInfo max(max_value, *edge.consumer, max_dependent);
                        edge.bounds.emplace_back(std::move(min), std::move(max));
                        edge.all_bounds_affine &= edge.bounds.back().first.affine;
                        edge.all_bounds_affine &= edge.bounds.back().second.affine;
                    }
                    edge.calls = checker.calls[edge.producer->func.name()];
                    any_incoming_edges = true;
                    node.is_pointwise &= checker.is_pointwise;
                    edges.emplace_back(std::move(edge));
                }
            }

            node.is_wrapper = node.func.is_wrapper();
            node.is_input = !node.is_output && !node.func.has_update_definition() && node.is_wrapper && !any_incoming_edges;
            node.dimensions = node.func.dimensions();
        }
    }

    int i = 0;
    for (auto &n : nodes) {
        for (auto &s : n.stages) {
            s.id = i;
            s.max_id = stage_count;
            i++;
        }
    }

    for (auto &edge : edges) {
        edge.producer->outgoing_edges.push_back(&edge);
        edge.consumer->incoming_edges.push_back(&edge);
    }

    for (size_t i = nodes.size(); i > 0; i--) {
        auto &n = nodes[i - 1];
        for (auto &s : n.stages) {
            s.dependencies.resize(nodes.size(), false);
            for (auto *e : s.incoming_edges) {
                s.dependencies[e->producer->id] = true;
                for (auto &s2 : e->producer->stages) {
                    for (size_t j = 0; j < nodes.size(); j++) {
                        s.dependencies[j] = s.dependencies[j] || s2.dependencies[j];
                    }
                }
            }
        }
    }

    featurize();
    build_graph_json();
}

void FunctionGraph::featurize() {
    for (Node &node : nodes) {
        for (size_t stage_idx = 0; stage_idx < node.stages.size(); stage_idx++) {
            Node::Stage &stage = node.stages[stage_idx];
            Featurizer featurizer(node.func, stage);

            if (node.func.extern_definition_proxy_expr().get()) {
                Expr v = simplify(node.func.extern_definition_proxy_expr());
                v = common_subexpression_elimination(v);
                v.accept(&featurizer);
            } else {
                Definition def = node.func.definition();
                if (stage_idx > 0) {
                    def = node.func.updates()[stage_idx - 1];
                }
                stage.features = PipelineFeatures();

                for (auto v : def.values()) {
                    featurizer.visit_store_args(node.func.name(), v.type(), def.args());
                    v = common_subexpression_elimination(simplify(v));
                    v.accept(&featurizer);
                }
                for (auto v : def.args()) {
                    v = common_subexpression_elimination(simplify(v));
                    v.accept(&featurizer);
                }
            }

            // Initialize schedule features (simplified for demonstration)
            stage.schedule_features = ScheduleFeatures();
            stage.schedule_features.num_realizations = 1.0;
            stage.schedule_features.num_productions = 1.0;
            stage.schedule_features.points_computed_total = node.dimensions * 100.0;
            // Add more schedule features as needed
        }
    }
}

void FunctionGraph::build_graph_json() {
    graph_json["nodes"] = json::array();
    graph_json["edges"] = json::array();
    graph_json["operations"] = json::array();
    graph_json["global_features"] = json::object();

    const vector<string> OP_HISTOGRAM_KEYS = {
        "Add", "And", "Cast", "Constant", "Div", "EQ", "ExternCall", "FuncCall",
        "ImageCall", "LE", "LT", "Let", "Max", "Min", "Mod", "Mul", "NE", "Not",
        "Or", "Param", "Select", "SelfCall", "Sub", "Variable"
    };

    const vector<string> MEMORY_PATTERN_KEYS = {
        "Pointwise", "Transpose", "Broadcast", "Slice"
    };

    const vector<string> SCHEDULING_KEYS = {
        "allocation_bytes_read_per_realization", "bytes_at_production",
        "bytes_at_realization", "bytes_at_root", "bytes_at_task",
        "inlined_calls", "inner_parallelism", "innermost_bytes_at_production",
        "innermost_bytes_at_realization", "innermost_bytes_at_root",
        "innermost_bytes_at_task", "innermost_loop_extent",
        "innermost_pure_loop_extent", "native_vector_size",
        "num_productions", "num_realizations", "num_scalars",
        "num_vectors", "outer_parallelism", "points_computed_minimum",
        "points_computed_per_production", "points_computed_per_realization",
        "points_computed_total", "scalar_loads_per_scalar",
        "scalar_loads_per_vector", "unique_bytes_read_per_realization",
        "unique_bytes_read_per_task", "unique_bytes_read_per_vector",
        "unique_lines_read_per_realization", "unique_lines_read_per_task",
        "unique_lines_read_per_vector", "unrolled_loop_extent",
        "vector_loads_per_vector", "vector_size", "working_set",
        "working_set_at_production", "working_set_at_realization",
        "working_set_at_root", "working_set_at_task"
    };

    for (Node &node : nodes) {
        json node_json;
        node_json["id"] = node.id;
        node_json["name"] = node.name;
        node_json["features"] = json::object();
        node_json["features"]["op_histogram"] = json::object();
        node_json["features"]["memory_patterns"] = json::object();
        node_json["features"]["scheduling"] = json::object();

        // Aggregate features from all stages
        for (const auto &stage : node.stages) {
            for (int i = 0; i < (int)PipelineFeatures::ScalarType::NumScalarTypes; i++) {
                if (!stage.features.types_in_use[i]) continue;
                for (const auto &key : OP_HISTOGRAM_KEYS) {
                    int idx = -1;
                    if (key == "Constant") idx = (int)PipelineFeatures::OpType::Const;
                    else if (key == "Cast") idx = (int)PipelineFeatures::OpType::Cast;
                    else if (key == "Variable") idx = (int)PipelineFeatures::OpType::Variable;
                    else if (key == "Param") idx = (int)PipelineFeatures::OpType::Param;
                    else if (key == "Add") idx = (int)PipelineFeatures::OpType::Add;
                    else if (key == "Sub") idx = (int)PipelineFeatures::OpType::Sub;
                    else if (key == "Mod") idx = (int)PipelineFeatures::OpType::Mod;
                    else if (key == "Mul") idx = (int)PipelineFeatures::OpType::Mul;
                    else if (key == "Div") idx = (int)PipelineFeatures::OpType::Div;
                    else if (key == "Min") idx = (int)PipelineFeatures::OpType::Min;
                    else if (key == "Max") idx = (int)PipelineFeatures::OpType::Max;
                    else if (key == "EQ") idx = (int)PipelineFeatures::OpType::EQ;
                    else if (key == "NE") idx = (int)PipelineFeatures::OpType::NE;
                    else if (key == "LT") idx = (int)PipelineFeatures::OpType::LT;
                    else if (key == "LE") idx = (int)PipelineFeatures::OpType::LE;
                    else if (key == "And") idx = (int)PipelineFeatures::OpType::And;
                    else if (key == "Or") idx = (int)PipelineFeatures::OpType::Or;
                    else if (key == "Not") idx = (int)PipelineFeatures::OpType::Not;
                    else if (key == "Select") idx = (int)PipelineFeatures::OpType::Select;
                    else if (key == "ImageCall") idx = (int)PipelineFeatures::OpType::ImageCall;
                    else if (key == "FuncCall") idx = (int)PipelineFeatures::OpType::FuncCall;
                    else if (key == "SelfCall") idx = (int)PipelineFeatures::OpType::SelfCall;
                    else if (key == "ExternCall") idx = (int)PipelineFeatures::OpType::ExternCall;
                    else if (key == "Let") idx = (int)PipelineFeatures::OpType::Let;
                    if (idx != -1) {
                        node_json["features"]["op_histogram"][key] = node_json["features"]["op_histogram"].value(key, 0) + stage.features.op_histogram[idx][i];
                    }
                }

                for (const auto &key : MEMORY_PATTERN_KEYS) {
                    vector<int> values(4, 0);
                    int type_idx = (int)PipelineFeatures::AccessType::LoadFunc;
                    if (key == "Pointwise") {
                        for (int t = 0; t < 4; t++) {
                            values[t] = stage.features.pointwise_accesses[t][i];
                        }
                    } else if (key == "Transpose") {
                        for (int t = 0; t < 4; t++) {
                            values[t] = stage.features.transpose_accesses[t][i];
                        }
                    } else if (key == "Broadcast") {
                        for (int t = 0; t < 4; t++) {
                            values[t] = stage.features.broadcast_accesses[t][i];
                        }
                    } else if (key == "Slice") {
                        for (int t = 0; t < 4; t++) {
                            values[t] = stage.features.slice_accesses[t][i];
                        }
                    }
                    for (size_t j = 0; j < 4; j++) {
                        node_json["features"]["memory_patterns"][key][j] = node_json["features"]["memory_patterns"].value(key, vector<int>(4, 0))[j] + values[j];
                    }
                }
            }

            // Scheduling features
            for (const auto &key : SCHEDULING_KEYS) {
                double value = 0.0;
                if (key == "num_realizations") value = stage.schedule_features.num_realizations;
                else if (key == "num_productions") value = stage.schedule_features.num_productions;
                else if (key == "points_computed_total") value = stage.schedule_features.points_computed_total;
                // Add more mappings as needed
                node_json["features"]["scheduling"][key] = node_json["features"]["scheduling"].value(key, 0.0) + value;
            }
        }

        node.features = node_json["features"];
        graph_json["nodes"].push_back(node_json);
    }

    for (Edge &edge : edges) {
        json edge_json;
        edge_json["source"] = edge.producer->name;
        edge_json["target"] = edge.consumer->name;
        edge_json["source_id"] = edge.producer->id;
        edge_json["target_id"] = edge.consumer->node->id;
        edge_json["features"] = json::object();
        edge_json["features"]["footprint"] = json::object();

        int j = 0;
        for (const auto &b : edge.bounds) {
            edge_json["features"]["footprint"]["Min " + std::to_string(j)] = b.first.expr.to_string();
            edge_json["features"]["footprint"]["Max " + std::to_string(j)] = b.second.expr.to_string();
            j++;
        }

        edge_json["features"]["load_jacobian"] = json::array();
        for (const auto &jac : edge.load_jacobians) {
            json jac_json = json::array();
            for (size_t i = 0; i < jac.producer_storage_dims(); i++) {
                json row = json::array();
                for (size_t j = 0; j < jac.consumer_loop_dims(); j++) {
                    auto c = jac(i, j);
                    if (!c.exists) {
                        row.push_back("_");
                    } else if (c.denominator == 1) {
                        row.push_back(c.numerator);
                    } else {
                        row.push_back(std::to_string(c.numerator) + "/" + std::to_string(c.denominator));
                    }
                }
                jac_json.push_back(row);
            }
            edge_json["features"]["load_jacobian"].push_back(jac_json);
        }

        edge.features = edge_json["features"];
        graph_json["edges"].push_back(edge_json);
    }

    // Add operations (simplified for demonstration)
    int op_id = 0;
    for (const auto &node : nodes) {
        if (node.func.name().find("conv") != string::npos) {
            json op;
            op["id"] = op_id++;
            op["type"] = "Convolution";
            op["kernel"] = vector<int>{3, 3}; // Placeholder
            graph_json["operations"].push_back(op);
        }
    }

    // Add global features (placeholder)
    graph_json["global_features"]["execution_time_ms"] = 0.0;
    graph_json["global_features"]["cache_hits"] = 0;
    graph_json["global_features"]["cache_misses"] = 0;
}

void FunctionGraph::dump(std::ostream &os) const {
    os << graph_json.dump(4) << "\n";
}

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide
