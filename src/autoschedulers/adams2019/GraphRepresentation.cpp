#include "GraphRepresentation.h"
#include "ASLog.h"
#include <nlohmann/json.hpp>
#include <sstream>

namespace Halide {
namespace Internal {
namespace Autoscheduler {

using json = nlohmann::json;

namespace {

class Featurizer : public IRVisitor {
    using IRVisitor::visit;

    Function& func;
    GraphRepresentation::Node::Stage& stage;

    int& op_bucket(PipelineFeatures::OpType op_type, Type scalar_type) {
        int type_bucket = (int)classify_type(scalar_type);
        stage.features.types_in_use[type_bucket] = true;
        return stage.features.op_histogram[(int)op_type][type_bucket];
    }

    PipelineFeatures::ScalarType classify_type(Type t) {
        if (t.is_float() && t.bits() > 32) return PipelineFeatures::ScalarType::Double;
        else if (t.is_float()) return PipelineFeatures::ScalarType::Float;
        else if (t.bits() == 1) return PipelineFeatures::ScalarType::Bool;
        else if (t.bits() <= 8) return PipelineFeatures::ScalarType::UInt8;
        else if (t.bits() <= 16) return PipelineFeatures::ScalarType::UInt16;
        else if (t.bits() <= 32) return PipelineFeatures::ScalarType::UInt32;
        else return PipelineFeatures::ScalarType::UInt64;
    }

    void visit(const Variable* op) override {
        if (op->param.defined()) op_bucket(PipelineFeatures::OpType::Param, op->type)++;
        else op_bucket(PipelineFeatures::OpType::Variable, op->type)++;
    }

    void visit(const IntImm* op) override { op_bucket(PipelineFeatures::OpType::Const, op->type)++; }
    void visit(const UIntImm* op) override { op_bucket(PipelineFeatures::OpType::Const, op->type)++; }
    void visit(const FloatImm* op) override { op_bucket(PipelineFeatures::OpType::Const, op->type)++; }
    void visit(const Add* op) override { op_bucket(PipelineFeatures::OpType::Add, op->type)++; IRVisitor::visit(op); }
    void visit(const Sub* op) override { op_bucket(PipelineFeatures::OpType::Sub, op->type)++; IRVisitor::visit(op); }
    void visit(const Mul* op) override { op_bucket(PipelineFeatures::OpType::Mul, op->type)++; IRVisitor::visit(op); }
    void visit(const Mod* op) override { op_bucket(PipelineFeatures::OpType::Mod, op->type)++; IRVisitor::visit(op); }
    void visit(const Div* op) override { op_bucket(PipelineFeatures::OpType::Div, op->type)++; IRVisitor::visit(op); }
    void visit(const Min* op) override { op_bucket(PipelineFeatures::OpType::Min, op->type)++; IRVisitor::visit(op); }
    void visit(const Max* op) override { op_bucket(PipelineFeatures::OpType::Max, op->type)++; IRVisitor::visit(op); }
    void visit(const EQ* op) override { op_bucket(PipelineFeatures::OpType::EQ, op->type)++; IRVisitor::visit(op); }
    void visit(const NE* op) override { op_bucket(PipelineFeatures::OpType::NE, op->type)++; IRVisitor::visit(op); }
    void visit(const LT* op) override { op_bucket(PipelineFeatures::OpType::LT, op->type)++; IRVisitor::visit(op); }
    void visit(const LE* op) override { op_bucket(PipelineFeatures::OpType::LE, op->type)++; IRVisitor::visit(op); }
    void visit(const GT* op) override { op_bucket(PipelineFeatures::OpType::LT, op->type)++; IRVisitor::visit(op); }
    void visit(const GE* op) override { op_bucket(PipelineFeatures::OpType::LE, op->type)++; IRVisitor::visit(op); }
    void visit(const And* op) override { op_bucket(PipelineFeatures::OpType::And, op->type)++; IRVisitor::visit(op); }
    void visit(const Or* op) override { op_bucket(PipelineFeatures::OpType::Or, op->type)++; IRVisitor::visit(op); }
    void visit(const Not* op) override { op_bucket(PipelineFeatures::OpType::Not, op->type)++; IRVisitor::visit(op); }
    void visit(const Select* op) override { op_bucket(PipelineFeatures::OpType::Select, op->type)++; IRVisitor::visit(op); }

    Scope<Expr> lets;

    void visit(const Let* op) override {
        ScopedBinding<Expr> bind(lets, op->name, op->value);
        op_bucket(PipelineFeatures::OpType::Let, op->type)++;
        IRVisitor::visit(op);
    }

    void visit(const Call* op) override {
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

    OptionalRational differentiate(const Expr& e, const std::string& v) {
        if (!expr_uses_var(e, v, lets)) return {true, 0, 1};
        else if (const Variable* var = e.as<Variable>()) {
            if (var->name == v) return {true, 1, 1};
            for (const auto& l : stage.loop) {
                if (var->name == l.var) return {true, 0, 1};
            }
            if (var->param.defined()) return {true, 0, 1};
            else if (lets.contains(var->name)) {
                std::string key = v + " " + var->name;
                if (dlets.contains(key)) return dlets.get(key);
                auto a = differentiate(lets.get(var->name), v);
                dlets.push(key, a);
                return a;
            }
            internal_error << "Encountered unbound variable in call args: " << var->name << "\n";
            return {false, 0, 0};
        } else if (const Add* op = e.as<Add>()) {
            auto a = differentiate(op->a, v);
            a += differentiate(op->b, v);
            return a;
        } else if (const Sub* op = e.as<Sub>()) {
            auto a = differentiate(op->a, v);
            auto b = differentiate(op->b, v);
            b.numerator = -b.numerator;
            a += b;
            return a;
        } else if (const Mul* op = e.as<Mul>()) {
            auto a = differentiate(op->a, v);
            if (const int64_t* ib = as_const_int(op->b)) {
                a.numerator *= *ib;
                return a;
            } else return {false, 0, 0};
        } else if (const Div* op = e.as<Div>()) {
            auto a = differentiate(op->a, v);
            if (const int64_t* ib = as_const_int(op->b)) {
                if (a.numerator != 0) a.denominator *= *ib;
                return a;
            } else return {false, 0, 0};
        } else if (const Call* op = e.as<Call>()) {
            if (op->is_intrinsic(Call::likely)) return differentiate(op->args[0], v);
        }
        return {false, 0, 0};
    }

    void visit_memory_access(const std::string& name, Type t, const std::vector<Expr>& args, PipelineFeatures::AccessType type) {
        std::vector<std::vector<OptionalRational>> matrix;
        std::vector<size_t> ones_per_row(args.size(), 0), zeros_per_row(args.size(), 0),
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
        bool is_transpose = args.size() == stage.loop.size();
        bool is_broadcast = true, is_slice = true;
        for (size_t i = 0; i < args.size(); i++) {
            bool single_one = ones_per_row[i] == 1 && zeros_per_row[i] == stage.loop.size() - 1;
            bool all_zero = zeros_per_row[i] == stage.loop.size();
            is_transpose &= single_one;
            is_broadcast &= single_one;
            is_slice &= single_one || all_zero;
        }
        for (size_t j = 0; j < stage.loop.size(); j++) {
            bool single_one = ones_per_col[j] == 1 && zeros_per_col[j] == args.size() - 1;
            bool all_zero = zeros_per_col[j] == args.size();
            is_transpose &= single_one || all_zero;
            is_broadcast &= single_one;
            is_slice &= single_one;
        }

        auto type_class = classify_type(t);
        stage.features.pointwise_accesses[(int)type][(int)type_class] += is_pointwise;
        stage.features.transpose_accesses[(int)type][(int)type_class] += is_transpose;
        stage.features.broadcast_accesses[(int)type][(int)type_class] += is_broadcast;
        stage.features.slice_accesses[(int)type][(int)type_class] += is_slice;

        for (auto* e : stage.incoming_edges) {
            if (e->producer->func.name() == name) {
                std::vector<std::vector<OptionalRational>> copy = matrix;
                e->add_load_jacobian(LoadJacobian(std::move(copy)));
            }
        }
    }

    Scope<OptionalRational> dlets;

public:
    Featurizer(Function& f, GraphRepresentation::Node::Stage& s) : func(f), stage(s) {}

    void visit_store_args(const std::string& name, Type t, std::vector<Expr> args) {
        for (auto& e : args) e = common_subexpression_elimination(simplify(e));
        visit_memory_access(name, t, args, PipelineFeatures::AccessType::Store);
    }
};

class DependsOnEstimate : public IRVisitor {
public:
    bool found_estimate = false;

private:
    void visit(const Variable* op) override {
        found_estimate |= op->param.defined();
    }
};

bool depends_on_estimate(const Expr& expr) {
    DependsOnEstimate checker;
    expr.accept(&checker);
    return checker.found_estimate;
}

}  // namespace

void GraphRepresentation::Node::required_to_computed(const Span* required, Span* computed) const {
    std::map<std::string, Expr> required_map;
    if (!region_computed_all_common_cases) {
        for (int i = 0; i < func.dimensions(); i++) {
            required_map[region_required[i].min.name()] = (int)required[i].min();
            required_map[region_required[i].max.name()] = (int)required[i].max();
        }
    }
    for (int i = 0; i < func.dimensions(); i++) {
        const auto& comp = region_computed[i];
        if (comp.equals_required) computed[i] = required[i];
        else if (comp.equals_union_of_required_with_constants) {
            computed[i] = Span(std::min(required[i].min(), comp.c_min),
                               std::max(required[i].max(), comp.c_max), false);
        } else {
            Expr min = simplify(substitute(required_map, comp.in.min));
            Expr max = simplify(substitute(required_map, comp.in.max));
            const int64_t* imin = as_const_int(min);
            const int64_t* imax = as_const_int(max);
            internal_assert(imin && imax) << min << ", " << max << "\n";
            computed[i] = Span(*imin, *imax, false);
        }
    }
}

void GraphRepresentation::Node::loop_nest_for_region(int stage_idx, const Span* computed, Span* loop) const {
    const auto& s = stages[stage_idx];
    std::map<std::string, Expr> computed_map;
    if (!s.loop_nest_all_common_cases) {
        for (int i = 0; i < func.dimensions(); i++) {
            computed_map[region_required[i].min.name()] = (int)computed[i].min();
            computed_map[region_required[i].max.name()] = (int)computed[i].max();
        }
    }
    for (size_t i = 0; i < s.loop.size(); i++) {
        const auto& l = s.loop[i];
        if (l.equals_region_computed) loop[i] = computed[l.region_computed_dim];
        else if (l.bounds_are_constant) loop[i] = Span(l.c_min, l.c_max, true);
        else {
            Expr min = simplify(substitute(computed_map, l.min));
            Expr max = simplify(substitute(computed_map, l.max));
            const int64_t* imin = as_const_int(min);
            const int64_t* imax = as_const_int(max);
            internal_assert(imin && imax) << min << ", " << max << "\n";
            loop[i] = Span(*imin, *imax, false);
        }
    }
}

void GraphRepresentation::Edge::BoundInfo::BoundInfo(const Expr& e, const Node::Stage& consumer, bool dependent)
    : expr(e), depends_on_estimate(dependent) {
    const Add* add = expr.as<Add>();
    const Mul* mul = add ? add->a.as<Mul>() : expr.as<Mul>();
    const IntImm* coeff_imm = mul ? mul->b.as<IntImm>() : nullptr;
    const IntImm* constant_imm = add ? add->b.as<IntImm>() : nullptr;
    Expr v = mul ? mul->a : add ? add->a : expr;
    const Variable* var = v.as<Variable>();

    if (const IntImm* c = e.as<IntImm>()) {
        affine = true;
        coeff = 0;
        constant = c->value;
    } else if (var && (!mul || coeff_imm) && (!add || constant_imm)) {
        affine = true;
        coeff = mul ? coeff_imm->value : 1;
        constant = add ? constant_imm->value : 0;
        consumer_dim = -1;
        for (int i = 0; i < (int)consumer.loop.size(); i++) {
            const auto& in = consumer.loop[i];
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

void GraphRepresentation::Edge::add_load_jacobian(LoadJacobian j1) {
    for (auto& j2 : load_jacobians) {
        if (j2.merge(j1)) return;
    }
    load_jacobians.emplace_back(std::move(j1));
}

void GraphRepresentation::Edge::expand_footprint(const Span* consumer_loop, Span* producer_required) const {
    const auto& symbolic_loop = consumer->loop;
    std::map<std::string, Expr> s;
    if (!all_bounds_affine) {
        for (size_t i = 0; i < symbolic_loop.size(); i++) {
            auto p = consumer_loop[i];
            const std::string& var = symbolic_loop[i].var;
            s[consumer->node->func.name() + "." + var + ".min"] = (int)p.min();
            s[consumer->node->func.name() + "." + var + ".max"] = (int)p.max();
        }
    }
    for (int i = 0; i < producer->func.dimensions(); i++) {
        bool bounds_are_constant = true;
        auto eval_bound = [&](const BoundInfo& b) {
            bounds_are_constant &= !b.depends_on_estimate;
            if (b.affine) {
                if (b.coeff == 0) return b.constant;
                const auto& src_pair = consumer_loop[b.consumer_dim];
                int64_t src = b.uses_max ? src_pair.max() : src_pair.min();
                bounds_are_constant &= src_pair.constant_extent();
                return src * b.coeff + b.constant;
            } else {
                Expr substituted = substitute(s, b.expr);
                Expr e = simplify(substituted);
                const int64_t* i = as_const_int(e);
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

void GraphRepresentation::LoadJacobian::dump(std::ostream& os, const char* prefix) const {
    if (count() > 1) os << prefix << count() << " x\n";
    for (size_t i = 0; i < producer_storage_dims(); i++) {
        os << prefix << "  [";
        for (size_t j = 0; j < consumer_loop_dims(); j++) {
            const auto& c = (*this)(i, j);
            if (!c.exists) os << " _  ";
            else if (c.denominator == 1) os << " " << c.numerator << "  ";
            else os << c.numerator << "/" << c.denominator << " ";
        }
        os << "]\n";
    }
    os << "\n";
}

GraphRepresentation::GraphRepresentation(const std::vector<Function>& outputs, const Target& target) {
    std::map<std::string, Function> env = build_environment(outputs);
    std::vector<std::string> order = topological_order(outputs, env);

    class ApplyParamEstimates : public IRMutator {
        Expr visit(const Variable* op) override {
            if (op->param.defined()) {
                Expr expr;
                if (!op->param.is_buffer()) expr = op->param.estimate();
                else {
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
            }
            return op;
        }
    } apply_param_estimates;

    nodes.resize(order.size());
    std::map<Function, Node*, Function::Compare> node_map;
    for (size_t i = 0; i < order.size(); i++) {
        Function f = env[order[order.size() - i - 1]];
        nodes[i].func = f;
        nodes[i].id = (int)i;
        nodes[i].max_id = (int)order.size();
        nodes[i].name = f.name();
        node_map[f] = &nodes[i];
    }

    int stage_count = 0;
    for (size_t i = order.size(); i > 0; i--) {
        Node& node = nodes[order.size() - i];
        Function consumer = node.func;
        Scope<Interval> scope;

        for (int j = 0; j < consumer.dimensions(); j++) {
            Halide::Var min_var(consumer.name() + "." + consumer.args()[j] + ".min");
            Halide::Var max_var(consumer.name() + "." + consumer.args()[j] + ".max");
            Interval interval(min_var, max_var);
            scope.push(consumer.args()[j], interval);
            node.region_required.emplace_back(Node::SymbolicInterval{min_var, max_var});
        }

        auto pure_args = node.func.args();
        for (int s = 0; s <= (int)consumer.updates().size(); s++) {
            stage_count++;
            if (s == 0) node.stages.emplace_back(Node::Stage(consumer.definition().schedule().get_stage(consumer)));
            else node.stages.emplace_back(Node::Stage(consumer.update(s - 1).schedule().get_stage(consumer)));
        }

        for (int s = 0; s <= (int)consumer.updates().size(); s++) {
            auto& stage = node.stages[s];
            stage.node = &node;
            stage.name = consumer.name();
            if (s > 0) stage.name += ".update(" + std::to_string(s - 1) + ")";
            stage.index = s;

            const Definition& def = (s == 0) ? consumer.definition() : consumer.update(s - 1);
            const StageSchedule& sched = def.schedule();

            Scope<Interval> stage_scope_with_concrete_rvar_bounds, stage_scope_with_symbolic_rvar_bounds;
            stage_scope_with_concrete_rvar_bounds.set_containing_scope(&scope);
            stage_scope_with_symbolic_rvar_bounds.set_containing_scope(&scope);
            for (const auto& rv : sched.rvars()) {
                Expr min = simplify(apply_param_estimates.mutate(rv.min));
                Expr max = simplify(apply_param_estimates.mutate(rv.min + rv.extent - 1));
                stage_scope_with_concrete_rvar_bounds.push(rv.var, Interval(min, max));
                min = Variable::make(Int(32), consumer.name() + "." + rv.var + ".min");
                max = Variable::make(Int(32), consumer.name() + "." + rv.var + ".max");
                stage_scope_with_symbolic_rvar_bounds.push(rv.var, Interval(min, max));
            }

            if (s == 0) node.region_computed.resize(consumer.dimensions());
            FuncValueBounds func_value_bounds = compute_function_value_bounds(order, env);
            for (int j = 0; j < consumer.dimensions(); j++) {
                Interval in = bounds_of_expr_in_scope(def.args()[j], stage_scope_with_concrete_rvar_bounds, func_value_bounds);
                internal_assert(in.is_bounded()) << "Region computed of " << consumer.name() << " is unbounded: [" << in.min << " " << in.max << "]\n";
                if (s == 0) node.region_computed[j].in = in;
                else node.region_computed[j].in.include(in);
            }
            if (s == (int)consumer.updates().size()) {
                node.region_computed_all_common_cases = true;
                for (int j = 0; j < consumer.dimensions(); j++) {
                    const auto& req = node.region_required[j];
                    auto& comp = node.region_computed[j];
                    comp.depends_on_estimate = depends_on_estimate(comp.in.min) || depends_on_estimate(comp.in.max);
                    comp.in.min = simplify(apply_param_estimates.mutate(comp.in.min));
                    comp.in.max = simplify(apply_param_estimates.mutate(comp.in.max));
                    if (equal(comp.in.min, req.min) && equal(comp.in.max, req.max)) comp.equals_required = true;
                    else {
                        const Min* min = comp.in.min.as<Min>();
                        const Max* max = comp.in.max.as<Max>();
                        const int64_t* min_b = min ? as_const_int(min->b) : nullptr;
                        const int64_t* max_b = max ? as_const_int(max->b) : nullptr;
                        if (min_b && max_b && equal(min->a, req.min) && equal(max->a, req.max)) {
                            comp.equals_union_of_required_with_constants = true;
                            comp.c_min = *min_b;
                            comp.c_max = *max_b;
                        } else node.region_computed_all_common_cases = false;
                    }
                }
            }

            user_assert(sched.splits().empty()) << "Func \"" << consumer.name() << "\" has scheduling directives; remove or conditionalize with `if (!auto_schedule)`.";
            stage.loop_nest_all_common_cases = true;
            for (size_t i = 0; i < sched.dims().size(); i++) {
                const auto& d = sched.dims()[i];
                if (!stage_scope_with_symbolic_rvar_bounds.contains(d.var)) continue;

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
                    if (l.var == pure_args[j]) l.pure_dim = j;
                    if (equal(l.min, node.region_computed[j].in.min) && equal(l.max, node.region_computed[j].in.max)) {
                        l.equals_region_computed = true;
                        l.region_computed_dim = j;
                        break;
                    }
                }

                if (!l.equals_region_computed) {
                    const int64_t* c_min = as_const_int(l.min), *c_max = as_const_int(l.max);
                    if (c_min && c_max) {
                        l.bounds_are_constant = true;
                        l.c_min = *c_min;
                        l.c_max = *c_max;
                    } else l.bounds_are_constant = false;
                }

                stage.loop_nest_all_common_cases &= (l.bounds_are_constant || l.equals_region_computed);
                stage.loop.emplace_back(std::move(l));
            }

            class CheckTypes : public IRVisitor {
                void visit(const IntImm* op) override { check_type(op->type); }
                void visit(const UIntImm* op) override { check_type(op->type); }
                void visit(const FloatImm* op) override { check_type(op->type); }
                void visit(const Variable* op) override { check_type(op->type); }
                void visit(const Call* op) override {
                    calls[op->name]++;
                    IRVisitor::visit(op);
                    check_type(op->type);
                    if (op->call_type == Call::Halide || op->call_type == Call::Image) {
                        is_pointwise &= op->args.size() == func.args().size();
                        for (size_t i = 0; i < op->args.size(); i++) {
                            const Variable* v = op->args[i].as<Variable>();
                            is_pointwise &= v && v->name == func.args()[i];
                        }
                    }
                }
                void visit(const Cast* op) override { IRVisitor::visit(op); check_type(op->type); }
                void visit(const Reinterpret* op) override { IRVisitor::visit(op); check_type(op->type); }
                void check_type(Type t) {
                    if (t.bits() > 1 && (!narrowest_type.bits() || t.bits() < narrowest_type.bits())) narrowest_type = t;
                }
                Function func;
            public:
                bool is_pointwise = true;
                Type narrowest_type;
                std::map<std::string, int> calls;
                explicit CheckTypes(const Function& f) : func(f) {}
            };

            std::vector<Expr> exprs_vector = def.args();
            exprs_vector.insert(exprs_vector.end(), def.values().begin(), def.values().end());
            if (def.predicate().defined()) exprs_vector.push_back(def.predicate());
            Expr exprs = Call::make(Int(32), "dummy", exprs_vector, Call::Extern);

            CheckTypes checker(consumer);
            exprs.accept(&checker);

            Type widest_output_type = def.values()[0].type();
            int bytes_per_point = 0;
            for (const auto& e : def.values()) {
                bytes_per_point += e.type().bytes();
                if (e.type().bytes() > widest_output_type.bytes()) widest_output_type = e;
            }
            if (s == 0) node.bytes_per_point = bytes_per_point;

            stage.vector_size = target.natural_vector_size(checker.narrowest_type);
            if (s == 0) node.vector_size = stage.vector_size;
            else node.vector_size = std::max(node.vector_size, stage.vector_size);

            node.is_output = false;
            for (const auto& o : outputs) node.is_output |= o.same_as(node.func);

            if (node.is_output) {
                std::map<std::string, Span> estimates;
                for (const auto& b : consumer.schedule().estimates()) {
                    const int64_t* i_min = as_const_int(b.min);
                    const int64_t* i_extent = as_const_int(b.extent);
                    user_assert(i_min && i_extent) << "Estimate/bound not constant in \"" << consumer.name() << "\", var:" << b.var;
                    estimates[b.var] = Span(*i_min, *i_min + *i_extent - 1, false);
                }
                for (const auto& b : consumer.schedule().bounds()) {
                    const int64_t* i_min = as_const_int(b.min);
                    const int64_t* i_extent = as_const_int(b.extent);
                    if (i_min && i_extent) estimates[b.var] = Span(*i_min, *i_min + *i_extent - 1, true);
                }
                for (int i = 0; i < consumer.dimensions(); i++) {
                    auto it = estimates.find(consumer.args()[i]);
                    user_assert(it != estimates.end()) << "Need estimate on dimension " << i << " of \"" << consumer.name() << "\"";
                    node.estimated_region_required.push_back(it->second);
                }
            }

            exprs = apply_param_estimates.mutate(exprs);
            node.is_pointwise = !node.func.has_update_definition();
            node.is_boundary_condition = node.is_pointwise && starts_with(node.func.name(), "repeat_edge");

            auto boxes = boxes_required(exprs, stage_scope_with_symbolic_rvar_bounds, func_value_bounds);
            bool any_incoming_edges = false;
            for (auto& p : boxes) {
                auto it = env.find(p.first);
                if (it != env.end() && p.first != consumer.name()) {
                    Edge edge;
                    edge.consumer = &stage;
                    edge.producer = node_map.at(env[p.first]);
                    edge.all_bounds_affine = true;
                    for (Interval& in : p.second.bounds) {
                        internal_assert(in.is_bounded()) << "Unbounded relationship: " << edge.producer->func.name() << " -> " << edge.consumer->name << "\n";
                        bool min_dependent = depends_on_estimate(in.min);
                        bool max_dependent = depends_on_estimate(in.max);
                        Expr min_value = simplify(apply_param_estimates.mutate(in.min));
                        Expr max_value = simplify(apply_param_estimates.mutate(in.max));
                        Edge::BoundInfo min(min_value, *edge.consumer, min_dependent);
                        Edge::BoundInfo max(max_value, *edge.consumer, max_dependent);
                        edge.bounds.emplace_back(std::move(min), std::move(max));
                        edge.all_bounds_affine &= edge.bounds.back().first.affine && edge.bounds.back().second.affine;
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
    for (auto& n : nodes) {
        for (auto& s : n.stages) {
            s.id = i++;
            s.max_id = stage_count;
        }
    }

    for (auto& edge : edges) {
        edge.producer->outgoing_edges.push_back(&edge);
        edge.consumer->incoming_edges.push_back(&edge);
    }

    for (size_t i = nodes.size(); i > 0; i--) {
        auto& n = nodes[i - 1];
        for (auto& s : n.stages) {
            s.dependencies.resize(nodes.size(), false);
            for (auto* e : s.incoming_edges) {
                s.dependencies[e->producer->id] = true;
                for (auto& s2 : e->producer->stages) {
                    for (size_t j = 0; j < nodes.size(); j++) {
                        s.dependencies[j] = s.dependencies[j] || s2.dependencies[j];
                    }
                }
            }
        }
    }

    featurize();
    to_json(std::stringstream());
}

void GraphRepresentation::featurize() {
    for (Node& node : nodes) {
        for (size_t stage_idx = 0; stage_idx < node.stages.size(); stage_idx++) {
            Node::Stage& stage = node.stages[stage_idx];
            Featurizer featurizer(node.func, stage);

            if (node.func.extern_definition_proxy_expr().get()) {
                Expr v = simplify(node.func.extern_definition_proxy_expr());
                v = common_subexpression_elimination(v);
                v.accept(&featurizer);
            } else {
                Definition def = node.func.definition();
                if (stage_idx > 0) def = node.func.updates()[stage_idx - 1];
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
        }
    }
}

void GraphRepresentation::dump(std::ostream& os) const {
    for (const Node& n : nodes) {
        os << "Node: " << n.func.name() << "\n";
        os << "  Symbolic region required:\n";
        for (const auto& i : n.region_required) os << "    " << i.min << ", " << i.max << "\n";
        os << "  Region computed:\n";
        for (const auto& i : n.region_computed) os << "    " << i.in.min << ", " << i.in.max << "\n";
        for (size_t i = 0; i < n.stages.size(); i++) {
            os << "  Stage " << i << ":\n";
            for (const auto& l : n.stages[i].loop) os << "    " << l.var << " " << l.min << " " << l.max << "\n";
            n.stages[i].features.dump(os);
        }
        os << "  pointwise: " << n.is_pointwise << " boundary condition: " << n.is_boundary_condition
           << " wrapper: " << n.is_wrapper << " input: " << n.is_input << " output: " << n.is_output << "\n";
    }
    for (const Edge& e : edges) {
        os << "Edge: " << e.producer->func.name() << " -> " << e.consumer->name << "\n";
        os << "  Footprint:\n";
        int j = 0;
        for (const auto& i : e.bounds) {
            os << "    Min " << j << ": " << i.first.expr << "\n";
            os << "    Max " << j << ": " << i.second.expr << "\n";
            j++;
        }
        os << "  Load Jacobians:\n";
        for (const auto& jac : e.load_jacobians) jac.dump(os, "  ");
    }
}

void GraphRepresentation::to_json(std::ostream& os) const {
    json graph;
    graph["nodes"] = json::array();
    graph["edges"] = json::array();
    graph["operations"] = json::array();
    graph["global_features"] = json::object();

    std::map<std::string, int> node_id_map;
    std::vector<json> nodes_json;
    std::vector<json> edges_json;

    const std::vector<std::string> op_histogram_keys = {
        "Add", "And", "Cast", "Constant", "Div", "EQ", "ExternCall", "FuncCall",
        "ImageCall", "LE", "LT", "Let", "Max", "Min", "Mod", "Mul", "NE", "Not",
        "Or", "Param", "Select", "SelfCall", "Sub", "Variable"
    };
    const std::vector<std::string> memory_pattern_keys = {"Pointwise", "Transpose", "Broadcast", "Slice"};

    for (const auto& node : nodes) {
        json node_json;
        node_json["id"] = node.id;
        node_json["name"] = node.name;
        node_json["features"] = json::object();
        json& features = node_json["features"];
        features["op_histogram"] = json::object();
        features["memory_patterns"] = json::object();
        features["scheduling"] = json::object();

        for (const auto& key : op_histogram_keys) features["op_histogram"][key] = 0;
        for (const auto& key : memory_pattern_keys) features["memory_patterns"][key] = std::vector<int>{0, 0, 0, 0};

        for (const auto& stage : node.stages) {
            for (int i = 0; i < (int)PipelineFeatures::ScalarType::NumScalarTypes; i++) {
                if (!stage.features.types_in_use[i]) continue;
                for (int j = 0; j < (int)PipelineFeatures::OpType::NumOpTypes; j++) {
                    std::string key;
                    switch (static_cast<PipelineFeatures::OpType>(j)) {
                        case PipelineFeatures::OpType::Add: key = "Add"; break;
                        case PipelineFeatures::OpType::And: key = "And"; break;
                        case PipelineFeatures::OpType::Cast: key = "Cast"; break;
                        case PipelineFeatures::OpType::Const: key = "Constant"; break;
                        case PipelineFeatures::OpType::Div: key = "Div"; break;
                        case PipelineFeatures::OpType::EQ: key = "EQ"; break;
                        case PipelineFeatures::OpType::ExternCall: key = "ExternCall"; break;
                        case PipelineFeatures::OpType::FuncCall: key = "FuncCall"; break;
                        case PipelineFeatures::OpType::ImageCall: key = "ImageCall"; break;
                        case PipelineFeatures::OpType::LE: key = "LE"; break;
                        case PipelineFeatures::OpType::LT: key = "LT"; break;
                        case PipelineFeatures::OpType::Let: key = "Let"; break;
                        case PipelineFeatures::OpType::Max: key = "Max"; break;
                        case PipelineFeatures::OpType::Min: key = "Min"; break;
                        case PipelineFeatures::OpType::Mod: key = "Mod"; break;
                        case PipelineFeatures::OpType::Mul: key = "Mul"; break;
                        case PipelineFeatures::OpType::NE: key = "NE"; break;
                        case PipelineFeatures::OpType::Not: key = "Not"; break;
                        case PipelineFeatures::OpType::Or: key = "Or"; break;
                        case PipelineFeatures::OpType::Param: key = "Param"; break;
                        case PipelineFeatures::OpType::Select: key = "Select"; break;
                        case PipelineFeatures::OpType::SelfCall: key = "SelfCall"; break;
                        case PipelineFeatures::OpType::Sub: key = "Sub"; break;
                        case PipelineFeatures::OpType::Variable: key = "Variable"; break;
                        default: continue;
                    }
                    features["op_histogram"][key] = features["op_histogram"][key].get<int>() + stage.features.op_histogram[j][i];
                }
                for (int j = 0; j < (int)PipelineFeatures::AccessType::NumAccessTypes; j++) {
                    features["memory_patterns"]["Pointwise"][j] += stage.features.pointwise_accesses[j][i];
                    features["memory_patterns"]["Transpose"][j] += stage.features.transpose_accesses[j][i];
                    features["memory_patterns"]["Broadcast"][j] += stage.features.broadcast_accesses[j][i];
                    features["memory_patterns"]["Slice"][j] += stage.features.slice_accesses[j][i];
                }
            }
        }

        nodes_json.push_back(node_json);
        node_id_map[node.name] = node.id;
    }

    for (const auto& edge : edges) {
        json edge_json;
        edge_json["source"] = edge.producer->name;
        edge_json["target"] = edge.consumer->name;
        edge_json["source_id"] = edge.producer->id;
        edge_json["target_id"] = edge.consumer->node->id;
        edge_json["features"] = json::object();
        json& e_features = edge_json["features"];

        json footprint;
        int j = 0;
        for (const auto& b : edge.bounds) {
            footprint["Min " + std::to_string(j)] = b.first.expr.to_string();
            footprint["Max " + std::to_string(j)] = b.second.expr.to_string();
            j++;
        }
        e_features["footprint"] = footprint;

        json jacobians = json::array();
        for (const auto& jac : edge.load_jacobians) {
            json jac_json = json::array();
            for (size_t i = 0; i < jac.producer_storage_dims(); i++) {
                json row = json::array();
                for (size_t k = 0; k < jac.consumer_loop_dims(); k++) {
                    auto c = jac(i, k);
                    if (!c.exists) row.push_back("_");
                    else if (c.denominator == 1) row.push_back(c.numerator);
                    else row.push_back(std::to_string(c.numerator) + "/" + std::to_string(c.denominator));
                }
                jac_json.push_back(row);
            }
            jacobians.push_back(jac_json);
        }
        e_features["load_jacobian"] = jacobians;

        edges_json.push_back(edge_json);
    }

    graph["nodes"] = nodes_json;
    graph["edges"] = edges_json;
    graph_json = graph;
    os << graph.dump();
}

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide
