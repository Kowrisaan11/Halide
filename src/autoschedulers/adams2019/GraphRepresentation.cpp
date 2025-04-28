#include "GraphRepresentation.h"

#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>

#include "ASLog.h"
#include "Errors.h"

namespace Halide {
namespace Internal {
namespace Autoscheduler {

namespace fs = std::filesystem;

namespace {

class Featurizer : public IRVisitor {
    using IRVisitor::visit;

    Function &func;
    PipelineFeatures &features;

    int &op_bucket(PipelineFeatures::OpType op_type, Type scalar_type) {
        int type_bucket = (int)classify_type(scalar_type);
        features.types_in_use[type_bucket] = true;
        return features.op_histogram[(int)op_type][type_bucket];
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

    OptionalRational differentiate(const Expr &e, const string &v) {
        if (!expr_uses_var(e, v, lets)) {
            return {true, 0, 1};
        } else if (const Variable *var = e.as<Variable>()) {
            if (var->name == v) {
                return {true, 1, 1};
            }
            return {true, 0, 1};
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
            }
            return {false, 0, 0};
        } else if (const Div *op = e.as<Div>()) {
            auto a = differentiate(op->a, v);
            if (const int64_t *ib = as_const_int(op->b)) {
                if (a.numerator != 0) {
                    a.denominator *= *ib;
                }
                return a;
            }
            return {false, 0, 0};
        } else if (const Call *op = e.as<Call>()) {
            if (op->is_intrinsic(Call::likely)) {
                return differentiate(op->args[0], v);
            }
        }
        return {false, 0, 0};
    }

    void visit_memory_access(const std::string &name, Type t, const vector<Expr> &args, PipelineFeatures::AccessType type) {
        vector<vector<OptionalRational>> matrix;
        matrix.resize(args.size());
        for (size_t i = 0; i < args.size(); i++) {
            matrix[i].resize(0);  // Simplified: no loop differentiation
        }
        auto type_class = classify_type(t);
        features.pointwise_accesses[(int)type][(int)type_class]++;
        features.transpose_accesses[(int)type][(int)type_class]++;
        features.broadcast_accesses[(int)type][(int)type_class]++;
        features.slice_accesses[(int)type][(int)type_class]++;
    }

public:
    Featurizer(Function &f, PipelineFeatures &pf)
        : func(f), features(pf) {
    }

    void visit_store_args(const std::string &name, Type t, vector<Expr> args) {
        for (auto &e : args) {
            e = common_subexpression_elimination(simplify(e));
        }
        visit_memory_access(name, t, args, PipelineFeatures::AccessType::Store);
    }
};

std::vector<int> parse_dimensions(const std::string &dim_str) {
    std::vector<int> dims;
    std::stringstream ss(dim_str);
    std::string num;
    while (std::getline(ss, num, ',')) {
        num.erase(std::remove_if(num.begin(), num.end(), [](unsigned char c) { return std::isspace(c); }), num.end());
        try {
            dims.push_back(std::stoi(num));
        } catch (const std::invalid_argument &e) {
            std::cerr << "Error parsing dimension: '" << num << "' in '" << dim_str << "'" << std::endl;
            return std::vector<int>();
        }
    }
    return dims;
}

std::vector<int> parse_kernel(const std::string &kernel_str) {
    std::vector<int> kernel;
    std::regex num_regex("[-]?\\d+");
    auto begin = std::sregex_iterator(kernel_str.begin(), kernel_str.end(), num_regex);
    auto end = std::sregex_iterator();
    for (auto i = begin; i != end; ++i) {
        try {
            kernel.push_back(std::stoi(i->str()));
        } catch (const std::invalid_argument &e) {
            std::cerr << "Error parsing kernel value: '" << i->str() << "' in '" << kernel_str << "'" << std::endl;
            return std::vector<int>();
        }
    }
    return kernel;
}

}  // namespace

void GraphRepresentation::featurize(const vector<Function> &outputs) {
    map<string, Function> env = build_environment(outputs);
    vector<string> order = topological_order(outputs, env);

    nodes.clear();
    for (size_t i = 0; i < order.size(); ++i) {
        Function f = env[order[order.size() - 1 - i]];
        Node node;
        node.name = f.name();
        node.id = static_cast<int>(i);
        node.features = PipelineFeatures();
        node.sched_features = ScheduleFeatures();
        Featurizer featurizer(f, node.features);

        if (f.extern_definition_proxy_expr().get()) {
            Expr v = simplify(f.extern_definition_proxy_expr());
            v = common_subexpression_elimination(v);
            v.accept(&featurizer);
        } else {
            Definition def = f.definition();
            for (auto v : def.values()) {
                featurizer.visit_store_args(f.name(), v.type(), def.args());
                v = common_subexpression_elimination(simplify(v));
                v.accept(&featurizer);
            }
            for (auto v : def.args()) {
                v = common_subexpression_elimination(simplify(v));
                v.accept(&featurizer);
            }
            for (size_t u = 0; u < f.updates().size(); ++u) {
                Definition update = f.updates()[u];
                for (auto v : update.values()) {
                    featurizer.visit_store_args(f.name(), v.type(), update.args());
                    v = common_subexpression_elimination(simplify(v));
                    v.accept(&featurizer);
                }
                for (auto v : update.args()) {
                    v = common_subexpression_elimination(simplify(v));
                    v.accept(&featurizer);
                }
            }
        }
        nodes.push_back(node);
    }
}

GraphRepresentation::GraphRepresentation(const vector<Function> &outputs, const Target &target,
                                         const string &stdout_content, double execution_time_ms,
                                         const vector<PipelineFeatures> &pipeline_features,
                                         const vector<ScheduleFeatures> &schedule_features) {
    featurize(outputs);

    // Update features from provided pipeline_features and schedule_features
    for (size_t i = 0; i < nodes.size() && i < pipeline_features.size() && i < schedule_features.size(); ++i) {
        nodes[i].features = pipeline_features[i];
        nodes[i].sched_features = schedule_features[i];
    }

    // Build edges (simplified: assume dependencies from topological order)
    map<string, int> node_id_map;
    for (size_t i = 0; i < nodes.size(); ++i) {
        node_id_map[nodes[i].name] = nodes[i].id;
    }

    for (const auto &n : nodes) {
        Function f = n.func;
        auto boxes = boxes_required(f.definition().values(), Scope<Interval>(), compute_function_value_bounds(outputs));
        for (const auto &p : boxes) {
            if (node_id_map.find(p.first) != node_id_map.end() && p.first != n.name) {
                Edge edge;
                edge.source = p.first;
                edge.target = n.name;
                edge.source_id = node_id_map[p.first];
                edge.target_id = node_id_map[n.name];
                edge.features = json::object();
                edge.features["footprint"] = json::object();
                edge.features["load_jacobian"] = json::array();
                edges.push_back(edge);
            }
        }
    }

    // Parse operations from stdout_content
    std::istringstream stdout_ss(stdout_content);
    std::string line;
    int op_id = 0;
    json current_op;
    operations = json::array();
    while (std::getline(stdout_ss, line)) {
        if (line.find("Convolving") != std::string::npos) {
            if (!current_op.empty()) operations.push_back(current_op);
            current_op = json{{"id", op_id++}, {"type", "Convolution"}};
            std::regex dim_kernel_regex("with kernel \\[([-]?\\d+,? ?)+\\]");
            std::smatch match;
            if (std::regex_search(line, match, dim_kernel_regex)) {
                current_op["kernel"] = parse_kernel(match[0].str().substr(match[0].str().find('[')));
            }
        } else if (line.find("Resampling") != std::string::npos) {
            if (!current_op.empty()) operations.push_back(current_op);
            current_op = json{{"id", op_id++}, {"type", "Resampling"}};
            std::regex dim_regex("from (\\d+, \\d+, \\d+) to (\\d+, \\d+, \\d+)");
            std::smatch match;
            if (std::regex_search(line, match, dim_regex)) {
                current_op["input_dims"] = parse_dimensions(match[1].str());
                current_op["output_dims"] = parse_dimensions(match[2].str());
            }
        } else if (line.find("Pooling") != std::string::npos) {
            if (!current_op.empty()) operations.push_back(current_op);
            current_op = json{{"id", op_id++}, {"type", "Pooling"}};
            std::regex stride_kernel_regex("stride: (\\d+) and kernel \\[([-]?\\d+,? ?)+\\]");
            std::smatch match;
            if (std::regex_search(line, match, stride_kernel_regex)) {
                current_op["stride"] = std::stoi(match[1].str());
                current_op["kernel"] = parse_kernel(match[2].str());
            }
        } else if (line.find("Approx size:") != std::string::npos) {
            std::regex size_regex("Approx size: (\\d+, \\d+, \\d+)");
            std::smatch match;
            if (std::regex_search(line, match, size_regex)) {
                if (!current_op.empty() && current_op.value("output_dims", std::vector<int>{}).empty()) {
                    current_op["output_dims"] = parse_dimensions(match[1].str());
                }
            }
        }
    }
    if (!current_op.empty()) operations.push_back(current_op);

    // Global features
    global_features = json::object();
    global_features["execution_time_ms"] = execution_time_ms;
    global_features["cache_hits"] = 0;  // Placeholder
    global_features["cache_misses"] = 0;  // Placeholder
}

json GraphRepresentation::to_json() const {
    json graph = json::object();
    graph["nodes"] = json::array();
    graph["edges"] = edges;
    graph["operations"] = operations;
    graph["global_features"] = global_features;

    const std::vector<std::string> OP_HISTOGRAM_KEYS = {
        "Const", "Cast", "Variable", "Param", "Add", "Sub", "Mod", "Mul", "Div",
        "Min", "Max", "EQ", "NE", "LT", "LE", "And", "Or", "Not", "Select",
        "ImageCall", "FuncCall", "SelfCall", "ExternCall", "Let"
    };
    const std::vector<std::string> MEMORY_PATTERN_KEYS = {
        "Pointwise", "Transpose", "Broadcast", "Slice"
    };
    const std::vector<std::string> SCHEDULING_KEYS = {
        "num_realizations", "num_productions", "points_computed_per_realization",
        "points_computed_per_production", "points_computed_total",
        "points_computed_minimum", "innermost_loop_extent",
        "innermost_pure_loop_extent", "unrolled_loop_extent",
        "inner_parallelism", "outer_parallelism", "bytes_at_realization",
        "bytes_at_production", "bytes_at_root", "innermost_bytes_at_realization",
        "innermost_bytes_at_production", "innermost_bytes_at_root",
        "inlined_calls", "unique_bytes_read_per_realization",
        "unique_lines_read_per_realization", "allocation_bytes_read_per_realization",
        "working_set", "vector_size", "native_vector_size", "num_vectors",
        "num_scalars", "scalar_loads_per_vector", "vector_loads_per_vector",
        "scalar_loads_per_scalar", "bytes_at_task", "innermost_bytes_at_task",
        "unique_bytes_read_per_vector", "unique_lines_read_per_vector",
        "unique_bytes_read_per_task", "unique_lines_read_per_task",
        "working_set_at_task", "working_set_at_production",
        "working_set_at_realization", "working_set_at_root"
    };

    for (const auto &n : nodes) {
        json node = json{{"id", n.id}, {"name", n.name}, {"features", json::object()}};
        json &features = node["features"];

        features["op_histogram"] = json::object();
        for (size_t i = 0; i < OP_HISTOGRAM_KEYS.size(); ++i) {
            features["op_histogram"][OP_HISTOGRAM_KEYS[i]] =
                n.features.op_histogram[i][(int)PipelineFeatures::ScalarType::UInt32];
        }

        features["memory_patterns"] = json::object();
        for (const auto &key : MEMORY_PATTERN_KEYS) {
            features["memory_patterns"][key] = json::array();
            for (size_t i = 0; i < 4; ++i) {
                if (key == "Pointwise") {
                    features["memory_patterns"][key].push_back(
                        n.features.pointwise_accesses[i][(int)PipelineFeatures::ScalarType::UInt32]);
                } else if (key == "Transpose") {
                    features["memory_patterns"][key].push_back(
                        n.features.transpose_accesses[i][(int)PipelineFeatures::ScalarType::UInt32]);
                } else if (key == "Broadcast") {
                    features["memory_patterns"][key].push_back(
                        n.features.broadcast_accesses[i][(int)PipelineFeatures::ScalarType::UInt32]);
                } else if (key == "Slice") {
                    features["memory_patterns"][key].push_back(
                        n.features.slice_accesses[i][(int)PipelineFeatures::ScalarType::UInt32]);
                }
            }
        }

        features["scheduling"] = json::object();
        features["scheduling"]["num_realizations"] = n.sched_features.num_realizations;
        features["scheduling"]["num_productions"] = n.sched_features.num_productions;
        features["scheduling"]["points_computed_per_realization"] = n.sched_features.points_computed_per_realization;
        features["scheduling"]["points_computed_per_production"] = n.sched_features.points_computed_per_production;
        features["scheduling"]["points_computed_total"] = n.sched_features.points_computed_total;
        features["scheduling"]["points_computed_minimum"] = n.sched_features.points_computed_minimum;
        features["scheduling"]["innermost_loop_extent"] = n.sched_features.innermost_loop_extent;
        features["scheduling"]["innermost_pure_loop_extent"] = n.sched_features.innermost_pure_loop_extent;
        features["scheduling"]["unrolled_loop_extent"] = n.sched_features.unrolled_loop_extent;
        features["scheduling"]["inner_parallelism"] = n.sched_features.inner_parallelism;
        features["scheduling"]["outer_parallelism"] = n.sched_features.outer_parallelism;
        features["scheduling"]["bytes_at_realization"] = n.sched_features.bytes_at_realization;
        features["scheduling"]["bytes_at_production"] = n.sched_features.bytes_at_production;
        features["scheduling"]["bytes_at_root"] = n.sched_features.bytes_at_root;
        features["scheduling"]["innermost_bytes_at_realization"] = n.sched_features.innermost_bytes_at_realization;
        features["scheduling"]["innermost_bytes_at_production"] = n.sched_features.innermost_bytes_at_production;
        features["scheduling"]["innermost_bytes_at_root"] = n.sched_features.innermost_bytes_at_root;
        features["scheduling"]["inlined_calls"] = n.sched_features.inlined_calls;
        features["scheduling"]["unique_bytes_read_per_realization"] = n.sched_features.unique_bytes_read_per_realization;
        features["scheduling"]["unique_lines_read_per_realization"] = n.sched_features.unique_lines_read_per_realization;
        features["scheduling"]["allocation_bytes_read_per_realization"] = n.sched_features.allocation_bytes_read_per_realization;
        features["scheduling"]["working_set"] = n.sched_features.working_set;
        features["scheduling"]["vector_size"] = n.sched_features.vector_size;
        features["scheduling"]["native_vector_size"] = n.sched_features.native_vector_size;
        features["scheduling"]["num_vectors"] = n.sched_features.num_vectors;
        features["scheduling"]["num_scalars"] = n.sched_features.num_scalars;
        features["scheduling"]["scalar_loads_per_vector"] = n.sched_features.scalar_loads_per_vector;
        features["scheduling"]["vector_loads_per_vector"] = n.sched_features.vector_loads_per_vector;
        features["scheduling"]["scalar_loads_per_scalar"] = n.sched_features.scalar_loads_per_scalar;
        features["scheduling"]["bytes_at_task"] = n.sched_features.bytes_at_task;
        features["scheduling"]["innermost_bytes_at_task"] = n.sched_features.innermost_bytes_at_task;
        features["scheduling"]["unique_bytes_read_per_vector"] = n.sched_features.unique_bytes_read_per_vector;
        features["scheduling"]["unique_lines_read_per_vector"] = n.sched_features.unique_lines_read_per_vector;
        features["scheduling"]["unique_bytes_read_per_task"] = n.sched_features.unique_bytes_read_per_task;
        features["scheduling"]["unique_lines_read_per_task"] = n.sched_features.unique_lines_read_per_task;
        features["scheduling"]["working_set_at_task"] = n.sched_features.working_set_at_task;
        features["scheduling"]["working_set_at_production"] = n.sched_features.working_set_at_production;
        features["scheduling"]["working_set_at_realization"] = n.sched_features.working_set_at_realization;
        features["scheduling"]["working_set_at_root"] = n.sched_features.working_set_at_root;

        graph["nodes"].push_back(node);
    }

    return graph;
}

void GraphRepresentation::generate(const Func &output, const string &output_dir,
                                  const string &pipeline_name, int beam_size) {
    fs::create_directories(output_dir);

    for (int i = 0; i < beam_size; ++i) {
        // Run autoscheduler
        AutoSchedulerResults results = apply_autoscheduler(output, {"Adams2019"}, {});
        vector<Function> outputs = {output.function()};
        Target target = get_host_target();

        // Get featurization data (placeholder)
        vector<PipelineFeatures> pipeline_features;
        vector<ScheduleFeatures> schedule_features;
        bool success = get_autoscheduler_features(output, i, pipeline_features, schedule_features);
        if (!success) {
            std::cerr << "Error: Failed to get featurization data for schedule " << i << " of pipeline " << pipeline_name << std::endl;
            continue;
        }

        // Get stdout content (placeholder)
        string stdout_content;
        success = get_pipeline_operations(output, stdout_content);
        if (!success) {
            std::cerr << "Error: Failed to get pipeline operations for schedule " << i << " of pipeline " << pipeline_name << std::endl;
            continue;
        }

        double execution_time_ms = results.execution_time;  // Placeholder

        GraphRepresentation graph(outputs, target, stdout_content, execution_time_ms,
                                 pipeline_features, schedule_features);
        json graph_data = graph.to_json();

        string output_file = output_dir + "/" + pipeline_name + "_schedule_" + std::to_string(i) + ".json";
        std::ofstream ofs(output_file);
        if (ofs.is_open()) {
            ofs << graph_data.dump(4);
            ofs.close();
            std::cout << "Graph representation saved to " << output_file << std::endl;
        } else {
            std::cerr << "Error: Could not open output file " << output_file << std::endl;
        }
    }
}

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide

// Example usage with a Halide tutorial pipeline
int main() {
    using namespace Halide;
    Var x("x"), y("y"), c("c");
    Func input("input");
    input(x, y, c) = cast<uint32_t>(x + y);

    Func conv("conv");
    RDom k(-1, 3);
    conv(x, y, c) = sum(input(x + k, y, c) * select(k == 0, 1, k == 1, 0, -1));

    Func output("output");
    output(x, y, c) = conv(x, y, c);

    std::string output_dir = "./graph_output";
    Halide::Internal::Autoscheduler::GraphRepresentation::generate(output, output_dir, "convolution_pipeline");

    return 0;
}
