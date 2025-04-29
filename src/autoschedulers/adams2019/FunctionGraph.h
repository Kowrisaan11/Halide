#ifndef FUNCTION_GRAPH_H
#define FUNCTION_GRAPH_H

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "Errors.h"
#include "Featurization.h"
#include "Halide.h"
#include "nlohmann/json.hpp"

namespace Halide {
namespace Internal {
namespace Autoscheduler {

using json = nlohmann::json;
using std::map;
using std::pair;
using std::string;
using std::vector;

struct Adams2019Params;

// A representation of the graph structure for a Halide pipeline
struct FunctionGraph {
    struct Node;
    struct Edge;

    // A symbolic interval for bounds
    struct SymbolicInterval {
        Halide::Var min;
        Halide::Var max;
    };

    // A span for concrete bounds
    class Span {
        int64_t min_, max_;
        bool constant_extent_;

    public:
        int64_t min() const { return min_; }
        int64_t max() const { return max_; }
        int64_t extent() const { return max_ - min_ + 1; }
        bool constant_extent() const { return constant_extent_; }

        void union_with(const Span &other) {
            min_ = std::min(min_, other.min());
            max_ = std::max(max_, other.max());
            constant_extent_ = constant_extent_ && other.constant_extent();
        }

        void set_extent(int64_t e) { max_ = min_ + e - 1; }
        void translate(int64_t x) { min_ += x; max_ += x; }

        Span(int64_t a, int64_t b, bool c) : min_(a), max_(b), constant_extent_(c) {}
        Span() = default;
        Span(const Span &other) = default;
        static Span empty_span() { return Span(INT64_MAX, INT64_MIN, true); }
    };

    // A node represents a single Func with its features
    struct Node {
        Function func;
        string name;
        int id, max_id;
        double bytes_per_point;
        vector<SymbolicInterval> region_required;
        vector<Span> estimated_region_required;
        vector<Span> region_computed;
        bool region_computed_all_common_cases = false;

        struct Stage {
            Node *node;
            int index;
            string name;
            vector<Loop> loop;
            bool loop_nest_all_common_cases = false;
            int vector_size;
            PipelineFeatures features;
            ScheduleFeatures schedule_features;
            Halide::Stage stage;
            int id, max_id;
            vector<Edge *> incoming_edges;
            vector<bool> dependencies;

            bool downstream_of(const Node &n) const { return dependencies[n.id]; }

            explicit Stage(Halide::Stage s) : stage(std::move(s)) {}
        };

        struct Loop {
            string var;
            bool pure, rvar;
            Expr min, max;
            int pure_dim;
            bool equals_region_computed = false;
            int region_computed_dim = 0;
            bool bounds_are_constant = false;
            int64_t c_min = 0, c_max = 0;
            string accessor;
        };

        vector<Stage> stages;
        vector<Edge *> outgoing_edges;
        int vector_size;
        int dimensions;
        bool is_wrapper;
        bool is_input;
        bool is_output;
        bool is_pointwise;
        bool is_boundary_condition;
        json features;

        void required_to_computed(const Span *required, Span *computed) const;
        void loop_nest_for_region(int stage_idx, const Span *computed, Span *loop) const;
    };

    // An edge represents a producer-consumer relationship
    struct Edge {
        struct BoundInfo {
            Expr expr;
            int64_t coeff, constant;
            int64_t consumer_dim;
            bool affine, uses_max, depends_on_estimate;
            BoundInfo(const Expr &e, const Node::Stage &consumer, bool dependent);
        };

        vector<pair<BoundInfo, BoundInfo>> bounds;
        Node *producer;
        Node::Stage *consumer;
        int calls;
        bool all_bounds_affine;
        vector<LoadJacobian> load_jacobians;
        json features;

        void add_load_jacobian(LoadJacobian j1);
        void expand_footprint(const Span *consumer_loop, Span *producer_required) const;
    };

    vector<Node> nodes;
    vector<Edge> edges;
    json graph_json;

    FunctionGraph(const vector<Function> &outputs, const Target &target);
    void dump(std::ostream &os) const;

private:
    void featurize();
    void build_graph_json();
};

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide

#endif  // FUNCTION_GRAPH_H
