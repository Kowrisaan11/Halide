#ifndef GRAPH_REPRESENTATION_H
#define GRAPH_REPRESENTATION_H

#include <map>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "Featurization.h"
#include "Halide.h"

namespace Halide {
namespace Internal {
namespace Autoscheduler {

using json = nlohmann::json;

struct GraphRepresentation {
    struct Node {
        std::string name;
        int id;
        int max_id;
        Function func;
        double bytes_per_point;
        int dimensions;
        bool is_input;
        bool is_output;
        bool is_pointwise;
        bool is_boundary_condition;
        bool is_wrapper;
        int vector_size;
        std::vector<SymbolicInterval> region_required;
        std::vector<Span> estimated_region_required;
        std::vector<RegionComputedInfo> region_computed;
        bool region_computed_all_common_cases;

        struct Stage {
            Node* node;
            int index;
            int id;
            int max_id;
            std::string name;
            Halide::Stage stage;
            std::vector<Loop> loop;
            bool loop_nest_all_common_cases;
            int vector_size;
            PipelineFeatures features;
            std::vector<bool> dependencies;
            std::vector<Edge*> incoming_edges;

            bool downstream_of(const Node& n) const {
                return dependencies[n.id];
            }

            explicit Stage(Halide::Stage s) : stage(std::move(s)) {}
        };
        std::vector<Stage> stages;
        std::vector<Edge*> outgoing_edges;

        struct Loop {
            std::string var;
            bool pure, rvar;
            Expr min, max;
            int pure_dim;
            bool equals_region_computed;
            int region_computed_dim;
            bool bounds_are_constant;
            int64_t c_min, c_max;
            std::string accessor;
        };

        struct SymbolicInterval {
            Halide::Var min;
            Halide::Var max;
        };

        struct RegionComputedInfo {
            Interval in;
            bool depends_on_estimate;
            bool equals_required;
            bool equals_union_of_required_with_constants;
            int64_t c_min, c_max;
        };

        void required_to_computed(const Span* required, Span* computed) const;
        void loop_nest_for_region(int stage_idx, const Span* computed, Span* loop) const;
    };

    struct Edge {
        struct BoundInfo {
            Expr expr;
            int64_t coeff, constant;
            int64_t consumer_dim;
            bool affine, uses_max, depends_on_estimate;
            BoundInfo(const Expr& e, const Node::Stage& consumer, bool dependent);
        };

        Node* producer;
        Node::Stage* consumer;
        std::vector<std::pair<BoundInfo, BoundInfo>> bounds;
        int calls;
        bool all_bounds_affine;
        std::vector<LoadJacobian> load_jacobians;

        void add_load_jacobian(LoadJacobian j1);
        void expand_footprint(const Span* consumer_loop, Span* producer_required) const;
    };

    struct Span {
        int64_t min_, max_;
        bool constant_extent_;

        int64_t min() const { return min_; }
        int64_t max() const { return max_; }
        int64_t extent() const { return max_ - min_ + 1; }
        bool constant_extent() const { return constant_extent_; }

        void union_with(const Span& other) {
            min_ = std::min(min_, other.min_);
            max_ = std::max(max_, other.max_);
            constant_extent_ = constant_extent_ && other.constant_extent_;
        }

        void set_extent(int64_t e) { max_ = min_ + e - 1; }
        void translate(int64_t x) { min_ += x; max_ += x; }

        Span(int64_t a, int64_t b, bool c) : min_(a), max_(b), constant_extent_(c) {}
        Span() = default;
        Span(const Span& other) = default;
        static Span empty_span() { return Span(INT64_MAX, INT64_MIN, true); }
    };

    struct LoadJacobian {
        std::vector<std::vector<OptionalRational>> coeffs;
        int64_t c;

        explicit LoadJacobian(std::vector<std::vector<OptionalRational>>&& matrix, int64_t c = 1)
            : coeffs(matrix), c(c) {}

        size_t producer_storage_dims() const { return coeffs.size(); }
        size_t consumer_loop_dims() const {
            return coeffs.empty() || coeffs[0].empty() ? 0 : coeffs[0].size();
        }

        OptionalRational operator()(int producer_storage_dim, int consumer_loop_dim) const {
            if (coeffs.empty()) return {true, 0, 1};
            internal_assert(producer_storage_dim < (int)coeffs.size());
            const auto& p = coeffs[producer_storage_dim];
            if (p.empty()) return {true, 0, 1};
            internal_assert(consumer_loop_dim < (int)p.size());
            return p[consumer_loop_dim];
        }

        int64_t count() const { return c; }

        bool merge(const LoadJacobian& other) {
            if (other.coeffs.size() != coeffs.size()) return false;
            for (size_t i = 0; i < coeffs.size(); i++) {
                if (other.coeffs[i].size() != coeffs[i].size()) return false;
                for (size_t j = 0; j < coeffs[i].size(); j++) {
                    if (!(other.coeffs[i][j] == coeffs[i][j])) return false;
                }
            }
            c += other.count();
            return true;
        }

        LoadJacobian operator*(const LoadJacobian& other) const {
            std::vector<std::vector<OptionalRational>> matrix;
            internal_assert(consumer_loop_dims() == 0 || consumer_loop_dims() == other.producer_storage_dims());
            matrix.resize(producer_storage_dims());
            for (size_t i = 0; i < producer_storage_dims(); i++) {
                matrix[i].resize(other.consumer_loop_dims());
                for (size_t j = 0; j < other.consumer_loop_dims(); j++) {
                    matrix[i][j] = OptionalRational{true, 0, 1};
                    for (size_t k = 0; k < consumer_loop_dims(); k++) {
                        matrix[i][j] += (*this)(i, k) * other(k, j);
                    }
                }
            }
            return LoadJacobian(std::move(matrix), count() * other.count());
        }

        void dump(std::ostream& os, const char* prefix) const;
    };

    std::vector<Node> nodes;
    std::vector<Edge> edges;
    json graph_json;

    GraphRepresentation(const std::vector<Function>& outputs, const Target& target);
    void dump(std::ostream& os) const;
    void to_json(std::ostream& os) const;

private:
    void featurize();
};

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide

#endif  // GRAPH_REPRESENTATION_H
