#ifndef GRAPH_REPRESENTATION_H
#define GRAPH_REPRESENTATION_H

#include <map>
#include <string>
#include <vector>

#include "Featurization.h"
#include "Halide.h"
#include "nlohmann/json.hpp"

namespace Halide {
namespace Internal {
namespace Autoscheduler {

using json = nlohmann::json;
using std::map;
using std::string;
using std::vector;

struct Adams2019Params;

// A representation of a Halide pipeline as a JSON-compatible graph for the cost model.
struct GraphRepresentation {
    // A node in the graph, corresponding to a Func.
    struct Node {
        string name;                     // Func name
        PipelineFeatures features;       // Op histogram and memory patterns
        ScheduleFeatures sched_features;  // Scheduling features
        int id;                          // Unique ID
    };

    // An edge in the graph, representing a producer-consumer relationship.
    struct Edge {
        string source;      // Producer Func name
        string target;      // Consumer Func name
        int source_id;      // Producer node ID
        int target_id;      // Consumer node ID
        json features;      // Footprint, load_jacobian, etc.
    };

    // The graph data
    vector<Node> nodes;
    vector<Edge> edges;
    json operations;        // Convolution, Resampling, etc.
    json global_features;   // Execution time, cache hits/misses

    // Construct the graph representation for a pipeline
    GraphRepresentation(const vector<Function> &outputs, const Target &target,
                        const string &stdout_content, double execution_time_ms,
                        const vector<PipelineFeatures> &pipeline_features,
                        const vector<ScheduleFeatures> &schedule_features);

    // Serialize to JSON
    json to_json() const;

    // Generate and save JSON for all schedules
    static void generate(const Func &output, const string &output_dir,
                        const string &pipeline_name, int beam_size = 32);

private:
    void featurize(const vector<Function> &outputs);
};

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide

#endif  // GRAPH_REPRESENTATION_H
