#ifndef TREE_REPRESENTATION_H
#define TREE_REPRESENTATION_H

#include <algorithm>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "nlohmann/json.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace Halide {
namespace Internal {
namespace Autoscheduler {

struct Stage {
    int stage_num;
    std::map<std::string, int> op_histogram;
    std::map<std::string, std::vector<int>> memory_patterns;
};

struct Node {
    std::string name;
    std::vector<Stage> stages;
    std::map<std::string, double> scheduling_features;
    int execution_order;
};

struct Edge {
    std::string source_name;
    std::string target_name;
    std::map<std::string, std::string> footprint;
    std::vector<std::vector<double>> load_jacobian;
};

struct GlobalFeatures {
    int cache_hits;
    int cache_misses;
    double execution_time_ms;
};

class TreeRepresentation {
public:
    std::vector<Node> nodes;
    std::vector<Edge> edges;
    GlobalFeatures global_features;
    std::map<std::string, std::map<std::string, double>> schedules;
    std::map<std::string, int> node_dict;
    std::vector<int> execution_order;

    // Constructor to parse stderr.txt and build the tree
    TreeRepresentation(const std::string& stderr_path);

    // Build and save the tree to a JSON file
    std::pair<int, std::vector<int>> build_and_save_tree(const std::string& output_path);

    // Dump the tree representation for debugging
    void dump(std::ostream& os) const;

private:
    // Parse stderr.txt to populate the tree
    void parse_stderr(const std::string& file_path);

    // Perform topological sort to determine execution order
    void topological_sort();

    // Utility function to split a string by delimiter
    std::vector<std::string> split(const std::string& s, char delim);
};

} // namespace Autoscheduler
} // namespace Internal
} // namespace Halide

#endif // TREE_REPRESENTATION_H
