#include "TreeRepresentation.h"

namespace Halide {
namespace Internal {
namespace Autoscheduler {

namespace {

// Default empty op_histogram
std::map<std::string, int> default_op_histogram = {
    {"Constant", 0}, {"Cast", 0}, {"Variable", 0}, {"Param", 0}, {"Add", 0}, {"Sub", 0}, {"Mod", 0}, {"Mul", 0}, {"Div", 0},
    {"Min", 0}, {"Max", 0}, {"EQ", 0}, {"NE", 0}, {"LT", 0}, {"LE", 0}, {"And", 0}, {"Or", 0}, {"Not", 0}, {"Select", 0},
    {"ImageCall", 0}, {"FuncCall", 0}, {"SelfCall", 0}, {"ExternCall", 0}, {"Let", 0}
};

// Default empty memory_patterns
std::map<std::string, std::vector<int>> default_memory_patterns = {
    {"Pointwise", {0, 0, 0, 0}},
    {"Transpose", {0, 0, 0, 0}},
    {"Broadcast", {0, 0, 0, 0}},
    {"Slice", {0, 0, 0, 0}}
};

// Scheduling feature names and default values
std::vector<std::string> scheduling_keys = {
    "num_realizations", "num_productions", "points_computed_per_realization", "points_computed_per_production",
    "points_computed_total", "points_computed_minimum", "innermost_loop_extent", "innermost_pure_loop_extent",
    "unrolled_loop_extent", "inner_parallelism", "outer_parallelism", "bytes_at_realization", "bytes_at_production",
    "bytes_at_root", "innermost_bytes_at_realization", "innermost_bytes_at_production", "innermost_bytes_at_root",
    "inlined_calls", "unique_bytes_read_per_realization", "unique_lines_read_per_realization",
    "allocation_bytes_read_per_realization", "working_set", "vector_size", "native_vector_size", "num_vectors",
    "num_scalars", "scalar_loads_per_vector", "vector_loads_per_vector", "scalar_loads_per_scalar", "bytes_at_task",
    "innermost_bytes_at_task", "unique_bytes_read_per_vector", "unique_lines_read_per_vector",
    "unique_bytes_read_per_task", "unique_lines_read_per_task", "working_set_at_task", "working_set_at_production",
    "working_set_at_realization", "working_set_at_root"
};

std::map<std::string, double> default_scheduling() {
    std::map<std::string, double> result;
    for (const auto& key : scheduling_keys) {
        result[key] = 0.0;
    }
    return result;
}

} // namespace

std::vector<std::string> TreeRepresentation::split(const std::string& s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        result.push_back(item);
    }
    return result;
}

void TreeRepresentation::parse_stderr(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << file_path << "\n";
        return;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    file.close();

    std::regex node_pattern(R"(Node: (\S+)\s+Symbolic region required:(.*?)(?=\nNode:|\nEdge:|\nCache))", std::regex::ECMAScript | std::regex::dotall);
    std::regex edge_pattern(R"(Edge: (\S+) -> (\S+)\s+Footprint:(.*?)(?=Load Jacobians:)(.*?)(?=\nEdge:|\nCache))", std::regex::ECMAScript | std::regex::dotall);
    std::regex global_features_pattern(R"(Cache \(block\) hits: (\d+)\s+Cache \(block\) misses: (\d+)\s+AutoSchedule\.cpp:\d+.*?AutoSchedule\.cpp:\d+ : ([\d.]+) ms)", std::regex::ECMAScript | std::regex::dotall);
    std::regex stage_pattern(R"(Stage (\d+):.*?(?=Stage|\n\s*pointwise:))", std::regex::ECMAScript | std::regex::dotall);
    std::regex op_histogram_pattern(R"(Op histogram:(.*?)(?=Memory access patterns))", std::regex::ECMAScript | std::regex::dotall);
    std::regex memory_pattern(R"(Memory access patterns.*?Pointwise:\s+(\d+\s+\d+\s+\d+\s+\d+).*?Transpose:\s+(\d+\s+\d+\s+\d+\s+\d+).*?Broadcast:\s+(\d+\s+\d+\s+\d+\s+\d+).*?Slice:\s+(\d+\s+\d+\s+\d+\s+\d+))", std::regex::ECMAScript | std::regex::dotall);
    std::regex schedule_pattern(R"(Schedule features for (\S+)(.*?)(?=Schedule features for|\Z))", std::regex::ECMAScript | std::regex::dotall);

    // Parse nodes
    std::sregex_iterator node_it(content.begin(), content.end(), node_pattern);
    std::sregex_iterator end;
    for (; node_it != end; ++node_it) {
        std::smatch match = *node_it;
        std::string node_name = match[1];
        node_dict[node_name] = nodes.size();
        std::vector<Stage> stages;

        std::string region = match[2];
        std::sregex_iterator stage_it(region.begin(), region.end(), stage_pattern);
        for (; stage_it != end; ++stage_it) {
            std::smatch stage_match = *stage_it;
            int stage_num = std::stoi(stage_match[1]);
            std::string stage_content = stage_match[0];

            std::smatch op_hist_match;
            std::map<std::string, int> op_histogram = default_op_histogram;
            if (std::regex_search(stage_content, op_hist_match, op_histogram_pattern)) {
                std::string op_text = op_hist_match[1];
                auto lines = split(op_text, '\n');
                for (const auto& line : lines) {
                    auto parts = split(line, ':');
                    if (parts.size() == 2) {
                        op_histogram[parts[0]] = std::stoi(parts[1]);
                    }
                }
            }

            std::smatch mem_match;
            std::map<std::string, std::vector<int>> memory_patterns = default_memory_patterns;
            if (std::regex_search(stage_content, mem_match, memory_pattern)) {
                memory_patterns["Pointwise"] = {std::stoi(split(mem_match[1], ' ')[0]), std::stoi(split(mem_match[1], ' ')[1]), std::stoi(split(mem_match[1], ' ')[2]), std::stoi(split(mem_match[1], ' ')[3])};
                memory_patterns["Transpose"] = {std::stoi(split(mem_match[2], ' ')[0]), std::stoi(split(mem_match[2], ' ')[1]), std::stoi(split(mem_match[2], ' ')[2]), std::stoi(split(mem_match[2], ' ')[3])};
                memory_patterns["Broadcast"] = {std::stoi(split(mem_match[3], ' ')[0]), std::stoi(split(mem_match[3], ' ')[1]), std::stoi(split(mem_match[3], ' ')[2]), std::stoi(split(mem_match[3], ' ')[3])};
                memory_patterns["Slice"] = {std::stoi(split(mem_match[4], ' ')[0]), std::stoi(split(mem_match[4], ' ')[1]), std::stoi(split(mem_match[4], ' ')[2]), std::stoi(split(mem_match[4], ' ')[3])};
            }

            stages.push_back({stage_num, op_histogram, memory_patterns});
        }
        nodes.push_back({node_name, stages, default_scheduling(), 0});
    }

    // Parse edges
    std::set<std::string> all_node_names(node_dict.begin(), node_dict.end());
    std::sregex_iterator edge_it(content.begin(), content.end(), edge_pattern);
    for (; edge_it != end; ++edge_it) {
        std::smatch match = *edge_it;
        std::string source_name = match[1];
        std::string target_name = match[2];
        all_node_names.insert(source_name);
        all_node_names.insert(target_name);

        auto footprint_lines = split(match[3], '\n');
        std::map<std::string, std::string> footprint;
        for (const auto& line : footprint_lines) {
            auto parts = split(line, ':');
            if (parts.size() == 2) {
                footprint[parts[0]] = parts[1];
            }
        }

        auto jacobian_lines = split(match[4], '\n');
        std::vector<std::vector<double>> load_jacobian;
        for (const auto& line : jacobian_lines) {
            if (line.find('[') != std::string::npos && line.find(']') != std::string::npos) {
                std::string values_str = line.substr(line.find('[') + 1, line.find(']') - line.find('[') - 1);
                auto values = split(values_str, ' ');
                std::vector<double> cleaned_values;
                for (const auto& val : values) {
                    if (val == "_") {
                        cleaned_values.push_back(0.0);
                    } else {
                        try {
                            if (val.find('/') != std::string::npos) {
                                auto frac = split(val, '/');
                                cleaned_values.push_back(std::stod(frac[0]) / std::stod(frac[1]));
                            } else {
                                cleaned_values.push_back(std::stod(val));
                            }
                        } catch (...) {
                            cleaned_values.push_back(0.0);
                        }
                    }
                }
                if (!cleaned_values.empty()) {
                    load_jacobian.push_back(cleaned_values);
                }
            }
        }

        edges.push_back({source_name, target_name, footprint, load_jacobian});
    }

    // Parse global features
    std::smatch gf_match;
    global_features = {0, 0, 0.0};
    if (std::regex_search(content, gf_match, global_features_pattern)) {
        global_features = {std::stoi(gf_match[1]), std::stoi(gf_match[2]), std::stod(gf_match[3])};
    }

    // Parse schedules
    std::sregex_iterator sched_it(content.begin(), content.end(), schedule_pattern);
    for (; sched_it != end; ++sched_it) {
        std::smatch match = *sched_it;
        std::string node_name = match[1];
        all_node_names.insert(node_name);
        auto sched_lines = split(match[2], '\n');
        std::map<std::string, double> scheduling = default_scheduling();
        for (const auto& line : sched_lines) {
            auto parts = split(line, ':');
            if (parts.size() == 2) {
                try {
                    scheduling[parts[0]] = parts[1].empty() ? 0.0 : std::stod(parts[1]);
                } catch (...) {
                    continue;
                }
            }
        }
        schedules[node_name] = scheduling;
    }

    // Ensure all nodes are accounted for
    for (const auto& node_name : all_node_names) {
        if (node_dict.find(node_name) == node_dict.end()) {
            node_dict[node_name] = nodes.size();
            nodes.push_back({node_name, {{0, default_op_histogram, default_memory_patterns}}, default_scheduling(), 0});
        }
    }

    // Assign scheduling features to nodes
    for (auto& node : nodes) {
        if (schedules.find(node.name) != schedules.end()) {
            node.scheduling_features = schedules[node.name];
        }
    }
}

void TreeRepresentation::topological_sort() {
    std::map<int, std::vector<int>> adj_list;
    std::map<int, int> in_degree;
    std::set<std::string> node_names;
    for (const auto& node : nodes) {
        node_names.insert(node.name);
    }

    for (const auto& edge : edges) {
        if (node_names.count(edge.source_name) && node_names.count(edge.target_name)) {
            int source_idx = node_dict.at(edge.source_name);
            int target_idx = node_dict.at(edge.target_name);
            adj_list[source_idx].push_back(target_idx);
            in_degree[target_idx]++;
        }
    }

    std::deque<int> queue;
    for (const auto& node : nodes) {
        int idx = node_dict.at(node.name);
        if (in_degree[idx] == 0) {
            queue.push_back(idx);
        }
    }

    execution_order.clear();
    while (!queue.empty()) {
        int current = queue.front();
        queue.pop_front();
        execution_order.push_back(current);

        for (int neighbor : adj_list[current]) {
            in_degree[neighbor]--;
            if (in_degree[neighbor] == 0) {
                queue.push_back(neighbor);
            }
        }
    }

    if (execution_order.size() != nodes.size()) {
        std::cerr << "Warning: Graph contains cycles or disconnected components.\n";
        execution_order.clear();
        for (const auto& node : nodes) {
            execution_order.push_back(node_dict.at(node.name));
        }
    }

    // Update execution order in nodes
    for (size_t i = 0; i < execution_order.size(); ++i) {
        nodes[execution_order[i]].execution_order = i;
    }
}

TreeRepresentation::TreeRepresentation(const std::string& stderr_path) {
    parse_stderr(stderr_path);
    topological_sort();
}

std::pair<int, std::vector<int>> TreeRepresentation::build_and_save_tree(const std::string& output_path) {
    std::map<int, int> exec_order_dict;
    for (size_t i = 0; i < execution_order.size(); ++i) {
        exec_order_dict[execution_order[i]] = i;
    }

    std::map<int, std::vector<Edge>> adj_list;
    for (const auto& edge : edges) {
        int source_idx = node_dict[edge.source_name];
        adj_list[source_idx].push_back(edge);
    }

    json root = {{"name", "Root"}, {"children", json::array()}};
    json global_features_node = {
        {"name", "Global Features"},
        {"cache_hits", global_features.cache_hits},
        {"cache_misses", global_features.cache_misses},
        {"execution_time_ms", global_features.execution_time_ms},
        {"children", json::array()}
    };
    root["children"].push_back(global_features_node);

    std::vector<int> sorted_exec_order = execution_order;
    std::sort(sorted_exec_order.begin(), sorted_exec_order.end(),
              [&exec_order_dict](int a, int b) { return exec_order_dict[a] < exec_order_dict[b]; });

    for (int node_idx : sorted_exec_order) {
        const auto& node = nodes[node_idx];
        json node_entry = {
            {"name", node.name},
            {"execution_order", exec_order_dict[node_idx]},
            {"op_histogram", node.stages[0].op_histogram},
            {"memory_patterns", node.stages[0].memory_patterns},
            {"scheduling", node.scheduling_features},
            {"children", json::array()}
        };

        if (adj_list.count(node_idx)) {
            auto& edge_list = adj_list[node_idx];
            std::sort(edge_list.begin(), edge_list.end(),
                      [&exec_order_dict, &node_dict](const Edge& a, const Edge& b) {
                          return exec_order_dict[node_dict.at(a.target_name)] <
                                 exec_order_dict[node_dict.at(b.target_name)];
                      });

            for (const auto& edge : edge_list) {
                json edge_child = {
                    {"target_name", edge.target_name},
                    {"target_execution_order", exec_order_dict[node_dict[edge.target_name]]},
                    {"footprint", edge.footprint},
                    {"load_jacobian", edge.load_jacobian}
                };
                node_entry["children"].push_back(edge_child);
            }
        }

        root["children"].push_back(node_entry);
    }

    fs::create_directories(fs::path(output_path).parent_path());
    std::ofstream out_file(output_path);
    out_file << root.dump(4);
    out_file.close();

    return {root["children"].size(), execution_order};
}

void TreeRepresentation::dump(std::ostream& os) const {
    os << "TreeRepresentation:\n";
    os << "Global Features:\n"
       << "  Cache Hits: " << global_features.cache_hits << "\n"
       << "  Cache Misses: " << global_features.cache_misses << "\n"
       << "  Execution Time (ms): " << global_features.execution_time_ms << "\n";

    for (const auto& node : nodes) {
        os << "Node: " << node.name << "\n"
           << "  Execution Order: " << node.execution_order << "\n"
           << "  Stages:\n";
        for (const auto& stage : node.stages) {
            os << "    Stage " << stage.stage_num << ":\n"
               << "      Op Histogram:\n";
            for (const auto& [op, count] : stage.op_histogram) {
                os << "        " << op << ": " << count << "\n";
            }
            os << "      Memory Patterns:\n";
            for (const auto& [pattern, values] : stage.memory_patterns) {
                os << "        " << pattern << ": [";
                for (size_t i = 0; i < values.size(); ++i) {
                    os << values[i] << (i < values.size() - 1 ? ", " : "");
                }
                os << "]\n";
            }
        }
        os << "  Scheduling Features:\n";
        for (const auto& [key, value] : node.scheduling_features) {
            os << "    " << key << ": " << value << "\n";
        }
    }

    for (const auto& edge : edges) {
        os << "Edge: " << edge.source_name << " -> " << edge.target_name << "\n"
           << "  Footprint:\n";
        for (const auto& [key, value] : edge.footprint) {
            os << "    " << key << ": " << value << "\n";
        }
        os << "  Load Jacobians:\n";
        for (const auto& row : edge.load_jacobian) {
            os << "    [ ";
            for (double val : row) {
                os << val << " ";
            }
            os << "]\n";
        }
    }
}

} // namespace Autoscheduler
} // namespace Internal
} // namespace Halide
