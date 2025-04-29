#include "GraphRepresentation.h"
#include "Halide.h"
#include <fstream>
#include <iostream>

namespace Halide {
namespace Internal {
namespace Autoscheduler {

void test_graph_representation() {
    Var x("x"), y("y");
    Func f("f"), g("g"), h("h");

    f(x, y) = x + y;
    g(x, y) = f(x, y) * 2;
    h(x, y) = g(x, y) + 1;

    Pipeline p({h});
    Target target("host");

    GraphRepresentation graph(p.outputs(), target);

    std::ofstream out("graph.json");
    graph.to_json(out);
    out.close();

    graph.dump(std::cout);

    std::cout << "GraphRepresentation test passed.\n";
}

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide

int main() {
    Halide::Internal::Autoscheduler::test_graph_representation();
    return 0;
}
