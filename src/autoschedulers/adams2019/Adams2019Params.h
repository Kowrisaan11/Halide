#ifndef ADAMS2019_PARAMS_H
#define ADAMS2019_PARAMS_H

namespace Halide {
namespace Internal {
namespace Autoscheduler {

struct Adams2019Params {
    int parallelism = 1;
    int disable_subtiling = 0;
    int disable_memoized_blocks = 0;
    int disable_memoized_features = 0;
    int64_t memory_limit = -1;
};

}  // namespace Autoscheduler
}  // namespace Internal
}  // namespace Halide

#endif  // ADAMS2019_PARAMS_H
