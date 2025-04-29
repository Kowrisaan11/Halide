##
# Build rules for the Adams2019 autoscheduler library
##

# =================================================================
# Add LibTorch configuration (NEW SECTION)
set(Torch_DIR "/home/kowrisaan/fyp/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
message(STATUS "Torch includes: ${TORCH_INCLUDE_DIRS}")
# =================================================================

set(COMMON_DIR "${Halide_SOURCE_DIR}/src/autoschedulers/common")

# =================================================================
# weights
set(WF_CPP baseline.cpp)
configure_file(baseline.weights baseline.weights COPYONLY)
add_custom_command(OUTPUT ${WF_CPP}
                   COMMAND binary2cpp baseline_weights < baseline.weights > ${WF_CPP}
                   DEPENDS baseline.weights binary2cpp
                   VERBATIM)

add_library(adams2019_weights_obj OBJECT ${WF_CPP})

# cost_model, train_cost_model
add_executable(adams2019_cost_model.generator cost_model_generator.cpp)
# Add Torch includes and libraries (MODIFIED)
target_include_directories(adams2019_cost_model.generator PRIVATE 
    ${TORCH_INCLUDE_DIRS}
    "${Halide_SOURCE_DIR}/src/autoschedulers/adams2019"
)
target_link_libraries(adams2019_cost_model.generator PRIVATE 
    Halide::Generator 
    ${TORCH_LIBRARIES}
)

# ... rest of existing code ...

# =================================================================
# Main autoscheduler library
add_autoscheduler(
    NAME Adams2019
    SOURCES
    AutoSchedule.cpp
    Cache.cpp
    DefaultCostModel.cpp
    FunctionDAG.cpp
    LoopNest.cpp
    State.cpp
    Weights.cpp
    $<TARGET_OBJECTS:adams2019_weights_obj>
)

# Add Torch includes and libraries (MODIFIED)
target_include_directories(Halide_Adams2019 PRIVATE 
    "${Halide_SOURCE_DIR}/src/autoschedulers/adams2019"
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(Halide_Adams2019 PRIVATE 
    adams2019_cost_model 
    adams2019_train_cost_model
    ${TORCH_LIBRARIES}
)

# =================================================================
# Auto-tuning support utilities
if (WITH_UTILS)
    add_executable(adams2019_retrain_cost_model
                   DefaultCostModel.cpp
                   Weights.cpp
                   retrain_cost_model.cpp
                   $<TARGET_OBJECTS:adams2019_weights_obj>)
    # Add Torch includes (MODIFIED)
    target_include_directories(adams2019_retrain_cost_model PRIVATE 
        "${Halide_SOURCE_DIR}/src/autoschedulers/adams2019"
        ${TORCH_INCLUDE_DIRS}
    )
    target_link_libraries(adams2019_retrain_cost_model PRIVATE 
        adams2019_cost_model 
        adams2019_train_cost_model 
        Halide::Plugin
        ${TORCH_LIBRARIES}
    )
endif ()
