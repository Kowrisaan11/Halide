# Specify minimum CMake version and project name
cmake_minimum_required(VERSION 3.10)
project(HalideAdams2019Autoscheduler)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define common directory
set(COMMON_DIR "${Halide_SOURCE_DIR}/src/autoschedulers/common")

# Find LibTorch
set(Torch_DIR "/home/kowrisaan/jathu/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Find nlohmann_json
find_package(nlohmann_json REQUIRED)

# =================================================================
# Weights
set(WF_CPP baseline.cpp)
configure_file(baseline.weights baseline.weights COPYONLY)
add_custom_command(OUTPUT ${WF_CPP}
                   COMMAND binary2cpp baseline_weights < baseline.weights > ${WF_CPP}
                   DEPENDS baseline.weights binary2cpp
                   VERBATIM)
add_library(adams2019_weights_obj OBJECT ${WF_CPP})

# =================================================================
# Cost Model and Train Cost Model
add_executable(adams2019_cost_model.generator cost_model_generator.cpp)
target_include_directories(adams2019_cost_model.generator PRIVATE 
    "${Halide_SOURCE_DIR}/src/autoschedulers/adams2019" 
    ${TORCH_INCLUDE_DIRS}
    ${COMMON_DIR}
)
target_link_libraries(adams2019_cost_model.generator PRIVATE 
    Halide::Generator 
    ${TORCH_LIBRARIES}
)
if(TORCH_CUDA_LIBRARIES)
    target_link_libraries(adams2019_cost_model.generator PRIVATE ${TORCH_CUDA_LIBRARIES})
endif()

add_halide_library(adams2019_cost_model 
                   FROM adams2019_cost_model.generator
                   GENERATOR cost_model
                   FUNCTION_NAME cost_model
                   FEATURES[x86-64-osx] avx2 sse41
                   FEATURES[arm-64-osx] arm_dot_prod-arm_fp16)

add_halide_library(adams2019_train_cost_model 
                   FROM adams2019_cost_model.generator
                   GENERATOR train_cost_model
                   FUNCTION_NAME train_cost_model
                   FEATURES[x86-64-osx] avx2 sse41
                   FEATURES[arm-64-osx] arm_dot_prod-arm_fp16
                   USE_RUNTIME adams2019_cost_model.runtime)

# =================================================================
# Retrain Cost Model (Optional Utility)
if(WITH_UTILS)
    add_executable(adams2019_retrain_cost_model
                   DefaultCostModel.cpp
                   Weights.cpp
                   retrain_cost_model.cpp
                   $<TARGET_OBJECTS:adams2019_weights_obj>)
    target_include_directories(adams2019_retrain_cost_model PRIVATE 
        "${Halide_SOURCE_DIR}/src/autoschedulers/adams2019" 
        ${TORCH_INCLUDE_DIRS}
        ${COMMON_DIR}
    )
    target_link_libraries(adams2019_retrain_cost_model PRIVATE 
        adams2019_cost_model 
        adams2019_train_cost_model 
        Halide::Plugin 
        ${TORCH_LIBRARIES}
    )
    if(TORCH_CUDA_LIBRARIES)
        target_link_libraries(adams2019_retrain_cost_model PRIVATE ${TORCH_CUDA_LIBRARIES})
    endif()
endif()

# =================================================================
# Main Autoscheduler Library
add_autoscheduler(
    NAME Adams2019
    SOURCES
        AutoSchedule.cpp
        Cache.cpp
        DefaultCostModel.cpp
        GraphRepresentation.cpp
        TreeRepresentation.cpp
        LoopNest.cpp
        State.cpp
        Weights.cpp
        $<TARGET_OBJECTS:adams2019_weights_obj>
)
target_include_directories(Halide_Adams2019 PRIVATE 
    "${Halide_SOURCE_DIR}/src/autoschedulers/adams2019" 
    ${TORCH_INCLUDE_DIRS}
    ${COMMON_DIR}
)
target_link_libraries(Halide_Adams2019 PRIVATE 
    adams2019_cost_model 
    adams2019_train_cost_model 
    ${TORCH_LIBRARIES}
    nlohmann_json::nlohmann_json
)
if(TORCH_CUDA_LIBRARIES)
    target_link_libraries(Halide_Adams2019 PRIVATE ${TORCH_CUDA_LIBRARIES})
endif()

# =================================================================
# Auto-tuning Support Utilities
if(WITH_UTILS)
    add_executable(adams2019_weightsdir_to_weightsfile 
                   weightsdir_to_weightsfile.cpp 
                   Weights.cpp)
    target_include_directories(adams2019_weightsdir_to_weightsfile PRIVATE ${COMMON_DIR})
    target_link_libraries(adams2019_weightsdir_to_weightsfile PRIVATE Halide::Runtime)
endif()

# =================================================================
# Tests for Adams2019 Internal Functionality
if(WITH_TESTS)
    add_executable(adams2019_test_graph_representation 
                   test_graph_representation.cpp 
                   GraphRepresentation.cpp)
    target_include_directories(adams2019_test_graph_representation PRIVATE 
        "${Halide_SOURCE_DIR}/src/autoschedulers/adams2019"
        ${COMMON_DIR}
    )
    target_link_libraries(adams2019_test_graph_representation PRIVATE 
        Halide::Plugin 
        nlohmann_json::nlohmann_json
    )
    add_test(NAME adams2019_test_graph_representation 
             COMMAND adams2019_test_graph_representation)
    set_tests_properties(adams2019_test_graph_representation 
                         PROPERTIES LABELS "adams2019;autoschedulers_cpu")
endif()
