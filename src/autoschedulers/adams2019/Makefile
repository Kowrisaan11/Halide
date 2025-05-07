# Makefile for Halide Adams2019 autoscheduler
# Located at /home/kowrisaan/fyp/Halide/src/autoschedulers/adams2019/Makefile

# Paths
HALIDE_SRC_ROOT = ../../../../..
HALIDE_DISTRIB_PATH ?= $(HALIDE_SRC_ROOT)/distrib
HALIDE_TOOLS_DIR = $(HALIDE_DISTRIB_PATH)/tools
HALIDE_INCLUDE_DIR = $(HALIDE_DISTRIB_PATH)/include
HALIDE_LIB_DIR = $(HALIDE_DISTRIB_PATH)/lib
COMMON_DIR = ../common
SRC = .
BIN = bin

# Compiler settings
CXX ?= g++
CXXFLAGS = -std=c++17 -fPIC -fvisibility=hidden -fvisibility-inlines-hidden -O3 \
           -Wall -Werror -Wno-unused-function -Wcast-qual -Wignored-qualifiers \
           -Wno-comment -Wsign-compare -Wno-unknown-warning-option -Wno-psabi \
           -I$(HALIDE_INCLUDE_DIR) -I$(HALIDE_TOOLS_DIR) -I$(COMMON_DIR)
LDFLAGS = -L$(HALIDE_LIB_DIR) -lHalide -lpthread -ldl -lz -lrt -lm

# LibTorch settings
TORCH_ROOT = /home/kowrisaan/libtorch
TORCH_INCLUDE = $(TORCH_ROOT)/include
TORCH_LIB = $(TORCH_ROOT)/lib
TORCH_LIBS = -ltorch -ltorch_cpu -lc10
CXXFLAGS += -I$(TORCH_INCLUDE)
LDFLAGS += -L$(TORCH_LIB) $(TORCH_LIBS) -Wl,-rpath,$(TORCH_LIB)

# nlohmann/json settings
JSON_INCLUDE = /usr/include
CXXFLAGS += -I$(JSON_INCLUDE)

# Determine plugin extension based on platform
UNAME = $(shell uname)
ifeq ($(UNAME), Darwin)
    PLUGIN_EXT = dylib
    HALIDE_RPATH_FOR_LIB = '-Wl,-rpath,@loader_path'
else
    PLUGIN_EXT = so
    HALIDE_RPATH_FOR_LIB = '-Wl,-rpath,$$ORIGIN'
endif

# Source files and objects
AUTOSCHED_SOURCES = \
    $(COMMON_DIR)/ASLog.cpp \
    $(SRC)/AutoSchedule.cpp \
    $(SRC)/Cache.cpp \
    $(SRC)/DefaultCostModel.cpp \
    $(SRC)/Weights.cpp \
    $(SRC)/FunctionDAG.cpp \
    $(SRC)/LoopNest.cpp \
    $(SRC)/State.cpp

AUTOSCHED_HEADERS = \
    $(SRC)/Cache.h \
    $(SRC)/DefaultCostModel.h \
    $(SRC)/Weights.h \
    $(SRC)/FunctionDAG.h \
    $(SRC)/LoopNest.h \
    $(SRC)/Featurization.h \
    $(SRC)/CostModel.h \
    $(SRC)/State.h \
    $(SRC)/Timer.h \
    $(COMMON_DIR)/PerfectHashMap.h

AUTOSCHED_OBJECTS = $(AUTOSCHED_SOURCES:.cpp=.o)

# Weight objects
AUTOSCHED_WEIGHT_OBJECTS = $(BIN)/baseline_weights.o

# Cost model libraries
AUTOSCHED_COST_MODEL_LIBS = \
    $(BIN)/cost_model/adams2019_cost_model.a \
    $(BIN)/cost_model/adams2019_train_cost_model.a

# Target
TARGET = $(BIN)/libautoschedule_adams2019.$(PLUGIN_EXT)

all: $(TARGET)

# Create binary2cpp tool
$(BIN)/binary2cpp: $(HALIDE_SRC_ROOT)/tools/binary2cpp.cpp
	@mkdir -p $(@D)
	$(CXX) $< -o $@

# Generate weights cpp file
$(BIN)/baseline_weights.cpp: $(BIN)/binary2cpp $(SRC)/baseline.weights
	@mkdir -p $(@D)
	$(BIN)/binary2cpp baseline_weights < $(SRC)/baseline.weights > $@

# Compile weights object
$(BIN)/baseline_weights.o: $(BIN)/baseline_weights.cpp
	$(CXX) -c $< -o $@

# Cost model generator
$(BIN)/cost_model.generator: $(SRC)/cost_model_generator.cpp \
				$(SRC)/cost_model_schedule.h \
				$(SRC)/NetworkSize.h
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(filter %.cpp,$^) -o $@ $(LDFLAGS)

# Generate auto schedule runtime
$(BIN)/auto_schedule_runtime.a: $(BIN)/cost_model.generator
	@mkdir -p $(@D)
	$^ -r auto_schedule_runtime -o $(BIN) target=$(HL_TARGET)

# Generate cost model libraries
$(BIN)/cost_model/adams2019_%.a: $(BIN)/cost_model.generator
	@mkdir -p $(@D)
	$^ -g $* -o $(BIN)/cost_model -f $* -n adams2019_$* target=$(HL_TARGET)-no_runtime -e stmt,static_library,h,assembly

# Main target - autoscheduler library
$(TARGET): $(AUTOSCHED_OBJECTS) $(AUTOSCHED_WEIGHT_OBJECTS) $(AUTOSCHED_COST_MODEL_LIBS) $(BIN)/auto_schedule_runtime.a
	@mkdir -p $(@D)
	$(CXX) -shared -fPIC -fvisibility=hidden -fvisibility-inlines-hidden $(CXXFLAGS) $(OPTIMIZE) -I $(BIN)/cost_model $(filter-out %.h,$^) -o $@ $(LDFLAGS) $(HALIDE_RPATH_FOR_LIB) -I $(SRC)

# Pattern rule for object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BIN) $(AUTOSCHED_OBJECTS)

.PHONY: all clean
