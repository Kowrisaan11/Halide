# Robust Makefile for adams2019 autoscheduler, supporting all dependencies as in the original

CXX := g++
THIS_MAKEFILE := $(realpath $(lastword $(MAKEFILE_LIST)))
SRC := $(strip $(shell dirname $(THIS_MAKEFILE)))
HALIDE_SRC_ROOT := $(realpath $(SRC)/../../../)
COMMON_DIR := $(realpath $(SRC)/../common/)
HALIDE_DISTRIB_PATH ?= $(HALIDE_SRC_ROOT)/distrib

BIN := bin
PLUGIN_EXT := so

# Print Halide distribution path for diagnostics
$(info Looking for Halide distro at $(HALIDE_DISTRIB_PATH). If this is incorrect, set the make variable HALIDE_DISTRIB_PATH)

CXXFLAGS := -std=c++17 -fPIC -O3 -Wall -Werror \
  -I$(HALIDE_DISTRIB_PATH)/include \
  -I$(HALIDE_DISTRIB_PATH)/tools \
  -I$(HALIDE_SRC_ROOT)/src \
  -I$(HALIDE_SRC_ROOT)/tools \
  -I$(COMMON_DIR) \
  -I$(SRC) \
  -I/home/kowrisaan/fyp/libtorch/include \
  -I/home/kowrisaan/fyp/libtorch/include/torch/csrc/api/include

LDFLAGS := -shared -L/home/kowrisaan/fyp/libtorch/lib \
  -ltorch -ltorch_cpu -lc10 \
  -Wl,-rpath,/home/kowrisaan/fyp/libtorch/lib \
  -L$(HALIDE_DISTRIB_PATH)/lib \
  -lHalide -ldl -lpthread -lz

ifeq ($(wildcard /home/kowrisaan/fyp/libtorch/lib/libtorch_cuda.so),/home/kowrisaan/fyp/libtorch/lib/libtorch_cuda.so)
  LDFLAGS += -ltorch_cuda -lcuda
  CXXFLAGS += -DTORCH_CUDA_AVAILABLE
endif

# Object and static library dependencies
AUTOSCHED_WEIGHT_OBJECTS := $(BIN)/baseline_weights.o
AUTOSCHED_COST_MODEL_LIBS := \
  $(BIN)/cost_model/adams2019_cost_model.a \
  $(BIN)/cost_model/adams2019_train_cost_model.a

# Source files (update as needed)
SRCS := \
  $(SRC)/AutoSchedule.cpp \
  $(SRC)/Cache.cpp \
  $(SRC)/DefaultCostModel.cpp \
  $(SRC)/Weights.cpp \
  $(SRC)/FunctionDAG.cpp \
  $(SRC)/LoopNest.cpp \
  $(SRC)/State.cpp

HDRS := \
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

OBJS := $(SRCS:$(SRC)/%.cpp=$(BIN)/%.o)

TARGET := $(BIN)/libautoschedule_adams2019.$(PLUGIN_EXT)

.PHONY: all clean

all: $(TARGET)

bin:
	mkdir -p bin

# Pattern rule for object files
$(BIN)/%.o: $(SRC)/%.cpp | bin
	$(CXX) $(CXXFLAGS) -c $< -o $@

# The main linking rule, matching the original structure and dependencies
$(TARGET): \
    $(COMMON_DIR)/ASLog.cpp \
    $(SRC)/AutoSchedule.cpp \
    $(SRC)/Cache.h \
    $(SRC)/Cache.cpp \
    $(SRC)/DefaultCostModel.h \
    $(SRC)/DefaultCostModel.cpp \
    $(SRC)/Weights.h \
    $(SRC)/Weights.cpp \
    $(SRC)/FunctionDAG.h \
    $(SRC)/FunctionDAG.cpp \
    $(SRC)/LoopNest.h \
    $(SRC)/LoopNest.cpp \
    $(SRC)/Featurization.h \
    $(SRC)/CostModel.h \
    $(SRC)/State.h \
    $(SRC)/State.cpp \
    $(SRC)/Timer.h \
    $(COMMON_DIR)/PerfectHashMap.h \
    $(AUTOSCHED_WEIGHT_OBJECTS) \
    $(AUTOSCHED_COST_MODEL_LIBS) \
    $(BIN)/auto_schedule_runtime.a
	@mkdir -p $(@D)
	$(CXX) -shared -fPIC -fvisibility=hidden -fvisibility-inlines-hidden $(CXXFLAGS) $(filter %.cpp %.o %.a,$^) $(LDFLAGS) -o $@ -I $(SRC)

clean:
	rm -rf bin
