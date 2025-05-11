# Makefile for adams2019 autoscheduler without baseline weights

CXX := g++
THIS_MAKEFILE := $(realpath $(lastword $(MAKEFILE_LIST)))
SRC := $(strip $(shell dirname $(THIS_MAKEFILE)))
HALIDE_SRC_ROOT := $(realpath $(SRC)/../../../)
COMMON_DIR := $(realpath $(SRC)/../common/)
HALIDE_DISTRIB_PATH ?= $(HALIDE_SRC_ROOT)/distrib

BIN := bin
PLUGIN_EXT := so

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

# If you still have these libraries, keep them; otherwise remove or comment out.
AUTOSCHED_COST_MODEL_LIBS := \
  $(BIN)/cost_model/adams2019_cost_model.a \
  $(BIN)/cost_model/adams2019_train_cost_model.a

SRCS := \
  AutoSchedule.cpp \
  Cache.cpp \
  DefaultCostModel.cpp \
  Weights.cpp \
  FunctionDAG.cpp \
  LoopNest.cpp \
  State.cpp

HDRS := \
  Cache.h \
  DefaultCostModel.h \
  Weights.h \
  FunctionDAG.h \
  LoopNest.h \
  Featurization.h \
  CostModel.h \
  State.h \
  Timer.h \
  $(COMMON_DIR)/PerfectHashMap.h

OBJS := $(SRCS:%.cpp=$(BIN)/%.o)

TARGET := $(BIN)/libautoschedule_adams2019.$(PLUGIN_EXT)

.PHONY: all clean

all: $(TARGET)

bin:
	mkdir -p bin

$(BIN)/%.o: %.cpp | bin
	$(CXX) $(CXXFLAGS) -c $< -o $@

# The main linking rule, WITHOUT baseline weights
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
    $(AUTOSCHED_COST_MODEL_LIBS) \
    $(BIN)/auto_schedule_runtime.a
	@mkdir -p $(@D)
	$(CXX) -shared -fPIC -fvisibility=hidden -fvisibility-inlines-hidden $(CXXFLAGS) $(filter %.cpp %.o %.a,$^) $(LDFLAGS) -o $@ -I $(SRC)

clean:
	rm -rf bin
