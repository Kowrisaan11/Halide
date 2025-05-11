# Makefile for adams2019 autoscheduler WITHOUT baseline weights or cost_model.a dependencies
CXX := g++
THIS_MAKEFILE := $(realpath $(lastword $(MAKEFILE_LIST)))
# Update SRC to point directly to the correct source directory
SRC := $(realpath $(dir $(THIS_MAKEFILE)))
HALIDE_SRC_ROOT := $(realpath $(SRC)/../../../)
COMMON_DIR := $(realpath $(SRC)/../common/)
HALIDE_DISTRIB_PATH ?= $(HALIDE_SRC_ROOT)/distrib
BIN := bin
PLUGIN_EXT := so

$(info Looking for Halide distro at $(HALIDE_DISTRIB_PATH). If this is incorrect, set the make variable HALIDE_DISTRIB_PATH)
$(info Source directory is $(SRC))

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

# Define source files with absolute paths for clarity
SRCS := \
  $(SRC)/AutoSchedule.cpp \
  $(SRC)/Cache.cpp \
  $(SRC)/DefaultCostModel.cpp \
  $(SRC)/FunctionDAG.cpp \
  $(SRC)/LoopNest.cpp \

HDRS := \
  $(SRC)/Cache.h \
  $(SRC)/DefaultCostModel.h \
  $(SRC)/FunctionDAG.h \
  $(SRC)/LoopNest.h \
  $(SRC)/Featurization.h \
  $(SRC)/CostModel.h \
  $(COMMON_DIR)/PerfectHashMap.h

# Transform the source paths to object paths
OBJS := $(patsubst $(SRC)/%.cpp,$(BIN)/%.o,$(SRCS))

TARGET := $(BIN)/libautoschedule_adams2019.$(PLUGIN_EXT)

.PHONY: all clean debug

all: $(TARGET)

debug:
	@echo "SRCS = $(SRCS)"
	@echo "OBJS = $(OBJS)"
	@echo "SRC = $(SRC)"
	@echo "BIN = $(BIN)"

$(BIN):
	mkdir -p $(BIN)

# Explicit rule for each object file to ensure correct paths
$(BIN)/%.o: $(SRC)/%.cpp | $(BIN)
	@echo "Compiling $< to $@"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Main linking rule for the shared library
$(TARGET): $(OBJS)
	@mkdir -p $(@D)
	@echo "Linking $@ from $(OBJS)"
	$(CXX) -shared -fPIC -fvisibility=hidden -fvisibility-inlines-hidden $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean:
	rm -rf $(BIN)

# Add verbose output to see what's happening
.PHONY: verbose
verbose:
	@echo "Current directory: $$(pwd)"
	@echo "THIS_MAKEFILE: $(THIS_MAKEFILE)"
	@echo "SRC: $(SRC)"
	@echo "Source files:"
	@for src in $(SRCS); do \
		echo "  $$src"; \
		if [ -f "$$src" ]; then \
			echo "    (exists)"; \
		else \
			echo "    (MISSING)"; \
		fi; \
	done
