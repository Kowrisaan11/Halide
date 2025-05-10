# Define paths
THIS_MAKEFILE = $(realpath $(filter %Makefile, $(MAKEFILE_LIST)))
SRC = $(strip $(shell dirname $(THIS_MAKEFILE)))
HALIDE_SRC_ROOT = $(realpath $(SRC)/../../../)
COMMON_DIR ?= $(realpath $(SRC)/../common/)

# Halide distribution path
HALIDE_DISTRIB_PATH ?= $(HALIDE_SRC_ROOT)/distrib

# Set up paths for includes
CXXFLAGS += -I$(HALIDE_DISTRIB_PATH)/include \
            -I$(HALIDE_DISTRIB_PATH)/tools \
            -I$(HALIDE_SRC_ROOT)/src \
            -I$(HALIDE_SRC_ROOT)/tools \
            -I$(COMMON_DIR) \
            -std=c++17 \
            -fPIC \
            -Wall -Werror \
            -Wno-unused-function \
            -Wcast-qual \
            -Wignored-qualifiers \
            -Wno-comment \
            -Wsign-compare \
            -Wno-unknown-warning-option \
            -O3

# LibTorch configuration
TORCH_DIR = /home/kowrisaan/fyp/libtorch

# Add LibTorch includes and flags
CXXFLAGS += -I$(TORCH_DIR)/include \
            -I$(TORCH_DIR)/include/torch/csrc/api/include

# Library flags
LDFLAGS += -L$(TORCH_DIR)/lib \
           -ltorch -ltorch_cpu -lc10 \
           -Wl,-rpath,$(TORCH_DIR)/lib \
           -L$(HALIDE_DISTRIB_PATH)/lib \
           -lHalide \
           -ldl -lpthread -lz

# Add CUDA support if available
ifneq ($(wildcard $(TORCH_DIR)/lib/libtorch_cuda.so),)
    LDFLAGS += -ltorch_cuda -lcuda
    CXXFLAGS += -DTORCH_CUDA_AVAILABLE
endif

# Source files
SRCS = AutoSchedule.cpp \
       DefaultCostModel.cpp

# Object files
OBJS = $(SRCS:%.cpp=bin/%.o)

.PHONY: all clean

all: bin/libautoschedule_adams2019.so

# Create directories
bin:
	mkdir -p bin

# Compile source files
bin/%.o: $(SRC)/%.cpp | bin
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link the shared library
bin/libautoschedule_adams2019.so: $(OBJS)
	$(CXX) -shared $(CXXFLAGS) $(OBJS) -o $@ $(LDFLAGS)

# Install the autoscheduler
install: bin/libautoschedule_adams2019.so
	cp $< $(HALIDE_DISTRIB_PATH)/lib/

# Clean target
clean:
	rm -rf bin
