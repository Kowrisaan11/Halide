# Minimal Makefile for adams2019 autoscheduler (no retraining tools)

SRC_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
HALIDE_SRC_ROOT := $(realpath $(SRC_DIR)/../../../)
HALIDE_DISTRIB_PATH ?= $(HALIDE_SRC_ROOT)/distrib

CXX      := g++
CXXFLAGS := -std=c++17 -fPIC -O3 -Wall -Werror \
            -I$(HALIDE_DISTRIB_PATH)/include \
            -I$(HALIDE_DISTRIB_PATH)/tools \
            -I$(HALIDE_SRC_ROOT)/src \
            -I$(HALIDE_SRC_ROOT)/tools \
            -I$(SRC_DIR)/../common \
            -I$(SRC_DIR) \
            -I/home/kowrisaan/fyp/libtorch/include \
            -I/home/kowrisaan/fyp/libtorch/include/torch/csrc/api/include

LDFLAGS  := -shared -L/home/kowrisaan/fyp/libtorch/lib \
            -ltorch -ltorch_cpu -lc10 \
            -Wl,-rpath,/home/kowrisaan/fyp/libtorch/lib \
            -L$(HALIDE_DISTRIB_PATH)/lib \
            -lHalide -ldl -lpthread -lz

# Add CUDA if available
ifneq ($(wildcard /home/kowrisaan/fyp/libtorch/lib/libtorch_cuda.so),)
    LDFLAGS += -ltorch_cuda -lcuda
    CXXFLAGS += -DTORCH_CUDA_AVAILABLE
endif

SRCS = AutoSchedule.cpp DefaultCostModel.cpp
OBJS = $(SRCS:%.cpp=bin/%.o)

.PHONY: all clean

all: bin/libautoschedule_adams2019.so

bin:
	mkdir -p bin

bin/%.o: %.cpp | bin
	$(CXX) $(CXXFLAGS) -c $< -o $@

bin/libautoschedule_adams2019.so: $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) $(LDFLAGS) -o $@

clean:
	rm -rf bin
