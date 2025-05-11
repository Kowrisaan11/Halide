# Makefile for adams2019 autoscheduler (corrected, no extraneous endif)

CXX := g++
THIS_MAKEFILE := $(realpath $(lastword $(MAKEFILE_LIST)))
SRC := $(strip $(shell dirname $(THIS_MAKEFILE)))
HALIDE_SRC_ROOT := $(realpath $(SRC)/../../../)
COMMON_DIR := $(realpath $(SRC)/../common/)
HALIDE_DISTRIB_PATH ?= $(HALIDE_SRC_ROOT)/distrib

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

SRCS := AutoSchedule.cpp DefaultCostModel.cpp
OBJS := $(patsubst %.cpp,bin/%.o,$(SRCS))

TARGET := bin/libautoschedule_adams2019.so

.PHONY: all clean

all: $(TARGET)

bin:
	mkdir -p bin

bin/%.o: %.cpp | bin
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

clean:
	rm -rf bin
