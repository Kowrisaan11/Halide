# Complete Makefile for adams2019 autoscheduler
# Place this in /home/kowrisaan/fyp/Halide/src/autoschedulers/adams2019/
# Supports building bin/libautoschedule_adams2019.so from C++ sources in this directory

CXX := g++
SRCDIR := .
HALIDE_SRC_ROOT := $(realpath $(SRCDIR)/../../../)
HALIDE_DISTRIB_PATH ?= $(HALIDE_SRC_ROOT)/distrib
COMMON_DIR := $(realpath $(SRCDIR)/../common/)

CXXFLAGS := -std=c++17 -fPIC -O3 -Wall -Werror \
  -I$(HALIDE_DISTRIB_PATH)/include \
  -I$(HALIDE_DISTRIB_PATH)/tools \
  -I$(HALIDE_SRC_ROOT)/src \
  -I$(HALIDE_SRC_ROOT)/tools \
  -I$(COMMON_DIR) \
  -I$(SRCDIR) \
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

# Source and object files
SRCS := \
  AutoSchedule.cpp \
  DefaultCostModel.cpp

OBJS := $(patsubst %.cpp,bin/%.o,$(SRCS))

# Target shared library
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
