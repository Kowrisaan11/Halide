# Makefile for Halide Adams2019 autoscheduler
# Located at /home/kowrisaan/fyp/Halide/src/autoschedulers/adams2019/Makefile

# Paths
HALIDE_SRC_ROOT = ../../../../..
HALIDE_DISTRIB_PATH ?= $(HALIDE_SRC_ROOT)/distrib
HALIDE_TOOLS_DIR = $(HALIDE_DISTRIB_PATH)/tools
HALIDE_INCLUDE_DIR = $(HALIDE_DISTRIB_PATH)/include
HALIDE_LIB_DIR = $(HALIDE_DISTRIB_PATH)/lib
COMMON_DIR = ../common

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

# Source files
SOURCES = \
    AutoSchedule.cpp \
    DefaultCostModel.cpp \
    FunctionDAG.cpp

OBJECTS = $(SOURCES:.cpp=.o)

# Target
TARGET = bin/libautoschedule_adams2019.so

all: $(TARGET)

$(TARGET): $(OBJECTS)
	@mkdir -p bin
	$(CXX) -shared -rdynamic $(OBJECTS) -o $@ $(LDFLAGS) -Wl,-rpath,'$$ORIGIN'

# Explicit rules for object files
AutoSchedule.o: AutoSchedule.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

DefaultCostModel.o: DefaultCostModel.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

FunctionDAG.o: FunctionDAG.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJECTS) $(TARGET) bin

.PHONY: all clean
