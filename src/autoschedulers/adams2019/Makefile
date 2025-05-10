THIS_MAKEFILE = $(realpath $(filter %Makefile, $(MAKEFILE_LIST)))
SRC = $(strip $(shell dirname $(THIS_MAKEFILE)))
HALIDE_SRC_ROOT = $(realpath $(SRC)/../../../)
COMMON_DIR ?= $(realpath $(SRC)/../common/)

HALIDE_DISTRIB_PATH ?= $(HALIDE_SRC_ROOT)/distrib

# Don't include an autoscheduler in the generator deps
AUTOSCHEDULER=
include $(HALIDE_SRC_ROOT)/apps/support/Makefile.inc

# Add the relative location of libHalide.so in the rpath in a distro
ifeq ($(UNAME), Darwin)
HALIDE_RPATH_FOR_BIN = '-Wl,-rpath,@executable_path/../lib'
HALIDE_RPATH_FOR_LIB = '-Wl,-rpath,@loader_path'
else
HALIDE_RPATH_FOR_BIN = '-Wl,-rpath,$$ORIGIN/../lib'
HALIDE_RPATH_FOR_LIB = '-Wl,-rpath,$$ORIGIN'
endif

# LibTorch configuration
TORCH_DIR = /home/kowrisaan/fyp/libtorch

# Add required includes
CXXFLAGS += -I$(COMMON_DIR) \
            -I$(TORCH_DIR)/include \
            -I$(TORCH_DIR)/include/torch/csrc/api/include \
            -I$(SRC) \
            -std=c++17 \
            -Wall -Werror \
            -Wno-unused-function \
            -Wcast-qual \
            -Wignored-qualifiers \
            -Wno-comment \
            -Wsign-compare \
            -Wno-unknown-warning-option \
            -O3 \
            -fPIC
	    -I$(HALIDE_DISTRIB_PATH)/include \
            -I$(HALIDE_DISTRIB_PATH)/tools \
            -I$(HALIDE_SRC_ROOT)/src \
            -I$(HALIDE_SRC_ROOT)/tools

# Add library paths and libraries
LDFLAGS += -L$(TORCH_DIR)/lib \
           -ltorch -ltorch_cpu -lc10 \
           -Wl,-rpath,$(TORCH_DIR)/lib

# Add CUDA support if available
ifneq ($(wildcard $(TORCH_DIR)/lib/libtorch_cuda.so),)
    LDFLAGS += -ltorch_cuda -lcuda
    CXXFLAGS += -DTORCH_CUDA_AVAILABLE
endif

# Required source files
SRCS = AutoSchedule.cpp \
       DefaultCostModel.cpp

OBJS = $(SRCS:%.cpp=bin/%.o)

# Model files
MODEL_FILES = model.pt scaler_params.json

# Main targets
.PHONY: all clean

all: bin/libautoschedule_adams2019.so

# Create bin directory
bin:
	mkdir -p bin

# Copy model files
bin/model_files: $(addprefix $(SRC)/, $(MODEL_FILES)) | bin
	cp $^ bin/
	@touch $@

# Compile source files
bin/%.o: $(SRC)/%.cpp | bin
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link the shared library
bin/libautoschedule_adams2019.so: $(OBJS) bin/model_files
	$(CXX) -shared $(CXXFLAGS) $(OBJS) -o $@ $(LDFLAGS) $(HALIDE_RPATH_FOR_LIB)

# Clean target
clean:
	rm -rf bin
