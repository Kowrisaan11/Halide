THIS_MAKEFILE = $(realpath $(filter %Makefile, $(MAKEFILE_LIST)))
SRC = $(strip $(shell dirname $(THIS_MAKEFILE)))
HALIDE_SRC_ROOT = $(realpath $(SRC)/../../../)
COMMON_DIR ?= $(realpath $(SRC)/../common/)

HALIDE_DISTRIB_PATH ?= $(HALIDE_SRC_ROOT)/distrib

TORCH_DIR = /home/kowrisaan/fyp/libtorch
JSON_DIR ?= /usr/local/include

CXXFLAGS += -I$(COMMON_DIR) \
            -I$(TORCH_DIR)/include \
            -I$(TORCH_DIR)/include/torch/csrc/api/include \
            -I$(JSON_DIR) \
            -I$(SRC) \
            -std=c++17

LDFLAGS += -L$(TORCH_DIR)/lib -ltorch -ltorch_cpu -lc10 -Wl,-rpath,$(TORCH_DIR)/lib

ifneq ($(wildcard $(TORCH_DIR)/lib/libtorch_cuda.so),)
    LDFLAGS += -ltorch_cuda -lcuda
    CXXFLAGS += -DTORCH_CUDA_AVAILABLE
endif

MODEL_FILES = model.pt scaler_params.json
$(BIN)/model_files: $(addprefix $(SRC)/, $(MODEL_FILES))
	@mkdir -p $(@D)
	cp $^ $(@D)/
	@touch $@

$(BIN)/libautoschedule_adams2019.$(PLUGIN_EXT): \
				$(SRC)/DefaultCostModel.cpp \
				$(BIN)/model_files
	@mkdir -p $(@D)
	$(CXX) -shared $(CXXFLAGS) $(OPTIMIZE) $< -o $@ $(LDFLAGS)

.PHONY: clean

clean:
	rm -rf $(BIN)
