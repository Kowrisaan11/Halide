THIS_MAKEFILE = $(realpath $(filter %Makefile, $(MAKEFILE_LIST)))
SRC = $(strip $(shell dirname $(THIS_MAKEFILE)))
HALIDE_SRC_ROOT = $(realpath $(SRC)/../../../)
COMMON_DIR ?= $(realpath $(SRC)/../common/)

HALIDE_DISTRIB_PATH ?= $(HALIDE_SRC_ROOT)/distrib

$(info Looking for Halide distro at $(HALIDE_DISTRIB_PATH). If this is incorrect, set the make variable HALIDE_DISTRIB_PATH)

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

# Set LibTorch directory
TORCH_DIR = /home/kowrisaan/fyp/libtorch
JSON_DIR ?= /usr/local/include

# Add timestamp and user information as compiler definitions
CURRENT_TIME = 2025-05-10 17:40:02
CURRENT_USER = Jathu03
CXXFLAGS += -DCURRENT_TIME=\"$(CURRENT_TIME)\" -DCURRENT_USER=\"$(CURRENT_USER)\"

# Add include paths
CXXFLAGS += -I$(COMMON_DIR) \
            -I$(TORCH_DIR)/include \
            -I$(TORCH_DIR)/include/torch/csrc/api/include \
            -I$(JSON_DIR)

# Add library paths and libraries
LDFLAGS += -L$(TORCH_DIR)/lib -ltorch -ltorch_cpu -lc10

# Add CUDA support if available
ifneq ($(wildcard $(TORCH_DIR)/lib/libtorch_cuda.so),)
    LDFLAGS += -ltorch_cuda -lcuda
    CXXFLAGS += -DTORCH_CUDA_AVAILABLE
endif

# Add rpath for LibTorch
LDFLAGS += -Wl,-rpath,$(TORCH_DIR)/lib

# Verify LibTorch installation
TORCH_CHECK := $(shell test -f $(TORCH_DIR)/lib/libtorch.so && echo "yes" || echo "no")
ifeq ($(TORCH_CHECK),no)
    $(error LibTorch not found in $(TORCH_DIR)/lib. Please check the installation)
endif

$(info Using LibTorch from $(TORCH_DIR))
$(info Current user: $(CURRENT_USER))
$(info Current time: $(CURRENT_TIME))

# Only required model files
MODEL_FILES = model.pt scaler_params.json
$(BIN)/model_files: $(addprefix $(SRC)/, $(MODEL_FILES))
	@mkdir -p $(@D)
	cp $^ $(@D)/
	@touch $@

$(BIN)/cost_model.generator: $(SRC)/cost_model_generator.cpp \
				$(SRC)/cost_model_schedule.h \
				$(SRC)/NetworkSize.h \
				$(GENERATOR_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(filter %.cpp,$^) -o $@ $(USE_EXPORT_DYNAMIC) $(LIBHALIDE_LDFLAGS)

$(BIN)/auto_schedule_runtime.a: $(BIN)/cost_model.generator
	@mkdir -p $(@D)
	$^ -r auto_schedule_runtime -o $(BIN) target=$(HL_TARGET)

$(BIN)/cost_model/adams2019_%.a: $(BIN)/cost_model.generator
	@mkdir -p $(@D)
	$^ -g $* -o $(BIN)/cost_model -f $* -n adams2019_$* target=$(HL_TARGET)-no_runtime -e stmt,static_library,h,assembly

# Updated build command for the autoscheduler plugin
$(BIN)/libautoschedule_adams2019.$(PLUGIN_EXT): \
				$(COMMON_DIR)/ASLog.cpp \
				$(SRC)/AutoSchedule.cpp \
				$(SRC)/Cache.h \
				$(SRC)/Cache.cpp \
				$(SRC)/DefaultCostModel.h \
				$(SRC)/DefaultCostModel.cpp \
				$(SRC)/CostModel.h \
				$(SRC)/FunctionDAG.h \
				$(SRC)/FunctionDAG.cpp \
				$(SRC)/LoopNest.h \
				$(SRC)/LoopNest.cpp \
				$(SRC)/Featurization.h \
				$(SRC)/State.h \
				$(SRC)/State.cpp \
				$(SRC)/Timer.h \
				$(COMMON_DIR)/PerfectHashMap.h \
				$(AUTOSCHED_COST_MODEL_LIBS) \
				$(BIN)/auto_schedule_runtime.a \
				$(BIN)/model_files \
				| $(LIB_HALIDE)
	@mkdir -p $(@D)
	@echo "Building autoscheduler plugin..."
	@echo "Using LibTorch: $(TORCH_DIR)"
	@echo "User: $(CURRENT_USER)"
	@echo "Time: $(CURRENT_TIME)"
	$(CXX) -shared $(USE_EXPORT_DYNAMIC) -fPIC -fvisibility=hidden -fvisibility-inlines-hidden \
		$(CXXFLAGS) $(OPTIMIZE) -I $(BIN)/cost_model \
		$(filter-out %.h $(LIBHALIDE_LDFLAGS) $(BIN)/model_files,$^) -o $@ \
		$(HALIDE_SYSTEM_LIBS) $(LDFLAGS) $(HALIDE_RPATH_FOR_LIB) -I $(SRC)
	@echo "Successfully built autoscheduler plugin"

# Build verification target
verify_torch:
	@echo "Verifying LibTorch installation..."
	@test -f $(TORCH_DIR)/lib/libtorch.so || (echo "Error: LibTorch not found in $(TORCH_DIR)/lib" && exit 1)
	@echo "LibTorch verified successfully"

.PHONY: clean verify_torch

clean:
	rm -rf $(BIN)
