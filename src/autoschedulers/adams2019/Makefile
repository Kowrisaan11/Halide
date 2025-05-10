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

# Add required libraries for JSON and PyTorch
TORCH_DIR ?= /usr/local/libtorch
JSON_DIR ?= /usr/local/include

CXXFLAGS += -I$(COMMON_DIR) -I$(TORCH_DIR)/include -I$(JSON_DIR) -I$(TORCH_DIR)/include/torch/csrc/api/include
LDFLAGS += -L$(TORCH_DIR)/lib -ltorch -ltorch_cpu -lc10
# Add CUDA support if available
ifneq ($(wildcard $(TORCH_DIR)/lib/libtorch_cuda.so),)
    LDFLAGS += -ltorch_cuda -lcuda
    CXXFLAGS += -DTORCH_CUDA_AVAILABLE
endif

AUTOSCHED_WEIGHT_OBJECTS=$(BIN)/baseline_weights.o

# Copy model files to bin directory
MODEL_FILES = model.pt scaler_params.json calibration_data.txt baseline.weights category_calibration.txt
$(BIN)/model_files: $(addprefix $(SRC)/, $(MODEL_FILES))
	@mkdir -p $(@D)
	cp $^ $(@D)/
	@touch $@

$(BIN)/binary2cpp: $(HALIDE_SRC_ROOT)/tools/binary2cpp.cpp
	@mkdir -p $(@D)
	$(CXX) $< -o $@

$(BIN)/baseline_weights.cpp: $(BIN)/binary2cpp $(SRC)/baseline.weights
	@mkdir -p $(@D)
	$(BIN)/binary2cpp baseline_weights < $(SRC)/baseline.weights > $@

$(BIN)/baseline_weights.o: $(BIN)/baseline_weights.cpp
	$(CXX) -c $< -o $@

AUTOSCHED_COST_MODEL_LIBS=\
$(BIN)/cost_model/adams2019_cost_model.a \
$(BIN)/cost_model/adams2019_train_cost_model.a \

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
				$(AUTOSCHED_WEIGHT_OBJECTS) \
				$(AUTOSCHED_COST_MODEL_LIBS) \
				$(BIN)/auto_schedule_runtime.a \
				$(BIN)/model_files \
				| $(LIB_HALIDE)
	@mkdir -p $(@D)
	$(CXX) -shared $(USE_EXPORT_DYNAMIC) -fPIC -fvisibility=hidden -fvisibility-inlines-hidden \
		$(CXXFLAGS) $(OPTIMIZE) -I $(BIN)/cost_model \
		$(filter-out %.h $(LIBHALIDE_LDFLAGS) $(BIN)/model_files,$^) -o $@ \
		$(HALIDE_SYSTEM_LIBS) $(LDFLAGS) $(HALIDE_RPATH_FOR_LIB) -I $(SRC)

# Updated retrain cost model build
$(BIN)/adams2019_retrain_cost_model: $(SRC)/retrain_cost_model.cpp \
				$(COMMON_DIR)/ASLog.cpp \
				$(SRC)/DefaultCostModel.h \
				$(SRC)/DefaultCostModel.cpp \
				$(SRC)/CostModel.h \
				$(SRC)/NetworkSize.h \
				$(AUTOSCHED_COST_MODEL_LIBS) \
				$(AUTOSCHED_WEIGHT_OBJECTS) \
				$(BIN)/auto_schedule_runtime.a \
				$(BIN)/model_files
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -frtti -Wall -I ../support -I $(BIN)/cost_model $(OPTIMIZE) \
		$(filter-out %.h $(BIN)/model_files,$^) -o $@ \
		$(LIBHALIDE_LDFLAGS) $(LDFLAGS) $(USE_OPEN_MP) $(HALIDE_RPATH_FOR_BIN) -I $(SRC)

.PHONY: clean

clean:
	rm -rf $(BIN)
