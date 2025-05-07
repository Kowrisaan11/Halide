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

CXXFLAGS += -I$(COMMON_DIR)

# LibTorch settings (adjust paths as needed)
TORCH_ROOT = /home/kowrisaan/libtorch
TORCH_INCLUDE = $(TORCH_ROOT)/include
TORCH_LIB = $(TORCH_ROOT)/lib
TORCH_LIBS = -ltorch -ltorch_cpu -lc10
CXXFLAGS += -I$(TORCH_INCLUDE)
LDFLAGS += -L$(TORCH_LIB) $(TORCH_LIBS)

# nlohmann/json settings (adjust paths as needed)
JSON_INCLUDE = /usr/include
CXXFLAGS += -I$(JSON_INCLUDE)

AUTOSCHED_WEIGHT_OBJECTS=$(BIN)/baseline_weights.o

$(BIN)/binary2cpp: $(HALIDE_SRC_ROOT)/tools/binary2cpp.cpp
	@mkdir -p $(@D)
	$(CXX) $< -o $@

$(BIN)/baseline_weights.cpp: $(BIN)/binary2cpp $(SRC)/baseline.weights
	@mkdir -p $(@D)
	$(BIN)/binary2cpp baseline_weights < $(SRC)/baseline.weights > $@

$(BIN)/baseline_weights.o: $(BIN)/baseline_weights.cpp
	$(CXX) -c $< -o $@

$(BIN)/libautoschedule_adams2019.$(PLUGIN_EXT): \
				$(COMMON_DIR)/ASLog.cpp \
				$(SRC)/AutoSchedule.cpp \
				$(SRC)/Cache.cpp \
				$(SRC)/DefaultCostModel.cpp \
				$(SRC)/Weights.cpp \
				$(SRC)/FunctionDAG.cpp \
				$(SRC)/LoopNest.cpp \
				$(SRC)/State.cpp \
				$(AUTOSCHED_WEIGHT_OBJECTS) \
				$(LIB_HALIDE)
	@mkdir -p $(@D)
	$(CXX) -shared $(USE_EXPORT_DYNAMIC) -fPIC -fvisibility=hidden -fvisibility-inlines-hidden $(CXXFLAGS) $(OPTIMIZE) \
	       $(filter-out %.h $(LIBHALIDE_LDFLAGS),$^) -o $@ $(HALIDE_SYSTEM_LIBS) $(HALIDE_RPATH_FOR_LIB) -I $(SRC)

$(BIN)/adams2019_weightsdir_to_weightsfile: $(SRC)/weightsdir_to_weightsfile.cpp $(SRC)/Weights.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $^ $(OPTIMIZE) -o $@ -I $(SRC)

.PHONY: clean

clean:
	rm -rf $(BIN)
