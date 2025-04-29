THIS_MAKEFILE = $(realpath $(filter %Makefile, $(MAKEFILE_LIST)))
SRC = $(strip $(shell dirname $(THIS_MAKEFILE)))
HALIDE_SRC_ROOT = $(realpath $(SRC)/../../../)
COMMON_DIR ?= $(realpath $(SRC)/../common/)

# ▼ Add LibTorch paths ▼
LIBTORCH_INCLUDE = /home/kowrisaan/fyp/libtorch/include/
LIBTORCH_CSRC_API = /home/kowrisaan/fyp/libtorch/include/torch/csrc/api/include/
LIBTORCH_LIB = /home/kowrisaan/fyp/libtorch/lib/

HALIDE_DISTRIB_PATH ?= $(HALIDE_SRC_ROOT)/distrib

$(info Looking for Halide distro at $(HALIDE_DISTRIB_PATH). If this is incorrect, set the make variable HALIDE_DISTRIB_PATH)

# Don't include an autoscheduler in the generator deps
AUTOSCHEDULER=
include $(HALIDE_SRC_ROOT)/apps/support/Makefile.inc

# Add LibTorch includes to compiler flags ▼
CXXFLAGS += -I$(LIBTORCH_INCLUDE) -I$(LIBTORCH_CSRC_API)
# Add LibTorch library path to linker flags ▼
LIBHALIDE_LDFLAGS += -L$(LIBTORCH_LIB) -Wl,-rpath,$(LIBTORCH_LIB) -ltorch

# Rest of original Makefile content remains the same until the clean rule...

$(BIN)/cost_model.generator: $(SRC)/cost_model_generator.cpp \
                $(SRC)/cost_model_schedule.h \
                $(SRC)/NetworkSize.h \
                $(GENERATOR_DEPS)
    @mkdir -p $(@D)
    $(CXX) $(CXXFLAGS) $(filter %.cpp,$^) -o $@ $(USE_EXPORT_DYNAMIC) $(LIBHALIDE_LDFLAGS)

$(BIN)/libautoschedule_adams2019.$(PLUGIN_EXT): \
                ...  # (rest of existing dependencies)
    @mkdir -p $(@D)
    $(CXX) -shared $(USE_EXPORT_DYNAMIC) -fPIC -fvisibility=hidden -fvisibility-inlines-hidden \
    $(CXXFLAGS) $(OPTIMIZE) -I $(BIN)/cost_model $(filter-out %.h $(LIBHALIDE_LDFLAGS),$^) -o $@ \
    $(HALIDE_SYSTEM_LIBS) $(HALIDE_RPATH_FOR_LIB) -I $(SRC) $(LIBHALIDE_LDFLAGS)  # ▼ Added LIBHALIDE_LDFLAGS

$(BIN)/adams2019_retrain_cost_model: ...  # (existing dependencies)
    @mkdir -p $(@D)
    $(CXX) $(CXXFLAGS) -frtti -Wall -I ../support -I $(BIN)/cost_model $(OPTIMIZE) \
    $(filter-out %.h,$^) -o $@ $(LIBHALIDE_LDFLAGS) $(USE_OPEN_MP) $(HALIDE_RPATH_FOR_BIN) -I $(SRC)

.PHONY: clean
clean:
    rm -rf $(BIN)
