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

CXXFLAGS += -I$(COMMON_DIR) \
            -I$(HALIDE_SRC_ROOT)/src \
            -I/home/kowrisaan/libtorch/include \
            -I/usr/include/nlohmann \
            -std=c++17 \
            -Wall \
            -Werror \
            -fno-rtti

LDFLAGS += -L/home/kowrisaan/libtorch/lib \
           -ltorch \
           -ltorch_cpu \
           -lc10 \
           -lpthread \
           -ldl

AUTOSCHEDULER_SRC = \
    $(COMMON_DIR)/ASLog.cpp \
    $(COMMON_DIR)/PerfectHashMap.h \
    $(SRC)/AutoSchedule.cpp \
    $(SRC)/Cache.cpp \
    $(SRC)/CostModel.h \
    $(SRC)/FunctionDAG.cpp \
    $(SRC)/SimpleLSTMModel.cpp \
    $(SRC)/State.cpp \
    $(SRC)/TreeRepresentation.cpp \
    $(SRC)/Timer.h

$(BIN)/libautoschedule_adams2019.$(PLUGIN_EXT): $(AUTOSCHEDULER_SRC) | $(LIB_HALIDE)
	@mkdir -p $(@D)
	$(CXX) -shared $(USE_EXPORT_DYNAMIC) -fPIC -fvisibility=hidden -fvisibility-inlines-hidden $(CXXFLAGS) $(OPTIMIZE) $(filter-out %.h, $^) -o $@ $(HALIDE_SYSTEM_LIBS) $(HALIDE_RPATH_FOR_LIB) $(LDFLAGS)

.PHONY: clean

clean:
	rm -rf $(BIN)
