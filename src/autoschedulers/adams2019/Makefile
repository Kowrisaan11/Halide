# Makefile for Adams2019 autoscheduler
CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Werror -Wno-unused-function -Wcast-qual -Wignored-qualifiers \
           -Wno-comment -Wsign-compare -Wno-unknown-warning-option -Wno-psabi \
           -fPIC -fvisibility=hidden -fvisibility-inlines-hidden \
           -I$(HALIDE_DISTRIB_PATH)/include -I$(HALIDE_DISTRIB_PATH)/tools \
           -I/home/kowrisaan/libtorch/include -I/usr/include/nlohmann \
           -I/home/kowrisaan/fyp/Halide/src/autoschedulers/common

LDFLAGS = -L$(HALIDE_DISTRIB_PATH)/lib -L/home/kowrisaan/libtorch/lib \
          -lHalide -ltorch -ltorch_cpu -lc10 -lpthread -ldl

NLOHMANN_JSON = -I/home/kowrisaan/fyp/libtorch/include/nlohmann

SRC_DIR = /home/kowrisaan/fyp/Halide/src/autoschedulers/adams2019
SOURCES = $(SRC_DIR)/AutoSchedule.cpp $(SRC_DIR)/FunctionDAG.cpp $(SRC_DIR)/Featurization.cpp $(SRC_DIR)/SimpleLSTMModel.cpp $(SRC_DIR)/TreeRepresentation.cpp
OBJECTS = $(notdir $(SOURCES:.cpp=.o))

all: $(HALIDE_DISTRIB_PATH)/bin/libautoschedule_adams2019.so

%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(NLOHMANN_JSON) -c $< -o $@

$(HALIDE_DISTRIB_PATH)/bin/libautoschedule_adams2019.so: $(OBJECTS)
	mkdir -p $(HALIDE_DISTRIB_PATH)/bin
	$(CXX) -shared -rdynamic $(OBJECTS) $(LDFLAGS) -o $@ '-Wl,-rpath,$$ORIGIN'

clean:
	rm -f *.o *.d $(HALIDE_DISTRIB_PATH)/bin/libautoschedule_adams2019.so

.PHONY: all clean
