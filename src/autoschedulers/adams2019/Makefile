# Makefile for Adams2019 autoscheduler

CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall -Werror -Wno-unused-function -Wcast-qual -Wignored-qualifiers \
           -Wno-comment -Wsign-compare -Wno-unknown-warning-option -Wno-psabi \
           -fPIC -fvisibility=hidden -fvisibility-inlines-hidden \
           -I$(HALIDE_DISTRIB_PATH)/include -I$(HALIDE_DISTRIB_PATH)/tools \
           -I/home/kowrisaan/libtorch/include -I/usr/include/nlohmann
LDFLAGS = -L$(HALIDE_DISTRIB_PATH)/lib -L/home/kowrisaan/libtorch/lib \
          -lHalide -ltorch -ltorch_cpu -lc10 -lpthread -ldl
NLOHMANN_JSON = -I/home/kowrisaan/fyp/libtorch/include/nlohmann

SOURCES = AutoSchedule.cpp FunctionDAG.cpp Featurization.cpp SimpleLSTMModel.cpp TreeRepresentation.cpp
OBJECTS = $(SOURCES:.cpp=.o)
DEPS = $(SOURCES:.cpp=.d)

all: libautoschedule_adams2019.a bin/libautoschedule_adams2019.so

libautoschedule_adams2019.a: $(OBJECTS)
	ar rcs $@ $^

bin/libautoschedule_adams2019.so: $(OBJECTS)
	$(CXX) -shared -rdynamic $(OBJECTS) $(LDFLAGS) -o $@ '-Wl,-rpath,$$ORIGIN'

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(NLOHMANN_JSON) -c $< -o $@

%.d: %.cpp
	$(CXX) $(CXXFLAGS) $(NLOHMANN_JSON) -M $< > $@

bin:
	mkdir -p bin

clean:
	rm -f *.o *.d libautoschedule_adams2019.a bin/libautoschedule_adams2019.so

-include $(DEPS)

.PHONY: all clean
