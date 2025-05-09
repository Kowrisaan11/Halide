# Adapted from Adams2019 autoscheduler Makefile

CXX = g++
CXXFLAGS = -g -std=c++17 -Wall -fno-rtti -I$(HALIDE_DISTRIB_PATH)/include -I$(HALIDE_DISTRIB_PATH)/tools -I/home/kowrisaan/libtorch/include -I/usr/include
LDFLAGS = -L$(HALIDE_DISTRIB_PATH)/lib -L/home/kowrisaan/libtorch/lib -lHalide -lpthread -ldl -ltorch -ltorch_cpu -lc10
NLOHMANN_JSON = -I/usr/include

SOURCES = AutoSchedule.cpp FunctionDAG.cpp Featurization.cpp SimpleLSTMModel.cpp TreeRepresentation.cpp
OBJECTS = $(SOURCES:.cpp=.o)
DEPS = $(SOURCES:.cpp=.d)

all: libautoschedule_adams2019.a

libautoschedule_adams2019.a: $(OBJECTS)
	ar rcs $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(NLOHMANN_JSON) -c $< -o $@

%.d: %.cpp
	$(CXX) $(CXXFLAGS) $(NLOHMANN_JSON) -M $< > $@

clean:
	rm -f *.o *.d libautoschedule_adams2019.a

-include $(DEPS)

.PHONY: all clean
