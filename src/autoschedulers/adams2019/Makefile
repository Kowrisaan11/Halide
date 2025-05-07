HALIDE_SRC_ROOT = ../..
HALIDE_BUILD_ROOT = $(HALIDE_SRC_ROOT)/build
HALIDE_LIB = $(HALIDE_BUILD_ROOT)/lib/libHalide.a

# Compiler settings
CXX = g++
CXXFLAGS = -g -Wall -Werror -I$(HALIDE_SRC_ROOT)/src -I$(HALIDE_SRC_ROOT)/tools -std=c++17
LDFLAGS = -L$(HALIDE_BUILD_ROOT)/lib -lHalide -lpthread -ldl

# LibTorch settings (adjust paths as needed)
TORCH_ROOT = /path/to/libtorch
TORCH_INCLUDE = $(TORCH_ROOT)/include
TORCH_LIB = $(TORCH_ROOT)/lib
TORCH_LIBS = -ltorch -ltorch_cpu -lc10
CXXFLAGS += -I$(TORCH_INCLUDE)
LDFLAGS += -L$(TORCH_LIB) $(TORCH_LIBS)

# nlohmann/json settings (adjust paths as needed)
JSON_INCLUDE = /path/to/nlohmann_json/include
CXXFLAGS += -I$(JSON_INCLUDE)

# Source files
SOURCES = \
    AutoSchedule.cpp \
    DefaultCostModel.cpp \
    FunctionDAG.cpp \
    Weights.cpp

OBJECTS = $(SOURCES:.cpp=.o)

TARGET = libautoschedule_adams2019.a

all: $(TARGET)

$(TARGET): $(OBJECTS)
	ar rcs $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)
