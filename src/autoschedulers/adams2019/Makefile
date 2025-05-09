

cat > /home/kowrisaan/fyp/Halide/src/autoschedulers/adams2019/Makefile << 'EOF'
# Makefile for Adams2019 autoscheduler

CXX = g++
CXXFLAGS = -g -O3 -std=c++17 -Wall -Werror -Wno-unused-function -Wcast-qual -Wignored-qualifiers \
           -Wno-comment -Wsign-compare -Wno-unknown-warning-option -Wno-psabi \
           -fPIC -fvisibility=hidden -fvisibility-inlines-hidden \
           -I$(HALIDE_DISTRIB_PATH)/include -I$(HALIDE_DISTRIB_PATH)/tools \
           -I/home/kowrisaan/libtorch/include -I/usr/include/nlohmann
LDFLAGS = -L$(HALIDE_DISTRIB_PATH)/lib -L/home/kowrisaan/libtorch/lib \
          -lHalide -ltorch -ltorch_cpu -lc10 -lpthread -ldl
NLOHMANN_JSON = -I/home/kowrisaan/fyp/libtorch/include/nlohmann

SOURCES = AutoSchedule.cpp FunctionDAG.cpp Featurization.cpp SimpleLSTMModel.cpp TreeRepresentation.cpp State.cpp
OBJECTS = $(SOURCES:.cpp=.o)
DEPS = $(SOURCES:.cpp=.d)

all: libautoschedule_adams2019.a bin/libautoschedule_adams2019.so

libautoschedule_adams2019.a: $(OBJECTS)
    @ar rcs $@ $^

bin/libautoschedule_adams2019.so: $(OBJECTS)
    @echo "Linking $@"
    @$(CXX) -shared -rdynamic $(OBJECTS) $(LDFLAGS) -o $@ '-Wl,-rpath,$$ORIGIN'

%.o: %.cpp
    @echo "Compiling $<"
    @$(CXX) $(CXXFLAGS) $(NLOHMANN_JSON) -c $< -o $@ -v 2> $(basename $<).err
    @if [ -s $(basename $<).err ]; then cat $(basename $<).err; exit 1; fi

%.d: %.cpp
    @echo "Generating dependencies for $<"
    @$(CXX) $(CXXFLAGS) $(NLOHMANN_JSON) -M $< > $@ -v

bin:
    @mkdir -p bin

clean:
    @rm -f *.o *.d *.err libautoschedule_adams2019.a bin/libautoschedule_adams2019.so

check-sources:
    @for src in $(SOURCES); do \
        if [ ! -f "$$src" ]; then \
            echo "Error: Source file $$src is missing"; \
            exit 1; \
        fi; \
        if [ ! -s "$$src" ]; then \
            echo "Error: Source file $$src is empty"; \
            exit 1; \
        fi; \
    done

all: check-sources

-include $(DEPS)

.PHONY: all clean check-sources
EOF
