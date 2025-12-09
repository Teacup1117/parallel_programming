#### Compiler & Flags #############################################
# On macOS, the default g++ is clang++ which doesn't support -fopenmp
# unless libomp is installed and special flags are provided. We therefore
# make OpenMP optional via USE_OPENMP=1.

CXX ?= g++
CXXFLAGS_BASE = -std=c++17 -O3 -pthread

# Enable with: make USE_OPENMP=1
ifeq ($(USE_OPENMP),1)
	CXXFLAGS_OPENMP = -fopenmp
else
	CXXFLAGS_OPENMP =
endif

CXXFLAGS = $(CXXFLAGS_BASE) $(CXXFLAGS_OPENMP)

TARGETS = hw1

.PHONY: all
all: $(TARGETS)

hw1: hw1.cpp
	$(CXX) $(CXXFLAGS_BASE) hw1.cpp -o hw1

hw1_parallel: hw1_parallel.cpp
	$(CXX) $(CXXFLAGS) hw1_parallel.cpp -o hw1_parallel

.PHONY: clean
clean:
	rm -f $(TARGETS)
