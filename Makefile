# CUDA compiler and standard C++ compiler
NVCC    = nvcc
CXX     = g++

# CUDA-specific flags for .cu files
CUFLAGS = -std=c++17 -arch=sm_80 -rdc=true -Iinclude -G -g

# Host C++ flags for .cpp files
CXXFLAGS = -std=c++17 -Iinclude

# Directories
SRC_DIR  = src

# Source files
CU_SRCS  = $(wildcard $(SRC_DIR)/*.cu) main.cu
CPP_SRCS = $(wildcard $(SRC_DIR)/*.cpp)

# Object files
CU_OBJS  = $(CU_SRCS:.cu=.cu.o)
CPP_OBJS = $(CPP_SRCS:.cpp=.cpp.o)
OBJS     = $(CU_OBJS) $(CPP_OBJS)

# Final executable
TARGET   = raytracer

# Default build
all: $(TARGET)

# Linking final executable (with nvcc so device relocatable code is linked properly)
$(TARGET): $(OBJS)
	$(NVCC) $(CUFLAGS) -o $@ $^ -lpng

# CUDA compile rule for .cu -> .cu.o
%.cu.o: %.cu
	$(NVCC) $(CUFLAGS) -c $< -o $@

# Host compile rule for .cpp -> .cpp.o
%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean
clean:
	rm -f $(TARGET) *.o $(SRC_DIR)/*.o

.PHONY: all clean
