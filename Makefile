CXX = g++
CXXFLAGS = -std=c++17 -Ofast -Wall -fopenmp
LIBS = -lpng 
TARGET = raytracer

# Source files
SRC = main.cpp $(wildcard src/*.cpp) $(wildcard lib/*.c)

# Object files
OBJ = $(SRC:.cpp=.o)

.PHONY: build run clean

# Build command
build: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJ) $(LIBS)

# Compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run command with file input
run: $(TARGET)
	./$(TARGET) $(file)

# Clean command
clean:
	rm -f $(OBJ) $(TARGET)
	rm -f ./*.png	