CXX = hipcc
CXXFLAGS = -std=c++17 -O3 --offload-arch=gfx908
TARGET = hw5
SRC = hw5.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

.PHONY: all clean
