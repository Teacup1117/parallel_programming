NVFLAGS := -std=c++11 -O3 -use_fast_math -Xcompiler -O3 -Xcompiler -march=native
TARGET := hw4

.PHONY: all
all: $(TARGET)

$(TARGET): sha256.o hw4.o
	nvcc $(NVFLAGS) -o hw4 hw4.o sha256.o

hw4.o: hw4.cu sha256.h
	nvcc $(NVFLAGS) -c hw4.cu

sha256.o: sha256.cu sha256.h
	nvcc $(NVFLAGS) -c sha256.cu

clean:
	rm -rf hw4 *.o




