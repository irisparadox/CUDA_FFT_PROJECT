CC = g++
NVCC = nvcc
CFLAGS = -Wall -I./include -I/usr/local/cuda/include
LDFLAGS = -L/usr/lib/x86_64-linux-gnu
LIBS = -lcudart -lcufft -lm
OUTPUT_DIR = ./output
SRC_DIR = ./src
KERNELS_DIR = ./kernels
INCLUDE_DIR = ./include
EXEC = $(OUTPUT_DIR)/fft_program

SRC_CPP = $(SRC_DIR)/main.cpp
KERNELS_CU = $(KERNELS_DIR)/spectra.cu
OBJ_CPP = $(SRC_CPP:.cpp=.o)
OBJ_CU = $(KERNELS_CU:.cu=.o)

all: $(EXEC)

$(OBJ_CPP): $(SRC_CPP)
	$(CC) $(CFLAGS) -c $(SRC_CPP) -o $(OBJ_CPP)

$(OBJ_CU): $(KERNELS_CU)
	$(NVCC) -c $(KERNELS_CU) -o $(OBJ_CU)

$(EXEC): $(OBJ_CPP) $(OBJ_CU)
	$(CC) $(OBJ_CPP) $(OBJ_CU) -o $(EXEC) $(LDFLAGS) $(LIBS)

clean:
	rm -f $(OBJ_CPP) $(OBJ_CU) $(EXEC)

run: $(EXEC)
	./$(EXEC)
