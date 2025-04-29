CC = g++
NVCC = nvcc
CFLAGS = -Wall -I./include -I/usr/local/cuda/include -I./external/glfw/include
LDFLAGS = -L/usr/lib/x86_64-linux-gnu -L./external/glfw/build/src
LIBS = -lcudart -lcufft -lm -lGLEW -lGL -lglfw3
OUTPUT_DIR = ./output
SRC_DIR = ./src
KERNELS_DIR = ./kernels
INCLUDE_DIR = ./include
EXEC = $(OUTPUT_DIR)/fft_program

SRC_CPP = $(wildcard $(SRC_DIR)/*.cpp)
KERNELS_CU = $(wildcard $(KERNELS_DIR)/*.cu)
OBJ_CPP = $(SRC_CPP:.cpp=.o)
OBJ_CU = $(KERNELS_CU:.cu=.o)

all: $(EXEC)

$(OBJ_CPP): $(SRC_DIR)/%.o : $(SRC_DIR)/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_CU): $(KERNELS_DIR)/%.o : $(KERNELS_DIR)/%.cu
	$(NVCC) -c $< -o $@

$(EXEC): $(OBJ_CPP) $(OBJ_CU)
	$(CC) $(OBJ_CPP) $(OBJ_CU) -o $(EXEC) $(LDFLAGS) $(LIBS)

clean:
	rm -f $(OBJ_CPP) $(OBJ_CU) $(EXEC)

run: $(EXEC)
	./$(EXEC)
