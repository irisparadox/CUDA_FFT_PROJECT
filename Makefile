CC = g++
NVCC = nvcc
CFLAGS = -Wall -I./include -I/usr/local/cuda/include -I./external/imgui -I./external/imgui/backends -I./external/glfw/include
LDFLAGS = -L/usr/lib/x86_64-linux-gnu -L./external/glfw/build/src
LIBS = -lcudart -lcufft -lm -lGLEW -lGL -lglfw3
OUTPUT_DIR = ./output
SRC_DIR = ./src
KERNELS_DIR = ./kernels
INCLUDE_DIR = ./include
IMGUI_DIR = ./external/imgui
EXEC = $(OUTPUT_DIR)/fft_program

SRC_CPP = $(wildcard $(SRC_DIR)/*.cpp)
KERNELS_CU = $(wildcard $(KERNELS_DIR)/*.cu)
IMGUI_SRC = \
    $(IMGUI_DIR)/imgui.cpp \
    $(IMGUI_DIR)/imgui_draw.cpp \
    $(IMGUI_DIR)/imgui_tables.cpp \
    $(IMGUI_DIR)/imgui_widgets.cpp \
    $(IMGUI_DIR)/backends/imgui_impl_glfw.cpp \
    $(IMGUI_DIR)/backends/imgui_impl_opengl3.cpp
OBJ_CPP = $(SRC_CPP:.cpp=.o)
OBJ_CU = $(KERNELS_CU:.cu=.o)
OBJ_IMGUI = $(IMGUI_SRC:.cpp=.o)

all: $(EXEC)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_CU): $(KERNELS_DIR)/%.o : $(KERNELS_DIR)/%.cu
	$(NVCC) -c $< -o $@

$(EXEC): $(OBJ_CPP) $(OBJ_IMGUI) $(OBJ_CU)
	$(CC) $(OBJ_CPP) $(OBJ_IMGUI) $(OBJ_CU) -o $(EXEC) $(LDFLAGS) $(LIBS)

clean:
	rm -f $(OBJ_CPP) $(OBJ_IMGUI) $(OBJ_CU) $(EXEC)

run: $(EXEC)
	./$(EXEC)
