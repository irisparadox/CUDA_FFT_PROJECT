CC = g++
NVCC = nvcc
CFLAGS = -Wall -g -I./include -I./external -I/usr/local/cuda/include -I./external/imgui -I./external/imgui/backends -I./external/glfw/include -I./external/glm
LDFLAGS = -L/usr/lib/x86_64-linux-gnu -L./external/glfw/build/src
LIBS = -lcudart -lcufft -lm -lGLEW -lGL -lglfw3
OUTPUT_DIR = ./output
OBJ_DIR = $(OUTPUT_DIR)/obj
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
OBJ_CPP = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/cxx/%.o,$(SRC_CPP))
OBJ_CU = $(patsubst $(KERNELS_DIR)/%.cu,$(OBJ_DIR)/cuda/%.o,$(KERNELS_CU))
OBJ_IMGUI = $(patsubst $(IMGUI_DIR)/%.cpp,$(OBJ_DIR)/imgui/%.o,$(IMGUI_SRC))

all: $(EXEC)

$(OBJ_DIR)/cxx/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/cuda/%.o: $(KERNELS_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) -c $< -o $@

$(OBJ_DIR)/imgui/%.o: $(IMGUI_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(EXEC): $(OBJ_CPP) $(OBJ_IMGUI) $(OBJ_CU)
	$(CC) $^ -o $@ $(LDFLAGS) $(LIBS)

clean:
	rm -rf $(OUTPUT_DIR)/obj
	rm -f $(EXEC)

run: $(EXEC)
	./$(EXEC)