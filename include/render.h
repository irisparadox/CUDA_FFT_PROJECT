#ifndef RENDER_H
#define RENDER_H

#include <GL/glew.h>
#include "simulation.h"
#include <mutex>
#include <atomic>
#include <thread>
#include <algorithm>

const int HISTORY_SIZE = 200;

struct StatBuffer {
    float values[HISTORY_SIZE] = {};
    float times[HISTORY_SIZE] = {};
    int offset = 0;
    std::mutex mtx;

    StatBuffer() {
        std::fill_n(values, HISTORY_SIZE, 0.0f);
        std::fill_n(times, HISTORY_SIZE, 0.0f);
    }

    void add_point(float time, float value) {
        std::lock_guard<std::mutex> lock(mtx);
        values[offset] = value;
        times[offset] = time;
        offset = (offset + 1) % HISTORY_SIZE;
    }

    void get_data(float* out_times, float* out_values, int& out_size) {
        std::lock_guard<std::mutex> lock(mtx);
        for (int i = 0; i < HISTORY_SIZE; ++i) {
            int idx = (offset + i) % HISTORY_SIZE;
            out_times[i] = times[idx];
            out_values[i] = values[idx];
        }
        out_size = HISTORY_SIZE;
    }
};

class Render {
public:
    Render(GLFWwindow* window);
    ~Render();

    void init();
    void update();
    void shutdown();

private:
    void init_imgui_context(GLFWwindow* window);
    void init_texture(GLuint* textureID, int width, int height, GLint type, GLint channels, GLint filter);
    void update_texture(GLuint vbo, GLuint* textureID, int width, int height, GLint type, GLint channels, GLint filter);
    void render_gui_window();
    void render_stats_plots();

    void start_stats_thread();
    void stop_stats_thread();
    float get_cpu_usage();
    float get_ram_usage();
    std::pair<float,float> get_gpu_usage();

    Simulation sim;
    GLuint ifft_heightmap;
    GLuint ifft_slope;
    GLuint initial_jonswap;
    GLuint h0t;
    int tex_width, tex_height;
    StatBuffer ram_usage, vram_usage, cpu_usage, gpu_usage;

    std::atomic<bool> stats_running;
    std::atomic<float> current_time;
    std::thread stats_thread;
};

#endif