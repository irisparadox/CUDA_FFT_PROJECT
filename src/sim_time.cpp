#include "../include/sim_time.h"
#include <GLFW/glfw3.h>

float Time::time = 0.0f;
float Time::delta_time = 0.0f;
float Time::last_frame = 0.0f;

void Time::init() {
    delta_time = 0.0f;
    time = 0.0f;
    last_frame = 0.0f;
}

void Time::update() {
    float current_frame = static_cast<float>(glfwGetTime());
    delta_time = current_frame - last_frame;
    last_frame = current_frame;

    time += delta_time;
}

void Time::end() {
    delta_time = 0;
    last_frame = 0;
    time = 0;
}