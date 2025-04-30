#include "../include/window.h"
#include <iostream>
#include <stdexcept>

Window::Window(int width, int height, const char* title)
    : width(width), height(height), handle(nullptr) {
    init_glfw();
    create_window(width, height, title);

    GLenum err = glewInit();
    if(err != GLEW_OK) {
        throw std::runtime_error("Failed to initialize GLEW");
    }

    setup_callbacks();
}

Window::~Window() {
    if(handle) {
        glfwDestroyWindow(handle);
    }
    glfwTerminate();
}

void Window::init_glfw() {
    if(!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    // OpenGL Core Version 4.6
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);

    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
}

void Window::create_window(int width, int height, const char* title) {
    handle = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if(!handle) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }

    glfwMakeContextCurrent(handle);
    glfwSetWindowUserPointer(handle, this);
}

void Window::setup_callbacks() {
    glfwSetFramebufferSizeCallback(handle, framebuffer_size_callback);
}

void Window::framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    auto* win = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (win) {
        glViewport(0, 0, width, height);
        win->width = width;
        win->height = height;
        if(win->framebuffer_resize_callback) {
            win->framebuffer_resize_callback(width, height);
        }
    }
}

bool Window::should_close() const {
    return glfwWindowShouldClose(handle);  
}

void Window::poll_events() const {
    glfwPollEvents();
}

void Window::swap_buffers() const {
    glfwSwapBuffers(handle);
}

void Window::close() {
    glfwSetWindowShouldClose(handle, true);
}

int Window::get_width() const {
    return width;
}

int Window::get_height() const {
    return height;
}

GLFWwindow* Window::get_native_handle() const {
    return handle;
}

void Window::set_framebuffer_resize_callback(std::function<void(int,int)> callback) {
    framebuffer_resize_callback = std::move(callback);
}