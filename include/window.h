#ifndef WINDOW_H
#define WINDOW_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <functional>

class Window {
public:
    Window(int width, int height, const char* title);
    ~Window();
    bool should_close() const;
    void swap_buffers() const;
    void poll_events() const;
    void close();

public:
    GLFWwindow* get_native_handle() const;
    int get_width() const;
    int get_height() const;
    void set_framebuffer_resize_callback(std::function<void(int,int)> callback);

private:
    void init_glfw();
    void create_window(int width, int height, const char* title);
    void setup_callbacks();

    static void framebuffer_size_callback(GLFWwindow* window, int width, int height);

private:
    GLFWwindow* handle;
    int width, height;

    std::function<void(int,int)> framebuffer_resize_callback;
};

#endif