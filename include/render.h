#ifndef RENDER_H
#define RENDER_H

#include <GL/glew.h>
#include "simulation.h"

class Render {
public:
    Render(GLFWwindow* window);
    ~Render();

    void init();
    void update();
    void shutdown();

private:
    void init_imgui_context(GLFWwindow* window);
    void init_texture(int width, int height);
    void update_texture(GLuint vbo, int width, int height);
    void render_gui_window();

    Simulation sim;
    GLuint ifft_heightmap;
    int tex_width, tex_height;
};

#endif