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
    void init_texture(GLuint* textureID, int width, int height, GLint type, GLint channels, GLint filter);
    void update_texture(GLuint vbo, GLuint* textureID, int width, int height, GLint type, GLint channels, GLint filter);
    void render_gui_window();

    Simulation sim;
    GLuint ifft_heightmap;
    GLuint ifft_slope;
    GLuint initial_jonswap;
    GLuint h0t;
    int tex_width, tex_height;
};

#endif