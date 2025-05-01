#include "../include/render.h"
#include "../external/imgui/imgui.h"
#include "../external/imgui/backends/imgui_impl_glfw.h"
#include "../external/imgui/backends/imgui_impl_opengl3.h"


Render::Render(GLFWwindow* window) : ifft_heightmap(0), tex_width(0), tex_height(0), sim(256, 100) {
    init_imgui_context(window);
}

Render::~Render() {
}

void Render::init_imgui_context(GLFWwindow* window) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 460");
    ImGui::StyleColorsDark();
}

void Render::init() {
    init_texture(&ifft_heightmap, sim.get_resolution(), sim.get_resolution());
    init_texture(&ifft_slope, sim.get_resolution(), sim.get_resolution());
}

void Render::update() {
    sim.sim_run();
    update_texture(sim.get_displacement_vbo(), &ifft_heightmap, sim.get_resolution(), sim.get_resolution());
    update_texture(sim.get_slope_vbo(), &ifft_slope, sim.get_resolution(), sim.get_resolution());
    render_gui_window();
}

void Render::shutdown() {
    if(ifft_heightmap)
        glDeleteTextures(1, &ifft_heightmap);

    ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

void Render::init_texture(GLuint* textureID, int width, int height) {
    tex_width = width;
    tex_height = height;

    glGenTextures(1, textureID);
    glBindTexture(GL_TEXTURE_2D, *textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, 
                GL_RGB, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

void Render::update_texture(GLuint vbo, GLuint* textureID, int width, int height) {
    if (width != tex_width || height != tex_height) {
        init_texture(textureID, width, height);
    }
    
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    float* data = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
    
    if (data) {
        glBindTexture(GL_TEXTURE_2D, *textureID);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, 
                       GL_RGB, GL_FLOAT, data);
        glUnmapBuffer(GL_ARRAY_BUFFER);
    }
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Render::render_gui_window() {
    ImGui_ImplGlfw_NewFrame();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui::NewFrame();

    static bool window_bool = true;
    static bool init_window = true;
    if (init_window) {
        ImGui::SetNextWindowSize(ImVec2(tex_width * 3.5f, tex_height * 1.85f), ImGuiCond_Always);
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
        init_window = false;
    }

    ImGui::Begin("Heightmap Visualization", &window_bool, ImGuiWindowFlags_NoResize);
    {
        ImVec2 image_size(tex_width * 1.7f, tex_height * 1.7f);
        ImTextureID tex_id = (ImTextureID)(intptr_t)ifft_heightmap;
        ImGui::Image(tex_id, image_size);

        ImGui::SameLine();
        ImTextureID tex_id2 = (ImTextureID)(intptr_t)ifft_slope;
        ImGui::Image(tex_id2, image_size);
    }
    ImGui::End();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}