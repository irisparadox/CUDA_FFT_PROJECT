#include "../include/render.h"
#include "../external/imgui/imgui.h"
#include "../external/imgui/backends/imgui_impl_glfw.h"
#include "../external/imgui/backends/imgui_impl_opengl3.h"
#include "../include/spectra_params.h"


Render::Render(GLFWwindow* window) : ifft_heightmap(0), tex_width(0), tex_height(0), sim(1024, 100) {
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
    init_texture(&ifft_heightmap, sim.get_resolution(), sim.get_resolution(), GL_RGB32F, GL_RGB, GL_LINEAR);
    init_texture(&ifft_slope, sim.get_resolution(), sim.get_resolution(), GL_RGB32F, GL_RGB, GL_LINEAR);
    init_texture(&initial_jonswap, sim.get_resolution(), sim.get_resolution(), GL_RG32F, GL_RG, GL_NEAREST);
    update_texture(sim.get_jonswap_vbo(), &initial_jonswap, sim.get_resolution(),
        sim.get_resolution(), GL_RG32F, GL_RG, GL_NEAREST);
    init_texture(&h0t, sim.get_resolution(), sim.get_resolution(), GL_RG32F, GL_RG, GL_NEAREST);
}

void Render::update() {
    sim.sim_run();
    update_texture(sim.get_displacement_vbo(), &ifft_heightmap,
        sim.get_resolution(), sim.get_resolution(), GL_RGB32F, GL_RGB, GL_LINEAR);
    update_texture(sim.get_slope_vbo(), &ifft_slope, sim.get_resolution(),
        sim.get_resolution(), GL_RGB32F, GL_RGB, GL_LINEAR);
    render_gui_window();
    update_texture(sim.get_h0t_vbo(), &h0t, sim.get_resolution(),
        sim.get_resolution(), GL_RG32F, GL_RG, GL_NEAREST);
    render_gui_window();
}

void Render::shutdown() {
    if(ifft_heightmap)
        glDeleteTextures(1, &ifft_heightmap);
    if(ifft_slope)
        glDeleteTextures(1, &ifft_slope);
    if(initial_jonswap)
        glDeleteTextures(1, &initial_jonswap);
    if(h0t)
        glDeleteTextures(1, &h0t);

    ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

void Render::init_texture(GLuint* textureID, int width, int height, GLint type, GLint channels, GLint filter) {
    tex_width = width;
    tex_height = height;

    glGenTextures(1, textureID);
    glBindTexture(GL_TEXTURE_2D, *textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, type, width, height, 0, 
                channels, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

void Render::update_texture(GLuint vbo, GLuint* textureID, int width, int height, GLint type, GLint channels, GLint filter) {
    if (width != tex_width || height != tex_height) {
        init_texture(textureID, width, height, type, channels, filter);
    }
    
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    float* data = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
    
    if (data) {
        glBindTexture(GL_TEXTURE_2D, *textureID);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, 
                       channels, GL_FLOAT, data);
        glUnmapBuffer(GL_ARRAY_BUFFER);
    }
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Render::render_gui_window() {
    ImGui_ImplGlfw_NewFrame();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui::NewFrame();

    static bool visualizer_bool = true;

    const float target_image_width = 400.0f;
    const float target_image_height = 400.0f;
    
    float image_aspect_ratio = (float)tex_width / (float)tex_height;
    
    ImVec2 image_size;
    if (image_aspect_ratio > 1.0f) {
        image_size.x = target_image_width;
        image_size.y = target_image_width / image_aspect_ratio;
    } else {
        image_size.y = target_image_height;
        image_size.x = target_image_height * image_aspect_ratio;
    }

    const float padding = 10.0f;
    float title_bar_height = ImGui::GetFrameHeight();

    ImVec2 window_size(
        (image_size.x * 2) + (padding * 3),
        image_size.y + (padding * 2) + title_bar_height
    );

    ImGui::SetNextWindowSize(window_size, ImGuiCond_Once);
    ImGui::SetNextWindowPos(ImVec2(440, 10), ImGuiCond_FirstUseEver);

    ImGui::Begin("Heightmap & Normalmap Visualizer", &visualizer_bool,
        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
    {
        ImGui::SetCursorPos(ImVec2(padding, title_bar_height + padding));
        ImTextureID tex_id = (ImTextureID)(intptr_t)ifft_heightmap;
        ImGui::Image(tex_id, image_size);

        ImGui::SetCursorPos(ImVec2(image_size.x + padding * 2, title_bar_height + padding));
        ImTextureID tex_id2 = (ImTextureID)(intptr_t)ifft_slope;
        ImGui::Image(tex_id2, image_size);
    }
    ImGui::End();

    static bool spectrum_bool = true;

    ImGui::SetNextWindowSize(window_size, ImGuiCond_Once);
    ImGui::SetNextWindowPos(ImVec2(440,470), ImGuiCond_FirstUseEver);

    ImGui::Begin("Spectrum Visualizer", &spectrum_bool, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
    {
        ImGui::SetCursorPos(ImVec2(padding, title_bar_height + padding));
        ImTextureID tex_id = (ImTextureID)(intptr_t)initial_jonswap;
        ImGui::Image(tex_id, image_size);

        ImGui::SetCursorPos(ImVec2(image_size.x + padding * 2, title_bar_height + padding));
        ImTextureID tex_id2 = (ImTextureID)(intptr_t)h0t;
        ImGui::Image(tex_id2, image_size);
    }
    ImGui::End();

    static bool parameters_bool = true;

    ImGui::SetNextWindowSize(ImVec2(250, 300), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(ImVec2(20,10), ImGuiCond_Once);

    ImGui::Begin("Parameters", &parameters_bool, ImGuiWindowFlags_None);
    {   
        static JONSWAP_params params = sim.get_params();
        //static int n = sim.get_resolution();
        //static int l = sim.get_l();

        //ImGui::InputInt("Resolution", &n);
        //ImGui::InputInt("Longitude", &l);

        //ImGui::Separator();

        ImGui::SliderFloat("Scale", &params.scale, 0.1f, 5.0f, "%.2f");
        ImGui::InputFloat("Wind Speed", &params.wind_speed, 5.0f, 15.0f, "%.1f");
        ImGui::SliderAngle("Angle", &params.angle, 0.0f, 360.0f, "%.0f deg");
        ImGui::SliderFloat("Spread Blend", &params.spread_blend, 0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Swell", &params.swell, 0.0f, 1.5f, "%.2f");
        ImGui::InputFloat("Fetch", &params.fetch, 50.0f, 100.0f, "%.1f");
        ImGui::InputFloat("Depth", &params.depth, 0.1f, 0.5f, "%.2f");
        ImGui::SliderFloat("Short Waves Fade", &params.short_waves_fade, 0.005f, 1.0f, "%.3f");
        ImGui::InputFloat("Gamma", &params.gamma, 0.1f, 0.3f, "%.1f");
        ImGui::InputFloat("Gravity", &params.g, 0.01f, 0.1f, "%.2f");

        ImGui::Dummy(ImVec2(0,20));

        if(ImGui::Button("Apply")) {
            //sim.set_resolution(n);
            //sim.set_l(l);
            sim.set_params(params);
            update_texture(sim.get_jonswap_vbo(), &initial_jonswap, sim.get_resolution(),
                sim.get_resolution(), GL_RG32F, GL_RG, GL_NEAREST);
        }

        ImGui::Separator();
        ImGui::Dummy(ImVec2(0,5));

        static float lambdaX = sim.get_lambda().x;
        static float lambdaY = sim.get_lambda().y;
        static float lambda[2] = {lambdaX, lambdaY};
        if(ImGui::SliderFloat2("Choppiness", lambda, 0.0f, 1.0f, "%.2f")) {
            sim.set_lambda(make_float2(lambda[0],lambda[1]));
        }
        
    }
    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}