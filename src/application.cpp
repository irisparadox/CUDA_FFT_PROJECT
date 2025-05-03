#include "../include/application.h"
#include "../include/sim_time.h"

#include <iostream>
#include <stdexcept>

Application* Application::instance = nullptr;

Application::Application(int width, int height, const char* title) : running(false) {
    if(instance != nullptr)
        throw std::runtime_error("Application already exists!");

    instance = this;
    window = new Window(width, height, title);
    render = new Render(window->get_native_handle());
}

Application::~Application() {
    delete window;
    delete render;
    instance = nullptr;
}

Application& Application::get() {
    if(!instance)
        throw std::runtime_error("Application has not ben created yet.");

    return *instance;
}

void Application::run() {
    running = true;
    on_init();

    while(running && !window->should_close()) {
        window->poll_events();
        on_update();
        on_render();
        window->swap_buffers();
    }

    running = false;
    on_shutdown();
}

void Application::on_init() {
    glEnable(GL_DEPTH_TEST);
    Time::init();
    render->init();
}

void Application::on_update() {
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Time::update();
}

void Application::on_render() {
    render->update();
}

void Application::on_shutdown() {
    Time::end();
    render->shutdown();
}