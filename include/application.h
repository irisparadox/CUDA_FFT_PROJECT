#ifndef APPLICATION_H
#define APPLICATION_H

#include "window.h"
#include "render.h"

/*
* Singleton Application Class
*/
class Application {
public:
    /*
    * Returns the instance of the Application
    */
    static Application& get();

    Application(int width, int height, const char* title);
    ~Application();

    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;

    void run();

protected:
    void on_init();
    void on_update();
    void on_render();
    void on_shutdown();

private:
    static Application* instance;
    Window* window;
    bool running;

    Render* render;
};

#endif