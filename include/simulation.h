#ifndef SIMULATION_H
#define SIMULATION_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include <GLFW/glfw3.h> 

#include "../include/spectra.h"

class Simulation {
public:
    Simulation(int simulation_res, int plane_longitude);
    Simulation(int simulation_res, int plane_longitude, float scale, float wind_speed, float angle,
        float spread_blend, float swell, float fetch, float depth, float short_waves_fade, float2 lambda);
    ~Simulation();

public:
    void sim_run();

private:
    void sim_init();
    void sim_end();
    void update_vbo();

private:
    float2* h0_k;
    float4* h0;
    float4* waves_data;
    
    float2* dx_dz;
    float2* dy_dxz;

    float3* displacement;

    GLuint vbo_displacement;
    cudaGraphicsResource* cuda_vbo_displacement;
    
    int resolution;
    int longitude;

    cufftHandle fft;
    
    JONSWAP_params params;
    float2 lambda;
};

#endif