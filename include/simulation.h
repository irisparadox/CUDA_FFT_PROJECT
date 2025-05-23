#ifndef SIMULATION_H
#define SIMULATION_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include <GLFW/glfw3.h> 

#include "spectra_params.h"
#include "spectra.h"

class Simulation {
public:
    Simulation(int simulation_res, int plane_longitude);
    Simulation(int simulation_res, int plane_longitude, float scale, float wind_speed, float angle,
        float spread_blend, float swell, float fetch, float depth, float short_waves_fade, float2 lambda);
    ~Simulation();

public:
    void sim_run();
    GLuint get_displacement_vbo() const;
    GLuint get_slope_vbo() const;
    GLuint get_jonswap_vbo() const;
    GLuint get_h0t_vbo() const;
    int get_resolution() const;
    int set_resolution(int n);
    int get_l() const;
    int set_l(int l);
    float get_normal_strength() const;
    void set_normal_strength(float n);
    JONSWAP_params get_params() const;
    void set_params(JONSWAP_params new_params);
    float2 get_lambda() const;
    void set_lambda(float2 lm);

private:
    void sim_init();
    void sim_end();
    template <typename _Ty>
    void update_vbo(cudaGraphicsResource** cuda_res, _Ty data);

private:
    float2* h0_k;
    float2* h0t;
    float4* h0;
    float4* waves_data;
    
    float2* dx_dz;
    float2* dy_dxz;
    float2* dyx_dyz;
    float2* dxx_dzz;

    float3* displacement;
    float2* slope;

    float3* normals;

    GLuint vbo_displacement;
    GLuint vbo_slope;
    GLuint vbo_init_jonswap;
    GLuint vbo_time_spectrum;
    cudaGraphicsResource* cuda_vbo_displacement;
    cudaGraphicsResource* cuda_vbo_slope;
    cudaGraphicsResource* cuda_vbo_init_jonswap;
    cudaGraphicsResource* cuda_vbo_h0t;
    
    int resolution;
    int longitude;

    cufftHandle fft;
    
    JONSWAP_params params;
    float2 lambda;
    float normal_strength;
};

#endif