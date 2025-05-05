#include <GL/glew.h>
#include <cuda_gl_interop.h> 

#include "../include/simulation.h"
#include "../include/time_spectra.h"
#include "../include/sim_time.h"
#include "../include/pack_spectrum.h"
#include "../include/compute_normals.h"

Simulation::Simulation(int simulation_res, int plane_longitude) :
    resolution(simulation_res), longitude(plane_longitude)
{
    params.scale = 1.0f;
    params.wind_speed = 125.0f;
    params.angle = 5.0f;
    params.spread_blend = 1.0f;
    params.swell = 1.0f;
    params.fetch = 100000.0f;
    params.depth = 10.0f;
    params.short_waves_fade = 0.01f;
    params.gamma = 3.3f;
    params.g = 9.81f;

    lambda = make_float2(.96f,.96f);
    normal_strength = 10.0f;

    sim_init();
}

Simulation::Simulation(int simulation_res, int plane_longitude, float scale, float wind_speed, float angle,
    float spread_blend, float swell, float fetch, float depth, float short_waves_fade, float2 lambda) :
    resolution(simulation_res), longitude(plane_longitude), lambda(lambda)
{
    params.scale = scale;
    params.wind_speed = wind_speed;
    params.angle = angle;
    params.spread_blend = spread_blend;
    params.swell = swell;
    params.fetch = fetch;
    params.depth = depth;
    params.short_waves_fade = short_waves_fade;
    params.gamma = 3.3f;
    params.g = 9.81f;

    normal_strength = 10.0f;

    sim_init();
}

Simulation::~Simulation() {
    sim_end();
}

void Simulation::sim_init() {
    cudaMalloc(&h0_k, sizeof(float2) * resolution * resolution);
    cudaMalloc(&h0t, sizeof(float2) * resolution * resolution);
    cudaMalloc(&dx_dz, sizeof(float2) * resolution * resolution);
    cudaMalloc(&dy_dxz, sizeof(float2) * resolution * resolution);
    cudaMalloc(&dyx_dyz, sizeof(float2) * resolution * resolution);
    cudaMalloc(&dxx_dzz, sizeof(float2) * resolution * resolution);

    cudaMalloc(&displacement, sizeof(float3) * resolution * resolution);
    cudaMalloc(&slope, sizeof(float2) * resolution * resolution);

    cudaMalloc(&normals, sizeof(float3) * resolution * resolution);

    cudaMalloc(&h0, sizeof(float4) * resolution * resolution);
    cudaMalloc(&waves_data, sizeof(float4) * resolution * resolution);

    glGenBuffers(1, &vbo_displacement);
    glGenBuffers(1, &vbo_slope);
    glGenBuffers(1, &vbo_init_jonswap);
    glGenBuffers(1, &vbo_time_spectrum);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_displacement);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * resolution * resolution, nullptr, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_slope);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * resolution * resolution, nullptr, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_init_jonswap);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float2) * resolution * resolution, nullptr, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, vbo_time_spectrum);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float2) * resolution * resolution, nullptr, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_displacement, vbo_displacement, cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_slope, vbo_slope, cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_init_jonswap, vbo_init_jonswap, cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_h0t, vbo_time_spectrum, cudaGraphicsMapFlagsWriteDiscard);

    launch_initial_JONSWAP(h0_k, h0, waves_data, resolution, longitude, params);
    apply_brightness(h0_k, 20.0f, resolution);
    update_vbo<float2*>(&cuda_vbo_init_jonswap, h0_k);

    cufftPlan2d(&fft, resolution, resolution, CUFFT_C2C);
}

void Simulation::sim_run() {
    update_spectra(h0, waves_data, h0t, dx_dz, dy_dxz, dyx_dyz, dxx_dzz, resolution, Time::time);
    apply_brightness(h0t, 20.0f, resolution);

    cufftExecC2C(fft, dx_dz, dx_dz, CUFFT_INVERSE);
    cufftExecC2C(fft, dy_dxz, dy_dxz, CUFFT_INVERSE);
    cufftExecC2C(fft, dyx_dyz, dyx_dyz, CUFFT_INVERSE);
    cufftExecC2C(fft, dxx_dzz, dxx_dzz, CUFFT_INVERSE);

    pack_and_assemble(dx_dz, dy_dxz, dyx_dyz, dxx_dzz, displacement, slope, resolution, lambda);
    compute_normals(slope, normals, normal_strength, resolution);

    update_vbo<float3*>(&cuda_vbo_displacement, displacement);
    update_vbo<float3*>(&cuda_vbo_slope, normals);
    update_vbo<float2*>(&cuda_vbo_h0t, h0t);
}

GLuint Simulation::get_displacement_vbo() const {
    return vbo_displacement;
}

GLuint Simulation::get_slope_vbo() const {
    return vbo_slope;
}

GLuint Simulation::get_jonswap_vbo() const {
    return vbo_init_jonswap;
}

GLuint Simulation::get_h0t_vbo() const {
    return vbo_time_spectrum;
}

int Simulation::get_resolution() const {
    return resolution;
}

int Simulation::set_resolution(int n) {
    resolution = n;
}

int Simulation::get_l() const {
    return longitude;
}

int Simulation::set_l(int l) {
    longitude = l;
}

float Simulation::get_normal_strength() const {
    return normal_strength;
}

void Simulation::set_normal_strength(float n) {
    normal_strength = n;
}

JONSWAP_params Simulation::get_params() const {
    return params;
}

void Simulation::set_params(JONSWAP_params new_params) {
    params = new_params;
    launch_initial_JONSWAP(h0_k, h0, waves_data, resolution, longitude, params);
    apply_brightness(h0_k, 20.0f, resolution);
    update_vbo<float2*>(&cuda_vbo_init_jonswap, h0_k);
}

float2 Simulation::get_lambda() const {
    return lambda;
}

void Simulation::set_lambda(float2 lm) {
    lambda = lm;
}

template <typename _Ty>
void Simulation::update_vbo(cudaGraphicsResource** cuda_res, _Ty data) {
    cudaGraphicsMapResources(1, cuda_res, 0);

    _Ty ptr_dev;
    size_t num_bytes;

    cudaGraphicsResourceGetMappedPointer((void**)&ptr_dev, &num_bytes, *cuda_res);
    cudaMemcpy(ptr_dev, data, num_bytes, cudaMemcpyDeviceToDevice);

    cudaGraphicsUnmapResources(1, cuda_res, 0);
}

void Simulation::sim_end() {
    cufftDestroy(fft);
    cudaFree(h0_k);
    cudaFree(h0t);
    cudaFree(dx_dz);
    cudaFree(dy_dxz);
    cudaFree(dyx_dyz);
    cudaFree(dxx_dzz);
    cudaFree(displacement);
    cudaFree(slope);
    cudaFree(normals);
    cudaFree(h0);
    cudaFree(waves_data);
    cudaGraphicsUnregisterResource(cuda_vbo_displacement);
    cudaGraphicsUnregisterResource(cuda_vbo_slope);
    cudaGraphicsUnregisterResource(cuda_vbo_init_jonswap);
    cudaGraphicsUnregisterResource(cuda_vbo_h0t);
    glDeleteBuffers(1, &vbo_displacement);
    glDeleteBuffers(1, &vbo_slope);
    glDeleteBuffers(1, &vbo_init_jonswap);
    glDeleteBuffers(1, &vbo_time_spectrum);
}