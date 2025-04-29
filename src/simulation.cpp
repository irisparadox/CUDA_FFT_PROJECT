#include <GL/glew.h>
#include <cuda_gl_interop.h> 

#include "../include/simulation.h"
#include "../include/time_spectra.h"
#include "../include/sim_time.h"

Simulation::Simulation(int simulation_res, int plane_longitude) :
    resolution(simulation_res), longitude(plane_longitude)
{
    params.scale = 1.0f;
    params.wind_speed = 90.0f;
    params.angle = 1.0f;
    params.spread_blend = 0.9f;
    params.swell = 1.0f;
    params.fetch = 50000.0f;
    params.depth = 0.5f;
    params.short_waves_fade = 0.01f;
    params.gamma = 3.3f;
    params.g = 9.81f;

    sim_init();
}

Simulation::Simulation(int simulation_res, int plane_longitude, float scale, float wind_speed, float angle,
    float spread_blend, float swell, float fetch, float depth, float short_waves_fade) :
    resolution(simulation_res), longitude(plane_longitude)
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

    sim_init();
}

Simulation::~Simulation() {
    sim_end();
}

float2* Simulation::get_dx_dz() {
    return dx_dz;
}

float2* Simulation::get_dy_dxz() {
    return dy_dxz;
}

void Simulation::sim_init() {
    cudaMalloc(&h0_k, sizeof(float2) * resolution * resolution);
    cudaMalloc(&dx_dz, sizeof(float2) * resolution * resolution);
    cudaMalloc(&dy_dxz, sizeof(float2) * resolution * resolution);

    cudaMalloc(&h0, sizeof(float4) * resolution * resolution);
    cudaMalloc(&waves_data, sizeof(float4) * resolution * resolution);

    glGenBuffers(1, &vbo_dx_dz);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_dx_dz);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float2) * resolution * resolution, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glGenBuffers(1, &vbo_dy_dxz);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_dy_dxz);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float2) * resolution * resolution, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(&cuda_vbo_dx_dz_resource, vbo_dx_dz, cudaGraphicsMapFlagsWriteDiscard);
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_dy_dxz_resource, vbo_dy_dxz, cudaGraphicsMapFlagsWriteDiscard);

    launch_initial_JONSWAP(h0_k, h0, waves_data, resolution, longitude, params);

    cufftPlan2d(&fft, resolution, resolution, CUFFT_C2C);
}

void Simulation::sim_run() {
    update_spectra(h0, waves_data, dx_dz, dy_dxz, resolution, Time::time);

    cufftExecC2C(fft, dx_dz, dx_dz, CUFFT_INVERSE);
    cufftExecC2C(fft, dy_dxz, dy_dxz, CUFFT_INVERSE);

    update_vbo();
}

void Simulation::update_vbo() {
    cudaGraphicsMapResources(1, &cuda_vbo_dx_dz_resource, 0);
    cudaGraphicsMapResources(1, &cuda_vbo_dy_dxz_resource, 0);

    float2* ptr_dx_dz;
    float2* ptr_dy_dxz;
    size_t num_bytes;

    cudaGraphicsResourceGetMappedPointer((void**)&ptr_dx_dz, &num_bytes, cuda_vbo_dx_dz_resource);
    cudaMemcpy(ptr_dx_dz, dx_dz, num_bytes, cudaMemcpyDeviceToDevice);

    cudaGraphicsResourceGetMappedPointer((void**)&ptr_dy_dxz, &num_bytes, cuda_vbo_dy_dxz_resource);
    cudaMemcpy(ptr_dy_dxz, dy_dxz, num_bytes, cudaMemcpyDeviceToDevice);

    cudaGraphicsUnmapResources(1, &cuda_vbo_dx_dz_resource, 0);
    cudaGraphicsUnmapResources(1, &cuda_vbo_dy_dxz_resource, 0);
}

void Simulation::sim_end() {
    cufftDestroy(fft);
    cudaFree(h0_k);
    cudaFree(dx_dz);
    cudaFree(dy_dxz);
    cudaFree(h0);
    cudaFree(waves_data);
}