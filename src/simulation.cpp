#include <GL/glew.h>
#include <cuda_gl_interop.h> 

#include "../include/simulation.h"
#include "../include/time_spectra.h"
#include "../include/sim_time.h"
#include "../include/pack_spectrum.h"

Simulation::Simulation(int simulation_res, int plane_longitude) :
    resolution(simulation_res), longitude(plane_longitude)
{
    params.scale = 1.0f;
    params.wind_speed = 75.0f;
    params.angle = 1.0f;
    params.spread_blend = 0.9f;
    params.swell = 0.9f;
    params.fetch = 50000.0f;
    params.depth = 5.0f;
    params.short_waves_fade = 0.01f;
    params.gamma = 3.3f;
    params.g = 9.81f;

    lambda = make_float2(1.0f,1.0f);

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

    sim_init();
}

Simulation::~Simulation() {
    sim_end();
}

void Simulation::sim_init() {
    cudaMalloc(&h0_k, sizeof(float2) * resolution * resolution);
    cudaMalloc(&dx_dz, sizeof(float2) * resolution * resolution);
    cudaMalloc(&dy_dxz, sizeof(float2) * resolution * resolution);

    cudaMalloc(&displacement, sizeof(float3) * resolution * resolution);

    cudaMalloc(&h0, sizeof(float4) * resolution * resolution);
    cudaMalloc(&waves_data, sizeof(float4) * resolution * resolution);

    glGenBuffers(1, &vbo_displacement);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_displacement);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * resolution * resolution, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    cudaGraphicsGLRegisterBuffer(&cuda_vbo_displacement, vbo_displacement, cudaGraphicsMapFlagsWriteDiscard);

    launch_initial_JONSWAP(h0_k, h0, waves_data, resolution, longitude, params);

    cufftPlan2d(&fft, resolution, resolution, CUFFT_C2C);
}

void Simulation::sim_run() {
    update_spectra(h0, waves_data, dx_dz, dy_dxz, resolution, Time::time);

    cufftExecC2C(fft, dx_dz, dx_dz, CUFFT_INVERSE);
    cufftExecC2C(fft, dy_dxz, dy_dxz, CUFFT_INVERSE);

    pack_and_assemble(dx_dz, dy_dxz, displacement, resolution, lambda);

    update_vbo();
}

GLuint Simulation::get_displacement_vbo() {
    return vbo_displacement;
}

int Simulation::get_resolution() {
    return resolution;
}

void Simulation::update_vbo() {
    cudaGraphicsMapResources(1, &cuda_vbo_displacement, 0);

    float3* ptr_displacement;
    size_t num_bytes;

    cudaGraphicsResourceGetMappedPointer((void**)&ptr_displacement, &num_bytes, cuda_vbo_displacement);
    cudaMemcpy(ptr_displacement, displacement, num_bytes, cudaMemcpyDeviceToDevice);

    cudaGraphicsUnmapResources(1, &cuda_vbo_displacement, 0);
}

void Simulation::sim_end() {
    cufftDestroy(fft);
    cudaFree(h0_k);
    cudaFree(dx_dz);
    cudaFree(dy_dxz);
    cudaFree(displacement);
    cudaFree(h0);
    cudaFree(waves_data);
    cudaGraphicsUnregisterResource(cuda_vbo_displacement);
    glDeleteBuffers(1, &vbo_displacement);
}