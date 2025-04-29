#include "../include/simulation.h"
#include "../include/time_spectra.h"
#include "../include/sim_time.h"

void sim_init(JONSWAP_params init_params) {
    sim.params = init_params;
    cudaMalloc(&sim.h0_k, sizeof(float2) * sim.resolution * sim.resolution);
    cudaMalloc(&sim.dx_dz, sizeof(float2) * sim.resolution * sim.resolution);
    cudaMalloc(&sim.dy_dxz, sizeof(float2) * sim.resolution * sim.resolution);

    cudaMalloc(&sim.h0, sizeof(float4) * sim.resolution * sim.resolution);
    cudaMalloc(&sim.waves_data, sizeof(float4) * sim.resolution * sim.resolution);

    launch_initial_JONSWAP(sim.h0_k, sim.h0, sim.waves_data, sim.resolution, sim.longitude, sim.params);
}

void sim_run() {
    
}

void sim_end() {
    cudaFree(sim.h0_k);
    cudaFree(sim.dx_dz);
    cudaFree(sim.dy_dxz);
    cudaFree(sim.h0);
    cudaFree(sim.waves_data);
}