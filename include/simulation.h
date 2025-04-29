#ifndef SIMULATION_H
#define SIMULATION_H

#include <cuda.h>
#include <cuda_runtime.h>

#include "../include/spectra.h"

typedef struct simulation {
    float2* h0_k;
    float4* h0;
    float4* waves_data;
    
    float2* dx_dz;
    float2* dy_dxz;
    
    int resolution;
    int longitude;
    
    JONSWAP_params params;
};

simulation sim;

void sim_init(JONSWAP_params init_params);
void sim_run();
void sim_end();

#endif