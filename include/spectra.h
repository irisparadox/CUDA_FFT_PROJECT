#ifndef SPECTRA_H
#define SPECTRA_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

typedef struct JONSWAP_params {
    float wind_speed;
    float angle;
    float spread_blend;
    float swell;
    float fetch;
    float gamma;
    float g;
};

void launch_initial_JONSWAP(float2* h0, float2* h0_x, float2* h0_z, int N, int L, JONSWAP_params params);

#endif // SPECTRA_H