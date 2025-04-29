#ifndef SPECTRA_H
#define SPECTRA_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

typedef struct JONSWAP_params {
    float scale;
    float wind_speed;
    float angle;
    float spread_blend;
    float swell;
    float fetch;
    float depth;
    float short_waves_fade;
    float gamma;
    float peak_omega;
    float alpha;
    float g;
};

void launch_initial_JONSWAP(float2* h0_k, float4* h0, float4* waves_data, int N, int L, JONSWAP_params params);

#endif // SPECTRA_H