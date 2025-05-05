#ifndef PACK_SPECTRUM_H
#define PACK_SPECTRUM_H

#include <cuda.h>
#include <cuda_runtime.h>

void pack_and_assemble(float2* dx_dz, float2* dy_dxz, float2* dyx_dyz, float2* dxx_dzz,
    float3* displacement, float2* slope, int N, float2 lambda);

void apply_brightness(float2* h0_k, float brightness, int N);
#endif