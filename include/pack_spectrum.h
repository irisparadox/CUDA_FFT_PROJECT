#ifndef PACK_SPECTRUM_H
#define PACK_SPECTRUM_H

#include <cuda.h>
#include <cuda_runtime.h>

void pack_and_assemble(float2* dx_dz, float2* dy_dxz, float3* displacement, int N, float2 lambda);

#endif