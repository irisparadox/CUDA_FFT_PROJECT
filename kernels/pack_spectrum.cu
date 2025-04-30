#include "../include/pack_spectrum.h"

__device__ float2 permute(float2 data, int2 id) {
    float sign = (1.0f - 2.0f * ((id.x + id.y) % 2));
    return make_float2(data.x * sign, data.y * sign);
}

__global__ void assemble_maps(float2* dx_dz, float2* dy_dxz, float3* displacement, int N, float2 lambda) {
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < N && j < N) {
        int idx = i * N + j;
        float2 htilde_displacement_X = permute(dx_dz[idx], make_int2(j, i));
        float2 htilde_displacement_Z = permute(dy_dxz[idx], make_int2(j, i));

        displacement[idx] = make_float3(lambda.x * htilde_displacement_X.x,
            htilde_displacement_Z.x, lambda.y * htilde_displacement_X.y);
    }
}

void pack_and_assemble(float2* dx_dz, float2* dy_dxz, float3* displacement, int N, float2 lambda) {
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    assemble_maps<<<gridDim, blockDim>>>(dx_dz, dy_dxz, displacement, N, lambda);
    cudaDeviceSynchronize();
}