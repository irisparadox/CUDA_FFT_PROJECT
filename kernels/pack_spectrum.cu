#include "../include/pack_spectrum.h"

__device__ float2 permute(float2 data, int2 id) {
    float sign = (1.0f - 2.0f * ((id.x + id.y) % 2));
    return make_float2(data.x * sign, data.y * sign);
}

__global__ void assemble_maps(float2* dx_dz, float2* dy_dxz, float2* dyx_dyz, float2* dxx_dzz,
    float3* displacement, float2* slope, int N, float2 lambda) {
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < N && j < N) {
        int idx = i * N + j;
        float2 htilde_displacement_X = permute(dx_dz[idx], make_int2(j, i));
        float2 htilde_displacement_Z = permute(dy_dxz[idx], make_int2(j, i));
        float2 htilde_slope_X = permute(dyx_dyz[idx], make_int2(j, i));
        float2 htilde_slope_Z = permute(dxx_dzz[idx], make_int2(j, i));

        float slopeX = htilde_slope_X.x / (1 + fabsf(htilde_slope_Z.x * lambda.x));
        float slopeY = htilde_slope_X.y / (1 + fabsf(htilde_slope_Z.y * lambda.y));

        displacement[idx] = make_float3(lambda.x * htilde_displacement_X.x,
            htilde_displacement_Z.x, lambda.y * htilde_displacement_X.y);
        slope[idx] = make_float2(slopeX, slopeY);       
    }
}

__global__ void apply_brightness_kernel(float2* h0_k, float brightness, int N) {
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < N && j < N) {
        int idx = i * N + j;
        h0_k[idx].x *= brightness;
        h0_k[idx].y *= brightness;

        h0_k[idx].x = fmaxf(0.0f, fminf(255.0f, h0_k[idx].x * 255.0f));
        h0_k[idx].y = fmaxf(0.0f, fminf(255.0f, h0_k[idx].y * 255.0f));
    }
}

void pack_and_assemble(float2* dx_dz, float2* dy_dxz, float2* dyx_dyz, float2* dxx_dzz,
    float3* displacement, float2* slope, int N, float2 lambda) {
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    assemble_maps<<<gridDim, blockDim>>>(dx_dz, dy_dxz, dyx_dyz, dxx_dzz, displacement, slope, N, lambda);
    cudaDeviceSynchronize();
}

void apply_brightness(float2* h0_k, float brightness, int N) {
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    apply_brightness_kernel<<<gridDim, blockDim>>>(h0_k, brightness, N);
    cudaDeviceSynchronize();
}