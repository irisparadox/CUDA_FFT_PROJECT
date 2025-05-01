#include "../include/compute_normals.h"

__device__ float saturate(float x) {
    return fminf(fmaxf(x, 0.0f), 1.0f);
}

__device__ float3 lerp(float3 a, float3 b, float t) {
    return make_float3(
        a.x + t * (b.x - a.x),
        a.y + t * (b.y - a.y),
        a.z + t * (b.z - a.z)
    );
}

__device__ float3 normalize3(float3 v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len > 0.0f)
        return make_float3(v.x / len, v.y / len, v.z / len);
    return make_float3(0.0f, 1.0f, 0.0f); // fallback
}

__global__ void compute_normal_from_slope(float2* slope, float3* meso_normals, int N, float depth, float attenuation) {
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < N && j < N) {
        int idx = i * N + j;
        float3 macroNormal = make_float3(0.0f, 1.0f, 0.0f);
        float3 mesoNormal = normalize3(make_float3(-slope[idx].x, .2f, -slope[idx].y));

        float weight = powf(saturate(depth), attenuation);
        meso_normals[idx] = normalize3(lerp(macroNormal, mesoNormal, weight));
    }
}

void compute_meso_normals(float2* slope, float3* meso_normals, int N) {
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    compute_normal_from_slope<<<gridDim, blockDim>>>(slope, meso_normals, N, 1.0f, 10.0f);
    cudaDeviceSynchronize();
}