#include "../include/time_spectra.h"

__device__ inline float2 complex_mult(const float2 a, const float2 b) {
    return make_float2(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

__global__ void calculate_amplitudes(float4* h0, float4* waves_data, float2* h0t,
    float2* dx_dz, float2* dy_dxz, float2* dyx_dyz, float2* dxx_dzz, int N, float time) {
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < N && j < N) {
        int idx = i * N + j;
        float4 wave = waves_data[idx];
        float phase = wave.w * time;
        float2 exponent = make_float2(cosf(phase), sinf(phase));
        float2 h0iw = complex_mult(make_float2(h0[idx].x, h0[idx].y), exponent);
        float2 h0minusiw = complex_mult(make_float2(h0[idx].z, h0[idx].w), make_float2(exponent.x, -exponent.y));
        float2 h = make_float2(h0iw.x + h0minusiw.x, h0iw.y + h0minusiw.y);

        h0t[idx] = h;
        
        float2 ih = make_float2(-h.y, h.x);

        float2 dis_x = make_float2(ih.x * wave.x * wave.y, ih.y * wave.x * wave.y);
        float2 dis_y = h;
        float2 dis_z = make_float2(ih.x * wave.z * wave.y, ih.y * wave.z * wave.y);

        float2 dis_x_dx = make_float2(-h.x * wave.x * wave.x * wave.y, -h.y * wave.x * wave.x * wave.y);
        float2 dis_y_dx = make_float2(ih.x * wave.x, ih.y * wave.x);
        float2 dis_z_dx = make_float2(-h.x * wave.x * wave.z * wave.y, -h.y * wave.x * wave.z * wave.y);

        float2 dis_y_dz = make_float2(ih.x * wave.z, ih.y * wave.z);
        float2 dis_z_dz = make_float2(-h.x * wave.z * wave.z * wave.y, -h.y * wave.z * wave.z * wave.y);

        dx_dz[idx] = make_float2(dis_x.x - dis_z.y, dis_x.y + dis_z.x);
        dy_dxz[idx] = make_float2(dis_y.x - dis_z_dx.y, dis_y.y + dis_z_dx.x);
        dyx_dyz[idx] = make_float2(dis_y_dx.x - dis_y_dz.y, dis_y_dx.y + dis_y_dz.x);
        dxx_dzz[idx] = make_float2(dis_x_dx.x - dis_z_dz.y, dis_x_dx.y + dis_z_dz.x);
    }
}

void update_spectra(float4* h0, float4* waves_data, float2* h0t,
    float2* dx_dz, float2* dy_dxz, float2* dyx_dyz, float2* dxx_dzz, int N, float time) {
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    calculate_amplitudes<<<gridDim, blockDim>>>(h0, waves_data, h0t, dx_dz, dy_dxz, dyx_dyz, dxx_dzz, N, time);
    cudaDeviceSynchronize();
}