#include "../include/spectra.h"

__global__ void setup_curand(curandState* states, int N, unsigned long seed) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N && j < N) {
        int idx = i * N + j;
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__device__ void gaussian_random(curandState* state, float* outX, float* outY) {
    float u1 = curand_uniform(state);
    float u2 = curand_uniform(state);

    float r = sqrtf(-2.0f * logf(u1));
    float theta = 2.0f * M_PI * u2;

    *outX = r * cosf(theta);
    *outY = r * cosf(theta);
}

__device__ inline float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

__device__ float NormalisationFactor(float s) {
    float s2 = s * s;
    float s3 = s2 * s;
    float s4 = s3 * s;
    if (s < 5.0f)
        return -0.000564f * s4 + 0.00776f * s3 - 0.044f * s2 + 0.192f * s + 0.163f;
    else
        return -4.80e-08f * s4 + 1.07e-05f * s3 - 9.53e-04f * s2 + 5.90e-02f * s + 3.93e-01f;
}

__device__ float Cosine2s(float theta, float s) {
    return NormalisationFactor(s) * powf(fabsf(cosf(0.5f * theta)), 2.0f * s);
}

__global__ void generate_initial_JONSWAP(float2* h0, float2* h0_x, float2* h0_z,
    curandState* states, int N, int L, JONSWAP_params params) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N && j < N) {
        int idx = i * N + j;
        curandState* localState = &states[idx];

        float kx = (j - N/2) * (2.0f * M_PI / L);
        float ky = (i - N/2) * (2.0f * M_PI / L);

        float k = sqrtf(kx*kx + ky*ky);
        if(k < 0.0001f) k = 0.0001f;

        float wave_angle = atan2f(ky, kx);

        // random gaussian indices for real and imaginary
        float2 r;
        gaussian_random(localState, &r.x, &r.y);

        float omega_p = 22.0f * powf(params.wind_speed * params.fetch / (params.g * params.g), -0.33f);
    
        float alpha = 0.076f * powf(params.wind_speed * params.wind_speed / (params.fetch * params.g), 0.22f);
        
        float omega = sqrtf(k * params.g);
        float PM = (alpha * params.g * params.g) / powf(omega, 5.0f) * 
                expf(-1.25f * powf(omega_p / omega, 4.0f));
        
        float sigma;
        if (omega <= omega_p) {
            sigma = 0.07f;
        } else {
            sigma = 0.09f;
        }
        
        float r_peak = expf(-0.5f * powf((omega - omega_p) / (sigma * omega_p), 2.0f));

        float spread_power = omega > omega_p ?
            .77f * powf(fabsf(omega / omega_p),-2.5f) : .97f * powf(fabsf(omega / omega_p),5.0f);
        
        float direction_spectrum = spread_power +
            16 * tanhf(min(omega / omega_p, 20.0f)) * params.swell * params.swell;
        
        float directional_spread = lerp(2.0f / M_PI * cosf(wave_angle) * cosf(wave_angle),
            Cosine2s(wave_angle - params.angle, spread_power), params.spread_blend);
        
        float JONSWAP = PM * powf(params.gamma, r_peak) * directional_spread;
        float sqrtPh = sqrtf(JONSWAP * 0.5f);
        
        // complex fourier amplitudes
        h0[idx].x = r.x * sqrtPh;
        h0[idx].y = r.y * sqrtPh;

        if(k > 0.0001f) {
            // horizontal step spectra
            float kx_over_k = kx / k;
            float ky_over_k = ky / k;

            // ~ik/k * h0
            h0_x[idx].x = -ky_over_k * h0[idx].y;
            h0_x[idx].y =  ky_over_k * h0[idx].x;

            h0_z[idx].x =  kx_over_k * h0[idx].y;
            h0_z[idx].y = -kx_over_k * h0[idx].x;
        } else {
            h0_x[idx] = make_float2(0.0f, 0.0f);
            h0_z[idx] = make_float2(0.0f, 0.0f);
        }
    }
}

void launch_initial_JONSWAP(float2* h0, float2* h0_x, float2* h0_z, int N, int L, JONSWAP_params params) {
    float2* d_h0,* d_h0_x,* d_h0_z;
    curandState* d_states;
    cudaMalloc(&d_h0, N * N * sizeof(float2));
    cudaMalloc(&d_h0_x, N * N * sizeof(float2));
    cudaMalloc(&d_h0_z, N * N * sizeof(float2));
    cudaMalloc(&d_states, N * N * sizeof(curandState));

    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    setup_curand<<<gridDim, blockDim>>>(d_states, N, 123456);
    cudaDeviceSynchronize();

    generate_initial_JONSWAP<<<gridDim, blockDim>>>(d_h0, d_h0_x, d_h0_z, d_states, N, L, params);
    cudaDeviceSynchronize();

    cudaMemcpy(h0, d_h0, N * N * sizeof(float2), cudaMemcpyDeviceToHost);
    cudaMemcpy(h0_x, d_h0_x, N * N * sizeof(float2), cudaMemcpyDeviceToHost);
    cudaMemcpy(h0_z, d_h0_z, N * N * sizeof(float2), cudaMemcpyDeviceToHost);

    cudaFree(d_h0);
    cudaFree(d_h0_x);
    cudaFree(d_h0_z);
    cudaFree(d_states);
}