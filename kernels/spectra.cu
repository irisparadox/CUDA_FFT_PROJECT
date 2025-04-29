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
    *outY = r * sinf(theta);
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

__device__ float frequency(float k, float g, float depth) {
    return sqrt(g * k * tanhf(min(k * depth, 20.0f)));
}

__device__ float frequency_derivative(float k, float g, float depth) {
    float th = tanhf(min(k * depth, 20.0f));
    float ch = coshf(k * depth);
    return g * (depth * k / ch / ch + th) / frequency(k, g, depth) / 2;
}

__device__ float donelan_banner_beta(float x) {
    if(x < 0.95f)
        return 2.61f * powf(fabsf(x), 1.3f);
    if(x < 1.6f)
        return 2.28f * powf(fabsf(x), -1.3f);
    
    float p = -0.4f + 0.8393f * expf(-0.567 * logf(x * x));
    return powf(10.0f, p);

}

__device__ float donelan_banner(float theta, float omega, float peak_omega) {
    float beta = donelan_banner_beta(omega / peak_omega);
    float sech = 1 / coshf(beta * theta);
    return beta / 2 / tanhf(beta * M_PI) * sech * sech;
}

__device__ float spread_power(float omega, float peak_omega) {
    if(omega > peak_omega)
        return 0.77f * powf(fabsf(omega / peak_omega),-2.5f);
    else
        return 0.97f * powf(fabsf(omega / peak_omega),5.0f);
}

__device__ float directional_spectrum(float theta, float omega, JONSWAP_params params) {
    float s = spread_power(omega, params.peak_omega) +
        16 * tanhf(min(omega / params.peak_omega, 20.0f)) * params.swell * params.swell;
    return lerp(2.0f / M_PI * cosf(theta) * cosf(theta),Cosine2s(theta - params.angle, s), params.spread_blend);
}

__device__ float TMA_correction(float omega, float g, float depth) {
    float omega_h = omega * sqrt(depth / g);
    if(omega_h <= 1)
        return 0.5f * omega_h * omega_h;
    if(omega_h < 2)
        return 1.0f - 0.5f * (2.0f - omega_h) * (2.0f - omega_h);

    return 1.0f;
}

__device__ float JONSWAP(float omega, JONSWAP_params* params) {
    float sigma;
    float one_over_fetch = 1 / powf(params->fetch, 0.3f);
    float one_over_wind  = 1 / powf(params->wind_speed, 0.4f);

    float wind_over_fetch = powf(params->wind_speed, 2.0f) / (params->fetch * params->g);

    // empirical JONSWAP peak omega (wp)
    params->peak_omega = 2.84f * powf(params->g, 0.7f) * one_over_fetch * one_over_wind;
    // empirical JONSWAP alpha
    params->alpha      = 0.076 * powf(wind_over_fetch, 0.22f);

    if(omega <= params->peak_omega)
        sigma = 0.07f;
    else
        sigma = 0.09;
    
    float r = expf(-(omega - params->peak_omega) * (omega - params->peak_omega)
        / 2 / sigma / sigma / params->peak_omega / params->peak_omega);

    float one_over_omega = 1 / omega;
    float peak_omega_over_omega = params->peak_omega / omega;

    return params->scale * TMA_correction(omega, params->g, params->depth) * params->alpha * params->g * params->g
        * one_over_omega * one_over_omega * one_over_omega * one_over_omega * one_over_omega
        * expf(-1.25 * peak_omega_over_omega * peak_omega_over_omega * peak_omega_over_omega * peak_omega_over_omega
        * powf(fabsf(params->gamma), r));

}

__device__ float short_waves_fade(float k_length, JONSWAP_params params) {
    return expf(-params.short_waves_fade * params.short_waves_fade * k_length * k_length);
}

__global__ void generate_initial_JONSWAP(float2* h0_k, float4* waves_data,
    curandState* states, int N, int L, JONSWAP_params params) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N && j < N) {

        float deltaK = 2.0f * M_PI / L;
        int nx = j - N/2;
        int nz = i - N/2;

        float2 k = make_float2(nx * deltaK, nz * deltaK);
        float k_length = sqrtf(k.x * k.x + k.y * k.y);

        int idx = i * N + j;
        curandState* localState = &states[idx];

        float cutoffLow = 0.0001f;
        float cutoffHigh = 2.0f * M_PI / 0.1f * 6.0f;

        if(k_length <= cutoffHigh && k_length >= cutoffLow) {
            float k_angle = atan2f(k.y, k.x);
            float omega   = frequency(k_length, params.g, params.depth);
            waves_data[idx] = make_float4(k.x, 1 / k_length, k.y, omega);
            float d_omegaK = frequency_derivative(k_length, params.g, params.depth);

            float spectrum = JONSWAP(omega, &params)
                * directional_spectrum(k_angle, omega, params)
                * short_waves_fade(k_length, params);

            // random gaussian indices for real and imaginary
            float2 r;
            gaussian_random(localState, &r.x, &r.y);

            float phillips_height = sqrtf(2 * spectrum * fabsf(d_omegaK) / k_length * deltaK * deltaK);
            float h0_k_x = r.x * phillips_height;
            float h0_k_y = r.y * phillips_height;
            h0_k[idx] = make_float2(h0_k_x, h0_k_y);
        } else {
            h0_k[idx] = make_float2(0.0f, 0.0f);
            waves_data[idx] = make_float4(k.x, 1.0f, k.y, 0.0f);
        }
    }
}

__global__ void calculate_conjugated_JONSWAP(float4* h0, float2* h0_k, int N) {
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < N && j < N) {
        int idx = i * N + j;
        float2 h0k = h0_k[idx];
        int conj_x = (N - j) % N;
        int conj_y = (N - i) % N;
        int conj_idx = conj_y * N + conj_x;
        float2 h0_minus_k = h0_k[conj_idx];
        h0[idx] = make_float4(h0k.x, h0k.y, h0_minus_k.x, -h0_minus_k.y);
    }
}

void launch_initial_JONSWAP(float2* h0_k, float4* h0, float4* waves_data, int N, int L, JONSWAP_params params) {
    curandState* d_states;
    cudaMalloc(&d_states, N * N * sizeof(curandState));

    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    setup_curand<<<gridDim, blockDim>>>(d_states, N, 123456);
    cudaDeviceSynchronize();

    generate_initial_JONSWAP<<<gridDim, blockDim>>>(h0_k, waves_data, d_states, N, L, params);
    cudaDeviceSynchronize();

    calculate_conjugated_JONSWAP<<<gridDim, blockDim>>>(h0, h0_k, N);
    cudaDeviceSynchronize();

    cudaFree(d_states);
}