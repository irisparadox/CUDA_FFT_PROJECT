#include <iostream>
#include <cstdlib>
#include <cufft.h>
#include <cuda_runtime.h>
#include "../include/spectra.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

const int N = 256;
const int L = 250;

void save_complex_to_image(const float2* h0, int N, const char* filename) {
    unsigned char* image = new unsigned char[N * N * 3];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            
            // real
            float real_part = h0[idx].x;
            unsigned char red = static_cast<unsigned char>(fminf(fmaxf(real_part * 2.5f * 255.0f, 0.0f), 255.0f));

            // imaginary
            float imag_part = h0[idx].y;
            unsigned char green = static_cast<unsigned char>(fminf(fmaxf(imag_part * 2.5f * 255.0f, 0.0f), 255.0f));

            unsigned char blue = 0;

            int pixel_idx = (i * N + j) * 3;
            image[pixel_idx + 0] = red;   // R
            image[pixel_idx + 1] = green; // G
            image[pixel_idx + 2] = blue;  // B
        }
    }

    stbi_write_png(filename, N, N, 3, image, N * 3);

    delete[] image;
}

void generate_heightmap(const float2* hx, int N, const char* filename) {
    unsigned char* image = new unsigned char[N * N * 3];

    float min_val = 1e20f;
    float max_val = -1e20f;

    // 1. Calculamos mínimo y máximo
    for (int i = 0; i < N * N; i++) {
        float val = fabsf(hx[i].x); // usamos valor absoluto
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }

    float range = max_val - min_val;
    if (range == 0.0f) range = 1.0f; // evitar división por cero

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;

            float real_part = fabsf(hx[idx].x); // trabajamos con el absoluto

            // Normalizamos a [0, 1]
            float normalized = (real_part - min_val) / range;

            // Aplicamos expansión de contraste real:
            normalized = sqrtf(normalized * normalized); // raíz cuadrada: expande valores pequeños

            unsigned char intensity = static_cast<unsigned char>(normalized * 255.0f);

            int pixel_idx = (i * N + j) * 3;
            image[pixel_idx + 0] = intensity; // R
            image[pixel_idx + 1] = intensity; // G
            image[pixel_idx + 2] = intensity; // B
        }
    }

    stbi_write_png(filename, N, N, 3, image, N * 3);

    delete[] image;
}

int main(int argc, char* argv[]) {
    if(argc != 9) {
        std::cout << "Error: too few arguments. " << 
        "Usage: ./fft_program scale wind_speed angle spread_blend swell fetch depth short_waves_fade\n Exiting...";
        return -1;
    }
    JONSWAP_params params;
    params.scale = atof(argv[1]);
    params.wind_speed = atof(argv[2]);
    params.angle = atof(argv[3]);
    params.spread_blend = atof(argv[4]);
    params.swell = atof(argv[5]);
    params.fetch = atof(argv[6]);
    params.depth = atof(argv[7]);
    params.short_waves_fade = atof(argv[8]);
    params.gamma = 3.3f;
    params.g = 9.81f;

    float2* h0_k = (float2*)malloc(N * N * sizeof(float2));
    float4* h0   = (float4*)malloc(N * N * sizeof(float4));
    float4* waves_data = (float4*)malloc(N * N * sizeof(float4));

    launch_initial_JONSWAP(h0_k, h0, waves_data, N, L, params);

    save_complex_to_image(h0_k, N, "output/initial_jonswap.png");

    float2* d_h0_k;
    float2* hk = (float2*)malloc(sizeof(float2) * N * N);
    cudaMalloc(&d_h0_k, sizeof(float2) * N * N);
    cudaMemcpy(d_h0_k, h0_k, sizeof(float2) * N * N, cudaMemcpyHostToDevice);
    cufftHandle plan;
    cufftPlan2d(&plan, N, N, CUFFT_C2C);
    cufftExecC2C(plan, d_h0_k, d_h0_k, CUFFT_INVERSE);
    cudaMemcpy(hk, d_h0_k, sizeof(float2) * N * N, cudaMemcpyDeviceToHost);
    cudaFree(d_h0_k);
    generate_heightmap(hk, N, "output/ifft_heightmap.png");

    free(h0);
    free(h0_k);
    free(waves_data);
    free(hk);
    return 0;
}