#include <iostream>
#include <cstdlib>
#include "../include/spectra.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

const int N = 256;
const int L = 512;

void save_complex_to_image(const float2* h0, int N, const char* filename) {
    // Crear una matriz de píxeles de tamaño N x N
    unsigned char* image = new unsigned char[N * N * 3];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            
            // Parte real mapeada al canal rojo (R)
            float real_part = h0[idx].x;
            unsigned char red = static_cast<unsigned char>(fminf(fmaxf(real_part * 2.5f * 255.0f, 0.0f), 255.0f));

            // Parte imaginaria mapeada al canal verde (G)
            float imag_part = h0[idx].y;
            unsigned char green = static_cast<unsigned char>(fminf(fmaxf(imag_part * 2.5f * 255.0f, 0.0f), 255.0f));

            // El componente azul (B) es 0
            unsigned char blue = 0;

            // Colocar los valores de color en la matriz de píxeles
            int pixel_idx = (i * N + j) * 3;
            image[pixel_idx + 0] = red;   // R
            image[pixel_idx + 1] = green; // G
            image[pixel_idx + 2] = blue;  // B
        }
    }

    // Guardar la imagen como archivo PNG
    stbi_write_png(filename, N, N, 3, image, N * 3);

    // Liberar memoria
    delete[] image;
}

int main(int argc, char* argv[]) {
    if(argc != 6) {
        std::cout << "Error: too few arguments. Usage: ./fft_program wind_speed angle spread_blend swell fetch\n Exiting...";
        return -1;
    }
    JONSWAP_params params;
    params.wind_speed = atof(argv[1]);
    params.angle = atof(argv[2]);
    params.spread_blend = atof(argv[3]);
    params.swell = atof(argv[4]);
    params.fetch = atof(argv[5]);
    params.gamma = 3.3f;
    params.g = 9.81f;

    float2* h0 = (float2*)malloc(N * N * sizeof(float2));
    float2* h0_x = (float2*)malloc(N * N * sizeof(float2));
    float2* h0_z = (float2*)malloc(N * N * sizeof(float2));

    launch_initial_JONSWAP(h0, h0_x, h0_z, N, L, params);

    save_complex_to_image(h0, N, "output/initial_jonswap.png");

    free(h0);
    free(h0_x);
    free(h0_z);
    return 0;
}