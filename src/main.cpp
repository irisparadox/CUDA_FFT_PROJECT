#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

#include "../include/application.h"

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

int main() {
    Application app(1280, 720, "IFFT Ocean Simulation");
    app.run();

    return 0;
}