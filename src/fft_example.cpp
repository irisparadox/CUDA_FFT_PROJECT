#include <cufft.h>
#include <cuda_runtime.h>
#include <iostream>

int main() {
    const int N = 8;

    cufftComplex h_data[N];

    for(int i = 0; i < N; ++i) {
        h_data[i].x = i;
        h_data[i].y = 0;
    }

    cufftComplex* d_data;

    cudaMalloc(&d_data, sizeof(cufftComplex) * N);
    cudaMemcpy(d_data, h_data, sizeof(cufftComplex) * N, cudaMemcpyHostToDevice);

    cufftHandle plan;
    if(cufftPlan1d(&plan, N, CUFFT_C2C, 1) != CUFFT_SUCCESS) {
        std::cout << "Error creando el plan FFT\n";
        return -1;
    }

    if(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        std::cout << "Error ejecutando el plan FFT\n";
        return -1;
    }

    cudaMemcpy(h_data, d_data, sizeof(cufftComplex) * N, cudaMemcpyDeviceToHost);

    std::cout << "FFT result:\n";
    for(int i = 0; i < N; ++i) {
        std::cout << h_data[i].x << ' ' << h_data[i].y << '\n';
    }

    cufftDestroy(plan);
    cudaFree(d_data);

    return 0;
}