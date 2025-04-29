#include <cuda.h>
#include <cuda_runtime.h>

void update_spectra(float4* h0, float4* waves_data, float2* dx_dz, float2* dy_dxz, float time);