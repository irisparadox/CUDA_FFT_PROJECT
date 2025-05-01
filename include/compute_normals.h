#ifndef COMPUTE_NORMALS_H
#define COMPUTE_NORMALS_H

#include <cuda.h>
#include <cuda_runtime.h>

void compute_meso_normals(float2* slope, float3* meso_normals, int N);

#endif