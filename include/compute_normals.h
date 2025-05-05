#ifndef COMPUTE_NORMALS_H
#define COMPUTE_NORMALS_H

#include <cuda.h>
#include <cuda_runtime.h>

void compute_normals(float2* slope, float3* normals, float strength, int N);

#endif