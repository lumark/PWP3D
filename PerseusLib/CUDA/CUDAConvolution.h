#pragma once 

#include <PerseusLib/Others/PerseusLibDefines.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "CUDADefines.h"

__host__ void initialiseConvolution(int width, int height);
__host__ void shutdownConvolution();

__host__ void computeDerivativeXY(float* function, float* derivativeX, float* derivativeY, int width, int height);

__global__ void convolutionRow(float *d_Result, float *d_Data, int dataW, int dataH);
__global__ void convolutionColumn(float *d_Result, float *d_Data, int dataW, int dataH, int smemStride, int gmemStride);
