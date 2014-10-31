#pragma once

#include <PerseusLib/Others/PerseusLibDefines.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "CUDADefines.h"

__host__ void initialiseEF(int width, int height, float* heavisideFunction, int heavisideFunctionSize);
__host__ void shutdownEF();

__host__ void registerViewGeometricData(float *invP_EF, float *projectionParams_EF, int *viewTransform_EF);
__host__ void registerObjectGeometricData(float* rotationQuaternion_EF, float* invPM_EF);

void processEFD1(float* dpose, int *roiNormalised, int *roiGenerated, float2* histogram, uchar4 *imageRegistered, unsigned char *imageObjects, 
                 bool isMultiobject, unsigned int *imageZBuffer, unsigned int *imageZBufferInverse,
                 float *dt, int *dtPosX, int *dtPosY, float *dtDX, float *dtDY, int objectId);

__global__ void processEFD1_global(float3 *dfxTranslation, float4 *dfxRotation, float2 *histogram, uchar4 *imageRegistered, unsigned char *imageObjects,
                                   bool isMultiobject, unsigned int *imageZBuffer, unsigned int *imageZBufferInverse,
                                   float *dt, int *dtPosX, int *dtPosY, float *dtDX, float *dtDY,
                                   int minX, int minY, int widthROI, int heightROI, int objectId);
