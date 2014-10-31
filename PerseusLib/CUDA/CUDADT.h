#pragma once

#include <PerseusLib/Others/PerseusLibDefines.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "CUDADefines.h"

__host__ void initialiseDT(int width, int height);
__host__ void shutdownDT();

__host__ void convertDTToImage(unsigned char* image, float* imageTransform, int widthFull, int heightFull);
__host__ void processDT(float *dt, int *dtPosX, int *dtPosY, unsigned char *grayImage, unsigned char* signMask, int *roi, int bandSize);
__host__ void getDT(float* dtROI, float *dt, int *roi, int widthFull, int heightFull);

__global__ void processDTT1(unsigned char* grayImage, float* dtImageT1, int* dtImagePosX, int* dtImagePosYT1, int dtWidth, 
                            int dtHeight, int dtWidthFull, int minxB, int minyB, int maxxB, int maxyB, int bandSize);
__global__ void processDTT2(float* dtImageT2, int dtWidthFull, int dtHeightFull, float* dtImageT1, float* zImage, int* vImage, 
                            unsigned char *signMask, int* dtImagePosX, int* dtImagePosYT1, int* dtImagePosYT2, int dtWidth, int dtHeight,
                            int minxB, int minyB, int maxxB, int maxyB, int bandSize);
