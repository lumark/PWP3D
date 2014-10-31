#pragma once

#include <PerseusLib/Others/PerseusLibDefines.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "CUDADefines.h"

__global__ void scharrTex(unsigned char*, unsigned int, int, int, float);
__global__ void sihluetteTex(unsigned char *, unsigned int, int, int, float);

__device__ unsigned char computeScharrPP(unsigned char, unsigned char, unsigned char, unsigned char, unsigned char, 
                                         unsigned char, unsigned char, unsigned char, unsigned char, unsigned char, float);
__device__ unsigned char computeSihluettePP(unsigned char, unsigned char, unsigned char, unsigned char, unsigned char, 
                                            unsigned char, unsigned char, unsigned char, unsigned char, unsigned char, float);

__host__ void initialiseScharr(int, int);
__host__ void shutdownScharr();
__host__ void computeSihluette(unsigned char *originalImage, unsigned char *scharrImage, int w, int h, float fScale);
__host__ void computeScharr(unsigned char *originalImage, unsigned char *sihlutteImage, int w, int h, float fScale);
