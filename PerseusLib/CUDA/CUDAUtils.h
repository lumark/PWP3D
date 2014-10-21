#pragma once

#include <PerseusLib/Others/PerseusLibDefines.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "CUDADefines.h"

//struct PixelRGB { CUDA_PIXEL r,b,g; };

__host__ int iDivUp(int a, int b);
__host__ int iDivDown(int a, int b);
__host__ int iAlignUp(int a, int b);
__host__ int iAlignDown(int a, int b);

//__host__ void MakeGrayScale(int*, CUDA_PIXEL*, int, int);
//__host__ void GrayToOutput(CUDA_PIXEL*, int*, int, int);
//__host__ void GetNormalizedRoi(int*, int*, int &, int &);
//__host__ void GetNormalizedImageParamatersCenter(int*, int*, int&, int&);
//__host__ void GetNormalizedImageParamaters(int*, int*, int *, int*, int&, int&);
//__host__ void GetCenteredRoi(int*, int, int, int*);
//__host__ void NormalizeWithRoi(CUDA_PIXEL*, int*, int, int, int*, int, int, CUDA_PIXEL*);
//__host__ void Add(CUDA_PIXEL* image1, CUDA_PIXEL* image2, CUDA_PIXEL *imageSum, int width, int height);
//__host__ void Sub(CUDA_PIXEL* image1, CUDA_PIXEL* image2, CUDA_PIXEL *imageDiff, int width, int height);
//__host__ void AddDT(CUDA_FLOAT* image1, CUDA_FLOAT* image2, CUDA_FLOAT* imageSum, int width, int height);
//__host__ void SubDT(CUDA_FLOAT* image1, CUDA_FLOAT* image2, CUDA_FLOAT* imageDiff, int width, int height);
//__host__ void CopyToOutputImageCentered(CUDA_PIXEL*, CUDA_PIXEL*, int, int, int, int);
//__host__ void CopyToOutputImageOriginal(CUDA_PIXEL*, CUDA_PIXEL*, int*, int*, int, int, int, int);
//__host__ void CombineRenderedWithRegistered3(CUDA_PIXEL*, CUDA_PIXEL*, int, int);
//
//__global__ void makeGrayScale(int*, CUDA_PIXEL*, int);
//__global__ void grayToOutput(CUDA_PIXEL*, int*, int);
//__global__ void normalizeWithRoi(CUDA_PIXEL*,int, int, int, int, int, CUDA_PIXEL*, int, int, int, int);
//__global__ void add(CUDA_PIXEL*, CUDA_PIXEL*, CUDA_PIXEL*, int, int);
//__global__ void sub(CUDA_PIXEL*, CUDA_PIXEL*, CUDA_PIXEL*, int, int);
//__global__ void addDT(CUDA_FLOAT*, CUDA_FLOAT*, CUDA_FLOAT*, int, int);
//__global__ void subDT(CUDA_FLOAT*, CUDA_FLOAT*, CUDA_FLOAT*, int, int);
//__global__ void combineRenderedWithRegistered3sm(CUDA_PIXEL*, CUDA_PIXEL*, int, int);
//
//__device__ inline void atomicFloatAdd(float *address, float val);
