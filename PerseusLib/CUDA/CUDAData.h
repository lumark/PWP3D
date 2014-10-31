#pragma once 

#include "CUDADefines.h"

struct CUDAData
{
  //int widthFull, heightFull;
  //int widthROI, heightROI;
  int bufferSize;

  int viewCount, objectCount;

  cudaArray *arrayScharr;

  int *dtVImage;
  float *dtZImage;
  int *dtImagePosYT1;
  float *dtImageT1;

  float *hKernelConvolution;

  cudaArray *arrayHeaviside;

  int histogramSize;
  float dpose[7];
  float2 *histograms;
  float3 *dfxTranslation, *dfxResultTranslation;
  float4 *dfxRotation, *dfxResultRotation;

  cudaChannelFormatDesc descRendererVertices;

  unsigned char *fill;
  unsigned char *objects;
  unsigned int *zbuffer;
  unsigned int *zbufferInverse;

  unsigned char *fillAll;
  unsigned char *objectsAll;
  unsigned int *zbufferAll;
  unsigned int *zbufferInverseAll;

  int roiGenerated[6];
  int roiGeneratedAll[6];

  int4 *d_rois, *d_roisAll, *h_rois, *h_roisAll;

  int roisSize;
};
