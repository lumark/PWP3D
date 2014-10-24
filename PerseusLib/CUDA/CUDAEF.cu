#include "CUDAEF.h"
#include "CUDAData.h"

extern CUDAData* cudaData;

texture<float, 1, cudaReadModeElementType> g_texHeaviside;

//a single object and a single view per video card at any one time
__device__ __constant__ int viewTransform[4];
__device__ __constant__ float invP[16];
__device__ __constant__ float Mplane[4][4];
__device__ __constant__ float projectionParams[5];
__device__ __constant__ float invPM[16];
__device__ __constant__ float q[4];

__device__ __constant__ int histOffsets[4];
__device__ __constant__ int histFactors[4];
__device__ __constant__ int histNoBins[4];

__host__ void initialiseEF(int width, int height, float* heavisideFunction, int heavisideFunctionSize)
{
  //TODO FIXME VARHISTBINS
  int noVarBinHistograms, noVarBinHistogramBins[4];
  int h_histOffsets[8], h_histFactors[8], h_noBins[8];

  noVarBinHistograms = 4;
  noVarBinHistogramBins[0] = 8; noVarBinHistogramBins[1] = 16; noVarBinHistogramBins[2] = 32; noVarBinHistogramBins[3] = 64;
  h_histFactors[0] = 5; h_histFactors[1] = 4; h_histFactors[2] = 3; h_histFactors[3] = 2;

  cudaData->histogramSize = 0;
  for (int i=0; i<noVarBinHistograms; i++)
  {
    h_noBins[i] = noVarBinHistogramBins[i];
    h_histOffsets[i] = cudaData->histogramSize;
    cudaData->histogramSize += h_noBins[i] * h_noBins[i] * h_noBins[i];
  }

  perseusSafeCall(cudaMemcpyToSymbol(histOffsets, h_histOffsets, noVarBinHistograms * sizeof(int), 0, cudaMemcpyHostToDevice));
  perseusSafeCall(cudaMemcpyToSymbol(histFactors, h_histFactors, noVarBinHistograms * sizeof(int), 0, cudaMemcpyHostToDevice));
  perseusSafeCall(cudaMemcpyToSymbol(histNoBins, h_noBins, noVarBinHistograms * sizeof(int), 0, cudaMemcpyHostToDevice));

  perseusSafeCall(cudaMalloc((void**)&cudaData->dfxTranslation, 10000 * sizeof(float3))); //1080p image 16x16 blocks
  perseusSafeCall(cudaMalloc((void**)&cudaData->dfxRotation, 10000 * sizeof(float4))); //1080p image 16x16 blocks

  perseusSafeCall(cudaMallocHost((void**)&cudaData->dfxResultTranslation, 10000 * sizeof(float3))); //1080p image 16x16 blocks
  perseusSafeCall(cudaMallocHost((void**)&cudaData->dfxResultRotation, 10000 * sizeof(float4))); //1080p image 16x16 blocks

  // copy the heviside function from the host memory into the device memory
  cudaChannelFormatDesc descTexHeaviside = cudaCreateChannelDesc<float>();

  perseusSafeCall(cudaMallocArray(&cudaData->arrayHeaviside, &descTexHeaviside, heavisideFunctionSize, 1));
  perseusSafeCall(cudaMemcpyToArray(cudaData->arrayHeaviside, 0, 0, heavisideFunction, heavisideFunctionSize * sizeof(float), cudaMemcpyHostToDevice));
}

__host__ void shutdownEF()
{
  perseusSafeCall(cudaFreeArray(cudaData->arrayHeaviside));

  perseusSafeCall(cudaFree(cudaData->dfxTranslation));
  perseusSafeCall(cudaFree(cudaData->dfxRotation));

  perseusSafeCall(cudaFreeHost(cudaData->dfxResultTranslation));
  perseusSafeCall(cudaFreeHost(cudaData->dfxResultRotation));
}

__host__ void registerViewGeometricData(float *invP_EF, float *projectionParams_EF, int *viewTransform_EF)
{
  perseusSafeCall(cudaMemcpyToSymbol(invP, invP_EF, 16 * sizeof(float), 0, cudaMemcpyHostToDevice));
  perseusSafeCall(cudaMemcpyToSymbol(projectionParams, projectionParams_EF, 5 * sizeof(float), 0, cudaMemcpyHostToDevice));
  perseusSafeCall(cudaMemcpyToSymbol(viewTransform, viewTransform_EF, 4 * sizeof(int), 0, cudaMemcpyHostToDevice));
}

__host__ void registerObjectGeometricData(float* rotationQuaternion_EF, float* invPM_EF)
{
  perseusSafeCall(cudaMemcpyToSymbol(invPM, invPM_EF, 16 * sizeof(float), 0,  cudaMemcpyHostToDevice));
  rotationQuaternion_EF[0] *= 2; rotationQuaternion_EF[1] *= 2; rotationQuaternion_EF[2] *= 2; rotationQuaternion_EF[3] *= 2;
  perseusSafeCall(cudaMemcpyToSymbol(q, rotationQuaternion_EF, 4 * sizeof(float), 0, cudaMemcpyHostToDevice));
}

__host__ void processEFD1(float* dpose, int *roiNormalised, int *roiGenerated, float2* histogram, uchar4 *imageRegistered, unsigned char *imageObjects, 
                          bool isMultiobject, unsigned int *imageZBuffer, unsigned int *imageZBufferInverse,
                          float *dt, int *dtPosX, int *dtPosY, float *dtDX, float *dtDY, int objectId)
{
  size_t i;

  dim3 threadSize(16,16);
  dim3 blockSize((int)ceil((float)roiNormalised[4] / (float)16), (int)ceil((float)roiNormalised[5] / (float)16));

  // copy heaviside function table into the GPU texture
  perseusSafeCall(cudaBindTextureToArray(g_texHeaviside, cudaData->arrayHeaviside));

  processEFD1_global<<<blockSize, threadSize>>>(cudaData->dfxTranslation, cudaData->dfxRotation, histogram, imageRegistered, imageObjects, isMultiobject,
                                                imageZBuffer, imageZBufferInverse, dt, dtPosX, dtPosY, dtDX, dtDY,
                                                roiGenerated[0], roiGenerated[1], roiGenerated[4], roiGenerated[5], objectId);

  perseusSafeCall(cudaUnbindTexture(g_texHeaviside));
  perseusSafeCall(cudaDeviceSynchronize());
  perseusSafeCall(cudaThreadSynchronize());

  cudaMemcpy(cudaData->dfxResultTranslation, cudaData->dfxTranslation, blockSize.x * blockSize.y * sizeof(float3), cudaMemcpyDeviceToHost);
  cudaMemcpy(cudaData->dfxResultRotation, cudaData->dfxRotation, blockSize.x * blockSize.y * sizeof(float4), cudaMemcpyDeviceToHost);

  for (i=0; i<7; i++) dpose[i] = 0;

  for (size_t i=0; i < blockSize.x * blockSize.y; i++)
  {
    dpose[0] += cudaData->dfxResultTranslation[i].x;
    dpose[1] += cudaData->dfxResultTranslation[i].y;
    dpose[2] += cudaData->dfxResultTranslation[i].z;
    dpose[3] += cudaData->dfxResultRotation[i].x;
    dpose[4] += cudaData->dfxResultRotation[i].y;
    dpose[5] += cudaData->dfxResultRotation[i].z;
    dpose[6] += cudaData->dfxResultRotation[i].w;
  }
}

__device__ float3 sdataTranslation[256];
__device__ float4 sdataRotation[256];

__global__ void processEFD1_global(
    float3 *dfxTranslation, float4 *dfxRotation, float2 *histogram, uchar4 *imageRegistered, unsigned char *imageObjects,
    bool isMultiobject, unsigned int *imageZBuffer, unsigned int *imageZBufferInverse,
    float *dt, int *dtPosX, int *dtPosY, float *dtDX, float *dtDY,
    int minX, int minY, int widthROI, int heightROI, int objectId)
{
  int offsetX = threadIdx.x + blockIdx.x * blockDim.x;
  int offsetY = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = offsetX + offsetY * widthROI;
  int offsetInBlock = threadIdx.x + blockDim.x * threadIdx.y;

  float3 dfPPTranslation; dfPPTranslation.x = 0; dfPPTranslation.y = 0; dfPPTranslation.z = 0;
  float4 dfPPRotation; dfPPRotation.x = 0; dfPPRotation.y = 0; dfPPRotation.z = 0; dfPPRotation.w = 0;

  sdataTranslation[offsetInBlock] = dfPPTranslation;
  sdataRotation[offsetInBlock] = dfPPRotation;

  if (offsetX < widthROI && offsetY < heightROI)
  {
    uchar4 imagePixel;

    int n_icX, n_icY, n_icZ;
    int n_greyPixel, n_currentHistogram;
    int n_hidx, n_pidx;

    float2 histogramPixel;
    float f_pYB, f_pYF;
    float f_xProjected[4], f_xUnprojected[4], f_xUnrotated[4];
    float f_fPPGeneric, f_dirac, f_heaviside;
    float f_otherInfo[2];

    float f_precalcX, d_precalcY, d_precalcXY;

    float f_dtIdx, f_norm;

    if (dtPosY[offset] >= 0)// && imageRegistered[offset].w > 128)
    {
      f_dtIdx = dt[offset];

      n_icX = offsetX; n_icY = offsetY;

      if (f_dtIdx < 0)
      {
        n_icX = dtPosX[offset];
        n_icY = dtPosY[offset];
      }

      n_icZ = n_icX + n_icY * widthROI;

      if (!isMultiobject || (isMultiobject && (imageObjects[n_icZ] - 1) == objectId &&
                             ((imageObjects[offsetX + offsetY * widthROI] - 1) == objectId || (imageObjects[offsetX + offsetY * widthROI] - 1) == -1 )))
      {
        n_hidx = 4096 + 512 * f_dtIdx;

        if (n_hidx >= 0 && n_hidx < 8192)
        {
          f_heaviside = tex1D(g_texHeaviside, n_hidx);

          imagePixel = imageRegistered[offset];
          n_greyPixel = int(float(imagePixel.x) * 0.3f + float(imagePixel.y) * 0.59f + float(imagePixel.z) * 0.11f);

          n_currentHistogram = 0;
          if (n_greyPixel < 128) n_currentHistogram = 3;
          else if (n_greyPixel < 192) n_currentHistogram = 2;
          else if (n_greyPixel < 224) n_currentHistogram = 1;

          //currentHistogram = 2;

          imagePixel.x = (imagePixel.x >> histFactors[n_currentHistogram]) & (histNoBins[n_currentHistogram] - 1);
          imagePixel.y = (imagePixel.y >> histFactors[n_currentHistogram]) & (histNoBins[n_currentHistogram] - 1);
          imagePixel.z = (imagePixel.z >> histFactors[n_currentHistogram]) & (histNoBins[n_currentHistogram] - 1);
          n_pidx = (imagePixel.x + imagePixel.y * histNoBins[n_currentHistogram]) * histNoBins[n_currentHistogram] + imagePixel.z;

          histogramPixel = histogram[histOffsets[n_currentHistogram] + n_pidx];

          f_pYF = histogramPixel.x + 0.0000001f; f_pYB = histogramPixel.y + 0.0000001f;

          f_dirac = (1.0f / float(PI)) * (1 / (f_dtIdx * f_dtIdx + 1.0f) + float(1e-3));
          f_fPPGeneric = f_dirac * (f_pYF - f_pYB) / (f_heaviside * (f_pYF - f_pYB) + f_pYB);

          f_xProjected[0] = 2 * (n_icX + minX - (float) viewTransform[0]) / (float) viewTransform[2] - 1;
          f_xProjected[1] = 2 * (n_icY + minY - (float) viewTransform[1]) / (float) viewTransform[3] - 1;
          f_xProjected[2] = 2 * ((float)imageZBuffer[n_icZ] / (float)MAX_INT) - 1;
          f_xProjected[3] = 1;

          f_xUnprojected[0] = invP[0] * f_xProjected[0] + invP[4] * f_xProjected[1] + invP[8] * f_xProjected[2] + invP[12] * f_xProjected[3];
          f_xUnprojected[1] = invP[1] * f_xProjected[0] + invP[5] * f_xProjected[1] + invP[9] * f_xProjected[2] + invP[13] * f_xProjected[3];
          f_xUnprojected[2] = invP[2] * f_xProjected[0] + invP[6] * f_xProjected[1] + invP[10] * f_xProjected[2] + invP[14] * f_xProjected[3];
          f_xUnprojected[3] = invP[3] * f_xProjected[0] + invP[7] * f_xProjected[1] + invP[11] * f_xProjected[2] + invP[15] * f_xProjected[3];
          f_norm = 1.0f/f_xUnprojected[3]; f_xUnprojected[0] *= f_norm; f_xUnprojected[1] *= f_norm; f_xUnprojected[2] *= f_norm; f_xUnprojected[3] *= f_norm;

          f_xUnrotated[0] = invPM[0] * f_xProjected[0] + invPM[4] * f_xProjected[1] + invPM[8] * f_xProjected[2] + invPM[12] * f_xProjected[3];
          f_xUnrotated[1] = invPM[1] * f_xProjected[0] + invPM[5] * f_xProjected[1] + invPM[9] * f_xProjected[2] + invPM[13] * f_xProjected[3];
          f_xUnrotated[2] = invPM[2] * f_xProjected[0] + invPM[6] * f_xProjected[1] + invPM[10] * f_xProjected[2] + invPM[14] * f_xProjected[3];
          f_xUnrotated[3] = invPM[3] * f_xProjected[0] + invPM[7] * f_xProjected[1] + invPM[11] * f_xProjected[2] + invPM[15] * f_xProjected[3];
          f_norm = 1.0f/f_xUnrotated[3];	f_xUnrotated[0] *= f_norm; f_xUnrotated[1] *= f_norm; f_xUnrotated[2] *= f_norm; f_xUnrotated[3] *= f_norm;

          f_otherInfo[0] = projectionParams[0] * dtDX[offset]; f_otherInfo[1] = projectionParams[1] * dtDY[offset];

          d_precalcXY = f_xUnprojected[2] * f_xUnprojected[2];

          dfPPTranslation.x = -f_otherInfo[0] / f_xUnprojected[2];
          dfPPTranslation.y = -f_otherInfo[1] / f_xUnprojected[2];
          dfPPTranslation.z = (f_otherInfo[0] * f_xUnprojected[0] + f_otherInfo[1] * f_xUnprojected[1]) / d_precalcXY;

          f_precalcX = -f_otherInfo[0] / d_precalcXY; d_precalcY = -f_otherInfo[1] / d_precalcXY;

          dfPPRotation.x =
              f_precalcX * (f_xUnprojected[2] * (q[1]*f_xUnrotated[1] + q[2]*f_xUnrotated[2]) -
              f_xUnprojected[0] * (q[2]*f_xUnrotated[0] + q[3]*f_xUnrotated[1] - 2*q[0]*f_xUnrotated[2])) +
              d_precalcY * (f_xUnprojected[2] * (q[1]*f_xUnrotated[0] - 2*q[0]*f_xUnrotated[1] - q[3]*f_xUnrotated[2]) -
              f_xUnprojected[1] * (q[2]*f_xUnrotated[0] + q[3]*f_xUnrotated[1] - 2*q[0]*f_xUnrotated[2]));

          dfPPRotation.y =
              f_precalcX * (f_xUnprojected[2] * (q[0]*f_xUnrotated[1] - 2*q[1]*f_xUnrotated[0] + q[3]*f_xUnrotated[2]) -
              f_xUnprojected[0] * (q[2]*f_xUnrotated[1] - q[3]*f_xUnrotated[0] - 2*q[1]*f_xUnrotated[2])) +
              d_precalcY * (f_xUnprojected[2] * (q[0]*f_xUnrotated[0] + q[2]*f_xUnrotated[2]) -
              f_xUnprojected[1] * (q[2]*f_xUnrotated[1] - q[3]*f_xUnrotated[0] - 2*q[1]*f_xUnrotated[2]));

          dfPPRotation.z =
              f_precalcX * (f_xUnprojected[2] * (q[0]*f_xUnrotated[2] - q[3]*f_xUnrotated[1] - 2*q[2]*f_xUnrotated[0]) -
              f_xUnprojected[0] * (q[0]*f_xUnrotated[0] + q[1]*f_xUnrotated[1])) +
              d_precalcY * (f_xUnprojected[2] * (q[3]*f_xUnrotated[0] - 2*q[2]*f_xUnrotated[1] + q[1]*f_xUnrotated[2]) -
              f_xUnprojected[1] * (q[0]*f_xUnrotated[0] + q[1]*f_xUnrotated[1]));

          dfPPRotation.w =
              f_precalcX * (f_xUnprojected[2] * (q[1]*f_xUnrotated[2] - q[2]*f_xUnrotated[1]) -
              f_xUnprojected[0] * (q[0]*f_xUnrotated[1] - q[1]*f_xUnrotated[0])) +
              d_precalcY * (f_xUnprojected[2] * (q[2]*f_xUnrotated[0] - q[0]*f_xUnrotated[2]) -
              f_xUnprojected[1] * (q[0]*f_xUnrotated[1] - q[1]*f_xUnrotated[0]));

          f_xProjected[0] = 2 * (n_icX + minX - (float) viewTransform[0]) / (float) viewTransform[2] - 1;
          f_xProjected[1] = 2 * (n_icY + minY - (float) viewTransform[1]) / (float) viewTransform[3] - 1;
          f_xProjected[2] = 2 * ((float)imageZBufferInverse[n_icZ] / (float)MAX_INT) - 1;
          f_xProjected[3] = 1;

          f_xUnprojected[0] = invP[0] * f_xProjected[0] + invP[4] * f_xProjected[1] + invP[8] * f_xProjected[2] + invP[12] * f_xProjected[3];
          f_xUnprojected[1] = invP[1] * f_xProjected[0] + invP[5] * f_xProjected[1] + invP[9] * f_xProjected[2] + invP[13] * f_xProjected[3];
          f_xUnprojected[2] = invP[2] * f_xProjected[0] + invP[6] * f_xProjected[1] + invP[10] * f_xProjected[2] + invP[14] * f_xProjected[3];
          f_xUnprojected[3] = invP[3] * f_xProjected[0] + invP[7] * f_xProjected[1] + invP[11] * f_xProjected[2] + invP[15] * f_xProjected[3];
          f_norm = 1.0f/f_xUnprojected[3]; f_xUnprojected[0] *= f_norm; f_xUnprojected[1] *= f_norm; f_xUnprojected[2] *= f_norm; f_xUnprojected[3] *= f_norm;

          f_xUnrotated[0] = invPM[0] * f_xProjected[0] + invPM[4] * f_xProjected[1] + invPM[8] * f_xProjected[2] + invPM[12] * f_xProjected[3];
          f_xUnrotated[1] = invPM[1] * f_xProjected[0] + invPM[5] * f_xProjected[1] + invPM[9] * f_xProjected[2] + invPM[13] * f_xProjected[3];
          f_xUnrotated[2] = invPM[2] * f_xProjected[0] + invPM[6] * f_xProjected[1] + invPM[10] * f_xProjected[2] + invPM[14] * f_xProjected[3];
          f_xUnrotated[3] = invPM[3] * f_xProjected[0] + invPM[7] * f_xProjected[1] + invPM[11] * f_xProjected[2] + invPM[15] * f_xProjected[3];
          f_norm = 1.0f/f_xUnrotated[3]; f_xUnrotated[0] *= f_norm; f_xUnrotated[1] *= f_norm; f_xUnrotated[2] *= f_norm; f_xUnrotated[3] *= f_norm;

          d_precalcXY = f_xUnprojected[2] * f_xUnprojected[2];

          dfPPTranslation.x += -f_otherInfo[0] / f_xUnprojected[2];
          dfPPTranslation.y += -f_otherInfo[1] / f_xUnprojected[2];
          dfPPTranslation.z += (f_otherInfo[0] * f_xUnprojected[0] + f_otherInfo[1] * f_xUnprojected[1]) / d_precalcXY;

          f_precalcX = -f_otherInfo[0] / d_precalcXY; d_precalcY = -f_otherInfo[1] / d_precalcXY;

          dfPPRotation.x +=
              f_precalcX * (f_xUnprojected[2] * (q[1]*f_xUnrotated[1] + q[2]*f_xUnrotated[2]) -
              f_xUnprojected[0] * (q[2]*f_xUnrotated[0] + q[3]*f_xUnrotated[1] - 2*q[0]*f_xUnrotated[2])) +
              d_precalcY * (f_xUnprojected[2] * (q[1]*f_xUnrotated[0] - 2*q[0]*f_xUnrotated[1] - q[3]*f_xUnrotated[2]) -
              f_xUnprojected[1] * (q[2]*f_xUnrotated[0] + q[3]*f_xUnrotated[1] - 2*q[0]*f_xUnrotated[2]));

          dfPPRotation.y +=
              f_precalcX * (f_xUnprojected[2] * (q[0]*f_xUnrotated[1] - 2*q[1]*f_xUnrotated[0] + q[3]*f_xUnrotated[2]) -
              f_xUnprojected[0] * (q[2]*f_xUnrotated[1] - q[3]*f_xUnrotated[0] - 2*q[1]*f_xUnrotated[2])) +
              d_precalcY * (f_xUnprojected[2] * (q[0]*f_xUnrotated[0] + q[2]*f_xUnrotated[2]) -
              f_xUnprojected[1] * (q[2]*f_xUnrotated[1] - q[3]*f_xUnrotated[0] - 2*q[1]*f_xUnrotated[2]));

          dfPPRotation.z +=
              f_precalcX * (f_xUnprojected[2] * (q[0]*f_xUnrotated[2] - q[3]*f_xUnrotated[1] - 2*q[2]*f_xUnrotated[0]) -
              f_xUnprojected[0] * (q[0]*f_xUnrotated[0] + q[1]*f_xUnrotated[1])) +
              d_precalcY * (f_xUnprojected[2] * (q[3]*f_xUnrotated[0] - 2*q[2]*f_xUnrotated[1] + q[1]*f_xUnrotated[2]) -
              f_xUnprojected[1] * (q[0]*f_xUnrotated[0] + q[1]*f_xUnrotated[1]));

          dfPPRotation.w +=
              f_precalcX * (f_xUnprojected[2] * (q[1]*f_xUnrotated[2] - q[2]*f_xUnrotated[1]) -
              f_xUnprojected[0] * (q[0]*f_xUnrotated[1] - q[1]*f_xUnrotated[0])) +
              d_precalcY * (f_xUnprojected[2] * (q[2]*f_xUnrotated[0] - q[0]*f_xUnrotated[2]) -
              f_xUnprojected[1] * (q[0]*f_xUnrotated[1] - q[1]*f_xUnrotated[0]));

          dfPPTranslation.x *= f_fPPGeneric; dfPPTranslation.y *= f_fPPGeneric; dfPPTranslation.z *= f_fPPGeneric;
          dfPPRotation.x *= f_fPPGeneric; dfPPRotation.y *= f_fPPGeneric;
          dfPPRotation.z *= f_fPPGeneric; dfPPRotation.w *= f_fPPGeneric;

          sdataTranslation[offsetInBlock].x = dfPPTranslation.x;
          sdataTranslation[offsetInBlock].y = dfPPTranslation.y;
          sdataTranslation[offsetInBlock].z = dfPPTranslation.z;

          sdataRotation[offsetInBlock].x = dfPPRotation.x;
          sdataRotation[offsetInBlock].y = dfPPRotation.y;
          sdataRotation[offsetInBlock].z = dfPPRotation.z;
          sdataRotation[offsetInBlock].w = dfPPRotation.w;
        }
      }
    }
  }

  __syncthreads();

  int sdataTargetOffset;

  for(unsigned int s = blockDim.x >> 1; s>0; s>>=1)
  {
    if (threadIdx.x < s)
    {
      sdataTargetOffset = (threadIdx.x + s) + blockDim.x * threadIdx.y;
      sdataTranslation[offsetInBlock].x += sdataTranslation[sdataTargetOffset].x;
      sdataTranslation[offsetInBlock].y += sdataTranslation[sdataTargetOffset].y;
      sdataTranslation[offsetInBlock].z += sdataTranslation[sdataTargetOffset].z;
    }
    __syncthreads();
  }

  for(unsigned int s = blockDim.y >> 1; s>0; s>>=1)
  {
    if (threadIdx.y < s)
    {
      sdataTargetOffset = threadIdx.x + blockDim.x * (threadIdx.y + s);
      sdataTranslation[offsetInBlock].x += sdataTranslation[sdataTargetOffset].x;
      sdataTranslation[offsetInBlock].y += sdataTranslation[sdataTargetOffset].y;
      sdataTranslation[offsetInBlock].z += sdataTranslation[sdataTargetOffset].z;
    }
    __syncthreads();
  }

  for(unsigned int s = blockDim.x >> 1; s>0; s>>=1)
  {
    if (threadIdx.x < s)
    {
      sdataTargetOffset = (threadIdx.x + s) + blockDim.x * threadIdx.y;
      sdataRotation[offsetInBlock].x += sdataRotation[sdataTargetOffset].x;
      sdataRotation[offsetInBlock].y += sdataRotation[sdataTargetOffset].y;
      sdataRotation[offsetInBlock].z += sdataRotation[sdataTargetOffset].z;
      sdataRotation[offsetInBlock].w += sdataRotation[sdataTargetOffset].w;
    }
    __syncthreads();
  }

  for(unsigned int s = blockDim.y >> 1; s>0; s>>=1)
  {
    if (threadIdx.y < s)
    {
      sdataTargetOffset = threadIdx.x + blockDim.x * (threadIdx.y + s);
      sdataRotation[offsetInBlock].x += sdataRotation[sdataTargetOffset].x;
      sdataRotation[offsetInBlock].y += sdataRotation[sdataTargetOffset].y;
      sdataRotation[offsetInBlock].z += sdataRotation[sdataTargetOffset].z;
      sdataRotation[offsetInBlock].w += sdataRotation[sdataTargetOffset].w;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0 && threadIdx.y == 0)
  {
    int offsetDfx = blockIdx.x + blockIdx.y * gridDim.x;
    dfxTranslation[offsetDfx] = sdataTranslation[offsetInBlock];
    dfxRotation[offsetDfx] = sdataRotation[offsetInBlock];
  }
}
