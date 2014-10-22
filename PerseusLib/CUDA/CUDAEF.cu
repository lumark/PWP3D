#include "CUDAEF.h"
#include "CUDAData.h"

extern CUDAData* cudaData;

texture<float, 1, cudaReadModeElementType> texHeaviside;

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

  perseusSafeCall(cudaBindTextureToArray(texHeaviside, cudaData->arrayHeaviside));
  perseusSafeCall(cudaUnbindTexture(texHeaviside));
  perseusSafeCall(cudaDeviceSynchronize());

  processEFD1_global<<<blockSize, threadSize>>>(cudaData->dfxTranslation, cudaData->dfxRotation, histogram, imageRegistered, imageObjects, isMultiobject,
                                                imageZBuffer, imageZBufferInverse, dt, dtPosX, dtPosY, dtDX, dtDY, roiGenerated[0], roiGenerated[1], roiGenerated[4], roiGenerated[5], objectId);
  cudaError cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    printf("Something was wrong! Error code: %d", cudaStatus);
  }

  perseusSafeCall(cudaUnbindTexture(texHeaviside));
  perseusSafeCall(cudaDeviceSynchronize());
  perseusSafeCall(cudaThreadSynchronize());

  cudaMemcpy(cudaData->dfxResultTranslation, cudaData->dfxTranslation, blockSize.x * blockSize.y * sizeof(float3), cudaMemcpyDeviceToHost);
  cudaMemcpy(cudaData->dfxResultRotation, cudaData->dfxRotation, blockSize.x * blockSize.y * sizeof(float4), cudaMemcpyDeviceToHost);

  for (i=0; i<7; i++) dpose[i] = 0;

  for (size_t i=0; i < blockSize.x * blockSize.y; i++)
  {
    dpose[0] += cudaData->dfxResultTranslation[i].x; dpose[1] += cudaData->dfxResultTranslation[i].y; dpose[2] += cudaData->dfxResultTranslation[i].z;
    dpose[3] += cudaData->dfxResultRotation[i].x; dpose[4] += cudaData->dfxResultRotation[i].y;
    dpose[5] += cudaData->dfxResultRotation[i].z; dpose[6] += cudaData->dfxResultRotation[i].w;
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

    int icX, icY, icZ;
    int greyPixel, currentHistogram;
    int hidx, pidx;

    float2 histogramPixel;
    float pYB, pYF;
    float xProjected[4], xUnprojected[4], xUnrotated[4];
    float dfPPGeneric, dirac, heaviside;
    float otherInfo[2];

    float precalcX, precalcY, precalcXY;

    float dtIdx, norm;

    if (dtPosY[offset] >= 0)// && imageRegistered[offset].w > 128)
    {
      dtIdx = dt[offset];

      icX = offsetX; icY = offsetY;
      if (dtIdx < 0) { icX = dtPosX[offset]; icY = dtPosY[offset]; }
      icZ = icX + icY * widthROI;

      if (!isMultiobject || (isMultiobject && (imageObjects[icZ] - 1) == objectId &&
                             ((imageObjects[offsetX + offsetY * widthROI] - 1) == objectId || (imageObjects[offsetX + offsetY * widthROI] - 1) == -1 )))
      {
        hidx = 4096 + 512 * dtIdx;
        if (hidx >= 0 && hidx < 8192)
        {
          heaviside = tex1D(texHeaviside, hidx);

          imagePixel = imageRegistered[offset];
          greyPixel = int(float(imagePixel.x) * 0.3f + float(imagePixel.y) * 0.59f + float(imagePixel.z) * 0.11f);

          currentHistogram = 0;
          if (greyPixel < 128) currentHistogram = 3;
          else if (greyPixel < 192) currentHistogram = 2;
          else if (greyPixel < 224) currentHistogram = 1;

          //currentHistogram = 2;

          imagePixel.x = (imagePixel.x >> histFactors[currentHistogram]) & (histNoBins[currentHistogram] - 1);
          imagePixel.y = (imagePixel.y >> histFactors[currentHistogram]) & (histNoBins[currentHistogram] - 1);
          imagePixel.z = (imagePixel.z >> histFactors[currentHistogram]) & (histNoBins[currentHistogram] - 1);
          pidx = (imagePixel.x + imagePixel.y * histNoBins[currentHistogram]) * histNoBins[currentHistogram] + imagePixel.z;

          histogramPixel = histogram[histOffsets[currentHistogram] + pidx];

          pYF = histogramPixel.x + 0.0000001f; pYB = histogramPixel.y + 0.0000001f;

          dirac = (1.0f / float(PI)) * (1 / (dtIdx * dtIdx + 1.0f) + float(1e-3));
          dfPPGeneric = dirac * (pYF - pYB) / (heaviside * (pYF - pYB) + pYB);

          xProjected[0] = 2 * (icX + minX - (float) viewTransform[0]) / (float) viewTransform[2] - 1;
          xProjected[1] = 2 * (icY + minY - (float) viewTransform[1]) / (float) viewTransform[3] - 1;
          xProjected[2] = 2 * ((float)imageZBuffer[icZ] / (float)MAX_INT) - 1;
          xProjected[3] = 1;

          xUnprojected[0] = invP[0] * xProjected[0] + invP[4] * xProjected[1] + invP[8] * xProjected[2] + invP[12] * xProjected[3];
          xUnprojected[1] = invP[1] * xProjected[0] + invP[5] * xProjected[1] + invP[9] * xProjected[2] + invP[13] * xProjected[3];
          xUnprojected[2] = invP[2] * xProjected[0] + invP[6] * xProjected[1] + invP[10] * xProjected[2] + invP[14] * xProjected[3];
          xUnprojected[3] = invP[3] * xProjected[0] + invP[7] * xProjected[1] + invP[11] * xProjected[2] + invP[15] * xProjected[3];
          norm = 1.0f/xUnprojected[3]; xUnprojected[0] *= norm; xUnprojected[1] *= norm; xUnprojected[2] *= norm; xUnprojected[3] *= norm;

          xUnrotated[0] = invPM[0] * xProjected[0] + invPM[4] * xProjected[1] + invPM[8] * xProjected[2] + invPM[12] * xProjected[3];
          xUnrotated[1] = invPM[1] * xProjected[0] + invPM[5] * xProjected[1] + invPM[9] * xProjected[2] + invPM[13] * xProjected[3];
          xUnrotated[2] = invPM[2] * xProjected[0] + invPM[6] * xProjected[1] + invPM[10] * xProjected[2] + invPM[14] * xProjected[3];
          xUnrotated[3] = invPM[3] * xProjected[0] + invPM[7] * xProjected[1] + invPM[11] * xProjected[2] + invPM[15] * xProjected[3];
          norm = 1.0f/xUnrotated[3];	xUnrotated[0] *= norm; xUnrotated[1] *= norm; xUnrotated[2] *= norm; xUnrotated[3] *= norm;

          otherInfo[0] = projectionParams[0] * dtDX[offset]; otherInfo[1] = projectionParams[1] * dtDY[offset];

          precalcXY = xUnprojected[2] * xUnprojected[2];

          dfPPTranslation.x = -otherInfo[0] / xUnprojected[2];
          dfPPTranslation.y = -otherInfo[1] / xUnprojected[2];
          dfPPTranslation.z = (otherInfo[0] * xUnprojected[0] + otherInfo[1] * xUnprojected[1]) / precalcXY;

          precalcX = -otherInfo[0] / precalcXY; precalcY = -otherInfo[1] / precalcXY;

          dfPPRotation.x =
              precalcX * (xUnprojected[2] * (q[1]*xUnrotated[1] + q[2]*xUnrotated[2]) -
              xUnprojected[0] * (q[2]*xUnrotated[0] + q[3]*xUnrotated[1] - 2*q[0]*xUnrotated[2])) +
              precalcY * (xUnprojected[2] * (q[1]*xUnrotated[0] - 2*q[0]*xUnrotated[1] - q[3]*xUnrotated[2]) -
              xUnprojected[1] * (q[2]*xUnrotated[0] + q[3]*xUnrotated[1] - 2*q[0]*xUnrotated[2]));

          dfPPRotation.y =
              precalcX * (xUnprojected[2] * (q[0]*xUnrotated[1] - 2*q[1]*xUnrotated[0] + q[3]*xUnrotated[2]) -
              xUnprojected[0] * (q[2]*xUnrotated[1] - q[3]*xUnrotated[0] - 2*q[1]*xUnrotated[2])) +
              precalcY * (xUnprojected[2] * (q[0]*xUnrotated[0] + q[2]*xUnrotated[2]) -
              xUnprojected[1] * (q[2]*xUnrotated[1] - q[3]*xUnrotated[0] - 2*q[1]*xUnrotated[2]));

          dfPPRotation.z =
              precalcX * (xUnprojected[2] * (q[0]*xUnrotated[2] - q[3]*xUnrotated[1] - 2*q[2]*xUnrotated[0]) -
              xUnprojected[0] * (q[0]*xUnrotated[0] + q[1]*xUnrotated[1])) +
              precalcY * (xUnprojected[2] * (q[3]*xUnrotated[0] - 2*q[2]*xUnrotated[1] + q[1]*xUnrotated[2]) -
              xUnprojected[1] * (q[0]*xUnrotated[0] + q[1]*xUnrotated[1]));

          dfPPRotation.w =
              precalcX * (xUnprojected[2] * (q[1]*xUnrotated[2] - q[2]*xUnrotated[1]) -
              xUnprojected[0] * (q[0]*xUnrotated[1] - q[1]*xUnrotated[0])) +
              precalcY * (xUnprojected[2] * (q[2]*xUnrotated[0] - q[0]*xUnrotated[2]) -
              xUnprojected[1] * (q[0]*xUnrotated[1] - q[1]*xUnrotated[0]));

          xProjected[0] = 2 * (icX + minX - (float) viewTransform[0]) / (float) viewTransform[2] - 1;
          xProjected[1] = 2 * (icY + minY - (float) viewTransform[1]) / (float) viewTransform[3] - 1;
          xProjected[2] = 2 * ((float)imageZBufferInverse[icZ] / (float)MAX_INT) - 1;
          xProjected[3] = 1;

          xUnprojected[0] = invP[0] * xProjected[0] + invP[4] * xProjected[1] + invP[8] * xProjected[2] + invP[12] * xProjected[3];
          xUnprojected[1] = invP[1] * xProjected[0] + invP[5] * xProjected[1] + invP[9] * xProjected[2] + invP[13] * xProjected[3];
          xUnprojected[2] = invP[2] * xProjected[0] + invP[6] * xProjected[1] + invP[10] * xProjected[2] + invP[14] * xProjected[3];
          xUnprojected[3] = invP[3] * xProjected[0] + invP[7] * xProjected[1] + invP[11] * xProjected[2] + invP[15] * xProjected[3];
          norm = 1.0f/xUnprojected[3]; xUnprojected[0] *= norm; xUnprojected[1] *= norm; xUnprojected[2] *= norm; xUnprojected[3] *= norm;

          xUnrotated[0] = invPM[0] * xProjected[0] + invPM[4] * xProjected[1] + invPM[8] * xProjected[2] + invPM[12] * xProjected[3];
          xUnrotated[1] = invPM[1] * xProjected[0] + invPM[5] * xProjected[1] + invPM[9] * xProjected[2] + invPM[13] * xProjected[3];
          xUnrotated[2] = invPM[2] * xProjected[0] + invPM[6] * xProjected[1] + invPM[10] * xProjected[2] + invPM[14] * xProjected[3];
          xUnrotated[3] = invPM[3] * xProjected[0] + invPM[7] * xProjected[1] + invPM[11] * xProjected[2] + invPM[15] * xProjected[3];
          norm = 1.0f/xUnrotated[3]; xUnrotated[0] *= norm; xUnrotated[1] *= norm; xUnrotated[2] *= norm; xUnrotated[3] *= norm;

          precalcXY = xUnprojected[2] * xUnprojected[2];

          dfPPTranslation.x += -otherInfo[0] / xUnprojected[2];
          dfPPTranslation.y += -otherInfo[1] / xUnprojected[2];
          dfPPTranslation.z += (otherInfo[0] * xUnprojected[0] + otherInfo[1] * xUnprojected[1]) / precalcXY;

          precalcX = -otherInfo[0] / precalcXY; precalcY = -otherInfo[1] / precalcXY;

          dfPPRotation.x +=
              precalcX * (xUnprojected[2] * (q[1]*xUnrotated[1] + q[2]*xUnrotated[2]) -
              xUnprojected[0] * (q[2]*xUnrotated[0] + q[3]*xUnrotated[1] - 2*q[0]*xUnrotated[2])) +
              precalcY * (xUnprojected[2] * (q[1]*xUnrotated[0] - 2*q[0]*xUnrotated[1] - q[3]*xUnrotated[2]) -
              xUnprojected[1] * (q[2]*xUnrotated[0] + q[3]*xUnrotated[1] - 2*q[0]*xUnrotated[2]));

          dfPPRotation.y +=
              precalcX * (xUnprojected[2] * (q[0]*xUnrotated[1] - 2*q[1]*xUnrotated[0] + q[3]*xUnrotated[2]) -
              xUnprojected[0] * (q[2]*xUnrotated[1] - q[3]*xUnrotated[0] - 2*q[1]*xUnrotated[2])) +
              precalcY * (xUnprojected[2] * (q[0]*xUnrotated[0] + q[2]*xUnrotated[2]) -
              xUnprojected[1] * (q[2]*xUnrotated[1] - q[3]*xUnrotated[0] - 2*q[1]*xUnrotated[2]));

          dfPPRotation.z +=
              precalcX * (xUnprojected[2] * (q[0]*xUnrotated[2] - q[3]*xUnrotated[1] - 2*q[2]*xUnrotated[0]) -
              xUnprojected[0] * (q[0]*xUnrotated[0] + q[1]*xUnrotated[1])) +
              precalcY * (xUnprojected[2] * (q[3]*xUnrotated[0] - 2*q[2]*xUnrotated[1] + q[1]*xUnrotated[2]) -
              xUnprojected[1] * (q[0]*xUnrotated[0] + q[1]*xUnrotated[1]));

          dfPPRotation.w +=
              precalcX * (xUnprojected[2] * (q[1]*xUnrotated[2] - q[2]*xUnrotated[1]) -
              xUnprojected[0] * (q[0]*xUnrotated[1] - q[1]*xUnrotated[0])) +
              precalcY * (xUnprojected[2] * (q[2]*xUnrotated[0] - q[0]*xUnrotated[2]) -
              xUnprojected[1] * (q[0]*xUnrotated[1] - q[1]*xUnrotated[0]));

          dfPPTranslation.x *= dfPPGeneric; dfPPTranslation.y *= dfPPGeneric; dfPPTranslation.z *= dfPPGeneric;
          dfPPRotation.x *= dfPPGeneric; dfPPRotation.y *= dfPPGeneric;
          dfPPRotation.z *= dfPPGeneric; dfPPRotation.w *= dfPPGeneric;

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
