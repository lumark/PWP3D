#include "CUDARenderer.h"
#include "CUDAUtils.h"
#include "CUDAData.h"

extern CUDAData* cudaData;

texture<float4, 2, cudaReadModeElementType> texRendererVertices;

__device__ __constant__ float pmMatrix[16];
__device__ __constant__ int viewTransformRender[4];

__host__ void initialiseRenderer(int width, int height)
{
  perseusSafeCall(cudaMalloc((void**)&cudaData->fill, width * height * sizeof(unsigned char)));
  perseusSafeCall(cudaMalloc((void**)&cudaData->objects, width * height * sizeof(unsigned char)));
  perseusSafeCall(cudaMalloc((void**)&cudaData->zbuffer, width * height * sizeof(unsigned int)));
  perseusSafeCall(cudaMalloc((void**)&cudaData->zbufferInverse, width * height * sizeof(unsigned int)));

  perseusSafeCall(cudaMalloc((void**)&cudaData->fillAll, width * height * sizeof(unsigned char)));
  perseusSafeCall(cudaMalloc((void**)&cudaData->objectsAll, width * height * sizeof(unsigned char)));
  perseusSafeCall(cudaMalloc((void**)&cudaData->zbufferAll, width * height * sizeof(unsigned int)));
  perseusSafeCall(cudaMalloc((void**)&cudaData->zbufferInverseAll, width * height * sizeof(unsigned int)));

  perseusSafeCall(cudaMalloc((void**)&cudaData->d_rois, 120000 * sizeof(int4))); //maximum 30720000 faces used in 16x16 blocks
  perseusSafeCall(cudaMallocHost((void**)&cudaData->h_rois, 120000 * sizeof(int4))); //maximum 30720000 faces used in 16x16 blocks

  perseusSafeCall(cudaMalloc((void**)&cudaData->d_roisAll, 120000 * sizeof(int4))); //maximum 30720000 faces used in 16x16 blocks
  perseusSafeCall(cudaMallocHost((void**)&cudaData->h_roisAll, 120000 * sizeof(int4))); //maximum 30720000 faces used in 16x16 blocks

  cudaData->descRendererVertices = cudaCreateChannelDesc<float4>();
}

__host__ void shutdownRenderer()
{
  perseusSafeCall(cudaFree(cudaData->fill));
  perseusSafeCall(cudaFree(cudaData->objects));
  perseusSafeCall(cudaFree(cudaData->zbuffer));
  perseusSafeCall(cudaFree(cudaData->zbufferInverse));

  perseusSafeCall(cudaFree(cudaData->fillAll));
  perseusSafeCall(cudaFree(cudaData->objectsAll));
  perseusSafeCall(cudaFree(cudaData->zbufferAll));
  perseusSafeCall(cudaFree(cudaData->zbufferInverseAll));

  perseusSafeCall(cudaFree(cudaData->d_rois));
  perseusSafeCall(cudaFreeHost(cudaData->h_rois));

  perseusSafeCall(cudaFree(cudaData->d_roisAll));
  perseusSafeCall(cudaFreeHost(cudaData->h_roisAll));
}


__host__ void renderObjectCUDA_one_EF(float4 *vertices, int faceCount, int objectId, float *h_pmMatrix, int *h_viewTransform, int widthRender, int heightRender)
{
  size_t texOffset;
  int i, roisSize;
  dim3 threads_in_block, blocks;

  threads_in_block = dim3(16,16);
  blocks = dim3(iDivUp(faceCount, 256));
  roisSize = iDivUp(faceCount, 256);

  perseusSafeCall(cudaMemcpyToSymbol(pmMatrix, h_pmMatrix, 16 * sizeof(float), 0, cudaMemcpyHostToDevice));
  perseusSafeCall(cudaMemcpyToSymbol(viewTransformRender, h_viewTransform, 4 * sizeof(int), 0, cudaMemcpyHostToDevice));

  perseusSafeCall(cudaMemset(cudaData->fill, 0, sizeof(unsigned char) * widthRender * heightRender));
  perseusSafeCall(cudaMemset(cudaData->objects, 0, sizeof(unsigned char) * widthRender * heightRender));
  perseusSafeCall(cudaMemset(cudaData->zbuffer, int(MAX_INT), sizeof(unsigned int) * widthRender * heightRender));
  perseusSafeCall(cudaMemset(cudaData->zbufferInverse, 0, sizeof(unsigned int) * widthRender * heightRender));

  cudaBindTexture2D(&texOffset, texRendererVertices, vertices, cudaData->descRendererVertices, 4, faceCount, 4 * sizeof(float4));

  renderObjectCUDA_EF_global<<<blocks, threads_in_block>>>(cudaData->fill, cudaData->objects, cudaData->zbuffer, cudaData->zbufferInverse,
                                                           cudaData->d_rois, faceCount, objectId, widthRender, heightRender);

  cudaUnbindTexture(texRendererVertices);

  perseusSafeCall(cudaMemcpy(cudaData->h_rois, cudaData->d_rois, sizeof(int4) * roisSize, cudaMemcpyDeviceToHost));
  for (i = 1; i<roisSize; i++)
  {
    cudaData->h_rois[0].x  = MIN(cudaData->h_rois[0].x, cudaData->h_rois[i].x);
    cudaData->h_rois[0].y  = MIN(cudaData->h_rois[0].y, cudaData->h_rois[i].y);
    cudaData->h_rois[0].z  = MAX(cudaData->h_rois[0].z, cudaData->h_rois[i].z);
    cudaData->h_rois[0].w  = MAX(cudaData->h_rois[0].w, cudaData->h_rois[i].w);
  }

  cudaData->roiGenerated[0] = cudaData->h_rois[0].x; cudaData->roiGenerated[1] = cudaData->h_rois[0].y;
  cudaData->roiGenerated[2] = cudaData->h_rois[0].z; cudaData->roiGenerated[3] = cudaData->h_rois[0].w;
  cudaData->roiGenerated[4] = cudaData->roiGenerated[2] - cudaData->roiGenerated[0] + 1;
  cudaData->roiGenerated[5] = cudaData->roiGenerated[3] - cudaData->roiGenerated[1] + 1;
}

__host__ void renderObjectCUDA_all_EF(float4 *vertices, int faceCount, int objectId, float *h_pmMatrix, int *h_viewTransform, int widthRender, int heightRender, bool clearData)
{
  size_t texOffset;
  int i, roisSize;
  dim3 threads_in_block, blocks;

  threads_in_block = dim3(16,16);
  blocks = dim3(iDivUp(faceCount, 256));
  roisSize = iDivUp(faceCount, 256);

  perseusSafeCall(cudaMemcpyToSymbol(pmMatrix, h_pmMatrix, 16 * sizeof(float), 0, cudaMemcpyHostToDevice));
  perseusSafeCall(cudaMemcpyToSymbol(viewTransformRender, h_viewTransform, 4 * sizeof(int), 0, cudaMemcpyHostToDevice));

  if (clearData)
  {
    perseusSafeCall(cudaMemset(cudaData->fillAll, 0, sizeof(unsigned char) * widthRender * heightRender));
    perseusSafeCall(cudaMemset(cudaData->objectsAll, 0, sizeof(unsigned char) * widthRender * heightRender));
    perseusSafeCall(cudaMemset(cudaData->zbufferAll, int(MAX_INT), sizeof(unsigned int) * widthRender * heightRender));
    perseusSafeCall(cudaMemset(cudaData->zbufferInverseAll, 0, sizeof(unsigned int) * widthRender * heightRender));
  }

  cudaBindTexture2D(&texOffset, texRendererVertices, vertices, cudaData->descRendererVertices, 4, faceCount, 4 * sizeof(float4));

  renderObjectCUDA_EF_global<<<blocks, threads_in_block>>>(cudaData->fillAll, cudaData->objectsAll, cudaData->zbufferAll, cudaData->zbufferInverseAll,
                                                           cudaData->d_roisAll, faceCount, objectId, widthRender, heightRender);

  cudaUnbindTexture(texRendererVertices);

  perseusSafeCall(cudaMemcpy(cudaData->h_roisAll, cudaData->d_roisAll, sizeof(int4) * roisSize, cudaMemcpyDeviceToHost));
  for (i = 1; i<roisSize; i++)
  {
    cudaData->h_roisAll[0].x  = MIN(cudaData->h_roisAll[0].x, cudaData->h_roisAll[i].x);
    cudaData->h_roisAll[0].y  = MIN(cudaData->h_roisAll[0].y, cudaData->h_roisAll[i].y);
    cudaData->h_roisAll[0].z  = MAX(cudaData->h_roisAll[0].z, cudaData->h_roisAll[i].z);
    cudaData->h_roisAll[0].w  = MAX(cudaData->h_roisAll[0].w, cudaData->h_roisAll[i].w);
  }

  if (clearData)
  {
    cudaData->roiGeneratedAll[0] = cudaData->h_roisAll[0].x; cudaData->roiGeneratedAll[1] = cudaData->h_roisAll[0].y;
    cudaData->roiGeneratedAll[2] = cudaData->h_roisAll[0].z; cudaData->roiGeneratedAll[3] = cudaData->h_roisAll[0].w;
  }
  else
  {
    cudaData->roiGeneratedAll[0] = MIN(cudaData->roiGenerated[0], cudaData->h_roisAll[0].x);
    cudaData->roiGeneratedAll[1] = MIN(cudaData->roiGenerated[1], cudaData->h_roisAll[0].y);
    cudaData->roiGeneratedAll[2] = MAX(cudaData->roiGenerated[2], cudaData->h_roisAll[0].z);
    cudaData->roiGeneratedAll[3] = MAX(cudaData->roiGenerated[3], cudaData->h_roisAll[0].w);
  }

  cudaData->roiGeneratedAll[4] = cudaData->roiGeneratedAll[2] - cudaData->roiGeneratedAll[0] + 1;
  cudaData->roiGeneratedAll[5] = cudaData->roiGeneratedAll[3] - cudaData->roiGeneratedAll[1] + 1;
}


__global__ void renderObjectCUDA_EF_global(unsigned char *fill, unsigned char *objects, unsigned int *zbuffer, unsigned int *zbufferInverse,
                                           int4 *rois, int faceCount, int objectId, int widthRender, int heightRender)
{
  __shared__ int4 sdataROI[256];

  int faceId = blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
  int offsetInBlock = threadIdx.x + blockDim.x * threadIdx.y;

  int sdataTargetOffset;

  sdataROI[offsetInBlock].x = 0xFFFF;
  sdataROI[offsetInBlock].y = 0xFFFF;
  sdataROI[offsetInBlock].z = -1;
  sdataROI[offsetInBlock].w = -1;

  if (faceId < faceCount)
  {
    int i;

    size_t index;

    unsigned int intZ, atomicVal;

    float norm;
    float4 buff1, buff2;
    float3 orderedPoints[3];
    float3 A, B, C;
    float3 S, E;
    float dx1, dx2, dx3, dz1, dz2, dz3, dxa, dxb, dza, dzb;
    float dzX, Sz, Sx, Sy, Ex;

    buff1 = tex2D(texRendererVertices, 0, faceId);
    buff2.x = pmMatrix[0] * buff1.x + pmMatrix[4] * buff1.y + pmMatrix[8] * buff1.z + pmMatrix[12] * buff1.w;
    buff2.y = pmMatrix[1] * buff1.x + pmMatrix[5] * buff1.y + pmMatrix[9] * buff1.z + pmMatrix[13] * buff1.w;
    buff2.z = pmMatrix[2] * buff1.x + pmMatrix[6] * buff1.y + pmMatrix[10] * buff1.z + pmMatrix[14] * buff1.w;
    buff2.w = pmMatrix[3] * buff1.x + pmMatrix[7] * buff1.y + pmMatrix[11] * buff1.z + pmMatrix[15] * buff1.w;
    norm = 1.0f / buff2.w;
    A.x = viewTransformRender[0] + viewTransformRender[2] * (buff2.x * norm + 1.0f) * 0.5f;
    A.y = viewTransformRender[1] + viewTransformRender[3] * (buff2.y * norm + 1.0f) * 0.5f;
    A.z = (buff2.z * norm + 1.0f) * 0.5f;

    buff1 = tex2D(texRendererVertices, 1, faceId);
    buff2.x = pmMatrix[0] * buff1.x + pmMatrix[4] * buff1.y + pmMatrix[8] * buff1.z + pmMatrix[12] * buff1.w;
    buff2.y = pmMatrix[1] * buff1.x + pmMatrix[5] * buff1.y + pmMatrix[9] * buff1.z + pmMatrix[13] * buff1.w;
    buff2.z = pmMatrix[2] * buff1.x + pmMatrix[6] * buff1.y + pmMatrix[10] * buff1.z + pmMatrix[14] * buff1.w;
    buff2.w = pmMatrix[3] * buff1.x + pmMatrix[7] * buff1.y + pmMatrix[11] * buff1.z + pmMatrix[15] * buff1.w;
    norm = 1.0f / buff2.w;
    B.x = viewTransformRender[0] + viewTransformRender[2] * (buff2.x * norm + 1.0f) * 0.5f;
    B.y = viewTransformRender[1] + viewTransformRender[3] * (buff2.y * norm + 1.0f) * 0.5f;
    B.z = (buff2.z * norm + 1.0f) * 0.5f;

    buff1 = tex2D(texRendererVertices, 2, faceId);
    buff2.x = pmMatrix[0] * buff1.x + pmMatrix[4] * buff1.y + pmMatrix[8] * buff1.z + pmMatrix[12] * buff1.w;
    buff2.y = pmMatrix[1] * buff1.x + pmMatrix[5] * buff1.y + pmMatrix[9] * buff1.z + pmMatrix[13] * buff1.w;
    buff2.z = pmMatrix[2] * buff1.x + pmMatrix[6] * buff1.y + pmMatrix[10] * buff1.z + pmMatrix[14] * buff1.w;
    buff2.w = pmMatrix[3] * buff1.x + pmMatrix[7] * buff1.y + pmMatrix[11] * buff1.z + pmMatrix[15] * buff1.w;
    norm = 1.0f / buff2.w;
    C.x = viewTransformRender[0] + viewTransformRender[2] * (buff2.x * norm + 1.0f) * 0.5f;
    C.y = viewTransformRender[1] + viewTransformRender[3] * (buff2.y * norm + 1.0f) * 0.5f;
    C.z = (buff2.z * norm + 1.0f) * 0.5f;

    sdataROI[offsetInBlock].x = MIN(A.x, B.x); sdataROI[offsetInBlock].y = MIN(A.y, B.y);
    sdataROI[offsetInBlock].z = MAX(A.x, B.x); sdataROI[offsetInBlock].w = MAX(A.y, B.y);

    sdataROI[offsetInBlock].x = MIN(sdataROI[offsetInBlock].x, C.x); sdataROI[offsetInBlock].y = MIN(sdataROI[offsetInBlock].y, C.y);
    sdataROI[offsetInBlock].z = MAX(sdataROI[offsetInBlock].z, C.x); sdataROI[offsetInBlock].w = MAX(sdataROI[offsetInBlock].w, C.y);

    if (A.y < B.y)
    {
      orderedPoints[0] = A; orderedPoints[1] = B; orderedPoints[2] = C;
      if (C.y < A.y) { orderedPoints[0] = C; orderedPoints[1] = A; orderedPoints[2] = B; }
      else if (C.y < B.y) { orderedPoints[0] = A; orderedPoints[1] = C; orderedPoints[2] = B; }
    }
    else
    {
      orderedPoints[0] = B; orderedPoints[1] = A;	orderedPoints[2] = C;
      if (C.y < B.y) { orderedPoints[0] = C; orderedPoints[1] = B; orderedPoints[2] = A; }
      else if (C.y < A.y) { orderedPoints[0] = B; orderedPoints[1] = C; orderedPoints[2] = A; }
    }

    A = orderedPoints[0]; B = orderedPoints[1]; C = orderedPoints[2];

    dx1 = (B.y - A.y) > 0 ? (B.x - A.x) / (B.y - A.y) : B.x - A.x;
    dx2 = (C.y - A.y) > 0 ? (C.x - A.x) / (C.y - A.y) : 0;
    dx3 = (C.y - B.y) > 0 ? (C.x - B.x) / (C.y - B.y) : 0;

    dz1 = (B.y - A.y) != 0 ? (B.z - A.z) / (B.y - A.y) : 0;
    dz2 = (C.y - A.y) != 0 ? (C.z - A.z) / (C.y - A.y) : 0;
    dz3 = (C.y - B.y) != 0 ? (C.z - B.z) / (C.y - B.y) : 0;

    S = E = A;

    B.y = floor(B.y - 0.5f); C.y = floor(C.y - 0.5f);

    if (dx1 > dx2) { dxa = dx2; dxb = dx1; dza = dz2; dzb = dz1; }
    else { dxa = dx1; dxb = dx2; dza = dz1; dzb = dz2; }

    for(; S.y <= B.y; S.y++, E.y++, S.x += dxa, E.x += dxb, S.z += dza, E.z += dzb)
    {
      dzX = (E.x != S.x) ? (E.z - S.z) / (E.x - S.x) : 0;
      Sz = S.z;

      Sy = CLAMP(S.y, 0, (VFLOAT) heightRender-1);
      Sx = CLAMP(S.x, 0, (VFLOAT) widthRender-1);
      Ex = CLAMP(E.x, 0, (VFLOAT) widthRender-1);

      for (i=(size_t)Sx; i<Ex; i++)
      {
        index = PIXELMATINDEX(i, Sy, widthRender);
        intZ = (unsigned int)(MAX_INT * Sz);

        atomicVal = atomicMin(&zbuffer[index], intZ);
        atomicMax(&zbufferInverse[index], intZ);
        if (atomicVal >= intZ) objects[index] = objectId + 1;

        Sz += dzX;
      }
    }

    if (dx1 > dx2) { dxa = dx2; dxb = dx3; dza = dz2; dzb = dz3; E = B; }
    else { dxa = dx3; dxb = dx2; dza = dz3; dzb = dz2; S = B; }

    for(; S.y <= C.y; S.y++, E.y++, S.x += dxa, E.x += dxb, S.z += dza, E.z += dzb)
    {
      dzX = (E.x != S.x) ? (E.z - S.z) / (E.x - S.x) : 0;
      Sz = S.z;

      Sy = CLAMP(S.y, 0, (VFLOAT) heightRender-1);
      Sx = CLAMP(S.x, 0, (VFLOAT) widthRender-1);
      Ex = CLAMP(E.x, 0, (VFLOAT) widthRender-1);

      for (i=(size_t)Sx; i<Ex; i++)
      {
        index = PIXELMATINDEX(i, Sy, widthRender);
        intZ = (unsigned int)(MAX_INT * Sz);

        atomicVal = atomicMin(&zbuffer[index], intZ);
        atomicMax(&zbufferInverse[index], intZ);
        if (atomicVal >= intZ) objects[index] = objectId + 1;

        Sz += dzX;
      }
    }
  }

  __syncthreads();

  for(unsigned int s = blockDim.x >> 1; s>0; s>>=1)
  {
    if (threadIdx.x < s)
    {
      sdataTargetOffset = (threadIdx.x + s) + blockDim.x * threadIdx.y;
      sdataROI[offsetInBlock].x = MIN(sdataROI[sdataTargetOffset].x, sdataROI[offsetInBlock].x);
      sdataROI[offsetInBlock].y = MIN(sdataROI[sdataTargetOffset].y, sdataROI[offsetInBlock].y);
      sdataROI[offsetInBlock].z = MAX(sdataROI[sdataTargetOffset].z, sdataROI[offsetInBlock].z);
      sdataROI[offsetInBlock].w = MAX(sdataROI[sdataTargetOffset].w, sdataROI[offsetInBlock].w);
    }
    __syncthreads();
  }

  for(unsigned int s = blockDim.y >> 1; s>0; s>>=1)
  {
    if (threadIdx.y < s)
    {
      sdataTargetOffset = threadIdx.x + blockDim.x * (threadIdx.y + s);
      sdataROI[offsetInBlock].x = MIN(sdataROI[sdataTargetOffset].x, sdataROI[offsetInBlock].x);
      sdataROI[offsetInBlock].y = MIN(sdataROI[sdataTargetOffset].y, sdataROI[offsetInBlock].y);
      sdataROI[offsetInBlock].z = MAX(sdataROI[sdataTargetOffset].z, sdataROI[offsetInBlock].z);
      sdataROI[offsetInBlock].w = MAX(sdataROI[sdataTargetOffset].w, sdataROI[offsetInBlock].w);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0 && threadIdx.y == 0)
  {
    int offsetROI = blockIdx.x + blockIdx.y * gridDim.x;
    rois[offsetROI] = sdataROI[offsetInBlock];
  }
}
