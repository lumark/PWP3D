#include "CUDAConvolution.h"
#include "CUDAUtils.h"
#include "CUDAData.h"

extern CUDAData* cudaData;

__device__ __constant__ float dKernelConvolution[KERNEL_WIDTH];
const int KERNEL_SIZE = KERNEL_WIDTH * sizeof(float);

// Loop unrolling templates, needed for best performance
template<int i> __device__ float convolutionRowT(float *data){return data[KERNEL_RADIUS-i]*dKernelConvolution[i]+convolutionRowT<i-1>(data);}
template<> __device__ float convolutionRowT<-1>(float *data){return 0;}
template<int i> __device__ float convolutionColT(float *data){return data[(KERNEL_RADIUS-i)*COLUMN_TILE_WIDTH]*dKernelConvolution[i]+convolutionColT<i-1>(data);}
template<> __device__ float convolutionColT<-1>(float *data){return 0;}

__global__ void convolutionRow(float *d_Result, float *d_Data, int dataW, int dataH)
{
  const int rowStart = IMUL(blockIdx.y, dataW);

  __shared__ float data[KERNEL_RADIUS + ROW_TILE_WIDTH + KERNEL_RADIUS];

  const int tileStart = IMUL(blockIdx.x, ROW_TILE_WIDTH);
  const int tileEnd = tileStart + ROW_TILE_WIDTH - 1;
  const int apronStart = tileStart - KERNEL_RADIUS;
  const int apronEnd = tileEnd + KERNEL_RADIUS;

  const int tileEndClamped = min(tileEnd, dataW - 1);
  const int apronStartClamped = max(apronStart, 0);
  const int apronEndClamped = min(apronEnd, dataW - 1);

  const int apronStartAligned = tileStart - KERNEL_RADIUS_ALIGNED;

  const int loadPos = apronStartAligned + threadIdx.x;

  if(loadPos >= apronStart)
  {
    const int smemPos = loadPos - apronStart;
    data[smemPos] = ((loadPos >= apronStartClamped) && (loadPos <= apronEndClamped)) ? d_Data[rowStart + loadPos] : 0;
  }

  __syncthreads();
  const int writePos = tileStart + threadIdx.x;

  if(writePos <= tileEndClamped)
  {
    const int smemPos = writePos - apronStart;
    float sum = 0;
    sum = convolutionRowT<2 * KERNEL_RADIUS>(data + smemPos);
    d_Result[rowStart + writePos] = sum;
  }
}

__global__ void convolutionColumn(float *d_Result, float *d_Data, int dataW, int dataH, int smemStride, int gmemStride)
{
  const int columnStart = IMUL(blockIdx.x, COLUMN_TILE_WIDTH) + threadIdx.x;

  __shared__ float data[COLUMN_TILE_WIDTH * (KERNEL_RADIUS + COLUMN_TILE_HEIGHT + KERNEL_RADIUS)];

  const int tileStart = IMUL(blockIdx.y, COLUMN_TILE_HEIGHT);
  const int tileEnd = tileStart + COLUMN_TILE_HEIGHT - 1;
  const int apronStart = tileStart - KERNEL_RADIUS;
  const int apronEnd = tileEnd   + KERNEL_RADIUS;

  const int tileEndClamped = min(tileEnd, dataH - 1);
  const int apronStartClamped = max(apronStart, 0);
  const int apronEndClamped = min(apronEnd, dataH - 1);

  int smemPos = IMUL(threadIdx.y, COLUMN_TILE_WIDTH) + threadIdx.x;
  int gmemPos = IMUL(apronStart + threadIdx.y, dataW) + columnStart;

  for(int y = apronStart + threadIdx.y; y <= apronEnd; y += blockDim.y)
  {
    data[smemPos] = ((y >= apronStartClamped) && (y <= apronEndClamped)) ?  d_Data[gmemPos] : 0;
    smemPos += smemStride;
    gmemPos += gmemStride;
  }

  __syncthreads();

  smemPos = IMUL(threadIdx.y + KERNEL_RADIUS, COLUMN_TILE_WIDTH) + threadIdx.x;
  gmemPos = IMUL(tileStart + threadIdx.y , dataW) + columnStart;

  for(int y = tileStart + threadIdx.y; y <= tileEndClamped; y += blockDim.y)
  {
    float sum = 0;
    sum = convolutionColT<2 * KERNEL_RADIUS>(data + smemPos);
    d_Result[gmemPos] = sum;
    smemPos += smemStride;
    gmemPos += gmemStride;
  }
}

__host__ void initialiseConvolution(int width, int height)
{
  cudaData->hKernelConvolution = (float *)malloc(KERNEL_SIZE);
  cudaData->hKernelConvolution[0] = 0.5f;
  cudaData->hKernelConvolution[1] = 0;
  cudaData->hKernelConvolution[2] = -0.5f;

  perseusSafeCall(cudaMemcpyToSymbol(dKernelConvolution, cudaData->hKernelConvolution, KERNEL_SIZE));
}
__host__ void shutdownConvolution()
{
  free(cudaData->hKernelConvolution);
}
__host__ void computeDerivativeXY(float* function, float* derivativeX, float* derivativeY, int width, int height)
{
  dim3 blockGridRows = dim3(iDivUp(width, ROW_TILE_WIDTH), height);
  dim3 blockGridColumns = dim3(iDivUp(width, COLUMN_TILE_WIDTH), iDivUp(height, COLUMN_TILE_HEIGHT));
  dim3 threadBlockRows = dim3(KERNEL_RADIUS_ALIGNED + ROW_TILE_WIDTH + KERNEL_RADIUS);
  dim3 threadBlockColumns = dim3(COLUMN_TILE_WIDTH, 8);

  convolutionRow<<<blockGridRows, threadBlockRows>>> (derivativeX, function, width, height);
  convolutionColumn<<<blockGridColumns, threadBlockColumns>>>( derivativeY, function, width, height,
                                                               COLUMN_TILE_WIDTH * threadBlockColumns.y, width * threadBlockColumns.y);
}
