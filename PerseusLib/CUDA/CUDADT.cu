#include "CUDADT.h"
#include "CUDAUtils.h"
#include "CUDAData.h"

extern CUDAData* cudaData;

__host__ void initialiseDT(int width, int height)
{
  //TODO MAX WIDTH ROI
  int maxWidthROI = width, maxHeightROI = height;

  perseusSafeCall(cudaMalloc((void**)&cudaData->dtImageT1, maxWidthROI * maxHeightROI * sizeof(float)));
  perseusSafeCall(cudaMalloc((void**)&cudaData->dtZImage, (maxWidthROI+1) * maxHeightROI * sizeof(float)));
  perseusSafeCall(cudaMalloc((void**)&cudaData->dtVImage, maxWidthROI * maxHeightROI * sizeof(int)));
  perseusSafeCall(cudaMalloc((void**)&cudaData->dtImagePosYT1, maxWidthROI * maxHeightROI * sizeof(int)));
}

__host__ void shutdownDT()
{
  perseusSafeCall(cudaFree(cudaData->dtImageT1));
  perseusSafeCall(cudaFree(cudaData->dtZImage));
  perseusSafeCall(cudaFree(cudaData->dtVImage));
  perseusSafeCall(cudaFree(cudaData->dtImagePosYT1));
}

__global__ void dtToImage(float* dtImage, unsigned char* grayImage, int dtWidth, int dtHeight)
{
  int i;
  float* currentDTRow = dtImage + threadIdx.x * dtWidth;
  unsigned char* currentRow = grayImage + threadIdx.x * dtWidth;

  for (i=0;i<dtWidth;i++) currentRow[i] = sqrt((VFLOAT)currentDTRow[i]);
}

__host__ void convertDTToImage(unsigned char* image, float* imageTransform, int widthFull, int heightFull)
{
  dtToImage<<<1, widthFull>>>(imageTransform, image, widthFull, heightFull);
}

__host__ void processDT(float *dt, int *dtPosX, int *dtPosY, unsigned char *grayImage, unsigned char* signMask, int *roi, int bandSize)
{
  int widthROI, heightROI;
  widthROI = roi[4]; heightROI = roi[5];

  dim3 blocks(8);
  dim3 threads_in_block_t1(iDivUp(widthROI, 8));
  dim3 threads_in_block_t2(iDivUp(heightROI, 8));

  perseusSafeCall(cudaMemset(cudaData->dtImageT1, 0, (widthROI+16) * (heightROI+16) * sizeof(float)));
  perseusSafeCall(cudaMemset(dt, 0, (widthROI+16) * (heightROI+16) * sizeof(float)));
  perseusSafeCall(cudaMemset(dtPosX, -1, (widthROI+16) * (heightROI+16) * sizeof(int)));
  perseusSafeCall(cudaMemset(dtPosY, -1, (widthROI+16) * (heightROI+16) * sizeof(int)));

  processDTT1<<<blocks, threads_in_block_t1>>>(grayImage, cudaData->dtImageT1, dtPosX, cudaData->dtImagePosYT1,
                                               widthROI, heightROI, widthROI, roi[0], roi[1], roi[2], roi[3], bandSize);

  processDTT2<<<blocks, threads_in_block_t2>>>(dt, widthROI, heightROI, cudaData->dtImageT1,
                                               cudaData->dtZImage, cudaData->dtVImage, signMask, dtPosX, cudaData->dtImagePosYT1, dtPosY,
                                               widthROI, heightROI, roi[0], roi[1], roi[2], roi[3], bandSize);
}

__global__ void processDTT1(unsigned char* grayImage, float* dtImageT1, int* dtImagePosX, int* dtImagePosYT1, int dtWidth, 
                            int dtHeight, int dtWidthFull, int minxB, int minyB, int maxxB, int maxyB, int bandSize)
{
  int j, iOriginal;
  int prevZero = minyB;

  int columnIndexI, columnIndexJ, rowIndex, dtHeightOriginal, columnIndexMin;

  int offsetX = minxB + threadIdx.x + blockIdx.x * blockDim.x;

  if (offsetX >= maxxB)
    return;

  rowIndex = offsetX;
  dtHeightOriginal = dtHeight + minyB;
  columnIndexMin = COLUMN(rowIndex, minyB, dtWidthFull);

  for (iOriginal = minyB; iOriginal<dtHeightOriginal; iOriginal++)
  {
    columnIndexI = COLUMN(rowIndex, iOriginal, dtWidthFull);
    dtImageT1[columnIndexI] = INF_INT;
    dtImagePosYT1[columnIndexI] = INF_INT;

    if (grayImage[columnIndexI] != 0 || iOriginal == dtHeightOriginal - 1 || iOriginal == minyB)
    {
      if ((prevZero == minyB) && grayImage[columnIndexMin] == 0 && iOriginal != minyB && iOriginal != dtHeightOriginal - 1)
      {
        for (j = iOriginal; j >= prevZero; j--)
        {
          columnIndexJ = COLUMN(rowIndex, j, dtWidthFull);
          dtImageT1[columnIndexJ] = SQRi(iOriginal - j);
          dtImagePosYT1[columnIndexJ] = iOriginal;
        }
        prevZero = iOriginal;

        continue;
      }

      if (iOriginal == dtHeightOriginal - 1 && prevZero == minyB && grayImage[columnIndexMin] != 0)
      {
        for (j = iOriginal; j >= prevZero; j--)
        {
          columnIndexJ = COLUMN(rowIndex, j, dtWidthFull);
          dtImageT1[columnIndexJ] = SQRi(iOriginal - j);
          dtImagePosYT1[columnIndexJ] = iOriginal;
        }
        prevZero = iOriginal;

        continue;
      }

      if (iOriginal == dtHeightOriginal - 1 && grayImage[columnIndexI] == 0 && prevZero != minyB)
      {
        for (j = iOriginal; j >= prevZero; j--)
        {
          columnIndexJ = COLUMN(rowIndex, j, dtWidthFull);
          dtImageT1[columnIndexJ] = SQRi(j - prevZero);
          dtImagePosYT1[columnIndexJ] = prevZero;
        }
        prevZero = iOriginal;

        continue;
      }

      if (((iOriginal == dtHeightOriginal - 1) || (prevZero == minyB && iOriginal == minyB ))
          && grayImage[columnIndexI] == 0)
      {
        continue;
      }

      for (j = iOriginal; j >= iOriginal - (iOriginal - prevZero)/2; j--)
      {
        columnIndexJ = COLUMN(rowIndex, j, dtWidthFull);
        dtImageT1[columnIndexJ] = SQRi(iOriginal - j);
        dtImagePosYT1[columnIndexJ] = iOriginal;
      }

      for (j = prevZero; j < iOriginal - (iOriginal - prevZero)/2; j++)
      {
        columnIndexJ = COLUMN(rowIndex, j, dtWidthFull);
        dtImageT1[columnIndexJ] = SQRi(j - prevZero);
        dtImagePosYT1[columnIndexJ] = prevZero;
      }
      prevZero = iOriginal;
    }
  }
}

__global__ void processDTT2(float* dtImageT2, int dtWidthFull, int dtHeightFull, float* dtImageT1, float* zImage, int* vImage, 
                            unsigned char *signMask, int* dtImagePosX, int* dtImagePosYT1, int* dtImagePosYT2, int dtWidth, int dtHeight,
                            int minxB, int minyB, int maxxB, int maxyB, int bandSize)
{
  int offsetX = minxB;
  int offsetY = threadIdx.x + (blockIdx.x * blockDim.x) + minyB;

  if (offsetX >= maxxB || offsetY >= maxyB)
    return;

  int offset = offsetX + offsetY * dtWidthFull;

  float* f = dtImageT1 + offset;

  float* d = dtImageT2 + offset;
  float* z = zImage + offset;
  int* v = vImage + offset;

  int* fPos = dtImagePosYT1 + offset;
  int* posXRow = dtImagePosX + offset;
  int* posYRow = dtImagePosYT2 + offset;

  unsigned char* signMaskRow = signMask + offset;

  int k = 0, q, dq;
  int dist, bandSizeSquared = IMUL(bandSize, bandSize);

  float fqpqq, s;
  v[0] = 0;
  z[0] = -INF_INT;
  z[1] = +INF_INT;

  for (q = 0; q < dtWidth; q++)
  {
    if (f[q] != +INF_INT)
    {
      fqpqq = f[q] + SQRi(q);
      dq = 2 * q;

      s = (q == 0) ? (fqpqq - (f[v[k]] + SQRi(v[k]))) : __fdividef((fqpqq - (f[v[k]] + SQRi(v[k]))),(dq - IMUL(2, v[k])));

      while (s <= z[k])
      {
        k--;
        if (k < 0) break;
        s = (q == 0) ? (fqpqq - (f[v[k]] + SQRi(v[k]))) : __fdividef((fqpqq - (f[v[k]] + SQRi(v[k]))),(dq - IMUL(2, v[k])));
      }
      k++;
      v[k] = q;
      z[k] = s;
    }
    else
    {
      s = (f[v[k]] == +INF_INT) ? 0 : +INF_INT;

      k++;
      v[k] = q;
      z[k] = s;
      z[k+1] = +INF_INT;
    }
  }

  k = 0;
  for (q = 0; q < dtWidth; q++)
  {
    while (z[k+1] < q && (k+1) < dtWidth) k++;

    if (v[k] < dtWidth)
    {
      if (f[v[k]] < bandSizeSquared)
      {
        dist = SQRi(q - v[k]) + f[v[k]];
        if (dist < bandSizeSquared)
        {
          posXRow[q] = v[k] + minxB;
          posYRow[q] = fPos[v[k]];
          d[q] = (signMaskRow[q] == signMask[posXRow[q] + posYRow[q] * dtWidth]) ? sqrtf(dist) : -sqrtf(dist); //was != 0
        }
      }
    }
  }
}

__host__ void getDT(float* dtROI, float *dt, int *roi, int widthFull, int heightFull)
{
  memset(dt, 0, widthFull * heightFull * sizeof(float));

  perseusSafeCall(cudaMemcpy2D(dt + roi[0] + roi[1] * widthFull, widthFull * sizeof(float),
      dtROI, roi[4] * sizeof(float), roi[4] * sizeof(float), roi[5], cudaMemcpyDeviceToDevice));
}
