//#include "CUDAScharr.h"
//#include "CUDAData.h"
//#include <thrust/detail/minmax.h>

//texture<unsigned char, 2, cudaReadModeElementType> texScharr;

//extern CUDAData* cudaData;

//__host__ void initialiseScharr(int width, int height)
//{
//  //TODO MAX WIDTH ROI
//  int maxWidthROI = width, maxHeightROI = height;

//  cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();
//  perseusSafeCall(cudaMallocArray(&cudaData->arrayScharr, &desc, maxWidthROI, maxHeightROI));
//}
//__host__ void shutdownScharr()
//{
//  perseusSafeCall(cudaFreeArray(cudaData->arrayScharr));
//}

//__host__ __device__ unsigned char
//computeScharrPP(unsigned char ul, unsigned char um, unsigned char ur, unsigned char ml, unsigned char mm, unsigned char mr, unsigned char ll,
//                unsigned char lm, unsigned char lr, unsigned char valm, unsigned char valok, float fScale)
//{
//  short Horz = 3*ur + 10*mr + 3*lr - 3*ul - 10*ml - 3*ll;
//  short Vert = 3*ul + 10*um + 3*ur - 3*ll - 10*lm - 3*lr;
//  short Sum = (short) ((fabsf(Horz) + fabsf(Vert))); //was fScale * abs..

//  Sum = max(0, Sum);
//  Sum = min(0xFF, Sum);

//  Sum = (Sum != 0 && mm == valm && (mm == valok || valok == 0)) ? 254 : 0;

//  return (unsigned char) Sum;
//}

//__host__ __device__ unsigned char
//computeSihluettePP(unsigned char ul, unsigned char um, unsigned char ur, unsigned char ml, unsigned char mm, unsigned char mr,
//                   unsigned char ll, unsigned char lm, unsigned char lr, unsigned char valm, unsigned char valok, float fScale)
//{
//  short Horz = 3*ur + 10*mr + 3*lr - 3*ul - 10*ml - 3*ll;
//  short Vert = 3*ul + 10*um + 3*ur - 3*ll - 10*lm - 3*lr;
//  short Sum = (short) ((fabsf(Horz) + fabsf(Vert))); //was fScale * abs..
//  bool bAnd = (ul == 0) || (um == 0) || (ur == 0) || (ml == 0) || (mr == 0) || (ll == 0) || (lm == 0) || (lr == 0) ;

//  Sum = fmaxf(0, Sum);
//  Sum = fminf(0xFF, Sum);

//  Sum = (Sum != 0 && mm != valm && bAnd) ? 254 : 0;

//  return (unsigned char) Sum;
//}


//__global__ void sihluetteTex( unsigned char *pScharrOriginal, unsigned int Pitch, int w, int h, float fScale )
//{
//  int i;
//  unsigned char val;

//  unsigned char *pSihluette = pScharrOriginal + blockIdx.x*Pitch;
//  for (i = threadIdx.x; i < w; i += blockDim.x)
//  {
//    unsigned char pix00 = tex2D(texScharr, (float) i-1, (float) blockIdx.x-1);
//    unsigned char pix01 = tex2D(texScharr, (float) i+0, (float) blockIdx.x-1);
//    unsigned char pix02 = tex2D(texScharr, (float) i+1, (float) blockIdx.x-1);
//    unsigned char pix10 = tex2D(texScharr, (float) i-1, (float) blockIdx.x+0);
//    unsigned char pix11 = tex2D(texScharr, (float) i+0, (float) blockIdx.x+0);
//    unsigned char pix12 = tex2D(texScharr, (float) i+1, (float) blockIdx.x+0);
//    unsigned char pix20 = tex2D(texScharr, (float) i-1, (float) blockIdx.x+1);
//    unsigned char pix21 = tex2D(texScharr, (float) i+0, (float) blockIdx.x+1);
//    unsigned char pix22 = tex2D(texScharr, (float) i+1, (float) blockIdx.x+1);

//    val = computeSihluettePP(pix00, pix01, pix02, pix10, pix11, pix12, pix20, pix21, pix22, 0, 0, fScale);
//    pSihluette[i] = val;
//  }
//}

//__global__ void scharrTex( unsigned char *pScharrOriginal, unsigned int Pitch, int w, int h, float fScale )
//{
//  int i;

//  unsigned char pix00, pix01, pix02, pix10, pix11, pix12, pix20, pix21, pix22;
//  unsigned char pixMax1, pixMax2, pixMax3, pixMax4;

//  unsigned char *pScharr = pScharrOriginal + blockIdx.x*Pitch;
//  for (i = threadIdx.x; i < w; i += blockDim.x)
//  {
//    pix00 = tex2D(texScharr, (float) i-1, (float) blockIdx.x-1);
//    pix01 = tex2D(texScharr, (float) i+0, (float) blockIdx.x-1);
//    pix02 = tex2D(texScharr, (float) i+1, (float) blockIdx.x-1);
//    pix10 = tex2D(texScharr, (float) i-1, (float) blockIdx.x+0);
//    pix11 = tex2D(texScharr, (float) i+0, (float) blockIdx.x+0);
//    pix12 = tex2D(texScharr, (float) i+1, (float) blockIdx.x+0);
//    pix20 = tex2D(texScharr, (float) i-1, (float) blockIdx.x+1);
//    pix21 = tex2D(texScharr, (float) i+0, (float) blockIdx.x+1);
//    pix22 = tex2D(texScharr, (float) i+1, (float) blockIdx.x+1);

//    pixMax1 = max(pix00, pix01);
//    pixMax2 = max(pix02, pix10);

//    pixMax3 = max(pix12, pix20);
//    pixMax4 = max(pix21, pix22);

//    pixMax1 = max(pixMax1, pixMax3);
//    pixMax2 = max(pixMax2, pixMax4);

//    pixMax1 = max(pixMax1, pixMax2);

//    pScharr[i] = computeScharrPP(pix00, pix01, pix02, pix10, pix11, pix12, pix20, pix21, pix22, pixMax1, 0, fScale);
//  }
//}

//__host__ void computeScharr(unsigned char *originalImage, unsigned char *scharrImage, int w, int h, float fScale)
//{
//  perseusSafeCall(cudaMemcpy2DToArray(cudaData->arrayScharr, 0, 0, originalImage, w, w, h, cudaMemcpyDeviceToDevice));
//  perseusSafeCall(cudaBindTextureToArray(texScharr, cudaData->arrayScharr));
//  perseusSafeCall(cudaMemset(scharrImage, 0, sizeof(unsigned char)*w*h));

//  scharrTex<<<h, 384>>>(scharrImage, w, w, h, fScale);

//  perseusSafeCall(cudaUnbindTexture(texScharr));
//}

//__host__ void computeSihluette(unsigned char *originalImage, unsigned char *sihlutteImage, int w, int h, float fScale)
//{
//  perseusSafeCall(cudaMemcpy2DToArray(cudaData->arrayScharr, 0, 0, originalImage, w, w, h, cudaMemcpyDeviceToDevice));
//  perseusSafeCall(cudaBindTextureToArray(texScharr, cudaData->arrayScharr));
//  perseusSafeCall(cudaMemset(sihlutteImage, 0, sizeof(unsigned char)*w*h));

//  sihluetteTex<<<h, 384>>>(sihlutteImage, w, w, h, fScale);

//  perseusSafeCall(cudaUnbindTexture(texScharr));
//}


#include "CUDAScharr.h"
#include "CUDAData.h"


#ifndef sabs

#define sabs( a ) ( ((a) > 0) ? (a) : (-a) )

#endif



texture<unsigned char, 2, cudaReadModeElementType> texScharr;

extern CUDAData* cudaData;

__host__ void initialiseScharr(int width, int height)
{
  //TODO MAX WIDTH ROI
  int maxWidthROI = width, maxHeightROI = height;

  cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();
  perseusSafeCall(cudaMallocArray(&cudaData->arrayScharr, &desc, maxWidthROI, maxHeightROI));
}
__host__ void shutdownScharr()
{
  perseusSafeCall(cudaFreeArray(cudaData->arrayScharr));
}

__host__ __device__ unsigned char
computeScharrPP(unsigned char ul, unsigned char um, unsigned char ur, unsigned char ml, unsigned char mm, unsigned char mr, unsigned char ll,
                unsigned char lm, unsigned char lr, unsigned char valm, unsigned char valok, float fScale)
{
  short Horz = 3*ur + 10*mr + 3*lr - 3*ul - 10*ml - 3*ll;
  short Vert = 3*ul + 10*um + 3*ur - 3*ll - 10*lm - 3*lr;
  short Sum = (short) ((sabs(Horz) + sabs(Vert))); //was fScale * abs..

  Sum = max(0, Sum);
  Sum = min(0xFF, Sum);

  Sum = (Sum != 0 && mm == valm && (mm == valok || valok == 0)) ? 254 : 0;

  return (unsigned char) Sum;
}

__host__ __device__ unsigned char
computeSihluettePP(unsigned char ul, unsigned char um, unsigned char ur, unsigned char ml, unsigned char mm, unsigned char mr,
                   unsigned char ll, unsigned char lm, unsigned char lr, unsigned char valm, unsigned char valok, float fScale)
{
  short Horz = 3*ur + 10*mr + 3*lr - 3*ul - 10*ml - 3*ll;
  short Vert = 3*ul + 10*um + 3*ur - 3*ll - 10*lm - 3*lr;
  short Sum = (short) ((sabs(Horz) + sabs(Vert))); //was fScale * abs..
  bool bAnd = (ul == 0) || (um == 0) || (ur == 0) || (ml == 0) || (mr == 0) || (ll == 0) || (lm == 0) || (lr == 0);

  Sum = max(0, Sum);
  Sum = min(0xFF, Sum);

  Sum = (Sum != 0 && mm != valm && bAnd) ? 254 : 0;

  return (unsigned char) Sum;
}


__global__ void sihluetteTex( unsigned char *pScharrOriginal, unsigned int Pitch, int w, int h, float fScale )
{
  int i;
  unsigned char val;

  unsigned char *pSihluette = pScharrOriginal + blockIdx.x*Pitch;
  for (i = threadIdx.x; i < w; i += blockDim.x)
  {
    unsigned char pix00 = tex2D(texScharr, (float) i-1, (float) blockIdx.x-1);
    unsigned char pix01 = tex2D(texScharr, (float) i+0, (float) blockIdx.x-1);
    unsigned char pix02 = tex2D(texScharr, (float) i+1, (float) blockIdx.x-1);
    unsigned char pix10 = tex2D(texScharr, (float) i-1, (float) blockIdx.x+0);
    unsigned char pix11 = tex2D(texScharr, (float) i+0, (float) blockIdx.x+0);
    unsigned char pix12 = tex2D(texScharr, (float) i+1, (float) blockIdx.x+0);
    unsigned char pix20 = tex2D(texScharr, (float) i-1, (float) blockIdx.x+1);
    unsigned char pix21 = tex2D(texScharr, (float) i+0, (float) blockIdx.x+1);
    unsigned char pix22 = tex2D(texScharr, (float) i+1, (float) blockIdx.x+1);

    val = computeSihluettePP(pix00, pix01, pix02, pix10, pix11, pix12, pix20, pix21, pix22, 0, 0, fScale);
    pSihluette[i] = val;
  }
}

__global__ void scharrTex( unsigned char *pScharrOriginal, unsigned int Pitch, int w, int h, float fScale )
{
  int i;

  unsigned char pix00, pix01, pix02, pix10, pix11, pix12, pix20, pix21, pix22;
  unsigned char pixMax1, pixMax2, pixMax3, pixMax4;

  unsigned char *pScharr = pScharrOriginal + blockIdx.x*Pitch;
  for (i = threadIdx.x; i < w; i += blockDim.x)
  {
    pix00 = tex2D(texScharr, (float) i-1, (float) blockIdx.x-1);
    pix01 = tex2D(texScharr, (float) i+0, (float) blockIdx.x-1);
    pix02 = tex2D(texScharr, (float) i+1, (float) blockIdx.x-1);
    pix10 = tex2D(texScharr, (float) i-1, (float) blockIdx.x+0);
    pix11 = tex2D(texScharr, (float) i+0, (float) blockIdx.x+0);
    pix12 = tex2D(texScharr, (float) i+1, (float) blockIdx.x+0);
    pix20 = tex2D(texScharr, (float) i-1, (float) blockIdx.x+1);
    pix21 = tex2D(texScharr, (float) i+0, (float) blockIdx.x+1);
    pix22 = tex2D(texScharr, (float) i+1, (float) blockIdx.x+1);

    pixMax1 = max(pix00, pix01);
    pixMax2 = max(pix02, pix10);

    pixMax3 = max(pix12, pix20);
    pixMax4 = max(pix21, pix22);

    pixMax1 = max(pixMax1, pixMax3);
    pixMax2 = max(pixMax2, pixMax4);

    pixMax1 = max(pixMax1, pixMax2);

    pScharr[i] = computeScharrPP(pix00, pix01, pix02, pix10, pix11, pix12, pix20, pix21, pix22, pixMax1, 0, fScale);
  }
}

__host__ void computeScharr(unsigned char *originalImage, unsigned char *scharrImage, int w, int h, float fScale)
{
  perseusSafeCall(cudaMemcpy2DToArray(cudaData->arrayScharr, 0, 0, originalImage, w, w, h, cudaMemcpyDeviceToDevice));
  perseusSafeCall(cudaBindTextureToArray(texScharr, cudaData->arrayScharr));
  perseusSafeCall(cudaMemset(scharrImage, 0, sizeof(unsigned char)*w*h));

  scharrTex<<<h, 384>>>(scharrImage, w, w, h, fScale);

  perseusSafeCall(cudaUnbindTexture(texScharr));
}

__host__ void computeSihluette(unsigned char *originalImage, unsigned char *sihlutteImage, int w, int h, float fScale)
{
  perseusSafeCall(cudaMemcpy2DToArray(cudaData->arrayScharr, 0, 0, originalImage, w, w, h, cudaMemcpyDeviceToDevice));
  perseusSafeCall(cudaBindTextureToArray(texScharr, cudaData->arrayScharr));
  perseusSafeCall(cudaMemset(sihlutteImage, 0, sizeof(unsigned char)*w*h));

  sihluetteTex<<<h, 384>>>(sihlutteImage, w, w, h, fScale);

  perseusSafeCall(cudaUnbindTexture(texScharr));
}
