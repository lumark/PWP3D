#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  include <windows.h>
#endif

#ifndef perseusSafeCall
#define perseusSafeCall(err) __perseusSafeCall(err, __FILE__, __LINE__)

inline void __perseusSafeCall( cudaError err, const char *file, const int line )
{
  if( cudaSuccess != err) {
    printf("%s(%i) : cudaSafeCall() Runtime API error : %s.\n",
           file, line, cudaGetErrorString(err) );
    exit(-1);
  }
}

#endif

#ifndef HISTOGRAM_NO_BINS
#define HISTOGRAM_NO_BINS 32
#endif

#ifndef EXECUTYIN512THREADS
#define EXECUTYIN512THREADS(counter, startPoint, func, params) \
  startPoint = 0;\
  if (counter / 512 > 0) \
{ \
  while (counter / 512 > 0) \
{ \
  func<<<1, 512>>> ## params; \
  startPoint += 512; \
  counter -= 512; \
  } \
  if (counter != 0) \
  func<<<1, counter>>> ## params; \
  } \
  else \
  func<<<1, counter>>> ## params;
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef MAX_BLOCKS_PER_DIM
#define MAX_BLOCKS_PER_DIM	65536	
#endif

#ifndef IMUL
#define IMUL(a, b) __mul24(a, b)
#endif

#ifndef SQR
#define SQR(a) (a)*(a)
#endif

#ifndef SQRi
#define SQRi(a) __mul24((a),(a))
#endif

#ifndef KERNEL_RADIUS
#define KERNEL_RADIUS 1
#endif

#ifndef KERNEL_WIDTH
#define KERNEL_WIDTH (2 * KERNEL_RADIUS + 1)
#endif

#ifndef ROW_TILE_WIDTH
#define ROW_TILE_WIDTH 128
#endif

#ifndef KERNEL_RADIUS_ALIGNED
#define KERNEL_RADIUS_ALIGNED 16
#endif

#ifndef COLUMN_TILE_WIDTH
#define COLUMN_TILE_WIDTH 16
#endif

#ifndef COLUMN_TILE_HEIGHT
#define COLUMN_TILE_HEIGHT 48
#endif
