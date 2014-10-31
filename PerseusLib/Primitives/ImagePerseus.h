#pragma once

#include <PerseusLib/Others/PerseusLibDefines.h>
#include <PerseusLib/CUDA/CUDADefines.h>

#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>

namespace PerseusLib
{
namespace Primitives
{
template <typename T>
class ImagePerseus
{
private:
  size_t totalSize, totalSizeGPU;

public:
  bool useCudaAlloc;

  int isAllocated;
  int width, height;

  T* pixels;
  T* pixelsGPU;

  ImagePerseus(int width, int height, bool useCudaAlloc = false, int allocWidthGPU = 0, int allocHeightGPU = 0) {
    this->width = width;
    this->height = height;
    this->useCudaAlloc = useCudaAlloc;

    totalSize = width * height * sizeof(T);
    if (allocWidthGPU != 0 && allocHeightGPU != 0) totalSizeGPU = allocWidthGPU * allocHeightGPU * sizeof(T);
    else totalSizeGPU = width * height * sizeof(T);

    if (useCudaAlloc)
    {
      perseusSafeCall(cudaMallocHost((void**)&pixels, totalSize));
      perseusSafeCall(cudaMalloc((void**)&pixelsGPU, totalSizeGPU));
      //          printf("use device alloc\n");
    }
    else
    { pixels = new T[width * height]; }

    this->Clear();

    isAllocated = true;
  }

  T &operator[](unsigned subscript) { return pixels[subscript]; }
  T operator[](unsigned subscript) const { return pixels[subscript]; }

  void Clear(T defaultValue = T(0))
  {
    memset(pixels, (int)defaultValue, totalSize);
    if (useCudaAlloc) perseusSafeCall(cudaMemset(pixelsGPU, (int)defaultValue, totalSizeGPU));
  }

  void UpdateGPUFromCPU() { if (useCudaAlloc) perseusSafeCall(cudaMemcpy(pixelsGPU, pixels, totalSizeGPU, cudaMemcpyHostToDevice)); }
  void UpdateCPUFromGPU() { if (useCudaAlloc) perseusSafeCall(cudaMemcpy(pixels, pixelsGPU, totalSizeGPU, cudaMemcpyDeviceToHost)); }

  void Free()
  {
    if (useCudaAlloc) {
      perseusSafeCall(cudaFree(pixelsGPU));
      perseusSafeCall(cudaFreeHost(pixels));
    }
    else delete pixels;

    this->isAllocated = false;
  }

  ~ImagePerseus() { if (isAllocated) this->Free(); }
};
}
}
