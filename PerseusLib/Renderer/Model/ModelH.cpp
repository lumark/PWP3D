#include "ModelH.h"

#include <PerseusLib/CUDA/CUDAEngine.h>

using namespace Renderer::Model3D;

ModelH::ModelH(void)
{
  isAllocated = false;
}

ModelH::~ModelH(void)
{
  if (isAllocated)
  {
    delete verticesVector;
    delete verticesVectorPreP;
    delete verticesGPUBuff;

    perseusSafeCall(cudaFree(verticesGPU));
  }
}
