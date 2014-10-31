#pragma once

#include <PerseusLib/Others/PerseusLibDefines.h>

#include <stdio.h>

#include <PerseusLib/Renderer/Model/ModelGroup.h>
#include <PerseusLib/Renderer/Model/ModelVertexH.h>

namespace Renderer
{
namespace Model3D
{
class ModelH
{
public:
  bool isAllocated;
  std::vector<ModelGroup*>* groups;

  VFLOAT *verticesVector;
  VFLOAT *verticesVectorPreP;
  VFLOAT *originalVerticesVector;
  size_t verticesVectorSize;

  float* verticesGPU, *verticesGPUBuff;

  VFLOAT minZ;

  int faceCount;

  ModelH(void);
  ~ModelH(void);
};
}
}
