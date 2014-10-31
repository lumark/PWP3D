#pragma once

#include <PerseusLib/Others/PerseusLibDefines.h>
#include <PerseusLib/Renderer/Model/ModelFace.h>
#include <PerseusLib/Renderer/Model/ModelVertex.h>

//#include <vector>

namespace Renderer
{
namespace Model3D
{
class ModelGroup
{
public:

  std::vector<ModelFace*> faces;
  char* groupName;

  ModelGroup(char* groupName);
  ModelGroup(void);
  ~ModelGroup(void);
};
}
}
