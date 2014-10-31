#pragma once

#include <stdio.h>
#include <vector>
#include <string>
#include <PerseusLib/Others/PerseusLibDefines.h>
#include <PerseusLib/Renderer/Model/ModelGroup.h>
#include <PerseusLib/Renderer/Model/ModelVertex.h>
#include <PerseusLib/Renderer/Model/ModelH.h>
#include <assimp/mesh.h>

using namespace Renderer::Model3D;

namespace Renderer
{
namespace Model3D
{

class Model
{
private:
  int createFromFile(FILE *file);
  int createFromMesh(aiMesh* pMesh);

public:
  std::vector<ModelGroup*> groups;
  std::vector<ModelVertex> vertices;

  VFLOAT* verticesVector;

  VFLOAT minZ;

  int faceCount;

  Model(void);
  Model(char* fileName);
  Model(std::string fileName);
  Model(aiMesh* pMesh);
  Model(FILE* f);

  Model* Clone();

  void ToModelH(ModelH* newModel);
  void ToModelHInit(ModelH* newModel);

  ~Model(void);
};
}
}
