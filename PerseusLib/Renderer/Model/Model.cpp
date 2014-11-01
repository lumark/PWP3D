#include "Model.h"

#include <PerseusLib/CUDA/CUDAEngine.h>

using namespace Renderer::Model3D;

Model::Model(void)
{

}

Model::~Model(void)
{
  size_t total = groups.size();
  size_t i;
  for (i=0; i<groups.size(); i++)
    delete groups[i];
  //for (i=0; i<vertices.size(); i++)
  //delete vertices[i];
  delete verticesVector;

  //TODO if shutdown optimiser this will be deallocated. find some way to check allocation.
  //perseusSafeCall(cudaFree(verticesGPU));
}

Model::Model(char *fileName)
{
  std::cout<<"Initializing Model from File "<< fileName<<std::endl;
  FILE* f = fopen(fileName, "r");

  if (f!=NULL)
  {
    faceCount = createFromFile(f);
    fclose(f);
  }
  else
  {
    std::cout<<"Fatal Error! File %s does not exist."<< fileName<<std::endl;
  }
}

Model::Model(std::string fileName)
{
  std::cout<<"Initializing Model from File "<< fileName<<std::endl;

  FILE* f = fopen(fileName.data(), "r");

  if (f!=NULL)
  {
    faceCount = createFromFile(f);
    fclose(f);
  }
  else
  {
    std::cout<<"Fatal Error! File %s does not exist."<< fileName<<std::endl;
  }
}

Model::Model(aiMesh* pMesh)
{
  faceCount = createFromMesh(pMesh);
}

Model::Model(FILE *f)
{
  faceCount = createFromFile(f);
}

int Model::createFromFile(FILE *file)
{
  std::cout<<"[createFromFile] init Model from File.."<<std::endl;

  int v;
  size_t i, j;
  float f[3];
  char buf[128];
  ModelGroup* group = new ModelGroup("default");
  ModelVertex vertex;
  ModelFace* face;
  int faceCount = 0;

  while(fscanf(file, "%s", buf) != EOF)
  {
    switch(buf[0])
    {
    case '#': fgets(buf, sizeof(buf), file); break;
    case 'v':
      switch (buf[1])
      {
      case '\0': // vertex
        fscanf(file, "%f %f %f", &f[0], &f[1], &f[2]);
        //vertex = new ModelVertex(f);
        vertices.push_back(ModelVertex(f));
        break;
      case 'n': fgets(buf, sizeof(buf), file); break;
      case 't': fgets(buf, sizeof(buf), file); break;
      }
      break;
    case 'm': fgets(buf, sizeof(buf), file); break;
    case 'u': fgets(buf, sizeof(buf), file); break;
    case 'g': //group
      fgets(buf, sizeof(buf), file);
      group = new ModelGroup(buf);
      this->groups.push_back(group);
      buf[strlen(buf)-1] = '\0';
      break;
    case 'o': //group, temp function
      fgets(buf, sizeof(buf), file);
      group = new ModelGroup(buf);
      this->groups.push_back(group);
      buf[strlen(buf)-1] = '\0';
      break;
    case 'f': //face
      fscanf(file, "%s", buf);
      sscanf(buf, "%d", &v);
      face = new ModelFace();
      face->vertices.push_back(v-1);
      while(fscanf(file, "%d", &v) > 0){
        face->vertices.push_back(v-1);
      }

      face->verticesVectorCount = face->vertices.size();
      face->verticesVector = new int[face->verticesVectorCount];
      for (i=0; i<face->verticesVectorCount; i++){
        face->verticesVector[i] = face->vertices[i];
      }
      group->faces.push_back(face);

      faceCount++;
      break;
    default: fgets(buf, sizeof(buf), file); break;
    }
  }

  j = 0;
  verticesVector = new VFLOAT[vertices.size()*4];

  this->minZ = vertices[0].vector3d.z;
  for (i=0; (size_t)i < vertices.size(); i++ )
  {
    verticesVector[j] = vertices[i].vector3d.x;
    verticesVector[j+1] = vertices[i].vector3d.y;
    verticesVector[j+2] = vertices[i].vector3d.z;

    if (this->minZ < vertices[i].vector3d.z)
      this->minZ = vertices[i].vector3d.z;

    verticesVector[j+3] = 1;
    j += 4;
  }

  std::cout<<"[createFromFile] Finished. Got faceCount num "<<faceCount<<std::endl;
  return faceCount;
}

int Model::createFromMesh(aiMesh* pMesh)
{
  std::cout<<"[createFromFile] initializing Model from mesh."<<std::endl;

  size_t i, j;
  ModelGroup* group = new ModelGroup("default");
  ModelFace* face;
  int nFaceCount = 0;

  // ---------------------------------------------------------------------------
  // init group
  this->groups.push_back(group);

  // ---------------------------------------------------------------------------
  // init vertices
  for(unsigned int i=0; i!=pMesh->mNumVertices; i++)
  {
    vertices.push_back(ModelVertex(pMesh->mVertices[i].x,
                                   pMesh->mVertices[i].y,
                                   pMesh->mVertices[i].z));
  }

  // ---------------------------------------------------------------------------
  // init face. When reading face from mesh, we need to do v instead of v-1
  // which is difference from init face from file process.
  for(unsigned int i=0; i!=pMesh->mNumFaces; i++)
  {
    face = new ModelFace();
    for(unsigned int j=0; j!= pMesh->mFaces[i].mNumIndices; j++)
    {
      face->vertices.push_back( pMesh->mFaces[i].mIndices[j]);
    }

    face->verticesVectorCount = face->vertices.size();
    face->verticesVector = new int[face->verticesVectorCount];
    for (unsigned int k=0; k<face->verticesVectorCount; k++)
    {
      face->verticesVector[k] = face->vertices[k];
    }

    group->faces.push_back(face);
    nFaceCount++;
  }

  // ---------------------------------------------------------------------------
  j = 0;
  verticesVector = new VFLOAT[vertices.size()*4];
  this->minZ = vertices[0].vector3d.z;
  for (i=0; (size_t)i < vertices.size(); i++ )
  {
    verticesVector[j] = vertices[i].vector3d.x;
    verticesVector[j+1] = vertices[i].vector3d.y;
    verticesVector[j+2] = vertices[i].vector3d.z;

    if (this->minZ < vertices[i].vector3d.z)
      this->minZ = vertices[i].vector3d.z;

    verticesVector[j+3] = 1;
    j += 4;
  }

  std::cout<<"[createFromMesh] Finished. Got faceCount num "<<nFaceCount<<std::endl;
  return nFaceCount;
}

Model* Model::Clone()
{
  size_t i, j, k;
  Model *newModel = new Model();
  newModel->faceCount = this->faceCount;
  for (i=0; i<this->groups.size(); i++)
  {
    char* newGroupName =  strdup(this->groups[i]->groupName);
    ModelGroup *group = new ModelGroup(newGroupName);
    for (j=0; j<this->groups[i]->faces.size(); j++)
    {
      ModelFace* face = new ModelFace();
      face->isVisible = groups[i]->faces[j]->isVisible;
      face->isInvisible = groups[i]->faces[j]->isInvisible;
      for (k=0; k<groups[i]->faces[j]->vertices.size(); k++)
        face->vertices.push_back(groups[i]->faces[j]->vertices[k]);
      group->faces.push_back(face);
    }
    newModel->groups.push_back(group);
  }
  for (i=0; i<this->vertices.size(); i++)
  {
    ModelVertex* vertex = this->vertices[i].Clone();
    newModel->vertices.push_back(*vertex);
  }
  return newModel;
}

void Model::ToModelH(ModelH* newModel)
{
  newModel->originalVerticesVector = this->verticesVector;
}




void Model::ToModelHInit(ModelH* newModel)
{
  int i, j, k;

  newModel->isAllocated = true;

  newModel->faceCount = this->faceCount;
  newModel->groups = &this->groups;
  newModel->verticesVectorSize = this->vertices.size();
  newModel->verticesVector = new VFLOAT[newModel->verticesVectorSize * 4];
  newModel->verticesVectorPreP = new VFLOAT[newModel->verticesVectorSize * 4];
  newModel->minZ = this->minZ;

  if(newModel->groups->size()<=0)
  {
    std::cerr<<"[Model::ToModelHInit] Fail init 3D model. exit."<<std::endl;
    exit(-1);
  }

  newModel->verticesGPUBuff = (float*) new float4[faceCount * 4];
  perseusSafeCall(cudaMalloc((void**)&newModel->verticesGPU, this->faceCount * 4 * sizeof(float4)));

  float4 *vb = (float4*) newModel->verticesGPUBuff;

  int faceId = 0;
  for (i=0; i<(int)this->groups.size(); i++)
  {
    ModelGroup* currentGroup = this->groups[i];
    for (j=0; j<(int)currentGroup->faces.size(); j++, faceId++)
    {
      for (k=0; k<3; k++)
      {
        if(faceId * 4 + k > faceCount * 4 )
        {
          printf("error! overflow! Req: %d, Max %d;\n", faceId * 4 + k, faceCount * 4);
        }

        vb[faceId * 4 + k].x = this->vertices[currentGroup->faces[j]->vertices[k]].vector3d.x;
        vb[faceId * 4 + k].y = this->vertices[currentGroup->faces[j]->vertices[k]].vector3d.y;
        vb[faceId * 4 + k].z = this->vertices[currentGroup->faces[j]->vertices[k]].vector3d.z;
        vb[faceId * 4 + k].w = 1;
      }
    }
  }

  perseusSafeCall(cudaMemcpy(newModel->verticesGPU, newModel->verticesGPUBuff, this->faceCount * 4 * sizeof(float4), cudaMemcpyHostToDevice));
}
