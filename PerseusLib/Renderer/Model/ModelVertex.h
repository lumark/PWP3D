#pragma once

#include <PerseusLib/Primitives/Vector3D.h>
#include <PerseusLib/Others/PerseusLibDefines.h>
#include <PerseusLib/Renderer/Model/ModelVertexH.h>

using namespace PerseusLib::Primitives;

namespace Renderer
{
namespace Model3D
{
class ModelVertex
{
public:
  VECTOR3DA vector3d;

  ModelVertex* Clone() { return new ModelVertex(vector3d); }

  ModelVertexH* ToModelVertexH() { return new ModelVertexH(vector3d.x, vector3d.y, vector3d.z, (VFLOAT) 1); }

  ModelVertex(VECTOR3DA v) { vector3d = VECTOR3DA(v.x, v.y, v.z); }
  ModelVertex(ModelVertexH* m) { vector3d.x = m->vector4d.x/m->vector4d.w; vector3d.y = m->vector4d.y/m->vector4d.w;; vector3d.z = m->vector4d.z/m->vector4d.w; }

  ModelVertex(float *v) { vector3d = VECTOR3DA(v); }
  ModelVertex(double *v) { vector3d = VECTOR3DA(v); }
  ModelVertex(long double *v) { vector3d = VECTOR3DA(v); }
  ModelVertex(int *v) { vector3d = VECTOR3DA(v); }

  ModelVertex(float v0, float v1, float v2) { vector3d = VECTOR3DA(v0, v1, v2); }
  ModelVertex(double v0, double  v1, double v2) { vector3d = VECTOR3DA(v0, v1, v2); }
  ModelVertex(long double v0, long double  v1, long double v2) { vector3d = VECTOR3DA(v0, v1, v2); }
  ModelVertex(int v0, int v1, int v2) { vector3d = VECTOR3DA(v0, v1, v2); }
  ModelVertex(void) { vector3d = VECTOR3DA(); }

  void FromModelVertexH(ModelVertexH* m) { vector3d.x = m->vector4d.x/m->vector4d.w; vector3d.y = m->vector4d.y/m->vector4d.w;; vector3d.z = m->vector4d.z/m->vector4d.w; }

  ~ModelVertex(void){ }
};
}
}
