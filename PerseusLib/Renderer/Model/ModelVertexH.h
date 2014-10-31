#pragma once

#include <PerseusLib/Others/PerseusLibDefines.h>

#include <PerseusLib/Primitives/Vector4D.h>

using namespace PerseusLib::Primitives;

namespace Renderer
{
namespace Model3D
{
class ModelVertexH
{
public:
  VECTOR4DA vector4d;

  ModelVertexH* Clone() { return new ModelVertexH(vector4d); }

  ModelVertexH(VECTOR4DA v) { vector4d = VECTOR4DA(v.x, v.y, v.z, v.w); }

  ModelVertexH(float *v) { vector4d = VECTOR4DA(v); }
  ModelVertexH(double *v) { vector4d = VECTOR4DA(v); }
  ModelVertexH(int *v) { vector4d = VECTOR4DA(v); }
  ModelVertexH(long double *v) { vector4d = VECTOR4DA(v); }

  ModelVertexH(float v0, float v1, float v2, float v3) { vector4d = VECTOR4DA(v0, v1, v2, v3); }
  ModelVertexH(double v0, double  v1, double v2, double v3) { vector4d = VECTOR4DA(v0, v1, v2, v3); }
  ModelVertexH(long double v0, long double  v1, long double v2, long double v3) { vector4d = VECTOR4DA(v0, v1, v2, v3); }
  ModelVertexH(int v0, int v1, int v2, int v3) { vector4d = VECTOR4DA(v0, v1, v2, v3); }
  ModelVertexH(void) { vector4d = VECTOR4DA(); }

  ~ModelVertexH(void){ }
};
}
}
