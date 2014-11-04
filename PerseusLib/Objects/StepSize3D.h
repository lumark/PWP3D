#pragma once

namespace PerseusLib
{
namespace Objects
{
class StepSize3D
{
public:
  float tX, tY, tZ, r;

  void SetFrom(float r, float tX, float tY, float tZ)
  {
    this->r = r; this->tX = tX; this->tY = tY; this->tZ = tZ;
  }

  void SetFrom(float *stepSize)
  {
    this->r = stepSize[0]; this->tX = stepSize[1];
    this->tY = stepSize[2]; this->tZ = stepSize[3];
  }

  StepSize3D(void) {
    this->SetFrom(0.0f, 0.0f, 0.0f, 0.0f);
  }

  StepSize3D(float r, float tX, float tY, float tZ) {
    this->SetFrom(r, tX, tY, tZ);
  }

  ~StepSize3D(void) { }
};
}
}
