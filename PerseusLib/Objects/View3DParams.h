#pragma once

namespace PerseusLib
{
namespace Objects
{
class View3DParams
{
public:
  float zNear, zFar;
  float zBufferOffset;

  View3DParams(void) {
    zBufferOffset = 0.0001f;
    zFar = 50.0f;
    zNear = 0.01f;
  }


  View3DParams(float f_zFar, float f_zNear) {
    zBufferOffset = 0.0001f;
    zFar = f_zFar;
    zNear = f_zNear;
  }

  ~View3DParams(void) {}
};
}
}
