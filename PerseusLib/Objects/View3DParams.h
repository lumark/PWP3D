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
  ~View3DParams(void) {}
};
}
}
