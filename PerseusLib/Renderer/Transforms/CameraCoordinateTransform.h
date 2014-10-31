#pragma once

#include <PerseusLib/Utils/MathUtils.h>
#include <PerseusLib/Others/PerseusLibDefines.h>
#include <PerseusLib/Renderer/Primitives/Camera3D.h>
#include <cmath>
#include <string.h>
#include <memory>

using namespace PerseusLib::Utils;
using namespace Renderer::Primitives;


namespace Renderer
{
namespace Transforms
{
class CameraCoordinateTransform
{
private:
  void decompKMatrix(VFLOAT source[3][4], VFLOAT cpara[3][4], VFLOAT trans[3][4]);

  VFLOAT norm(VFLOAT a, VFLOAT b, VFLOAT c) {
    return ((VFLOAT)sqrtf(a * a + b * b + c * c));
  }
  VFLOAT dot(VFLOAT a1, VFLOAT a2, VFLOAT a3, VFLOAT b1, VFLOAT b2, VFLOAT b3) {
    return (a1 * b1 + a2 * b2 + a3 * b3);
  }

public:
  struct ProjectionParams
  {
    VFLOAT all[6];
    VFLOAT A,B,C,D,E,F;

  }projectionParams;

  VFLOAT *projectionMatrix, *projectionMatrixGL;

  VFLOAT zFar, zNear, fovy;

  void SetProjectionMatrix(VFLOAT *projectionMatrix);
  void SetProjectionMatrix();
  void SetProjectionMatrix(VFLOAT fovy, VFLOAT aspect, VFLOAT zNear, VFLOAT zFar);
  void SetProjectionMatrix(Camera3D* camera, VFLOAT zNear, VFLOAT zFar);
  void SetProjectionMatrix(char* cameraCalibrationFile, VFLOAT zNear, VFLOAT zFar);

  void GetProjectionMatrix(VFLOAT *pmatrix) { for (int i=0; i< 16; i++) pmatrix[i] = projectionMatrix[i]; }
  void GetProjectionMatrixGL(VFLOAT *pmatrix) { for (int i=0; i< 16; i++) pmatrix[i] = projectionMatrixGL[i]; }
  void GetProjectionParameters(ProjectionParams *params);

  void GetInvPMatrix(VFLOAT* prod);

  CameraCoordinateTransform(void);
  ~CameraCoordinateTransform(void);
};
}
}
