#pragma once

#include <PerseusLib/Utils/MathUtils.h>
#include <PerseusLib/Primitives/Vector3D.h>

#include <PerseusLib/Others/PerseusLibDefines.h>

#include <PerseusLib/Renderer/Primitives/Camera3D.h>
#include <PerseusLib/Renderer/Primitives/Quaternion.h>

#include <PerseusLib/Renderer/Transforms/CameraCoordinateTransform.h>
#include <PerseusLib/Renderer/Transforms/ObjectCoordinateTransform.h>

using namespace PerseusLib::Utils;
using namespace PerseusLib::Primitives;

using namespace Renderer::Primitives;

namespace Renderer
{
namespace Transforms
{
class CoordinateTransform
{
  std::vector<CameraCoordinateTransform*> cameraTransforms;
  std::vector<ObjectCoordinateTransform*> objectTransforms;

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

  Quaternion *rotation;

  VFLOAT *modelViewMatrix;
  VFLOAT *projectionMatrix;
  VFLOAT *pmMatrix;

  int* view;

  VFLOAT zFar, zNear, fovy;

  Quaternion qrotation;
  VECTOR3DA translation;

  void SetTranslation(VFLOAT* translation) {
    this->translation.x = translation[0]; this->translation.y = translation[1]; this->translation.z = translation[2];
  }
  void SetTranslation(VECTOR3DA translation) { this->translation = translation; }
  void SetTranslation(VECTOR3DA* translation) { this->translation = *translation; }

  void AddTranslation(VFLOAT *translation) {
    this->translation.x += translation[0]; this->translation.y += translation[1]; this->translation.z += translation[2];
  }
  void AddTranslation(VECTOR3DA translation) {
    this->translation.x += translation.x; this->translation.y += translation.y; this->translation.z += translation.z;
  }
  void AddTranslation(VECTOR3DA *translation) {
    this->translation.x += translation->x; this->translation.y += translation->y; this->translation.z += translation->z;
  }

  void SetRotation(Quaternion* rotation) { this->rotation = rotation; }

  void SetProjectionMatrix(VFLOAT *projectionMatrix);
  void SetProjectionMatrix();
  void SetProjectionMatrix(VFLOAT fovy, VFLOAT aspect, VFLOAT zNear, VFLOAT zFar);
  void SetProjectionMatrix(Camera3D* camera, VFLOAT zNear, VFLOAT zFar);

  void SetViewPort(int x, int y, int width, int height) { view[0] = x; view[1] = y; view[2] = width; view[3] = height; }

  void GetProjectionMatrix(VFLOAT *pmatrix) { for (int i=0; i< 16; i++) pmatrix[i] = projectionMatrix[i]; }
  void GetModelViewMatrix(VFLOAT* );
  void GetPMMatrix(VFLOAT*);
  void GetProjectionParameters(ProjectionParams *params);

  void GetInvPMMatrix(VFLOAT*);
  void GetInvPMatrix(VFLOAT* prod);

  //void GetViewPort(int* mview) { mview = view; }

  CoordinateTransform(void);
  ~CoordinateTransform(void);
};
}
}
