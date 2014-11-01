#pragma once

#include <PerseusLib/Others/PerseusLibDefines.h>
#include <PerseusLib/Utils/ImageUtils.h>
#include <PerseusLib/Renderer/Primitives/Camera3D.h>
#include <PerseusLib/Renderer/Transforms/CameraCoordinateTransform.h>

using namespace PerseusLib::Primitives;
using namespace PerseusLib::Utils;

using namespace Renderer::Transforms;
using namespace Renderer::Primitives;

namespace Renderer
{
namespace Objects
{
class Renderer3DView
{
public:
  Camera3D* camera3D;
  CameraCoordinateTransform* cameraCoordinateTransform;
  CameraCoordinateTransform::ProjectionParams projectionParams;

  int viewId;
  int view[4];
  int roiGenerated[6];

  VFLOAT invP[16];

  void SetViewPort(int x, int y, int width, int height) {
    view[0] = x;
    view[1] = y;
    view[2] = width;
    view[3] = height;
  }

  Renderer3DView(int width, int height, Camera3D* camera, VFLOAT zNear,
                 VFLOAT zFar, int viewId)
  {
    this->camera3D = camera;
    cameraCoordinateTransform = new CameraCoordinateTransform();
    printf("[Renderer3DView] Set zNear %f, zFar %f; \n", zNear, zFar);

    cameraCoordinateTransform->SetProjectionMatrix(camera, zNear, zFar);

    this->viewId = viewId;

    this->SetViewPort(0, 0, width, height);
  }

  Renderer3DView(int width, int height, char* cameraCalibrationFile,
                 VFLOAT zNear, VFLOAT zFar, int viewId)
  {
    camera3D = new Camera3D(cameraCalibrationFile);

    cameraCoordinateTransform = new CameraCoordinateTransform();
    printf("[Renderer3DView] Set zNear %f, zFar %f; \n", zNear, zFar);

    cameraCoordinateTransform->SetProjectionMatrix(cameraCalibrationFile, zNear, zFar);
    cameraCoordinateTransform->GetProjectionParameters(&projectionParams);

    //setup invP matrix (inverse of projection matrix ... needed in energy function)
    cameraCoordinateTransform->GetInvPMatrix(invP);

    this->viewId = viewId;

    this->SetViewPort(0, 0, width, height);
  }

  Renderer3DView(int width, int height, float fSizeX, float fSizeY,
                 float fFocalLengthX, float fFocalLengthY,
                 float fCenterPointX, float fCenterPointY,
                 VFLOAT zNear, VFLOAT zFar, int viewId)
  {
    camera3D = new Camera3D(fSizeX, fSizeY, fFocalLengthX, fFocalLengthY,
                            fCenterPointX, fCenterPointY);

    cameraCoordinateTransform = new CameraCoordinateTransform();
    printf("[Renderer3DView] Set zNear %f, zFar %f; \n", zNear, zFar);

    cameraCoordinateTransform->SetProjectionMatrix(camera3D, zNear, zFar);
    cameraCoordinateTransform->GetProjectionParameters(&projectionParams);

    //setup invP matrix (inverse of projection matrix ... needed in energy function)
    cameraCoordinateTransform->GetInvPMatrix(invP);

    this->viewId = viewId;

    this->SetViewPort(0, 0, width, height);
  }


  ~Renderer3DView(void) {
    delete cameraCoordinateTransform;
  }
};
}
}
