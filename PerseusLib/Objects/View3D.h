#pragma once

#include <PerseusLib/Others/PerseusLibDefines.h>

#include <PerseusLib/Renderer/Transforms/CoordinateTransform.h>
#include <PerseusLib/Renderer/Objects/Renderer3DView.h>

#include <PerseusLib/Primitives/Vector3D.h>
#include <PerseusLib/Primitives/Vector4D.h>

#include <PerseusLib/Objects/Pose3D.h>
#include <PerseusLib/Objects/View3DParams.h>
#include <PerseusLib/Objects/ImageRender.h>

#include <PerseusLib/Utils/ImageUtils.h>

using namespace PerseusLib::Objects;

using namespace Renderer::Primitives;
using namespace Renderer::Transforms;
using namespace Renderer::Objects;

using namespace PerseusLib::Utils;

namespace PerseusLib
{
namespace Objects
{
class View3D
{
public:
  int viewId;

  float zBufferOffset;

  VUINT roiGeneratedAll[6];

  Renderer3DView* renderView;

  ImageRender *imageRenderAll;

  ImageRender *imageHistogramMaskAll;
  ImageUChar *imageWireframeAll;

  ImageUChar *imagePosteriors;
  ImageUChar4 *imageRegistered;
  ImageUChar4 *imageRegisteredPrev;
  ImageUChar4 *imageProximity;

  ImageUChar *videoMask;

  View3D(int viewIdx, char* cameraCalibFileName, int width, int height, View3DParams* params = NULL) {
    if (params == NULL)
    {
      params = new View3DParams();
    }

    if(params->zFar <0 || params->zNear<0)
    {
      printf("error! invalid zFar value. \n");
    }

    this->viewId = viewIdx;
    this->zBufferOffset = params->zBufferOffset;

    renderView = new Renderer3DView(width, height, cameraCalibFileName, params->zNear, params->zFar, viewIdx);

    imageRenderAll = new ImageRender(width, height, true);

    imageHistogramMaskAll = new ImageRender(width, height, false);
    imageWireframeAll = new ImageUChar(width, height, false);

    imagePosteriors = new ImageUChar(width, height, false);

    imageRegistered = new ImageUChar4(width, height, true);
    imageRegisteredPrev = new ImageUChar4(width, height, false);
    imageProximity = new ImageUChar4(width, height, false);

    videoMask = new ImageUChar(width, height, false);
  }

  View3D(int viewIdx, float fSizeX, float fSizeY,
         float fFocalLengthX, float fFocalLengthY,
         float fCenterPointX, float fCenterPointY,
         int width, int height, float fzNear, float fzFar, View3DParams* params = NULL)
  {
    if (params == NULL)
    {
      params = new View3DParams(fzFar, fzNear);
    }

    if(params->zFar <0 || params->zNear<0)
    {
      printf("error! invalid zFar value. \n");
    }

    this->viewId = viewIdx;
    this->zBufferOffset = params->zBufferOffset;

    renderView = new Renderer3DView(width, height, fSizeX, fSizeY,
                                    fFocalLengthX, fFocalLengthY,
                                    fCenterPointX, fCenterPointY,
                                    params->zNear, params->zFar, viewIdx);

    imageRenderAll = new ImageRender(width, height, true);

    imageHistogramMaskAll = new ImageRender(width, height, false);
    imageWireframeAll = new ImageUChar(width, height, false);

    imagePosteriors = new ImageUChar(width, height, false);

    imageRegistered = new ImageUChar4(width, height, true);
    imageRegisteredPrev = new ImageUChar4(width, height, false);
    imageProximity = new ImageUChar4(width, height, false);

    videoMask = new ImageUChar(width, height, false);
  }


  ~View3D() {
    delete imageRenderAll;

    delete imageHistogramMaskAll;
    delete imageWireframeAll;

    delete imagePosteriors;

    delete imageRegistered;
    delete imageRegisteredPrev;
    delete imageProximity;

    delete videoMask;

    delete renderView;
  }
};
}
}
