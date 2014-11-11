#pragma once

#include <iostream>
#include <PerseusLib/Others/PerseusLibDefines.h>

#include <PerseusLib/Renderer/Transforms/CoordinateTransform.h>
#include <PerseusLib/Renderer/Objects/Renderer3DObject.h>
#include <PerseusLib/Renderer/Objects/Renderer3DView.h>

#include <PerseusLib/Objects/HistogramVarBin.h>

#include <PerseusLib/Primitives/Vector3D.h>
#include <PerseusLib/Primitives/Vector4D.h>

#include <PerseusLib/Objects/View3D.h>
#include <PerseusLib/Objects/StepSize3D.h>
#include <PerseusLib/Objects/Pose3D.h>
#include <PerseusLib/Objects/Object3DParams.h>
#include <PerseusLib/Objects/View3DParams.h>
#include <PerseusLib/Objects/HistogramVarBin.h>

#include <PerseusLib/Objects/ImageRender.h>

#include <PerseusLib/Utils/ImageUtils.h>

using namespace PerseusLib::Objects;
using namespace PerseusLib::Utils;

using namespace Renderer::Primitives;
using namespace Renderer::Objects;
using namespace Renderer::Transforms;

namespace PerseusLib
{
namespace Objects
{
class Object3D
{
public:
  int objectId;
  int viewCount;

  Pose3D **pose, **dpose, **initialPose;

  VFLOAT **invPMMatrix, **pmMatrix;

  VINT** roiGenerated; VINT** roiNormalised;

  Renderer3DObject* renderObject;

  HistogramVarBin **histogramVarBin;

  ImageRender **imageRender;
  ImageUChar **imageWireframe;
  ImageUChar **imageSihluette;
  ImageUChar **imageHistogramMask;

  ImageUChar4 **imageCamera;

  ImageFloat **dt, **dtDX, **dtDY;
  ImageInt **dtPosX, **dtPosY;

  ImageUChar **histMasks;
  ImageUChar4 **histSources;

  ImageFloat **imagePosteriorsPFPB;

  StepSize3D **stepSize;

  void UpdatePoseFromRenderer(View3D* view) {
    pose[view->viewId]->SetFrom(renderObject->objectCoordinateTransform[view->viewId]->translation,
        renderObject->objectCoordinateTransform[view->viewId]->rotation);
  }

  void UpdateRendererFromPose(View3D* view) {
    renderObject->objectCoordinateTransform[view->viewId]->SetFrom(pose[view->viewId]->translation, pose[view->viewId]->rotation);
  }

  Object3D(int objectId, int viewCount, char *objectFileName, int width, int height, Object3DParams* objectParams = NULL)
  {
    //    std::cout<<"[Object3D] Initializing.."<<std::endl;

    if (objectParams == NULL) {
      objectParams = new Object3DParams();
    }

    int i;

    this->objectId = objectId;
    this->viewCount = viewCount;

    renderObject = new Renderer3DObject(objectFileName, viewCount, objectId);

    histogramVarBin = new HistogramVarBin*[viewCount];

    invPMMatrix = new VFLOAT*[viewCount];
    pmMatrix = new VFLOAT*[viewCount];

    roiGenerated = new VINT*[viewCount];
    roiNormalised = new VINT*[viewCount];

    pose = new Pose3D*[viewCount];
    dpose = new Pose3D*[viewCount];
    initialPose = new Pose3D*[viewCount];
    stepSize = new StepSize3D*[viewCount];

    imageRender = new ImageRender*[viewCount];
    imageWireframe = new ImageUChar*[viewCount];
    imageSihluette = new ImageUChar*[viewCount];
    imageHistogramMask = new ImageUChar*[viewCount];

    imageCamera = new ImageUChar4*[viewCount];

    dt = new ImageFloat*[viewCount];
    dtDX = new ImageFloat*[viewCount];
    dtDY = new ImageFloat*[viewCount];

    dtPosX = new ImageInt*[viewCount];
    dtPosY = new ImageInt*[viewCount];

    histMasks = new ImageUChar*[viewCount];
    histSources = new ImageUChar4*[viewCount];

    imagePosteriorsPFPB = new ImageFloat*[viewCount];

    for (i=0; i<viewCount; i++)
    {
      histogramVarBin[i] = new HistogramVarBin();
      histogramVarBin[i]->Set(objectParams->noVarBinHistograms, objectParams->noVarBinHistogramBins);

      cudaMallocHost((void**)&invPMMatrix[i], sizeof(VFLOAT) * 16);
      cudaMallocHost((void**)&pmMatrix[i], sizeof(VFLOAT) * 16);

      roiGenerated[i] = new VINT[6];
      roiNormalised[i] = new VINT[6];

      pose[i] = new Pose3D();
      dpose[i] = new Pose3D();
      initialPose[i] = new Pose3D();
      stepSize[i] = new StepSize3D();

      imageRender[i] = new ImageRender(width, height, true);

      imageWireframe[i] = new ImageUChar(width, height, false);
      imageSihluette[i] = new ImageUChar(width, height, true);
      imageHistogramMask[i] = new ImageUChar(width, height, false);

      imageCamera[i] = new ImageUChar4(width, height, true);

      dt[i] = new ImageFloat(width, height, true);
      dtDX[i] = new ImageFloat(width, height, true);
      dtDY[i] = new ImageFloat(width, height, true);

      dtPosX[i] = new ImageInt(width, height, true);
      dtPosY[i] = new ImageInt(width, height, true);

      histMasks[i] = new ImageUChar(width, height, false);
      histSources[i] = new ImageUChar4(width, height, false);

      imagePosteriorsPFPB[i] = new ImageFloat(width, height, false);
    }

    //    std::cout<<"[Object3D] finish init object.."<<std::endl;
  }

  Object3D(int objectId, int viewCount, aiMesh* pMesh, int width, int height, Object3DParams* objectParams = NULL)
  {
    //    std::cout<<"[Object3D] Initializing.."<<std::endl;
    if (objectParams == NULL) {
      objectParams = new Object3DParams();
    }

    int i;

    this->objectId = objectId;
    this->viewCount = viewCount;

    renderObject = new Renderer3DObject(pMesh, viewCount, objectId);

    histogramVarBin = new HistogramVarBin*[viewCount];

    invPMMatrix = new VFLOAT*[viewCount];
    pmMatrix = new VFLOAT*[viewCount];

    roiGenerated = new VINT*[viewCount];
    roiNormalised = new VINT*[viewCount];

    pose = new Pose3D*[viewCount];
    dpose = new Pose3D*[viewCount];
    initialPose = new Pose3D*[viewCount];
    stepSize = new StepSize3D*[viewCount];

    imageRender = new ImageRender*[viewCount];
    imageWireframe = new ImageUChar*[viewCount];
    imageSihluette = new ImageUChar*[viewCount];
    imageHistogramMask = new ImageUChar*[viewCount];

    imageCamera = new ImageUChar4*[viewCount];

    dt = new ImageFloat*[viewCount];
    dtDX = new ImageFloat*[viewCount];
    dtDY = new ImageFloat*[viewCount];

    dtPosX = new ImageInt*[viewCount];
    dtPosY = new ImageInt*[viewCount];

    histMasks = new ImageUChar*[viewCount];
    histSources = new ImageUChar4*[viewCount];

    imagePosteriorsPFPB = new ImageFloat*[viewCount];

    for (i=0; i<viewCount; i++)
    {
      histogramVarBin[i] = new HistogramVarBin();
      histogramVarBin[i]->Set(objectParams->noVarBinHistograms, objectParams->noVarBinHistogramBins);

      cudaMallocHost((void**)&invPMMatrix[i], sizeof(VFLOAT) * 16);
      cudaMallocHost((void**)&pmMatrix[i], sizeof(VFLOAT) * 16);

      roiGenerated[i] = new VINT[6];
      roiNormalised[i] = new VINT[6];

      pose[i] = new Pose3D();
      dpose[i] = new Pose3D();
      initialPose[i] = new Pose3D();
      stepSize[i] = new StepSize3D();

      imageRender[i] = new ImageRender(width, height, true);

      imageWireframe[i] = new ImageUChar(width, height, false);
      imageSihluette[i] = new ImageUChar(width, height, true);
      imageHistogramMask[i] = new ImageUChar(width, height, false);

      imageCamera[i] = new ImageUChar4(width, height, true);

      dt[i] = new ImageFloat(width, height, true);
      dtDX[i] = new ImageFloat(width, height, true);
      dtDY[i] = new ImageFloat(width, height, true);

      dtPosX[i] = new ImageInt(width, height, true);
      dtPosY[i] = new ImageInt(width, height, true);

      histMasks[i] = new ImageUChar(width, height, false);
      histSources[i] = new ImageUChar4(width, height, false);

      imagePosteriorsPFPB[i] = new ImageFloat(width, height, false);
    }

    //    std::cout<<"[Object3D] finish init object.."<<std::endl;
  }

  ~Object3D(void)
  {
    delete renderObject;

    for (int i=0; i<viewCount; i++)
    {
      histogramVarBin[i]->Free();
      delete histogramVarBin[i];

      cudaFreeHost(invPMMatrix[i]);
      cudaFreeHost(pmMatrix[i]);

      delete roiGenerated[i];
      delete roiNormalised[i];

      delete pose[i];
      delete dpose[i];
      delete initialPose[i];
      delete stepSize[i];

      delete imageRender[i];
      delete imageWireframe[i];
      delete imageSihluette[i];
      delete imageHistogramMask[i];

      delete imageCamera[i];

      delete dt[i];
      delete dtDX[i];
      delete dtDY[i];

      delete dtPosX[i];
      delete dtPosY[i];

      delete histMasks[i];
      delete histSources[i];

      delete imagePosteriorsPFPB[i];
    }

    delete pose;
    delete dpose;
    delete initialPose;

    delete histogramVarBin;

    delete invPMMatrix;
    delete pmMatrix;

    delete roiGenerated;
    delete roiNormalised;

    delete stepSize;

    delete imageRender;
    delete imageWireframe;
    delete imageSihluette;
    delete imageHistogramMask;

    delete imageCamera;

    delete dt;
    delete dtDX;
    delete dtDY;

    delete dtPosX;
    delete dtPosY;

    delete histMasks;
    delete histSources;

    delete imagePosteriorsPFPB;
  }
};
}
}
