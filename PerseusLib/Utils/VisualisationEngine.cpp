#include "VisualisationEngine.h"

using namespace PerseusLib::Utils;

VisualisationEngine* VisualisationEngine::instance;


void VisualisationEngine::Initialise(int width, int height)
{

}

void VisualisationEngine::Shutdown()
{

}

void VisualisationEngine::GetImage(ImageUChar4* image, GetImageType getImageType, Object3D* object, View3D* view, Pose3D *pose)
{
  switch (getImageType)
  {
  case GETIMAGE_WIREFRAME:
    DrawingEngine::Instance()->Draw(object, view, pose, object->imageWireframe[view->viewId], DrawingEngine::RENDERING_WIREFRAME);
    ImageUtils::Instance()->Copy(object->imageWireframe[view->viewId], image);
    break;
  case GETIMAGE_FILL:
    ImageUtils::Instance()->Copy(view->imageRegistered, image);
    DrawingEngine::Instance()->Draw(object, view, pose, object->imageRender[view->viewId]->imageFill, DrawingEngine::RENDERING_FILL);
    ImageUtils::Instance()->Overlay(object->imageRender[view->viewId]->imageFill, image);
    break;
  case GETIMAGE_ORIGINAL:
    ImageUtils::Instance()->Copy(view->imageRegistered, image);
    break;
  case GETIMAGE_POSTERIORS:
    this->computePosteriors(object, view);
    ImageUtils::Instance()->ScaleToGray(object->imagePosteriorsPFPB[view->viewId], image);
    break;
  case GETIMAGE_SIHLUETTE:
    ImageUtils::Instance()->Copy(view->imageRegistered, image);
    getProcessedDataDTSihluetteLSDXDY(object, view);
    ImageUtils::Instance()->Overlay(object->imageSihluette[view->viewId], image);
    break;
  case GETIMAGE_DT:
    ImageUtils::Instance()->Copy(view->imageRegistered, image);
    getProcessedDataDTSihluetteLSDXDY(object, view);
    ImageUtils::Instance()->ScaleToGray(object->dt[view->viewId], image);
    break;
  case GETIMAGE_OBJECTS:
    //DrawingEngine::Instance()->Draw(object, view, pose, object->imageWireframe[view->viewId], DrawingEngine::RENDERING_FILL);
    break;
  case GETIMAGE_PROXIMITY:
    ImageUtils::Instance()->Copy(view->imageRegistered, image);
    DrawingEngine::Instance()->Draw(object, view, pose, object->imageWireframe[view->viewId], DrawingEngine::RENDERING_WIREFRAME);
    ImageUtils::Instance()->Overlay(object->imageWireframe[view->viewId], image);
    break;
  }
}

void VisualisationEngine::computePosteriors(Object3D* object, View3D* view)
{
  int i, j, idx;
  unsigned char r, b, g;

  float pYB, pYF, pF, pB;
  float etaF, etaB;

  etaF = object->histogramVarBin[view->viewId]->etaF; etaB = object->histogramVarBin[view->viewId]->etaB;

  for (j=0, idx=0; j<view->imageRegistered->height; j++) for (i=0; i<view->imageRegistered->width; idx++, i++)
  {
    r = view->imageRegistered->pixels[idx].x; g = view->imageRegistered->pixels[idx].y; b = view->imageRegistered->pixels[idx].z;

    object->histogramVarBin[view->viewId]->GetValue(&pYF, &pYB, r, g, b, i, j);
    pYF += 0.0000001f; pYB += 0.0000001f;
    pF = pYF / (etaF * pYF + etaB * pYB); pB = pYB / (etaF * pYF + etaB * pYB);
    object->imagePosteriorsPFPB[view->viewId]->pixels[idx] = pF - pB;
  }
}
