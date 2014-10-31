#include "HistogramEngine.h"
#include "ImageUtils.h"

#include <omp.h>

using namespace PerseusLib::Utils;

HistogramEngine* HistogramEngine::instance;

HistogramEngine::HistogramEngine(void) { }

HistogramEngine::~HistogramEngine(void) { delete instance; }

void HistogramEngine::NormaliseHistogramVarBin(HistogramVarBin *histogram)
{
  int i, j, k, histIndex, histOffset, histNo;
  float sumHistogramForeground, sumHistogramBackground;

  sumHistogramForeground = histogram->totalForegroundPixels;
  sumHistogramBackground = histogram->totalBackgroundPixels;

  sumHistogramForeground = (sumHistogramForeground != 0) ? 1.0f / sumHistogramForeground : 0;
  sumHistogramBackground = (sumHistogramBackground != 0) ? 1.0f / sumHistogramBackground : 0;

  for (histNo = 0; histNo<histogram->noHistograms; histNo++)
  {
    histOffset = histogram->histOffsets[histNo];

    for (i=0; i<histogram->noBins[histNo]; i++) for (j=0; j<histogram->noBins[histNo]; j++) for (k=0; k<histogram->noBins[histNo]; k++)
    {
      histIndex = (i + j * histogram->noBins[histNo]) * histogram->noBins[histNo] + k;
      if (histogram->alreadyInitialised)
      {
        histogram->normalised[histOffset + histIndex].x =
            histogram->normalised[histOffset + histIndex].x * (1.0f - histogram->mergeAlphaForeground) +
            histogram->notnormalised[histOffset + histIndex].x * sumHistogramForeground * histogram->mergeAlphaForeground;

        histogram->normalised[histOffset + histIndex].y =
            histogram->normalised[histOffset + histIndex].y * (1.0f - histogram->mergeAlphaBackground) +
            histogram->notnormalised[histOffset + histIndex].y * sumHistogramBackground * histogram->mergeAlphaBackground;
      }
      else
      {
        histogram->normalised[histOffset + histIndex].x = histogram->notnormalised[histOffset + histIndex].x * sumHistogramForeground;
        histogram->normalised[histOffset + histIndex].y = histogram->notnormalised[histOffset + histIndex].y * sumHistogramBackground;
      }
    }
  }

  if (!histogram->alreadyInitialised) histogram->alreadyInitialised = true;
}


void HistogramEngine::BuildHistogramVarBin(HistogramVarBin *histogram, ImageUChar *mask, ImageUChar4* image, int objectId)
{
  int i, j, idx;
  PixelUCHAR4 pixel;

  for (j = 0; j < image->height; j++) for (i = 0; i < image->width; i++)
  {
    idx = i + j * mask->width;

    pixel = image->pixels[idx];

    if (mask->pixels[idx] != 0)
    {
      if ((mask->pixels[idx] - 1) == objectId)
        histogram->AddPoint(1, 0, pixel.x, pixel.y, pixel.z, i, j);
    }
    else
      histogram->AddPoint(0, 1, pixel.x, pixel.y, pixel.z, i, j);
  }
  printf("BuildHistogramVarBin success.\n");
}

void HistogramEngine::BuildHistogramVarBin(HistogramVarBin *histogram, ImageUChar *mask, ImageUChar *videoMask, ImageUChar4* image, int objectId)
{
  int i, j, idx;
  PixelUCHAR4 pixel;

  for (j = 0; j < image->height; j++) for (i = 0; i < image->width; i++)
  {
    idx = i + j * mask->width;

    pixel = image->pixels[idx];

    if (videoMask->pixels[idx] > 128)
    {
      if (mask->pixels[idx] != 0 && (mask->pixels[idx] - 1) == objectId)
        histogram->AddPoint(1, 0, pixel.x, pixel.y, pixel.z, i, j);
      else
        histogram->AddPoint(0, 1, pixel.x, pixel.y, pixel.z, i, j);
    }
  }
}

void HistogramEngine::UpdateVarBinHistogram(Object3D* object, View3D* view, ImageUChar4* originalImage, ImageUChar* mask)
{
  ImageUtils::Instance()->Copy(mask, object->imageHistogramMask[view->viewId]);
  this->BuildHistogramVarBin(object->histogramVarBin[view->viewId], mask, originalImage, object->objectId);
  this->NormaliseHistogramVarBin(object->histogramVarBin[view->viewId]);
  object->histogramVarBin[view->viewId]->UpdateGPUFromCPU();
}

void HistogramEngine::UpdateVarBinHistogram(Object3D* object, View3D* view, ImageUChar4* originalImage, ImageUChar* mask, ImageUChar* videoMask)
{
  ImageUtils::Instance()->Copy(mask, object->imageHistogramMask[view->viewId]);
  this->BuildHistogramVarBin(object->histogramVarBin[view->viewId], mask, videoMask, originalImage, object->objectId);
  this->NormaliseHistogramVarBin(object->histogramVarBin[view->viewId]);
  object->histogramVarBin[view->viewId]->UpdateGPUFromCPU();
}

void HistogramEngine::UpdateVarBinHistogram(Object3D* object, View3D* view, ImageUChar4* originalImage, Pose3D* pose)
{
  DrawingEngine::Instance()->Draw(object, view, pose, object->imageHistogramMask[view->viewId], DrawingEngine::RENDERING_FILL);
  this->BuildHistogramVarBin(object->histogramVarBin[view->viewId], object->imageHistogramMask[view->viewId], originalImage, object->objectId);
  this->NormaliseHistogramVarBin(object->histogramVarBin[view->viewId]);
  object->histogramVarBin[view->viewId]->UpdateGPUFromCPU();
}

void HistogramEngine::SetVarBinHistogram(Object3D* object, View3D* view, float2 *normalised)
{
  memcpy(object->histogramVarBin[view->viewId]->normalised, normalised, sizeof(float2) * object->histogramVarBin[view->viewId]->fullHistSize);
  object->histogramVarBin[view->viewId]->UpdateGPUFromCPU();
}

void HistogramEngine::MergeHistograms(Object3D** objects, int objectCount, View3D** views, int viewCount, float mergeAlphaForeground, float mergeAlphaBackground)
{
  int viewIdx, objectIdx;

  for (viewIdx = 0; viewIdx < viewCount; viewIdx++)
  {
    for (objectIdx = 0; objectIdx < objectCount; objectIdx++)
    {
      objects[objectIdx]->histogramVarBin[views[viewIdx]->viewId]->mergeAlphaForeground = mergeAlphaForeground;
      objects[objectIdx]->histogramVarBin[views[viewIdx]->viewId]->mergeAlphaBackground = mergeAlphaBackground;
      objects[objectIdx]->histogramVarBin[views[viewIdx]->viewId]->ClearNotNormalisedPartial();

      DrawingEngine::Instance()->Draw(objects[objectIdx], views[viewIdx], objects[objectIdx]->initialPose[views[viewIdx]->viewId],
          views[viewIdx]->imageHistogramMaskAll, (objectIdx == 0));
    }

    for (objectIdx = 0; objectIdx < objectCount; objectIdx++)
    { this->UpdateVarBinHistogram(objects[objectIdx], views[viewIdx], views[viewIdx]->imageRegistered, views[viewIdx]->imageHistogramMaskAll->imageObjects); }
  }
}
