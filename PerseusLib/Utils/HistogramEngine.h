#pragma once

#include <PerseusLib/Utils/ImageUtils.h>

#include <PerseusLib/Objects/HistogramVarBin.h>
#include <PerseusLib/Objects/Object3D.h>
#include <PerseusLib/Objects/View3D.h>
#include <PerseusLib/Objects/Pose3D.h>

#include <PerseusLib/Renderer/Engine/DrawingEngine.h>

using namespace PerseusLib::Primitives;
using namespace PerseusLib::Objects;
using namespace PerseusLib::Utils;

using namespace Renderer::Engine;

namespace PerseusLib
{
namespace Utils
{
class HistogramEngine
{
private:
  static HistogramEngine* instance;

  void NormaliseHistogramVarBin(HistogramVarBin *histogram);

  void BuildHistogramVarBin(HistogramVarBin *histogram, ImageUChar *mask, ImageUChar4 *image, int objectId);
  void BuildHistogramVarBin(HistogramVarBin *histogram, ImageUChar *mask, ImageUChar *videoMask, ImageUChar4* image, int objectId);
public:
  static HistogramEngine* Instance(void) {
    if (instance == NULL) instance = new HistogramEngine();
    return instance;
  }

  void UpdateVarBinHistogram(Object3D* object, View3D* view, ImageUChar4* originalImage, ImageUChar* mask);
  void UpdateVarBinHistogram(Object3D* object, View3D* view, ImageUChar4* originalImage, ImageUChar* mask, ImageUChar* videoMask);
  void UpdateVarBinHistogram(Object3D* object, View3D* view, ImageUChar4* originalImage, Pose3D* pose);

  void SetVarBinHistogram(Object3D* object, View3D* view, float2 *normalised);

  void MergeHistograms(Object3D** objects, int objectCount, View3D** views, int viewCount, float mergeAlphaForeground, float mergeAlphaBackground);

  HistogramEngine(void);
  ~HistogramEngine(void);
};
}
}
