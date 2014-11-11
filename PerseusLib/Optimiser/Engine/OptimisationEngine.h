#pragma once

#include <PerseusLib/Utils/ImageUtils.h>

#include <PerseusLib/Objects/IterationConfiguration.h>
#include <PerseusLib/Objects/Object3D.h>
#include <PerseusLib/Objects/View3D.h>
#include <PerseusLib/Objects/StepSize3D.h>
#include <PerseusLib/Optimiser/EFs/IEnergyFunction.h>
#include <PerseusLib/Optimiser/EFs/EFStandard.h>

using namespace PerseusLib::Primitives;
using namespace PerseusLib::Objects;
using namespace PerseusLib::Utils;

namespace PerseusLib
{
namespace Optimiser
{
class OptimisationEngine
{
private:
  static OptimisationEngine* instance;

  IEnergyFunction *energyFunction;
  IEnergyFunction *energyFunction_standard;

  IterationConfiguration *iterConfig;

  int viewCount;
  int *objectCount;
  Object3D*** objects;
  View3D** views;

  StepSize3D **stepSizes;

  bool HasConverged();

  void SetPresetStepSizes();

  void NormaliseRotation();

  void DescendWithGradient(StepSize3D *stepSize, IterationConfiguration *iterConfig);
  void AdvanceTranslation(Object3D* object, View3D* view, StepSize3D* stepSize);
  void AdvanceRotation(Object3D* object, View3D* view, StepSize3D* stepSize);

  void RunOneMultiIteration(IterationConfiguration* iterConfig);
  void RunOneSingleIteration(StepSize3D* stepSize, IterationConfiguration* iterConfig);

public:
  static OptimisationEngine* Instance(void) {
    if (instance == NULL) instance = new OptimisationEngine();
    return instance;
  }

  void Initialise(int width, int heigh);
  void Shutdown();

  void RegisterViewImage(View3D *view, ImageUChar4* image);
  void Minimise(Object3D **objects, View3D **views, IterationConfiguration *iterConfig);
  void MinimiseSingle(Object3D **objects, View3D **views, IterationConfiguration *iterConfig, int nIterNum);

  OptimisationEngine(void);
  ~OptimisationEngine(void);
};
}
}
