#pragma once

#include <PerseusLib/Optimiser/EFs/IEnergyFunction.h>

namespace PerseusLib
{
namespace Optimiser
{
class EFStandard: public IEnergyFunction
{
private:
  void GetFirstDerivativeValues_CPU_6DoF(Object3D ***objects, int *objectCount, View3D** views, int viewCount, IterationConfiguration* iterConfig);

public:
  void PrepareIteration(Object3D ***objects, int *objectCount, View3D** views, int viewCount, IterationConfiguration* iterConfig);
  void GetFirstDerivativeValues(Object3D ***objects, int *objectCount, View3D** views, int viewCount, IterationConfiguration* iterConfig);

  EFStandard(void);
  ~EFStandard(void);
};
}
}
