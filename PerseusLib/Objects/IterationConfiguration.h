#pragma once

#include <PerseusLib/Objects/Pose3D.h>
#include <PerseusLib/Objects/StepSize3D.h>
#include <PerseusLib/Others/PerseusLibDefines.h>

using namespace PerseusLib::Objects;

#ifndef PERSEUS_MAX_OBJECT_COUNT
#define PERSEUS_MAX_OBJECT_COUNT 20
#endif

#ifndef PERSEUS_MAX_VIEW_COUNT
#define PERSEUS_MAX_VIEW_COUNT 10
#endif


#ifndef PERSEUS_MAX_ITER_COUNT
#define PERSEUS_MAX_ITER_COUNT 160
#endif

namespace PerseusLib
{
namespace Objects
{
class IterationConfiguration
{
public:
  int iterCount;

  int width;
  int height;

  int levelSetBandSize;

  int iterObjectCount[PERSEUS_MAX_VIEW_COUNT];
  int iterViewCount;

  int iterObjectIds[PERSEUS_MAX_VIEW_COUNT][PERSEUS_MAX_OBJECT_COUNT];
  int iterViewIds[PERSEUS_MAX_VIEW_COUNT];

  IterationTarget iterTarget[PERSEUS_MAX_ITER_COUNT];

  bool useCUDARender;
  bool useCUDAEF;

  IterationConfiguration(void) {
    int i;
    iterViewCount = 0; iterCount = 1;
    for (i=0; i<PERSEUS_MAX_VIEW_COUNT; i++) iterObjectCount[i] = 0;
    for (i=0; i<PERSEUS_MAX_ITER_COUNT; i++) iterTarget[i] = ITERATIONTARGET_BOTH;
    useCUDARender = false;
    useCUDAEF = false;
  }
  ~IterationConfiguration(void) {
  }
};
}
}
