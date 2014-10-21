#pragma once

#include <PerseusLib/Others/PerseusLibDefines.h>
#include <PerseusLib/Objects/Object3D.h>
#include <PerseusLib/Objects/View3D.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

using namespace PerseusLib::Objects;

#include "CUDADefines.h"

void initialiseCUDA(int width, int height, float* heavisideFunction, int heavisideFunctionSize);
void shutdownCUDA();

void registerObjectImage(Object3D* object, View3D* view, bool renderingFromGPU, bool isMultiobject);
void registerObjectAndViewGeometricData(Object3D* object, View3D* view);

void processDTSihluetteLSDXDY(Object3D* object, View3D* view, int bandSize);
void processAndGetEFFirstDerivatives(Object3D* object, View3D* view, bool isMultiobject);

void getProcessedDataDTSihluetteLSDXDY(Object3D* object, View3D* view);
void getProcessedDataEFFirstDerivatives(Object3D* object, View3D* view);
void getProcessedDataRendering(Object3D* object, View3D* view);
void getProcessedDataRenderingAll(View3D* view);

void renderObjectCUDA(Object3D *object, View3D *view);
void renderObjectAllCUDA(Object3D **objects, int objectCount, View3D *view);
