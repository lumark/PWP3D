#include "CUDAEngine.h"

#include "CUDAData.h"
#include "CUDADT.h"
#include "CUDAConvolution.h"
#include "CUDAScharr.h"
#include "CUDAEF.h"
#include "CUDARenderer.h"

CUDAData *cudaData;

void initialiseCUDA(int width, int height, float* heavisideFunction, int heavisideFunctionSize)
{
  //  printf("init cuda, heaviside Function size is %d\n", heavisideFunctionSize);
  cudaData = new CUDAData();

  initialiseRenderer(width, height);
  initialiseScharr(width, height);
  initialiseDT(width, height);
  initialiseConvolution(width, height);
  initialiseEF(width, height, heavisideFunction, heavisideFunctionSize);
}

void shutdownCUDA()
{
  shutdownRenderer();
  shutdownScharr();
  shutdownDT();
  shutdownConvolution();
  shutdownEF();
}

void registerObjectImage(Object3D* object, View3D* view, bool renderingFromGPU, bool isMultobject)
{
  //  printf("\n== registerObjectImage ==\n");
  int viewId = view->viewId;

  int *roiGenerated = object->roiGenerated[viewId];
  int *roiNormalised = object->roiNormalised[viewId];

  //  printf("roiGenerated is %d, %d, %d, %d, %d, %d; \n", roiGenerated[0], roiGenerated[1], roiGenerated[2], roiGenerated[3], roiGenerated[4], roiGenerated[5]);

  int widthROI, heightROI, widthFull;
  widthROI = roiGenerated[4]; heightROI = roiGenerated[5];
  widthFull = view->imageRenderAll->imageFill->width;

  unsigned char *objects;
  unsigned int *zbuffer, *zbufferInverse;
  cudaMemcpyKind renderingSource;
  if (renderingFromGPU)
  {
    objects = cudaData->objects;
    zbuffer = cudaData->zbuffer;
    zbufferInverse = cudaData->zbufferInverse;
    renderingSource = cudaMemcpyDeviceToDevice;
  }
  else
  {
    objects = object->imageRender[viewId]->imageObjects->pixels;
    zbuffer = object->imageRender[viewId]->imageZBuffer->pixels;
    zbufferInverse = object->imageRender[viewId]->imageZBufferInverse->pixels;
    renderingSource = cudaMemcpyHostToDevice;
  }

  unsigned char *objectsGPUROI = object->imageRender[viewId]->imageObjects->pixelsGPU;
  unsigned int *zbufferGPUROI = object->imageRender[viewId]->imageZBuffer->pixelsGPU;
  unsigned int *zbufferInverseGPUROI = object->imageRender[viewId]->imageZBufferInverse->pixelsGPU;

  uchar4 *cameraGPU = (uchar4*) view->imageRegistered->pixelsGPU;
  uchar4 *cameraGPUROI = (uchar4*) object->imageCamera[viewId]->pixelsGPU;

  roiNormalised[0] = 0; roiNormalised[1] = 0;
  roiNormalised[2] = roiGenerated[4]; roiNormalised[3] = roiGenerated[5];
  roiNormalised[4] = roiGenerated[4]; roiNormalised[5] = roiGenerated[5];

  //  perseusSafeCall(cudaMemcpy2D(objectsGPUROI, widthROI, objects + roiGenerated[0] + roiGenerated[1] * widthFull,
  //    widthFull, widthROI, heightROI, renderingSource));

  //  printf("[registerObjectImage] offset is: %d\n",roiGenerated[0] + roiGenerated[1] * widthFull);
  //  printf("[registerObjectImage] roiGenerated[0] is: %d\n",roiGenerated[0]);
  //  printf("[registerObjectImage] roiGenerated[1] * widthFull is: %d\n", roiGenerated[1] * widthFull);

  perseusSafeCall(cudaMemcpy2D(objectsGPUROI, widthROI* sizeof(uchar1), objects + roiGenerated[0] + roiGenerated[1] * widthFull,
      widthFull * sizeof(uchar1), widthROI * sizeof(uchar1), heightROI, renderingSource));

  perseusSafeCall(cudaMemcpy2D(zbufferGPUROI, widthROI * sizeof(uint1), zbuffer + roiGenerated[0] + roiGenerated[1] * widthFull,
      widthFull * sizeof(uint1), widthROI * sizeof(uint1), heightROI, renderingSource));

  perseusSafeCall(cudaMemcpy2D(zbufferInverseGPUROI, widthROI * sizeof(uint1), zbufferInverse + roiGenerated[0] + roiGenerated[1] * widthFull,
      widthFull * sizeof(uint1), widthROI * sizeof(uint1), heightROI, renderingSource));

  perseusSafeCall(cudaMemcpy2D(cameraGPUROI, widthROI * sizeof(uchar4), cameraGPU + roiGenerated[0] + roiGenerated[1] * widthFull,
      widthFull * sizeof(uchar4), widthROI * sizeof(uchar4), heightROI, cudaMemcpyDeviceToDevice));
  perseusSafeCall(cudaDeviceSynchronize());

  if (isMultobject)
  {
    unsigned char *objectsAll;
    unsigned char *objectsAllGPUROI = object->imageRender[viewId]->imageObjects->pixelsGPU;

    if (renderingFromGPU) { objectsAll = cudaData->objectsAll; renderingSource = cudaMemcpyDeviceToDevice; }
    else { objectsAll = view->imageRenderAll->imageObjects->pixels; renderingSource = cudaMemcpyHostToDevice; }

    perseusSafeCall(cudaMemcpy2D(objectsAllGPUROI, widthROI, objectsAll + roiGenerated[0] + roiGenerated[1] * widthFull,
        widthFull, widthROI, heightROI, renderingSource));
  }
  //  printf("[registerObjectImage] finish \n");
}

void registerObjectAndViewGeometricData(Object3D* object, View3D* view)
{
  float rotationParameters[7];
  //  printf("[registerObjectAndViewGeometricData] change rotation.\n");
  object->pose[view->viewId]->rotation->Get(rotationParameters);
  registerObjectGeometricData(rotationParameters, object->invPMMatrix[view->viewId]);

  registerViewGeometricData(view->renderView->invP, view->renderView->projectionParams.all, view->renderView->view);
}

void processDTSihluetteLSDXDY(Object3D* object, View3D* view, int bandSize)
{
  int viewId = view->viewId;

  int *roi = object->roiNormalised[viewId];

  unsigned char *objectsGPUROI = object->imageRender[viewId]->imageObjects->pixelsGPU;
  unsigned char *sihluetteGPUROI = object->imageSihluette[viewId]->pixelsGPU;

  float *dtGPUROI = object->dt[viewId]->pixelsGPU;
  int *dtPosXGPUROI = object->dtPosX[viewId]->pixelsGPU;
  int *dtPosYGPUROI = object->dtPosY[viewId]->pixelsGPU;
  float *dtDXGPUROI = object->dtDX[viewId]->pixelsGPU;
  float *dtDYGPUROI = object->dtDY[viewId]->pixelsGPU;

  computeSihluette(objectsGPUROI, sihluetteGPUROI, roi[4], roi[5], 1.0f);
  processDT(dtGPUROI, dtPosXGPUROI, dtPosYGPUROI, sihluetteGPUROI, objectsGPUROI, roi, bandSize);
  computeDerivativeXY(dtGPUROI, dtDXGPUROI, dtDYGPUROI, roi[4], roi[5]);
}

void processAndGetEFFirstDerivatives(Object3D* object, View3D* view, bool isMultiobject)
{
  int viewId = view->viewId;

  float dpose[7];

  int *roiNormalised = object->roiNormalised[viewId];
  int *roiGenerated = object->roiGenerated[viewId];

  float2 *histogram = (float2*) object->histogramVarBin[viewId]->normalisedGPU;

  uchar4 *cameraGPUROI = (uchar4*) object->imageCamera[viewId]->pixelsGPU;

  unsigned char *objectsGPUROI = object->imageRender[viewId]->imageObjects->pixelsGPU;
  unsigned int *zbufferGPUROI = object->imageRender[viewId]->imageZBuffer->pixelsGPU;
  unsigned int *zbufferInverseGPUROI = object->imageRender[viewId]->imageZBufferInverse->pixelsGPU;

  float *dtGPUROI = object->dt[viewId]->pixelsGPU;
  int *dtPosXGPUROI = object->dtPosX[viewId]->pixelsGPU;
  int *dtPosYGPUROI = object->dtPosY[viewId]->pixelsGPU;
  float *dtDXGPUROI = object->dtDX[viewId]->pixelsGPU;
  float *dtDYGPUROI = object->dtDY[viewId]->pixelsGPU;

  // launch cuda kernel to compute the pose
  processEFD1(dpose, roiNormalised, roiGenerated, histogram, cameraGPUROI, objectsGPUROI, isMultiobject, zbufferGPUROI, zbufferInverseGPUROI,
              dtGPUROI, dtPosXGPUROI, dtPosYGPUROI, dtDXGPUROI, dtDYGPUROI, object->objectId);

  object->dpose[view->viewId]->SetFrom(dpose, 7);
}

void getProcessedDataDTSihluetteLSDXDY(Object3D* object, View3D* view)
{
  int viewId = view->viewId;

  int *roi = object->roiGenerated[viewId];

  float *dtGPUROI = object->dt[viewId]->pixelsGPU;
  int *dtPosXGPUROI = object->dtPosX[viewId]->pixelsGPU;
  int *dtPosYGPUROI = object->dtPosY[viewId]->pixelsGPU;
  float *dtDXGPUROI = object->dtDX[viewId]->pixelsGPU;
  float *dtDYGPUROI = object->dtDY[viewId]->pixelsGPU;

  unsigned char *sihluetteGPUROI = object->imageSihluette[viewId]->pixelsGPU;

  float *dt = object->dt[viewId]->pixels;
  int *dtPosX = object->dtPosX[viewId]->pixels;
  int *dtPosY = object->dtPosY[viewId]->pixels;
  float *dtDX = object->dtDX[viewId]->pixels;
  float *dtDY = object->dtDY[viewId]->pixels;

  unsigned char* sihluette = object->imageSihluette[viewId]->pixels;

  int widthFull, heightFull;
  widthFull = view->imageRenderAll->imageFill->width; heightFull = view->imageRenderAll->imageFill->height;

  memset(dt, 0, widthFull * heightFull * sizeof(float));
  memset(dtPosX, -1, widthFull * heightFull * sizeof(int));
  memset(dtPosY, -1, widthFull * heightFull * sizeof(int));
  memset(sihluette, 0, widthFull * heightFull * sizeof(unsigned char));
  memset(dtDX, 0, widthFull * heightFull * sizeof(float));
  memset(dtDY, 0, widthFull * heightFull * sizeof(float));

  perseusSafeCall(cudaMemcpy2D(dt + roi[0] + roi[1] * widthFull, widthFull * sizeof(float),
      dtGPUROI, roi[4] * sizeof(float), roi[4] * sizeof(float), roi[5], cudaMemcpyDeviceToHost));
  perseusSafeCall(cudaMemcpy2D(dtPosX + roi[0] + roi[1] * widthFull, widthFull * sizeof(int),
      dtPosXGPUROI, roi[4] * sizeof(int), roi[4] * sizeof(int), roi[5], cudaMemcpyDeviceToHost));
  perseusSafeCall(cudaMemcpy2D(dtPosY + roi[0] + roi[1] * widthFull, widthFull * sizeof(int),
      dtPosYGPUROI, roi[4] * sizeof(int), roi[4] * sizeof(int), roi[5], cudaMemcpyDeviceToHost));
  perseusSafeCall(cudaMemcpy2D(sihluette + roi[0] + roi[1] * widthFull, widthFull * sizeof(unsigned char),
      sihluetteGPUROI, roi[4] * sizeof(unsigned char), roi[4] * sizeof(unsigned char), roi[5], cudaMemcpyDeviceToHost));
  perseusSafeCall(cudaMemcpy2D(dtDX + roi[0] + roi[1] * widthFull, widthFull * sizeof(float),
      dtDXGPUROI, roi[4] * sizeof(float), roi[4] * sizeof(float), roi[5], cudaMemcpyDeviceToHost));
  perseusSafeCall(cudaMemcpy2D(dtDY + roi[0] + roi[1] * widthFull, widthFull * sizeof(float),
      dtDYGPUROI, roi[4] * sizeof(float), roi[4] * sizeof(float), roi[5], cudaMemcpyDeviceToHost));
}

void getProcessedDataRenderingAll(View3D* view)
{
  int widthFull = view->imageRenderAll->imageFill->width;
  int heightFull = view->imageRenderAll->imageFill->height;

  unsigned char *fill = view->imageRenderAll->imageFill->pixels;
  unsigned char *objects = view->imageRenderAll->imageObjects->pixels;
  unsigned int *zbuffer = view->imageRenderAll->imageZBuffer->pixels;
  unsigned int *zbufferInverse = view->imageRenderAll->imageZBufferInverse->pixels;

  perseusSafeCall(cudaMemcpy(fill, cudaData->fillAll, sizeof(unsigned char) * widthFull * heightFull, cudaMemcpyDeviceToHost));
  perseusSafeCall(cudaMemcpy(objects, cudaData->objectsAll, sizeof(unsigned char) * widthFull * heightFull, cudaMemcpyDeviceToHost));
  perseusSafeCall(cudaMemcpy(zbuffer, cudaData->zbufferAll, sizeof(unsigned int) * widthFull * heightFull, cudaMemcpyDeviceToHost));
  perseusSafeCall(cudaMemcpy(zbufferInverse, cudaData->zbufferInverseAll, sizeof(unsigned int) * widthFull * heightFull, cudaMemcpyDeviceToHost));
}

void getProcessedDataRendering(Object3D* object, View3D* view)
{
  int viewId = view->viewId;

  int widthFull = view->imageRenderAll->imageFill->width;
  int heightFull = view->imageRenderAll->imageFill->height;

  unsigned char *fill = object->imageRender[viewId]->imageFill->pixels;
  unsigned char *objects = object->imageRender[viewId]->imageObjects->pixels;
  unsigned int *zbuffer = object->imageRender[viewId]->imageZBuffer->pixels;
  unsigned int *zbufferInverse = object->imageRender[viewId]->imageZBufferInverse->pixels;

  perseusSafeCall(cudaMemcpy(fill, cudaData->fill, sizeof(unsigned char) * widthFull * heightFull, cudaMemcpyDeviceToHost));
  perseusSafeCall(cudaMemcpy(objects, cudaData->objects, sizeof(unsigned char) * widthFull * heightFull, cudaMemcpyDeviceToHost));
  perseusSafeCall(cudaMemcpy(zbuffer, cudaData->zbuffer, sizeof(unsigned int) * widthFull * heightFull, cudaMemcpyDeviceToHost));
  perseusSafeCall(cudaMemcpy(zbufferInverse, cudaData->zbufferInverse, sizeof(unsigned int) * widthFull * heightFull, cudaMemcpyDeviceToHost));
}

void renderObjectCUDA(Object3D *object, View3D *view)
{
  int viewId = view->viewId;
  int width = view->imageRenderAll->imageFill->width;
  int height = view->imageRenderAll->imageFill->height;

  Renderer3DObject* renderObject = object->renderObject;

  renderObjectCUDA_one_EF((float4*)renderObject->drawingModel[viewId]->verticesGPU, renderObject->drawingModel[viewId]->faceCount,
                          object->objectId, object->pmMatrix[viewId], view->renderView->view, width, height);

  memcpy(object->roiGenerated[viewId], cudaData->roiGenerated, 6 * sizeof(int));
}

void renderObjectAllCUDA(Object3D **objects, int objectCount, View3D *view)
{
  int objectIdx, viewId, objectId, width, height;

  Object3D* object;
  Renderer3DObject* renderObject; Renderer3DView* renderView;

  width = view->imageRenderAll->imageFill->width;
  height = view->imageRenderAll->imageFill->height;

  renderView = view->renderView;
  object = objects[0]; renderObject = object->renderObject;

  viewId = view->viewId; objectId = object->objectId;

  renderObjectCUDA_all_EF((float4*)renderObject->drawingModel[viewId]->verticesGPU, renderObject->drawingModel[viewId]->faceCount,
                          objectId, object->pmMatrix[viewId], renderView->view, width, height, true);

  for (objectIdx = 1; objectIdx<objectCount; objectIdx++)
  {
    object = objects[objectIdx]; renderObject = object->renderObject; objectId = object->objectId;

    renderObjectCUDA_all_EF((float4*)renderObject->drawingModel[viewId]->verticesGPU, renderObject->drawingModel[viewId]->faceCount,
                            objectId, object->pmMatrix[viewId], renderView->view, width, height, false);
  }

  memcpy(view->roiGeneratedAll, cudaData->roiGeneratedAll, 6 * sizeof(int));
}
