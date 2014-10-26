#include "DrawingEngine.h"
#include <algorithm>
#include <math.h>
#include <omp.h>

#include <PerseusLib/CUDA/CUDAEngine.h>

using namespace PerseusLib::Utils;

using namespace Renderer::Engine;
using namespace Renderer::Model3D;
using namespace Renderer::Primitives;
using namespace Renderer::Transforms;
using namespace Renderer::Objects;

using namespace std;

DrawingEngine* DrawingEngine::instance;

DrawingEngine::DrawingEngine(void) { }
DrawingEngine::~DrawingEngine(void) { }

template <class T>
inline T min3(T t1, T t2, T t3) 
{ T minim; minim = (t1 < t2) ? t1 : t2; minim = (t3 < minim) ? t3 : minim; return minim;}

template <class T>
inline T max3(T t1, T t2, T t3) 
{ T maxim; maxim = (t1 > t2) ? t1 : t2; maxim = (t3 > maxim) ? t3 : maxim; return maxim;}

void DrawingEngine::drawWireframe(ImageUChar* imageWireframe, ModelH* drawingModel, int* roiGenerated)
{
  std::cout<<"[DrawingEngine/drawWireframe] start"<<endl;
  size_t i, j;
  int localExtrems[4];
  localExtrems[0] = 0;
  localExtrems[1] = 0;
  localExtrems[2] = 0;
  localExtrems[3] = 0;

  ModelFace* currentFace;
  VBYTE currentColor;

  roiGenerated[0] = 0xFFFF; roiGenerated[1] = 0xFFFF; roiGenerated[2] = -1; roiGenerated[3] = -1;

  std::cout<<"drawingModel->groups->size() is "<<drawingModel->groups->size()<<std::endl;
  for (i=0; i<drawingModel->groups->size(); i++)
  {
    currentColor = 254;

    for (j=0; j<(*drawingModel->groups)[i]->faces.size(); j++)
    {
      currentFace = (*drawingModel->groups)[i]->faces[j];

      //      if (currentFace->isVisible)
      //      {
      printf("check if draw face;\n");
      this->drawFaceEdges(imageWireframe, currentFace, drawingModel, currentColor, localExtrems);
      roiGenerated[0] = MIN(roiGenerated[0], localExtrems[0]);
      roiGenerated[1] = MIN(roiGenerated[1], localExtrems[1]);
      roiGenerated[2] = MAX(roiGenerated[2], localExtrems[2]);
      roiGenerated[3] = MAX(roiGenerated[3], localExtrems[3]);
      //      }
      //      else
      //      {
      //        printf("face is invasable;\n");
      //      }
    }
  }

  roiGenerated[4] = roiGenerated[2] - roiGenerated[0] + 1;
  roiGenerated[5] = roiGenerated[3] - roiGenerated[1] + 1;

  this->drawFaceEdges(imageWireframe, currentFace, drawingModel, currentColor, localExtrems);

  std::cout<<"[DrawingEngine/drawWireframe] Finish "<<endl;
}

void DrawingEngine::drawFilled(ImageRender* imageRender, ModelH* drawingModel, int objectId)
{
  size_t i, j;

  VBYTE currentColor;

  for (i=0; i<drawingModel->groups->size(); i++)
  {
    currentColor = 24 * (objectId+1);

    for (j=0; j<(*drawingModel->groups)[i]->faces.size(); j++)
      this->drawFaceFilled(imageRender, (*drawingModel->groups)[i]->faces[j], drawingModel, objectId, currentColor, (int)i);
  }
}

void DrawingEngine::drawFilled(ImageUChar* imageRender, ModelH* drawingModel, int objectId)
{
  size_t i, j;

  VBYTE currentColor;

  for (i=0; i<drawingModel->groups->size(); i++)
  {
    currentColor = 24 * (objectId+1);

    for (j=0; j<(*drawingModel->groups)[i]->faces.size(); j++)
      this->drawFaceFilled(imageRender, (*drawingModel->groups)[i]->faces[j], drawingModel, objectId, currentColor, (int)i);
  }
}

void DrawingEngine::Draw(Object3D* object, View3D* view, Pose3D *pose, ImageUChar *imageRender, RenderingType renderingType, bool clearImage)
{
  int roi[6];

  Renderer3DObject *renderObject = object->renderObject;
  Renderer3DView *renderView = view->renderView;

  this->ComputeAndSetPMMatrices(object, view, pose);

  this->applyCoordinateTransform(renderView, renderObject, object->pmMatrix[view->viewId]);
  std::cout<<"[DrawingEngine::Draw] applyCoordinateTransform"<<std::endl;

  if (clearImage) imageRender->Clear();

  if (renderingType == RENDERING_FILL)
  {
    std::cout<<"[DrawingEngine::Draw] use drawFilled."<<std::endl;
    drawFilled(imageRender, renderObject->drawingModel[view->viewId], object->objectId);
  }
  else
  {
    std::cout<<"[DrawingEngine::Draw] use drawWireframe."<<std::endl;
    drawWireframe(imageRender, renderObject->drawingModel[view->viewId], roi);
  }
}

void DrawingEngine::Draw(Object3D* object, View3D* view, Pose3D *pose, ImageRender *imageRender, bool clearImage)
{
  //int roi[6];

  Renderer3DObject *renderObject = object->renderObject;
  Renderer3DView *renderView = view->renderView;

  this->ComputeAndSetPMMatrices(object, view, pose);

  this->applyCoordinateTransform(renderView, renderObject, object->pmMatrix[view->viewId]);

  if (clearImage)
  {
    imageRender->Clear();
    imageRender->ClearZBuffer();
  }

  drawFilled(imageRender, renderObject->drawingModel[view->viewId], object->objectId);
}


void DrawingEngine::GetPMMatrices(Object3D *object, View3D *view, Pose3D* pose, float *projectionMatrix, float *modelViewMatrix, float *pmMatrix)
{
  Renderer3DObject *renderObject = object->renderObject;
  Renderer3DView *renderView = view->renderView;

  pose->GetModelViewMatrix(modelViewMatrix);
  renderView->cameraCoordinateTransform->GetProjectionMatrix(projectionMatrix);

  MathUtils::Instance()->SquareMatrixProduct(pmMatrix, projectionMatrix, modelViewMatrix, 4);
}

void DrawingEngine::ComputeAndSetPMMatrices(Object3D *object, View3D *view, Pose3D* pose)
{
  float pmMatrix[16], invPMMatrix[16];
  float modelViewMatrixFull[16];

  if (pose != NULL)
  {
    pose->GetModelViewMatrix(modelViewMatrixFull);
  }
  else
  {
    object->renderObject->GetModelViewMatrix(modelViewMatrixFull, view->viewId);
  }

  view->renderView->cameraCoordinateTransform->GetProjectionMatrix(projectionMatrix);

  MathUtils::Instance()->SquareMatrixProduct(pmMatrix, projectionMatrix, modelViewMatrixFull, 4);
  MathUtils::Instance()->InvertMatrix4(invPMMatrix, pmMatrix);

  memcpy(object->pmMatrix[view->viewId], pmMatrix, 16 * sizeof(float));
  memcpy(object->invPMMatrix[view->viewId], invPMMatrix, 16 * sizeof(float));
}

void DrawingEngine::DrawAllInView(Object3D** objects, int objectCount, View3D* view, bool useCUDA, bool getBackData)
{
  //  printf("\n== DrawAllInView ==");
  int objectIdx;

  Object3D* object;
  Renderer3DObject* renderObject;
  Renderer3DView* renderView;

  for (objectIdx = 0; objectIdx < objectCount; objectIdx++)
  { this->ComputeAndSetPMMatrices(objects[objectIdx], view); }

  if (useCUDA)
  {
    renderObjectAllCUDA(objects, objectCount, view);
    if (getBackData) getProcessedDataRenderingAll(view);
  }
  else
  {
    view->imageWireframeAll->Clear();
    view->imageRenderAll->Clear();
    view->imageRenderAll->ClearZBuffer();

    view->roiGeneratedAll[0] = 0xFFFF; view->roiGeneratedAll[1] = 0xFFFF;
    view->roiGeneratedAll[2] = -1; view->roiGeneratedAll[3] = -1;

    for (objectIdx = 0; objectIdx < objectCount; objectIdx++)
    {
      object = objects[objectIdx];
      renderObject = object->renderObject; renderView = view->renderView;

      this->applyCoordinateTransform(renderView, renderObject, object->pmMatrix[view->viewId]);

      drawWireframe(view->imageWireframeAll, renderObject->drawingModel[view->viewId], (int*)view->roiGeneratedAll);
      drawFilled(view->imageRenderAll, renderObject->drawingModel[view->viewId], object->objectId);
    }
  }
}

void DrawingEngine::Draw(Object3D* object, View3D* view, bool useCUDA, bool getBackData)
{
  Renderer3DObject *renderObject = object->renderObject;
  Renderer3DView *renderView = view->renderView;
  int objectId = object->objectId;
  int viewId = view->viewId;

  this->ComputeAndSetPMMatrices(object, view);

  if (useCUDA)
  {
    renderObjectCUDA(object, view);
    if (getBackData) getProcessedDataRendering(object, view);
  }
  else
  {
    this->applyCoordinateTransform(renderView, renderObject, object->pmMatrix[view->viewId]);

    object->imageWireframe[viewId]->Clear();
    object->imageRender[viewId]->Clear();
    object->imageRender[viewId]->ClearZBuffer();

    object->roiGenerated[viewId][0] = 0xFFFF; object->roiGenerated[viewId][1] = 0xFFFF;
    object->roiGenerated[viewId][2] = -1; object->roiGenerated[viewId][3] = -1;

    drawWireframe(object->imageWireframe[viewId], renderObject->drawingModel[view->viewId], object->roiGenerated[viewId]);
    drawFilled(object->imageRender[viewId], renderObject->drawingModel[view->viewId], objectId);
  }
}

void DrawingEngine::ChangeROIWithBand(Object3D* object, View3D *view, int bandSize, int width, int height)
{
  //  printf("\n== ChangeROIWithBand == \n");

  int *roiGenerated = object->roiGenerated[view->viewId];
  //  printf("[ChangeROIWithBand] roiGenerated value is (%d,%d,%d,%d,%d,%d)\n",
  //         roiGenerated[0],roiGenerated[1] ,roiGenerated[2] ,roiGenerated[3] ,roiGenerated[4] ,roiGenerated[5]   );

  int roiTest[6];
  memcpy(roiTest, roiGenerated, 6 * sizeof(int));

  //  printf("[ChangeROIWithBand] bandSize:%d,width:%d,height:%d\n",bandSize, width, height);
  roiGenerated[0] = CLAMP(roiGenerated[0] - bandSize, 0, width);
  roiGenerated[1] = CLAMP(roiGenerated[1] - bandSize, 0, height);
  roiGenerated[2] = CLAMP(roiGenerated[2] + bandSize, 0, width);
  roiGenerated[3] = CLAMP(roiGenerated[3] + bandSize, 0, height);

  roiGenerated[4] = roiGenerated[2] - roiGenerated[0];
  roiGenerated[5] = roiGenerated[3] - roiGenerated[1];

  //  printf("[ChangeROIWithBand] change roiGenerated value to (%d,%d,%d,%d,%d,%d)\n", roiGenerated[0],roiGenerated[1] ,roiGenerated[2] ,roiGenerated[3] ,roiGenerated[4] ,roiGenerated[5]   );
}

void DrawingEngine::ChangeROIWithBand(View3D *view3D, int bandSize, int width, int height)
{
  Renderer3DView* view = view3D->renderView;

  int roiTest[6];
  memcpy(roiTest, view->roiGenerated, 6 * sizeof(int));

  view->roiGenerated[0] = CLAMP(view->roiGenerated[0] - bandSize, 0, width);
  view->roiGenerated[1] = CLAMP(view->roiGenerated[1] - bandSize, 0, height);
  view->roiGenerated[2] = CLAMP(view->roiGenerated[2] + bandSize, 0, width);
  view->roiGenerated[3] = CLAMP(view->roiGenerated[3] + bandSize, 0, height);

  view->roiGenerated[4] = view->roiGenerated[2] - view->roiGenerated[0];
  view->roiGenerated[5] = view->roiGenerated[3] - view->roiGenerated[1];
}

void DrawingEngine::drawFaceEdges(ImageUChar *image, ModelFace* currentFace, ModelH* drawingModel, VBYTE color, int* extrems)
{
  if (currentFace->verticesVectorCount != 3)
  {
    return;
  }

  VFLOAT x1 = drawingModel->verticesVector[currentFace->verticesVector[0]*4 + 0];
  VFLOAT y1 = drawingModel->verticesVector[currentFace->verticesVector[0]*4 + 1];

  if(isfinite(x1)==false)
  {
    printf("  [drawFaceEdges] fatal error! invalid value of x.\n");
    exit(-1);
  }

  VFLOAT x2 = drawingModel->verticesVector[currentFace->verticesVector[1]*4 + 0];
  VFLOAT y2 = drawingModel->verticesVector[currentFace->verticesVector[1]*4 + 1];

  VFLOAT x3 = drawingModel->verticesVector[currentFace->verticesVector[2]*4 + 0];
  VFLOAT y3 = drawingModel->verticesVector[currentFace->verticesVector[2]*4 + 1];

  DRAWLINE(image, x1, y1, x2, y2, color);
  DRAWLINE(image, x2, y2, x3, y3, color);
  DRAWLINE(image, x1, y1, x3, y3, color);

  extrems[0] = (VINT) x1;
  extrems[1] = (VINT) y1;
  extrems[2] = (VINT) x1;
  extrems[3] = (VINT) y1;

  extrems[0] = (VINT) MIN(extrems[0], x2);
  extrems[1] = (VINT) MIN(extrems[1], y2);
  extrems[2] = (VINT) MAX(extrems[2], x2);
  extrems[3] = (VINT) MAX(extrems[3], y2);

  extrems[0] = (VINT) MIN(extrems[0], x3);
  extrems[1] = (VINT) MIN(extrems[1], y3);
  extrems[2] = (VINT) MAX(extrems[2], x3);
  extrems[3] = (VINT) MAX(extrems[3], y3);

  printf("extrems (%d,%d,%d,%d)\n",extrems[0], extrems[1], extrems[2], extrems[3] );
  std::cout<<"finish draw face "<<std::endl;
}

void DrawingEngine::drawFaceFilled(ImageRender *imageRender, ModelFace* currentFace, ModelH* drawingModel, int objectId, VBYTE color, VINT meshId)
{
  if (currentFace->verticesVectorCount != 3) return;

  size_t i;
  size_t index;
  VUINT intZ;
  VFLOAT dx1, dx2, dx3, dz1, dz2, dz3, dxa, dxb, dza, dzb;
  VFLOAT dzX, Sz, Sx, Sy, Ex;

  VECTOR3DA S, E;
  VECTOR3DA A, B, C;
  VECTOR3DA orderedPoints[3];

  VFLOAT x1 = drawingModel->verticesVector[currentFace->verticesVector[0]*4 + 0];
  VFLOAT y1 = drawingModel->verticesVector[currentFace->verticesVector[0]*4 + 1];
  VFLOAT z1 = drawingModel->verticesVector[currentFace->verticesVector[0]*4 + 2];

  VFLOAT x2 = drawingModel->verticesVector[currentFace->verticesVector[1]*4 + 0];
  VFLOAT y2 = drawingModel->verticesVector[currentFace->verticesVector[1]*4 + 1];
  VFLOAT z2 = drawingModel->verticesVector[currentFace->verticesVector[1]*4 + 2];

  VFLOAT x3 = drawingModel->verticesVector[currentFace->verticesVector[2]*4 + 0];
  VFLOAT y3 = drawingModel->verticesVector[currentFace->verticesVector[2]*4 + 1];
  VFLOAT z3 = drawingModel->verticesVector[currentFace->verticesVector[2]*4 + 2];

  A = VECTOR3DA(x1, y1, z1);
  B = VECTOR3DA(x2, y2, z2);
  C = VECTOR3DA(x3, y3, z3);

  if (y1 < y2)
  {
    if (y3 < y1) { orderedPoints[0] = C; orderedPoints[1] = A; orderedPoints[2] = B; }
    else if (y3 < y2) { orderedPoints[0] = A; orderedPoints[1] = C; orderedPoints[2] = B; }
    else { orderedPoints[0] = A; orderedPoints[1] = B; orderedPoints[2] = C; }
  }
  else
  {
    if (y3 < y2) { orderedPoints[0] = C; orderedPoints[1] = B; orderedPoints[2] = A; }
    else if (y3 < y1) { orderedPoints[0] = B; orderedPoints[1] = C; orderedPoints[2] = A; }
    else { orderedPoints[0] = B; orderedPoints[1] = A;	orderedPoints[2] = C; }
  }

  A = orderedPoints[0]; B = orderedPoints[1]; C = orderedPoints[2];

  dx1 = (B.y - A.y) > 0 ? (B.x - A.x) / (B.y - A.y) : B.x - A.x;
  dx2 = (C.y - A.y) > 0 ? (C.x - A.x) / (C.y - A.y) : 0;
  dx3 = (C.y - B.y) > 0 ? (C.x - B.x) / (C.y - B.y) : 0;

  dz1 = (B.y - A.y) != 0 ? (B.z - A.z) / (B.y - A.y) : 0;
  dz2 = (C.y - A.y) != 0 ? (C.z - A.z) / (C.y - A.y) : 0;
  dz3 = (C.y - B.y) != 0 ? (C.z - B.z) / (C.y - B.y) : 0;

  S = E = A;

  B.y = floor(B.y - 0.5f); C.y = floor(C.y - 0.5f);

  if (dx1 > dx2) { dxa = dx2; dxb = dx1; dza = dz2; dzb = dz1; }
  else { dxa = dx1; dxb = dx2; dza = dz1; dzb = dz2; }

  for(; S.y <= B.y; S.y++, E.y++, S.x += dxa, E.x += dxb, S.z += dza, E.z += dzb)
  {
    dzX = (E.x != S.x) ? (E.z - S.z) / (E.x - S.x) : 0;
    Sz = S.z;

    Sy = CLAMP(S.y, 0, (VFLOAT) imageRender->height-1);
    Sx = CLAMP(S.x, 0, (VFLOAT) imageRender->width-1);
    Ex = CLAMP(E.x, 0, (VFLOAT) imageRender->width-1);

    for (i=(size_t)Sx; i<Ex; i++)
    {
      index = PIXELMATINDEX(i, Sy, imageRender->width);
      intZ = (unsigned int) (MAX_INT * Sz);

      if (intZ < imageRender->imageZBuffer->pixels[index])
      {
        imageRender->imageFill->pixels[index] = color;
        imageRender->imageZBuffer->pixels[index] = intZ;
        imageRender->imageObjects->pixels[index] = objectId + 1;
      }

      if (intZ > imageRender->imageZBufferInverse->pixels[index])
      { imageRender->imageZBufferInverse->pixels[index] = intZ; }

      Sz += dzX;
    }
  }

  if (dx1 > dx2) { dxa = dx2; dxb = dx3; dza = dz2; dzb = dz3; E = B; }
  else { dxa = dx3; dxb = dx2; dza = dz3; dzb = dz2; S = B; }

  for(; S.y <= C.y; S.y++, E.y++, S.x += dxa, E.x += dxb, S.z += dza, E.z += dzb)
  {
    dzX = (E.x != S.x) ? (E.z - S.z) / (E.x - S.x) : 0;
    Sz = S.z;

    Sy = CLAMP(S.y, 0, (VFLOAT) imageRender->height-1);
    Sx = CLAMP(S.x, 0, (VFLOAT) imageRender->width-1);
    Ex = CLAMP(E.x, 0, (VFLOAT) imageRender->width-1);

    for (i=(size_t)Sx; i<Ex; i++)
    {
      index = PIXELMATINDEX(i, Sy, imageRender->width);
      intZ = (unsigned int) (MAX_INT * Sz);

      if (intZ < imageRender->imageZBuffer->pixels[index])
      {
        imageRender->imageFill->pixels[index] = color;
        imageRender->imageZBuffer->pixels[index] = intZ;
        imageRender->imageObjects->pixels[index] = objectId + 1;
      }

      if (intZ > imageRender->imageZBufferInverse->pixels[index])
      { imageRender->imageZBufferInverse->pixels[index] = intZ; }

      Sz += dzX;
    }
  }
}

void DrawingEngine::drawFaceFilled(ImageUChar *imageRender, ModelFace* currentFace, ModelH* drawingModel, int objectId, VBYTE color, VINT meshId)
{
  if (currentFace->verticesVectorCount != 3) return;

  size_t i, index;
  VFLOAT dx1, dx2, dx3, dxa, dxb;
  VFLOAT Sx, Sy, Ex;

  VECTOR3DA S, E;
  VECTOR3DA A, B, C;
  VECTOR3DA orderedPoints[3];

  VFLOAT x1 = drawingModel->verticesVector[currentFace->verticesVector[0]*4 + 0];
  VFLOAT y1 = drawingModel->verticesVector[currentFace->verticesVector[0]*4 + 1];

  VFLOAT x2 = drawingModel->verticesVector[currentFace->verticesVector[1]*4 + 0];
  VFLOAT y2 = drawingModel->verticesVector[currentFace->verticesVector[1]*4 + 1];

  VFLOAT x3 = drawingModel->verticesVector[currentFace->verticesVector[2]*4 + 0];
  VFLOAT y3 = drawingModel->verticesVector[currentFace->verticesVector[2]*4 + 1];

  A = VECTOR3DA(x1, y1, 0.0f); B = VECTOR3DA(x2, y2, 0.0f); C = VECTOR3DA(x3, y3, 0.0f);

  if (y1 < y2)
  {
    if (y3 < y1) { orderedPoints[0] = C; orderedPoints[1] = A; orderedPoints[2] = B; }
    else if (y3 < y2) { orderedPoints[0] = A; orderedPoints[1] = C; orderedPoints[2] = B; }
    else { orderedPoints[0] = A; orderedPoints[1] = B; orderedPoints[2] = C; }
  }
  else
  {
    if (y3 < y2) { orderedPoints[0] = C; orderedPoints[1] = B; orderedPoints[2] = A; }
    else if (y3 < y1) { orderedPoints[0] = B; orderedPoints[1] = C; orderedPoints[2] = A; }
    else { orderedPoints[0] = B; orderedPoints[1] = A;	orderedPoints[2] = C; }
  }

  A = orderedPoints[0]; B = orderedPoints[1]; C = orderedPoints[2];

  dx1 = (B.y - A.y) > 0 ? (B.x - A.x) / (B.y - A.y) : B.x - A.x;
  dx2 = (C.y - A.y) > 0 ? (C.x - A.x) / (C.y - A.y) : 0;
  dx3 = (C.y - B.y) > 0 ? (C.x - B.x) / (C.y - B.y) : 0;

  S = E = A;

  B.y = floor(B.y - 0.5f); C.y = floor(C.y - 0.5f);

  if (dx1 > dx2) { dxa = dx2; dxb = dx1; } else { dxa = dx1; dxb = dx2; }

  for(; S.y <= B.y; S.y++, E.y++, S.x += dxa, E.x += dxb)
  {
    Sy = CLAMP(S.y, 0, (VFLOAT) imageRender->height-1); Sx = CLAMP(S.x, 0, (VFLOAT) imageRender->width-1);
    Ex = CLAMP(E.x, 0, (VFLOAT) imageRender->width-1);

    for (i=(size_t)Sx; i<Ex; i++) { index = PIXELMATINDEX(i, Sy, imageRender->width); imageRender->pixels[index] = color; }
  }

  if (dx1 > dx2) { dxa = dx2; dxb = dx3; E = B; } else { dxa = dx3; dxb = dx2; S = B; }

  for(; S.y <= C.y; S.y++, E.y++, S.x += dxa, E.x += dxb)
  {
    Sy = CLAMP(S.y, 0, (VFLOAT) imageRender->height-1); Sx = CLAMP(S.x, 0, (VFLOAT) imageRender->width-1);
    Ex = CLAMP(E.x, 0, (VFLOAT) imageRender->width-1);

    for (i=(size_t)Sx; i<Ex; i++)
    {
      index = PIXELMATINDEX(i, Sy, imageRender->width);
      for (i=(size_t)Sx; i<Ex; i++) { index = PIXELMATINDEX(i, Sy, imageRender->width); imageRender->pixels[index] = color; }
    }
  }
}

void DrawingEngine::applyCoordinateTransform(Renderer3DView* view, Renderer3DObject* object, float *pmMatrix)
{
  size_t i;

  object->model->ToModelH(object->drawingModel[view->viewId]);


  for (i=0; i < object->drawingModel[view->viewId]->verticesVectorSize; i++)
  {
    VFLOAT* originalVertexAsDouble = &object->drawingModel[view->viewId]->originalVerticesVector[i*4];
    VFLOAT* vertexAsDouble = &object->drawingModel[view->viewId]->verticesVector[i*4];

    MathUtils::Instance()->MatrixVectorProduct4(pmMatrix, originalVertexAsDouble, vertexAsDouble);

    vertexAsDouble[0] = view->view[0] + view->view[2] * (vertexAsDouble[0] + 1.f)/2.f;
    vertexAsDouble[1] = view->view[1] + view->view[3] * (vertexAsDouble[1] + 1.f)/2.f;
    vertexAsDouble[2] = (vertexAsDouble[2] + 1)/2;
  }
}
