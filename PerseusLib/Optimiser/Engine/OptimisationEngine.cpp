#include "OptimisationEngine.h"

#include <PerseusLib/CUDA/CUDAEngine.h>

using namespace PerseusLib::Optimiser;

OptimisationEngine* OptimisationEngine::instance;

OptimisationEngine::OptimisationEngine(void) { }
OptimisationEngine::~OptimisationEngine(void) { }

void OptimisationEngine::Initialise(int width, int height)
{
	int i;

	objects = new Object3D**[PERSEUS_MAX_VIEW_COUNT];
	views = new View3D*[PERSEUS_MAX_VIEW_COUNT];
	for (i=0; i<PERSEUS_MAX_VIEW_COUNT; i++) objects[i] = new Object3D*[PERSEUS_MAX_OBJECT_COUNT];

	objectCount = new int[PERSEUS_MAX_VIEW_COUNT];

	stepSizes = new StepSize3D*[8];
	for (i=0; i<8; i++) stepSizes[i] = new StepSize3D();

	energyFunction_standard = new EFStandard();

	this->SetPresetStepSizes();

	MathUtils::Instance()->ReadAndAllocateHeaviside(8192, "/Users/luma/Code/Luma/PWP3D/Files/Others/heaviside.txt");

	initialiseCUDA(width, height, MathUtils::Instance()->heavisideFunction, MathUtils::Instance()->heavisideSize);

  printf("init cuda success.\n");
}

void OptimisationEngine::Shutdown()
{
	int i;

	shutdownCUDA();

	for (i=0; i<8; i++) delete stepSizes[i];

	delete objectCount;

	delete stepSizes;
	delete energyFunction_standard;

	MathUtils::Instance()->DeallocateHeaviside();

	delete instance;
}

void OptimisationEngine::SetPresetStepSizes()
{
	stepSizes[0]->tX = -0.005f; stepSizes[0]->tY = -0.005f; stepSizes[0]->tZ = -0.005f; stepSizes[0]->r = -0.0008f;

	stepSizes[1]->tX = -0.003f; stepSizes[1]->tY = -0.003f; stepSizes[1]->tZ = -0.003f; stepSizes[1]->r = -0.0003f;
	stepSizes[2]->tX = -0.003f; stepSizes[2]->tY = -0.003f; stepSizes[2]->tZ = -0.003f; stepSizes[2]->r = -0.0003f;
	
	stepSizes[3]->tX = -0.002f; stepSizes[3]->tY = -0.002f; stepSizes[3]->tZ = -0.003f; stepSizes[3]->r = -0.0003f;
	stepSizes[4]->tX = -0.002f; stepSizes[4]->tY = -0.002f; stepSizes[4]->tZ = -0.003f; stepSizes[4]->r = -0.0003f;
	stepSizes[5]->tX = -0.002f; stepSizes[5]->tY = -0.002f; stepSizes[5]->tZ = -0.003f; stepSizes[5]->r = -0.0003f;

	stepSizes[6]->tX = -0.001f; stepSizes[6]->tY = -0.001f; stepSizes[6]->tZ = -0.002f; stepSizes[6]->r = -0.0002f;
	stepSizes[7]->tX = -0.001f; stepSizes[7]->tY = -0.001f; stepSizes[7]->tZ = -0.002f; stepSizes[7]->r = -0.0002f;
}

void OptimisationEngine::RegisterViewImage(View3D *view, ImageUChar4* image)
{
	ImageUtils::Instance()->Copy(image, view->imageRegistered);
	view->imageRegistered->UpdateGPUFromCPU();
}

void OptimisationEngine::Minimise(Object3D **objects, View3D **views, IterationConfiguration *iterConfig)
{
	int objectIdx, viewIdx, iterIdx;

	this->iterConfig = iterConfig;

	viewCount = iterConfig->iterViewCount;
	for (viewIdx=0; viewIdx<viewCount; viewIdx++) 
	{
		this->views[viewIdx] = views[iterConfig->iterViewIds[viewIdx]];
		this->objectCount[viewIdx] = iterConfig->iterObjectCount[viewIdx];
	}

	for (viewIdx=0; viewIdx<viewCount; viewIdx++) for (objectIdx=0; objectIdx<objectCount[viewIdx]; objectIdx++) 
	{
		this->objects[viewIdx][objectIdx] = objects[iterConfig->iterObjectIds[viewIdx][objectIdx]];
    this->objects[viewIdx][objectIdx]->initialPose[viewIdx]->CopyInto(this->objects[viewIdx][objectIdx]->pose[viewIdx]);
		this->objects[viewIdx][objectIdx]->UpdateRendererFromPose(views[viewIdx]);
	}

  energyFunction = energyFunction_standard;

  for (iterIdx=0; iterIdx<iterConfig->iterCount; iterIdx++) this->RunOneMultiIteration(iterConfig);
}

void OptimisationEngine::RunOneMultiIteration(IterationConfiguration* iterConfig)
{
	this->RunOneSingleIteration(stepSizes[0], iterConfig); if (this->HasConverged()) return;
	this->RunOneSingleIteration(stepSizes[1], iterConfig); if (this->HasConverged()) return;
	this->RunOneSingleIteration(stepSizes[2], iterConfig); if (this->HasConverged()) return;
	this->RunOneSingleIteration(stepSizes[3], iterConfig); if (this->HasConverged()) return;
	this->RunOneSingleIteration(stepSizes[4], iterConfig); if (this->HasConverged()) return;
	this->RunOneSingleIteration(stepSizes[5], iterConfig); if (this->HasConverged()) return;
	this->RunOneSingleIteration(stepSizes[6], iterConfig); if (this->HasConverged()) return;
	this->RunOneSingleIteration(stepSizes[7], iterConfig); if (this->HasConverged()) return;

	this->NormaliseRotation();
}

void OptimisationEngine::RunOneSingleIteration(StepSize3D* presetStepSize, IterationConfiguration* iterConfig)
{
  energyFunction->PrepareIteration(objects, objectCount, views, viewCount, iterConfig);

  // update the pose of the object
  energyFunction->GetFirstDerivativeValues(objects, objectCount, views, viewCount, iterConfig);

  this->DescendWithGradient(presetStepSize, iterConfig);
}

void OptimisationEngine::DescendWithGradient(StepSize3D *presetStepSize, IterationConfiguration *iterConfig)
{
  //  printf("[DescendWithGradient]\n");
	int objectIdx, viewIdx;

	StepSize3D actualStepSize;

	for (viewIdx = 0; viewIdx < viewCount; viewIdx++) for (objectIdx = 0; objectIdx < objectCount[viewIdx]; objectIdx++)
	{
		actualStepSize.r = presetStepSize->r * objects[viewIdx][objectIdx]->stepSize[viewIdx]->r;
		actualStepSize.tX = presetStepSize->tX * objects[viewIdx][objectIdx]->stepSize[viewIdx]->tX;
		actualStepSize.tY = presetStepSize->tY * objects[viewIdx][objectIdx]->stepSize[viewIdx]->tY;
		actualStepSize.tZ = presetStepSize->tZ * objects[viewIdx][objectIdx]->stepSize[viewIdx]->tZ;

		switch (iterConfig->iterTarget[0])
		{
		case ITERATIONTARGET_BOTH:
      AdvanceTranslation(objects[viewIdx][objectIdx], views[viewIdx], &actualStepSize);

			AdvanceRotation(objects[viewIdx][objectIdx], views[viewIdx], &actualStepSize);

			break;
		case ITERATIONTARGET_TRANSLATION:
			AdvanceTranslation(objects[viewIdx][objectIdx], views[viewIdx], &actualStepSize);
			break;
		case ITERATIONTARGET_ROTATION:
			AdvanceRotation(objects[viewIdx][objectIdx], views[viewIdx], &actualStepSize);
			break;
		}

		objects[viewIdx][objectIdx]->UpdateRendererFromPose(views[viewIdx]);
	}
}
void OptimisationEngine::AdvanceTranslation(Object3D* object, View3D* view, StepSize3D* stepSize)
{
  object->pose[view->viewId]->translation->x -= stepSize->tX * object->dpose[view->viewId]->translation->x;
  object->pose[view->viewId]->translation->y -= stepSize->tY * object->dpose[view->viewId]->translation->y;
  object->pose[view->viewId]->translation->z -= stepSize->tZ * object->dpose[view->viewId]->translation->z;
}
void OptimisationEngine::AdvanceRotation(Object3D* object, View3D* view, StepSize3D* stepSize)
{
  object->pose[view->viewId]->rotation->vector4d.x -= stepSize->r * object->dpose[view->viewId]->rotation->vector4d.x;
  object->pose[view->viewId]->rotation->vector4d.y -= stepSize->r * object->dpose[view->viewId]->rotation->vector4d.y;
  object->pose[view->viewId]->rotation->vector4d.z -= stepSize->r * object->dpose[view->viewId]->rotation->vector4d.z;
  object->pose[view->viewId]->rotation->vector4d.w -= stepSize->r * object->dpose[view->viewId]->rotation->vector4d.w;
}

void OptimisationEngine::NormaliseRotation()
{
	int objectIdx, viewIdx;
	for (viewIdx = 0; viewIdx < viewCount; viewIdx++) for (objectIdx = 0; objectIdx < objectCount[viewIdx]; objectIdx++)
	{
		objects[viewIdx][objectIdx]->pose[viewIdx]->rotation->Normalize();
		objects[viewIdx][objectIdx]->UpdateRendererFromPose(views[viewIdx]);
	}
}

bool OptimisationEngine::HasConverged()
{
	return false;
}
