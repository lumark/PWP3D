#include "EFStandard.h"

using namespace PerseusLib::Optimiser;

#include <PerseusLib/CUDA/CUDAEngine.h>

#include <PerseusLib/Utils/ImageUtils.h>
using namespace PerseusLib::Utils;


EFStandard::EFStandard(void)
{
}

EFStandard::~EFStandard(void)
{
}

void EFStandard::PrepareIteration(Object3D ***objects, int *objectCount, View3D** views, int viewCount, IterationConfiguration* iterConfig)
{
	Object3D* object; View3D* view; int objectIdx, viewIdx;

	for (viewIdx = 0; viewIdx<viewCount; viewIdx++) 
	{
		view = views[viewIdx];

    if (objectCount[viewIdx] > 1) DrawingEngine::Instance()->DrawAllInView(objects[viewIdx], objectCount[viewIdx], view, iterConfig->useCUDARender, true);

    for (objectIdx = 0; objectIdx < objectCount[viewIdx]; objectIdx++)
    {
      object = objects[viewIdx][objectIdx];

      DrawingEngine::Instance()->Draw(object, view, iterConfig->useCUDARender, !iterConfig->useCUDAEF);
      DrawingEngine::Instance()->ChangeROIWithBand(object, view, iterConfig->levelSetBandSize, iterConfig->width, iterConfig->height);

      registerObjectImage(object, view, iterConfig->useCUDARender, (objectCount[viewIdx] > 1));

      processDTSihluetteLSDXDY(object, view, iterConfig->levelSetBandSize);
    }
	}
}

void EFStandard::GetFirstDerivativeValues(Object3D ***objects, int *objectCount, View3D** views, int viewCount, IterationConfiguration* iterConfig)
{
	int objectIdx, viewIdx;
	Object3D* object; View3D* view;

	if (iterConfig->useCUDAEF)
	{
		for (viewIdx = 0; viewIdx < viewCount; viewIdx++) for (objectIdx = 0; objectIdx < objectCount[viewIdx]; objectIdx++)
		{
			object = objects[viewIdx][objectIdx]; view = views[viewIdx];

			registerObjectAndViewGeometricData(object, view);

			processAndGetEFFirstDerivatives(object, view, (objectCount[viewIdx] > 1));
		}
		return;
	}

	this->GetFirstDerivativeValues_CPU_6DoF(objects, objectCount, views, viewCount, iterConfig);
}

void EFStandard::GetFirstDerivativeValues_CPU_6DoF(Object3D ***objects, int *objectCount, View3D** views, int viewCount, IterationConfiguration* iterConfig)
{
	int objectIdx, viewIdx, objectId, viewId;
	Object3D* object; View3D* view;

	int width = iterConfig->width, height = iterConfig->height;

	int i, j, k, idx, icX, icY, icZ, hidx; 
	float pYB, pYF, dtIdx, dfPPGeneric, dirac, heaviside;
	unsigned char r, b, g;
	int *dtPosX, *dtPosY;
	float *dt, *dtDX, *dtDY;
	float xProjected[4], xUnprojected[4], xUnrotated[4], dfPP[7], dpose[7], otherInfo[2];

	for (viewIdx = 0; viewIdx<viewCount; viewIdx++) for (objectIdx = 0; objectIdx<objectCount[viewIdx]; objectIdx++)
	{
		view = views[viewIdx]; object = objects[viewIdx][objectIdx];

		viewId = view->viewId; objectId = object->objectId;

		dt = object->dt[viewId]->pixels;
		dtPosX = object->dtPosX[viewId]->pixels; dtPosY = object->dtPosY[viewId]->pixels;
		dtDX = object->dtDX[viewId]->pixels; dtDY = object->dtDY[viewId]->pixels;

		getProcessedDataDTSihluetteLSDXDY(object, view);

		for (i=0; i<7; i++) dpose[i] = 0;

		for (j=0, idx=0; j<height; j++) for (i=0; i<width; idx++, i++)
		{
			if (dtPosY[idx] >= 0)// && view->videoMask->pixels[idx] > 128)
			{
				dtIdx = dt[idx];

				icX = i; icY = j;
				if (dtIdx < 0) { icX = dtPosX[idx] + object->roiGenerated[viewId][0]; icY = dtPosY[idx] + object->roiGenerated[viewId][1]; }
				icZ = icX + icY * width;


				if (objectCount[viewIdx] > 1) 
					if (((view->imageRenderAll->imageObjects->pixels[icZ]-1) != objectId) ||
						((view->imageRenderAll->imageObjects->pixels[i + j * width] - 1) != objectId && (view->imageRenderAll->imageObjects->pixels[i + j * width] - 1) != -1 ))
						continue;

				hidx = int(4096 + 512 * dtIdx);
				if (hidx >= 0 && hidx < MathUtils::Instance()->heavisideSize)
				{
					heaviside = MathUtils::Instance()->heavisideFunction[hidx];

					r = view->imageRegistered->pixels[idx].x; g = view->imageRegistered->pixels[idx].y; b = view->imageRegistered->pixels[idx].z;

					object->histogramVarBin[viewId]->GetValue(&pYF, &pYB, r, g, b, i, j);

					pYF += 0.0000001f; pYB += 0.0000001f;

					dirac = (1.0f / float(PI)) * (1 / (dtIdx * dtIdx + 1.0f) + float(1e-3));
					dfPPGeneric = dirac * (pYF - pYB) / (heaviside * (pYF - pYB) + pYB);

					// run 1
					xProjected[0] = (float) 2 * (icX - view->renderView->view[0]) / view->renderView->view[2] - 1;
					xProjected[1] = (float) 2 * (icY - view->renderView->view[1]) / view->renderView->view[3] - 1;
					xProjected[2] = (float) 2 * ((float)object->imageRender[viewId]->imageZBuffer->pixels[icZ] / (float)MAX_INT) - 1;
					xProjected[3] = 1;

					MathUtils::Instance()->MatrixVectorProduct4(view->renderView->invP, xProjected, xUnprojected);
					MathUtils::Instance()->MatrixVectorProduct4(object->invPMMatrix[viewId], xProjected, xUnrotated);

					otherInfo[0] = view->renderView->projectionParams.A * dtDX[idx];
					otherInfo[1] = view->renderView->projectionParams.B * dtDY[idx];

					dfPP[0] = -otherInfo[0] / xUnprojected[2]; 
					dfPP[1] = -otherInfo[1] / xUnprojected[2];
					dfPP[2] = (otherInfo[0] * xUnprojected[0] + otherInfo[1] * xUnprojected[1]) / (xUnprojected[2] * xUnprojected[2]);

					object->renderObject->objectCoordinateTransform[viewId]->rotation->GetDerivatives(dfPP + 3, xUnprojected, xUnrotated, 
						view->renderView->projectionParams.all, otherInfo);

					for (k=0; k<7; k++) { dfPP[k] *= dfPPGeneric; dpose[k] += dfPP[k]; }

					// run 2
					xProjected[0] = (float) 2 * (icX - view->renderView->view[0]) / view->renderView->view[2] - 1;
					xProjected[1] = (float) 2 * (icY - view->renderView->view[1]) / view->renderView->view[3] - 1;
					xProjected[2] = (float) 2 * ((float)object->imageRender[viewId]->imageZBufferInverse->pixels[icZ] / (float)MAX_INT) - 1;
					xProjected[3] = 1;

					MathUtils::Instance()->MatrixVectorProduct4(view->renderView->invP, xProjected, xUnprojected);
					MathUtils::Instance()->MatrixVectorProduct4(object->invPMMatrix[viewId], xProjected, xUnrotated);

					otherInfo[0] = view->renderView->projectionParams.A * dtDX[idx];
					otherInfo[1] = view->renderView->projectionParams.B * dtDY[idx];

					dfPP[0] = -otherInfo[0] / xUnprojected[2];
					dfPP[1] = -otherInfo[1] / xUnprojected[2];
					dfPP[2] = (otherInfo[0] * xUnprojected[0] + otherInfo[1] * xUnprojected[1]) / (xUnprojected[2] * xUnprojected[2]);

					object->renderObject->objectCoordinateTransform[viewId]->rotation->GetDerivatives(dfPP + 3, xUnprojected, xUnrotated, 
						view->renderView->projectionParams.all, otherInfo);

					for (k=0; k<7; k++) { dfPP[k] *= dfPPGeneric; dpose[k] += dfPP[k]; }
				}
			}
		}

		object->dpose[viewId]->SetFrom(dpose, 7);

		//char rez[200];
		//sprintf(rez, "%4.5f %4.5f %4.5f %4.5f %4.5f %4.5f %4.5f", object->dpose[viewId]->translation->x, object->dpose[viewId]->translation->y,
		//	object->dpose[viewId]->translation->z, object->dpose[viewId]->rotation->vector4d.x, object->dpose[viewId]->rotation->vector4d.y,
		//	object->dpose[viewId]->rotation->vector4d.z, object->dpose[viewId]->rotation->vector4d.w);

		//DEBUGBREAK;
	}
}
