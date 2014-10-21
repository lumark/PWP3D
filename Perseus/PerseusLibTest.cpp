#include "../PerseusLib/PerseusLib.h"

#include "Utils/Timer.h"

using namespace Perseus::Utils;

int main(void)
{
	char str[100];
	int i;

	int width = 640, height = 480;
	int viewCount = 1, objectCount = 1;
	int objectId = 0, viewIdx = 0, objectIdx = 0;

	Timer t;

  std::cout<<"pass 1"<<std::endl;

	//result visualisation
	ImageUChar4* result = new ImageUChar4(width, height);

	//input image
	//camera = 24 bit colour rgb
	ImageUChar4* camera = new ImageUChar4(width, height);
  ImageUtils::Instance()->LoadImageFromFile(camera, "/Users/luma/Code/Luma/PWP3D/Files/Images/Red.png");

  std::cout<<"pass 2"<<std::endl;

	//objects allocation + initialisation: 3d model in obj required
	Object3D **objects = new Object3D*[objectCount];
  objects[objectIdx] = new Object3D(objectId, viewCount, "/Users/luma/Code/Luma/PWP3D/Files/Models/Renderer/long.obj", width, height);

  std::cout<<"pass 3"<<std::endl;

	//views allocation + initialisation: camera calibration (artoolkit format) required
	View3D **views = new View3D*[viewCount];
  views[viewIdx] = new View3D(0, "/Users/luma/Code/Luma/PWP3D/Files/CameraCalibration/900nc.cal", width, height);

  std::cout<<"pass 3.5"<<std::endl;


	//histogram initialisation
	//source = 24 bit colour rgb
	//mask = 24 bit black/white png - white represents object
	//videoMask = 24 bit black/white png - white represents parts of the image that are usable
  ImageUtils::Instance()->LoadImageFromFile(views[viewIdx]->videoMask, "/Users/luma/Code/Luma/PWP3D/Files/Masks/480p_All_VideoMask.png");
  ImageUtils::Instance()->LoadImageFromFile(objects[objectIdx]->histSources[viewIdx], "/Users/luma/Code/Luma/PWP3D/Files/Masks/Red_Source.png");
  ImageUtils::Instance()->LoadImageFromFile(objects[objectIdx]->histMasks[viewIdx], "/Users/luma/Code/Luma/PWP3D/Files/Masks/Red_Mask.png", objectIdx+1);
	HistogramEngine::Instance()->UpdateVarBinHistogram(objects[objectIdx], views[viewIdx], objects[objectIdx]->histSources[viewIdx], 
		objects[objectIdx]->histMasks[viewIdx], views[viewIdx]->videoMask);

  std::cout<<"pass 4"<<std::endl;


	//iteration configuration for one object
	IterationConfiguration *iterConfig = new IterationConfiguration();
	iterConfig->width = width; iterConfig->height = height;
	iterConfig->iterViewIds[viewIdx] = 0;
	iterConfig->iterObjectCount[viewIdx] = 1;
	iterConfig->levelSetBandSize = 8;
	iterConfig->iterObjectIds[viewIdx][objectIdx] = 0;
	iterConfig->iterViewCount = 1;
	iterConfig->iterCount = 1;

  std::cout<<"pass 5"<<std::endl;

	//step size per object and view
	objects[objectIdx]->stepSize[viewIdx] = new StepSize3D(0.2f, 0.5f, 0.5f, 10.0f);

	//initial pose per object and view
	objects[objectIdx]->initialPose[viewIdx]->SetFrom(-1.98f, -2.90f, 37.47f, -40.90f, -207.77f, 27.48f);

	//primary initilisation
	OptimisationEngine::Instance()->Initialise(width, height);

	//register camera image with main engine
	OptimisationEngine::Instance()->RegisterViewImage(views[viewIdx], camera);

  for (i=0; i<4; i++)
  {
    switch (i)
    {
    case 0:
      iterConfig->useCUDAEF = true;
      iterConfig->useCUDARender = true;
      break;
    case 1:
      iterConfig->useCUDAEF = true;
      iterConfig->useCUDARender = false;
      break;
    case 2:
      iterConfig->useCUDAEF = false;
      iterConfig->useCUDARender = true;
      break;
    case 3:
      iterConfig->useCUDAEF = false;
      iterConfig->useCUDARender = false;
      break;
    }

    sprintf(str, "/Users/luma/Code/Luma/PWP3D/Files/Results/result%04d.png", i);

    t.restart();
    //main processing
    OptimisationEngine::Instance()->Minimise(objects, views, iterConfig);
    t.check("Iteration");

//    //result plot
//    VisualisationEngine::Instance()->GetImage(result, GETIMAGE_PROXIMITY, objects[objectIdx], views[viewIdx], objects[objectIdx]->pose[viewIdx]);

//    //result save to file
//    ImageUtils::Instance()->SaveImageToFile(result, str);

    printf("%f %f %f %f %f %f %f\n",
      objects[objectIdx]->pose[viewIdx]->translation->x, objects[objectIdx]->pose[viewIdx]->translation->y, objects[objectIdx]->pose[viewIdx]->translation->z,
      objects[objectIdx]->pose[viewIdx]->rotation->vector4d.x, objects[objectIdx]->pose[viewIdx]->rotation->vector4d.y,
      objects[objectIdx]->pose[viewIdx]->rotation->vector4d.z, objects[objectIdx]->pose[viewIdx]->rotation->vector4d.w);
  }

  //posteriors plot
  sprintf(str, "/Users/luma/Code/Luma/PWP3D/iles/Results/posteriors.png");
  VisualisationEngine::Instance()->GetImage(result, GETIMAGE_POSTERIORS, objects[objectIdx], views[viewIdx], objects[objectIdx]->pose[viewIdx]);
  ImageUtils::Instance()->SaveImageToFile(result, str);

	//primary engine destructor
	OptimisationEngine::Instance()->Shutdown();

	for (i = 0; i<objectCount; i++) delete objects[i];
	delete objects;

	for (i = 0; i<viewCount; i++) delete views[i];
	delete views;

	delete result;

	return 0;
}
