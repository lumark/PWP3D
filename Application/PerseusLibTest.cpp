#include "../PerseusLib/PerseusLib.h"

#include "Utils/Timer.h"
#include <opencv2/opencv.hpp>

using namespace Perseus::Utils;

int main(void)
{
//    std::string sModelPath = "/Users/luma/Code/Luma/PWP3D/Files/Models/Renderer/long.obj";
//  std::string sModelPath = "/Users/luma/Code/DataSet/Mesh/BlueCar.obj";
    std::string sModelPath = "/Users/luma/Code/DataSet/Mesh/RedCan.obj";

  std::string sSrcImage = "/Users/luma/Code/Luma/PWP3D/Files/Images/Red.png";
  std::string sCameraMatrix = "/Users/luma/Code/Luma/PWP3D/Files/CameraCalibration/900nc.cal";
  std::string sTargetMask = "/Users/luma/Code/Luma/PWP3D/Files/Masks/480p_All_VideoMask.png";
  std::string sHistSrc = "/Users/luma/Code/Luma/PWP3D/Files/Masks/Red_Source.png";
  std::string sHistMask = "/Users/luma/Code/Luma/PWP3D/Files/Masks/Red_Mask.png";

  char str[100];
  int i;

  int width = 640, height = 480;
  int viewCount = 1, objectCount = 1;
  int objectId = 0, viewIdx = 0, objectIdx = 0;

  Timer t;

  //result visualisation
  ImageUChar4* ResultImage = new ImageUChar4(width, height);

  // ---------------------------------------------------------------------------
  //input image
  //camera = 24 bit colour rgb
  ImageUChar4* camera = new ImageUChar4(width, height);
  ImageUtils::Instance()->LoadImageFromFile(camera, (char*)sSrcImage.c_str());

  //objects allocation + initialisation: 3d model in obj required
  Object3D **objects = new Object3D*[objectCount];

  std::cout<<"\n==[APP] Init Model =="<<std::endl;
  objects[objectIdx] = new Object3D(objectId, viewCount, (char*)sModelPath.c_str(), width, height);

  // ---------------------------------------------------------------------------
  //views allocation + initialisation: camera calibration (artoolkit format) required
  std::cout<<"\n==[APP] Init CameraMatrix =="<<std::endl;
  View3D **views = new View3D*[viewCount];
  views[viewIdx] = new View3D(0, (char*)sCameraMatrix.c_str(), width, height);


  // ---------------------------------------------------------------------------
  //histogram initialisation
  //source = 24 bit colour rgb
  //mask = 24 bit black/white png - white represents object
  //videoMask = 24 bit black/white png - white represents parts of the image that are usable
  std::cout<<"\n==[APP] Init Target ROI =="<<std::endl;
  ImageUtils::Instance()->LoadImageFromFile(views[viewIdx]->videoMask,(char*)sTargetMask.c_str());

  ImageUtils::Instance()->LoadImageFromFile(objects[objectIdx]->histSources[viewIdx],
                                            (char*)sHistSrc.c_str());

  ImageUtils::Instance()->LoadImageFromFile(objects[objectIdx]->histMasks[viewIdx],
                                            (char*)sHistMask.c_str(), objectIdx+1);

  HistogramEngine::Instance()->UpdateVarBinHistogram(
        objects[objectIdx], views[viewIdx], objects[objectIdx]->histSources[viewIdx],
        objects[objectIdx]->histMasks[viewIdx], views[viewIdx]->videoMask);


  // ---------------------------------------------------------------------------
  //iteration configuration for one object
  IterationConfiguration *iterConfig = new IterationConfiguration();
  iterConfig->width = width; iterConfig->height = height;
  iterConfig->iterViewIds[viewIdx] = 0;
  iterConfig->iterObjectCount[viewIdx] = 1;
  iterConfig->levelSetBandSize = 8;
  iterConfig->iterObjectIds[viewIdx][objectIdx] = 0;
  iterConfig->iterViewCount = 1;
  iterConfig->iterCount = 1;

  //step size per object and view
  objects[objectIdx]->stepSize[viewIdx] = new StepSize3D(0.2f, 0.5f, 0.5f, 10.0f);

  //initial pose per object and view
  objects[objectIdx]->initialPose[viewIdx]->SetFrom(
        -1.98f, -2.90f, 37.47f, -40.90f, -207.77f, 27.48f);

  //primary initilisation
  OptimisationEngine::Instance()->Initialise(width, height);

  //register camera image with main engine
  OptimisationEngine::Instance()->RegisterViewImage(views[viewIdx], camera);

  // ---------------------------------------------------------------------------
  std::cout<<"\n==[APP] Rendering object initial pose.. =="<<std::endl;
  VisualisationEngine::Instance()->GetImage(
        ResultImage, GETIMAGE_PROXIMITY,
        objects[objectIdx], views[viewIdx],
        objects[objectIdx]->initialPose[viewIdx]);

  cv::Mat ResultMat(height,width,CV_8UC4, ResultImage->pixels);
  cv::imshow("initial pose", ResultMat);
  cv::waitKey(2000);

  std::cout<<"[App] Finish Rendered object initial pose."<<std::endl;

  for (i=0; i<1; i++)
  {
    switch (i)
    {
    case 0:
      iterConfig->useCUDAEF = true;
      iterConfig->useCUDARender = true;
      break;
    case 1:
      iterConfig->useCUDAEF = false;
      iterConfig->useCUDARender = true;
      break;
    case 2:
      iterConfig->useCUDAEF = true;
      iterConfig->useCUDARender = false;
      break;
    case 3:
      iterConfig->useCUDAEF = true;
      iterConfig->useCUDARender = true;
      break;
    }

    printf("===== mode: useCUDAAEF: %d, use CUDARender %d =====;\n",
           iterConfig->useCUDAEF, iterConfig->useCUDARender);

    sprintf(str, "/Users/luma/Code/Luma/PWP3D/Files/Results/result%04d.png", i);

    //main processing
    t.restart();
    OptimisationEngine::Instance()->Minimise(objects, views, iterConfig);
    t.check("Iteration");

    //result plot
    VisualisationEngine::Instance()->GetImage(
          ResultImage, GETIMAGE_PROXIMITY,
          objects[objectIdx], views[viewIdx], objects[objectIdx]->pose[viewIdx]);

    //result save to file
    //    ImageUtils::Instance()->SaveImageToFile(result, str);
    cv::Mat ResultMat(height,width,CV_8UC4, ResultImage->pixels);
    cv::imshow("result", ResultMat);
    cv::waitKey(10000);

    printf("final pose result %f %f %f %f %f %f %f\n\n",
           objects[objectIdx]->pose[viewIdx]->translation->x,
           objects[objectIdx]->pose[viewIdx]->translation->y,
           objects[objectIdx]->pose[viewIdx]->translation->z,
           objects[objectIdx]->pose[viewIdx]->rotation->vector4d.x,
           objects[objectIdx]->pose[viewIdx]->rotation->vector4d.y,
           objects[objectIdx]->pose[viewIdx]->rotation->vector4d.z,
           objects[objectIdx]->pose[viewIdx]->rotation->vector4d.w);
  }

  //posteriors plot
  sprintf(str, "/Users/luma/Code/Luma/PWP3D/iles/Results/posteriors.png");
  VisualisationEngine::Instance()->GetImage(
        ResultImage, GETIMAGE_POSTERIORS,
        objects[objectIdx], views[viewIdx], objects[objectIdx]->pose[viewIdx]);

  ImageUtils::Instance()->SaveImageToFile(ResultImage, str);

  //primary engine destructor
  OptimisationEngine::Instance()->Shutdown();

  for (i = 0; i<objectCount; i++) delete objects[i];
  delete objects;

  for (i = 0; i<viewCount; i++) delete views[i];
  delete views;

  delete ResultImage;

  std::cout<<"Exit pwp3D app successfully."<<std::endl;

  return 0;
}
