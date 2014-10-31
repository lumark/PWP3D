#pragma once

#include <PerseusLib/Others/PerseusLibDefines.h>

#include <stdio.h>
#include <string>
#include <vector>
#include <math.h>

namespace Renderer
{
namespace Primitives
{
class Camera3D
{
public:
  VFLOAT cameraParameters[5];

  VFLOAT invFocal[2];
  VFLOAT d2Tan, dOneOver2Tan, dWinv, dLargestRadius, dMaxR;
  VFLOAT dLastR, dLastFactor, dLastDistR;

  VFLOAT focalLength[2];
  VFLOAT centerPoint[2];
  VFLOAT distortionW;

  std::vector<float> vLastIm;
  std::vector<float> vLastCam;
  std::vector<float> vLastDistCam;

  float invrtrans(float r)
  {
    if (distortionW == 0.0f) return r;
    return (tanf(r * distortionW) * dOneOver2Tan);
  }

  float rtrans_factor(float r)
  {
    if ( r <0.001f || distortionW == 0.0f ) return 1.0f;
    return (dWinv * atanf(r * d2Tan) / r);
  }

  std::vector<float> makeVector(float val1, float val2) {
    std::vector<float> val(2);
    val[0] = val1;
    val[1] = val2;
    return val;
  }

  std::vector<float> Project(const std::vector<float> &vCam)
  {
    float dLastR = sqrtf(vCam[0] * vCam[0] + vCam[1] * vCam[1]);
    float dLastFactor = rtrans_factor(dLastR);

    vLastIm[0] = centerPoint[0] + focalLength[0] * dLastFactor * vCam[0];
    vLastIm[1] = centerPoint[1] + focalLength[1] * dLastFactor * vCam[1];

    return vLastIm;
  }

  void Project(float* vInput, float *vOutput)
  {
    float dLastR = sqrtf(vInput[0] * vInput[0] + vInput[1] * vInput[1]);
    float dLastFactor = rtrans_factor(dLastR);

    vOutput[0] = centerPoint[0] + focalLength[0] * dLastFactor * vInput[0];
    vOutput[1] = centerPoint[1] + focalLength[1] * dLastFactor * vInput[1];
  }

  std::vector<float> UnProject(const std::vector<float> &v2Im)
  {
    vLastDistCam[0] = (v2Im[0] - centerPoint[0]) * invFocal[0];
    vLastDistCam[1] = (v2Im[1] - centerPoint[1]) * invFocal[1];
    dLastDistR = sqrtf(vLastDistCam[0] * vLastDistCam[0] + vLastDistCam[1] * vLastDistCam[1]);
    dLastR = invrtrans(dLastDistR);

    float dFactor;
    if (dLastDistR > 0.01f) dFactor = dLastR / dLastDistR;
    else dFactor = 1.0f;

    dLastFactor = 1.0f / dFactor;
    vLastCam[0] = dFactor * vLastDistCam[0];
    vLastCam[1] = dFactor * vLastDistCam[1];

    return vLastCam;
  }

public:
  VFLOAT SizeX;
  VFLOAT SizeY;
  VFLOAT K[3][4], KGL[3][4];
  VFLOAT vImplaneTL[2], vImplaneBR[2];

  std::string CameraName;

  Camera3D(char *fileName)
  {
    vLastIm = std::vector<float>(2);
    vLastCam = std::vector<float>(2);
    vLastDistCam = std::vector<float>(2);

    VBYTE cameraName[100];
    int i, j;
    FILE* f = fopen(fileName, "r");
    fscanf(f, "%s", cameraName);

    fscanf(f, "%f %f", &SizeX, &SizeY);
    fscanf(f, "%f %f", &focalLength[0], &focalLength[1]);
    fscanf(f, "%f %f", &centerPoint[0], &centerPoint[1]);
    fclose(f);

    for (i=0; i<3; i++) for (j=0; j<4; j++) K[i][j] = 0;
    for (i=0; i<3; i++) for (j=0; j<4; j++) KGL[i][j] = 0;

    K[0][0] = focalLength[0]; K[1][1] = focalLength[1];
    K[0][2] = centerPoint[0]; K[1][2] = centerPoint[1];
    K[2][2] = 1;

    KGL[0][0] = focalLength[0]; KGL[1][1] = focalLength[1];
    KGL[0][2] = -centerPoint[0]; KGL[1][2] = -centerPoint[1];
    KGL[2][2] = -1;

    for (i = 0; i < 4; i++) K[1][i] = (SizeY - 1) * (K[2][i]) - K[1][i];
    for (i = 0; i < 4; i++) KGL[1][i] = (SizeY - 1) * (KGL[2][i]) - KGL[1][i];
  }

  Camera3D(float fSizeX, float fSizeY, float fFocalLengthX,
           float fFocalLengthY, float fCenterPointX, float fCenterPointY)
  {
    vLastIm = std::vector<float>(2);
    vLastCam = std::vector<float>(2);
    vLastDistCam = std::vector<float>(2);

    int i, j;

    SizeX = fSizeX;
    SizeY = fSizeY;
    focalLength[0] = fFocalLengthX;
    focalLength[1] = fFocalLengthY;
    centerPoint[0] = fCenterPointX;
    centerPoint[1] = fCenterPointY;

    for (i=0; i<3; i++) for (j=0; j<4; j++) K[i][j] = 0;
    for (i=0; i<3; i++) for (j=0; j<4; j++) KGL[i][j] = 0;

    K[0][0] = focalLength[0]; K[1][1] = focalLength[1];
    K[0][2] = centerPoint[0]; K[1][2] = centerPoint[1];
    K[2][2] = 1;

    KGL[0][0] = focalLength[0]; KGL[1][1] = focalLength[1];
    KGL[0][2] = -centerPoint[0]; KGL[1][2] = -centerPoint[1];
    KGL[2][2] = -1;

    for (i = 0; i < 4; i++) K[1][i] = (SizeY - 1) * (K[2][i]) - K[1][i];
    for (i = 0; i < 4; i++) KGL[1][i] = (SizeY - 1) * (KGL[2][i]) - KGL[1][i];
  }


  ~Camera3D(void) { }
};
}
}
