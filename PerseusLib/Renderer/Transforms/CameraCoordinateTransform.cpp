#include "CameraCoordinateTransform.h"

using namespace Renderer::Transforms;

CameraCoordinateTransform::CameraCoordinateTransform(void)
{
  projectionMatrix = new VFLOAT[16];
  projectionMatrixGL = new VFLOAT[16];

  for (int i=0; i<16; i++)
  {
    projectionMatrix[i] = 0;
    projectionMatrixGL[i] = 0;
  }
}

CameraCoordinateTransform::~CameraCoordinateTransform(void)
{
  delete projectionMatrix;
  delete projectionMatrixGL;
}

void CameraCoordinateTransform::SetProjectionMatrix(VFLOAT *projectionMatrix)
{
  this->projectionMatrix[0] = projectionMatrix[0];
  this->projectionMatrix[5] = projectionMatrix[5];
  this->projectionMatrix[8] = projectionMatrix[8];
  this->projectionMatrix[9] = projectionMatrix[9];
  this->projectionMatrix[10] = projectionMatrix[10];
  this->projectionMatrix[14] = projectionMatrix[14];
}

void CameraCoordinateTransform::SetProjectionMatrix()
{
  this->projectionMatrix[0] = 1;
  this->projectionMatrix[5] = 1;
  this->projectionMatrix[10] = 1;
  this->projectionMatrix[15] = 1;
}

void CameraCoordinateTransform::GetProjectionParameters(ProjectionParams *params)
{
  params->A = projectionMatrix[0];
  params->B = projectionMatrix[5];
  params->C = projectionMatrix[8];
  params->D = projectionMatrix[9];
  params->E = projectionMatrix[10];
  params->F = projectionMatrix[14];

  params->all[0] = params->A;
  params->all[1] = params->B;
  params->all[2] = params->C;
  params->all[3] = params->D;
  params->all[4] = params->E;
  params->all[5] = params->F;
}

void CameraCoordinateTransform::decompKMatrix(VFLOAT source[3][4], VFLOAT cpara[3][4], VFLOAT trans[3][4])
{
  int       r, c;
  VFLOAT    Cpara[3][4];
  VFLOAT    rem1, rem2, rem3;

  if( source[2][3] >= 0 )
  {
    for( r = 0; r < 3; r++ )
      for( c = 0; c < 4; c++ )
        Cpara[r][c] = source[r][c];
  }
  else
  {
    for( r = 0; r < 3; r++ )
      for( c = 0; c < 4; c++ )
        Cpara[r][c] = -(source[r][c]);
  }

  for( r = 0; r < 3; r++ )
    for( c = 0; c < 4; c++ )
      cpara[r][c] = 0.0;

  cpara[2][2] = norm( Cpara[2][0], Cpara[2][1], Cpara[2][2] );
  trans[2][0] = Cpara[2][0] / cpara[2][2];
  trans[2][1] = Cpara[2][1] / cpara[2][2];
  trans[2][2] = Cpara[2][2] / cpara[2][2];
  trans[2][3] = Cpara[2][3] / cpara[2][2];

  cpara[1][2] = dot( trans[2][0], trans[2][1], trans[2][2],
      Cpara[1][0], Cpara[1][1], Cpara[1][2] );
  rem1 = Cpara[1][0] - cpara[1][2] * trans[2][0];
  rem2 = Cpara[1][1] - cpara[1][2] * trans[2][1];
  rem3 = Cpara[1][2] - cpara[1][2] * trans[2][2];
  cpara[1][1] = norm( rem1, rem2, rem3 );
  trans[1][0] = rem1 / cpara[1][1];
  trans[1][1] = rem2 / cpara[1][1];
  trans[1][2] = rem3 / cpara[1][1];

  cpara[0][2] = dot( trans[2][0], trans[2][1], trans[2][2],
      Cpara[0][0], Cpara[0][1], Cpara[0][2] );
  cpara[0][1] = dot( trans[1][0], trans[1][1], trans[1][2],
      Cpara[0][0], Cpara[0][1], Cpara[0][2] );
  rem1 = Cpara[0][0] - cpara[0][1]*trans[1][0] - cpara[0][2]*trans[2][0];
  rem2 = Cpara[0][1] - cpara[0][1]*trans[1][1] - cpara[0][2]*trans[2][1];
  rem3 = Cpara[0][2] - cpara[0][1]*trans[1][2] - cpara[0][2]*trans[2][2];
  cpara[0][0] = norm( rem1, rem2, rem3 );
  trans[0][0] = rem1 / cpara[0][0];
  trans[0][1] = rem2 / cpara[0][0];
  trans[0][2] = rem3 / cpara[0][0];

  trans[1][3] = (Cpara[1][3] - cpara[1][2]*trans[2][3]) / cpara[1][1];
  trans[0][3] = (Cpara[0][3] - cpara[0][1]*trans[1][3]
      - cpara[0][2]*trans[2][3]) / cpara[0][0];

  for( r = 0; r < 3; r++ )
    for( c = 0; c < 3; c++ )
      cpara[r][c] /= cpara[2][2];
}

void CameraCoordinateTransform::SetProjectionMatrix(char *cameraCalibrationFile, VFLOAT zNear, VFLOAT zFar)
{
  Camera3D* cam3D = new Camera3D(cameraCalibrationFile);

  this->zFar = zFar;
  this->zNear = zNear;

  this->SetProjectionMatrix(cam3D, zNear, zFar);

  delete cam3D;
}

void CameraCoordinateTransform::SetProjectionMatrix(Camera3D* camera, VFLOAT zNear, VFLOAT zFar)
{
  int i,j;

  if(zNear<=0 || zFar<=0)
  {
    printf("Set projection matrix to:  zNear:%f, zFar:%f \n", zNear, zFar);
    printf("Error! Invalid projectionMatrix.\n");
    exit(-1);
  }

  VFLOAT icpara[3][4];
  VFLOAT trans[3][4];
  VFLOAT p[3][3];
  VFLOAT q[4][4], qGL[4][4];

  decompKMatrix(camera->K, icpara, trans);
  for (i = 0; i < 3; i++) for (j = 0; j < 3; j++) p[i][j] = icpara[i][j] / icpara[2][2];

  q[0][0] = (2.0f * p[0][0] / camera->SizeX);
  q[0][1] = (2.0f * p[0][1] / camera->SizeX);
  q[0][2] = ((2.0f * p[0][2] / camera->SizeX) - 1.0f);
  q[0][3] = 0.0f;

  q[1][0] = 0.0f;
  q[1][1] = -(2.0f * p[1][1] / camera->SizeY);
  q[1][2] = ((2.0f * p[1][2] / camera->SizeY) - 1.0f);
  q[1][3] = 0.0f;

  q[2][0] = 0.0f;
  q[2][1] = 0.0f;
  q[2][2] = (zFar + zNear) / (zFar - zNear);
  q[2][3] = -2.0f * zFar * zNear / (zFar - zNear);

  q[3][0] = 0.0f;
  q[3][1] = 0.0f;
  q[3][2] = 1.0f;
  q[3][3] = 0.0f;

  for (i = 0; i < 4; i++)
  {
    for (j = 0; j < 3; j++)
    { projectionMatrix[i + j * 4] = q[i][0] * trans[0][j] + q[i][1] * trans[1][j] + q[i][2] * trans[2][j]; }
    projectionMatrix[i + 3 * 4] = q[i][0] * trans[0][3] + q[i][1] * trans[1][3] + q[i][2] * trans[2][3] + q[i][3];
  }

  decompKMatrix(camera->KGL, icpara, trans);
  for (i = 0; i < 3; i++) for (j = 0; j < 3; j++) { p[i][j] = icpara[i][j] / icpara[2][2]; }

  qGL[0][0] = (2.0f * p[0][0] / camera->SizeX);
  qGL[0][1] = (2.0f * p[0][1] / camera->SizeX);
  qGL[0][2] = 1.0f - (2.0f * p[0][2] / camera->SizeX);
  qGL[0][3] = 0.0f;

  qGL[1][0] = 0.0f;
  qGL[1][1] = (2.0f * p[1][1] / camera->SizeY);
  qGL[1][2] = ((2.0f * p[1][2] / camera->SizeY) - 1.0f);
  qGL[1][3] = 0.0f;

  qGL[2][0] = 0.0f;
  qGL[2][1] = 0.0f;
  qGL[2][2] = (zFar + zNear) / (zNear - zFar);
  qGL[2][3] = 2.0f * zFar * zNear / (zNear - zFar);

  qGL[3][0] = 0.0f;
  qGL[3][1] = 0.0f;
  qGL[3][2] = -1.0f;
  qGL[3][3] = 0.0f;

  for (i = 0; i < 4; i++)
  {
    for (j = 0; j < 3; j++)
    { projectionMatrixGL[i + j * 4] = qGL[i][0] * trans[0][j] + qGL[i][1] * trans[1][j] + qGL[i][2] * trans[2][j]; }
    projectionMatrixGL[i + 3 * 4] = qGL[i][0] * trans[0][3] + qGL[i][1] * trans[1][3] + qGL[i][2] * trans[2][3] + qGL[i][3];
  }
}

void CameraCoordinateTransform::SetProjectionMatrix(VFLOAT fovy, VFLOAT aspect, VFLOAT zNear, VFLOAT zFar)
{
  this->zFar = zFar;
  this->zNear = zNear;
  this->fovy = fovy;

  memset(projectionMatrix, 0, 16 * sizeof(float));
  VFLOAT f = (VFLOAT) 1.0 / (VFLOAT) tan(PI/180 * fovy/2);
  projectionMatrix[0] = f/aspect;
  projectionMatrix[5] = f;
  projectionMatrix[10] = -1 * (zFar + zNear)/(zFar - zNear);
  projectionMatrix[11] = -1;
  projectionMatrix[14] = -2 * (zFar * zNear)/(zFar - zNear);
}

void CameraCoordinateTransform::GetInvPMatrix(VFLOAT* prod)
{
  MathUtils::Instance()->InvertMatrix4(prod, projectionMatrix);
}
