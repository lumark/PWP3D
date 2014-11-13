#include "Quaternion.h"
#include <math.h>

using namespace Renderer::Primitives;

void Quaternion::FromEuler(VFLOAT rotationX, VFLOAT rotationY, VFLOAT rotationZ)
{
  rotationX = rotationX * (VFLOAT) DEGTORAD;
  rotationY = rotationY * (VFLOAT) DEGTORAD;
  rotationZ = rotationZ * (VFLOAT) DEGTORAD;

  VFLOAT c1 = cos(rotationY / 2.f);
  VFLOAT c2 = cos(rotationZ / 2.f);
  VFLOAT c3 = cos(rotationX / 2.f);

  VFLOAT s1 = sin(rotationY / 2.f);
  VFLOAT s2 = sin(rotationZ / 2.f);
  VFLOAT s3 = sin(rotationX / 2.f);

  VFLOAT c1c2 = c1*c2;
  VFLOAT s1s2 = s1*s2;
  this->vector4d.w = c1c2*c3 - s1s2*s3;
  this->vector4d.x = c1c2*s3 + s1s2*c3;
  this->vector4d.y = s1*c2*c3 + c1*s2*s3;
  this->vector4d.z = c1*s2*c3 - s1*c2*s3;

  VFLOAT norm = 1 / sqrtf(vector4d.x * vector4d.x + vector4d.y * vector4d.y + vector4d.z * vector4d.z + vector4d.w * vector4d.w);
  vector4d = VECTOR4DA(vector4d.x * norm, vector4d.y * norm, vector4d.z * norm, vector4d.w * norm);
}

void Quaternion::FromMatrix(VFLOAT* rotMatrix)
{
  float matrix[3][3];
  for (int i=0; i<3; i++) for (int j=0; j<3; j++) matrix[j][i] = rotMatrix[i + j * 3]; //was [j][i]

  float trace = matrix[0][0] + matrix[1][1] + matrix[2][2];
  if( trace > 0 )
  {
    float s = 0.5f / sqrtf(trace + 1.0f);
    vector4d.w = 0.25f / s;
    vector4d.x = ( matrix[2][1] - matrix[1][2] ) * s;
    vector4d.y = ( matrix[0][2] - matrix[2][0] ) * s;
    vector4d.z = ( matrix[1][0] - matrix[0][1] ) * s;
  } else {
    if ( matrix[0][0] > matrix[1][1] && matrix[0][0] > matrix[2][2] ) {
      float s = 2.0f * sqrtf( 1.0f + matrix[0][0] - matrix[1][1] - matrix[2][2]);
      vector4d.w = (matrix[2][1] - matrix[1][2] ) / s;
      vector4d.x = 0.25f * s;
      vector4d.y = (matrix[0][1] + matrix[1][0] ) / s;
      vector4d.z = (matrix[0][2] + matrix[2][0] ) / s;
    } else if (matrix[1][1] > matrix[2][2]) {
      float s = 2.0f * sqrtf( 1.0f + matrix[1][1] - matrix[0][0] - matrix[2][2]);
      vector4d.w = (matrix[0][2] - matrix[2][0] ) / s;
      vector4d.x = (matrix[0][1] + matrix[1][0] ) / s;
      vector4d.y = 0.25f * s;
      vector4d.z = (matrix[1][2] + matrix[2][1] ) / s;
    } else {
      float s = 2.0f * sqrtf( 1.0f + matrix[2][2] - matrix[0][0] - matrix[1][1] );
      vector4d.w = (matrix[1][0] - matrix[0][1] ) / s;
      vector4d.x = (matrix[0][2] + matrix[2][0] ) / s;
      vector4d.y = (matrix[1][2] + matrix[2][1] ) / s;
      vector4d.z = 0.25f * s;
    }
  }
}

void Quaternion::ToOpenGLMatrix(VFLOAT* M)
{
  VFLOAT m[4][4];

  toOpenGLMatrix(m);

  m[0][3] = 0;
  m[1][3] = 0;
  m[2][3] = 0;

  m[3][0] = 0;
  m[3][1] = 0;
  m[3][2] = 0;

  m[3][3] = 1;

  matrixToOpenGLMatrix(m, M);

  DEBUGBREAK;
}

void Quaternion::toOpenGLMatrix(VFLOAT m[][4])
{
  VFLOAT sqw = vector4d.w * vector4d.w;
  VFLOAT sqx = vector4d.x * vector4d.x;
  VFLOAT sqy = vector4d.y * vector4d.y;
  VFLOAT sqz = vector4d.z * vector4d.z;

  VFLOAT invs = 1 / (sqx + sqy + sqz + sqw);
  m[0][0] = ( sqx - sqy - sqz + sqw)*invs ;
  m[1][1] = (-sqx + sqy - sqz + sqw)*invs ;
  m[2][2] = (-sqx - sqy + sqz + sqw)*invs ;

  VFLOAT tmp1 = vector4d.x*vector4d.y;
  VFLOAT tmp2 = vector4d.z*vector4d.w;
  m[1][0] = (VFLOAT) 2.0 * (tmp1 + tmp2)*invs ;
  m[0][1] = (VFLOAT) 2.0 * (tmp1 - tmp2)*invs ;

  tmp1 = vector4d.x*vector4d.z;
  tmp2 = vector4d.y*vector4d.w;
  m[2][0] = (VFLOAT) 2.0 * (tmp1 - tmp2)*invs ;
  m[0][2] = (VFLOAT) 2.0 * (tmp1 + tmp2)*invs ;
  tmp1 = vector4d.y*vector4d.z;
  tmp2 = vector4d.x*vector4d.w;
  m[2][1] = (VFLOAT) 2.0 * (tmp1 + tmp2)*invs ;
  m[1][2] = (VFLOAT) 2.0 * (tmp1 - tmp2)*invs ;
}

void Quaternion::matrixToOpenGLMatrix(VFLOAT m[][4], VFLOAT* M)
{
  int i, j, k;
  k=0;
  for (i=0;i<4;i++)
  {
    for (j=0;j<4;j++)
    {
      M[k] = m[j][i];
      k++;
    }
  }
}

void Quaternion::GetDerivatives(VFLOAT* derivatives, VFLOAT* xUnprojected,
                                VFLOAT* xSource, VFLOAT* projectionParams,
                                VFLOAT* otherInfo)
{
  VFLOAT qx2, qy2, qz2, qw2;
  VFLOAT precalcX, precalcY, precalcXY;
  VFLOAT d0x, d0y, d0z;
  VFLOAT d1x, d1y, d1z;
  VFLOAT d2x, d2y, d2z;
  VFLOAT d3x, d3y, d3z;

  qx2 = 2 * vector4d.x; qy2 = 2 * vector4d.y; qz2 = 2 * vector4d.z; qw2 = 2 * vector4d.w;

  d0x = qy2*xSource[1] + qz2*xSource[2];
  d0y = qy2*xSource[0] - 2*qx2*xSource[1] - qw2*xSource[2];
  d0z = qz2*xSource[0] + qw2*xSource[1] - 2*qx2*xSource[2];

  d1x = qx2*xSource[1] - 2*qy2*xSource[0] + qw2*xSource[2];
  d1y = qx2*xSource[0] + qz2*xSource[2];
  d1z = qz2*xSource[1] - qw2*xSource[0] - 2*qy2*xSource[2];

  d2x = qx2*xSource[2] - qw2*xSource[1] - 2*qz2*xSource[0];
  d2y = qw2*xSource[0] - 2*qz2*xSource[1] + qy2*xSource[2];
  d2z = qx2*xSource[0] + qy2*xSource[1];

  d3x = qy2*xSource[2] - qz2*xSource[1];
  d3y = qz2*xSource[0] - qx2*xSource[2];
  d3z = qx2*xSource[1] - qy2*xSource[0];

  precalcXY = xUnprojected[2] * xUnprojected[2];
  precalcX = -otherInfo[0] / precalcXY;
  precalcY = -otherInfo[1] / precalcXY;

  derivatives[0] = precalcX * (xUnprojected[2] * d0x - xUnprojected[0] * d0z) + precalcY * (xUnprojected[2] * d0y - xUnprojected[1] * d0z);
  derivatives[1] = precalcX * (xUnprojected[2] * d1x - xUnprojected[0] * d1z) + precalcY * (xUnprojected[2] * d1y - xUnprojected[1] * d1z);
  derivatives[2] = precalcX * (xUnprojected[2] * d2x - xUnprojected[0] * d2z) + precalcY * (xUnprojected[2] * d2y - xUnprojected[1] * d2z);
  derivatives[3] = precalcX * (xUnprojected[2] * d3x - xUnprojected[0] * d3z) + precalcY * (xUnprojected[2] * d3y - xUnprojected[1] * d3z);
}
