#include "MathUtils.h"
#include "FileUtils.h"

#include <math.h>

using namespace PerseusLib::Utils;

MathUtils* MathUtils::instance;

MathUtils::MathUtils(void)
{
}

MathUtils::~MathUtils(void)
{
}

void MathUtils::SquareMatrixProduct(double *c, const double *a, const double *b, int dim)
{
  int i, j, k;

  for (i = 0; i < dim; i++)
  {
    for (j = 0; j < dim; j++)
    {
      c[i + j*dim] = 0;
      for (k = 0; k < dim; k++) c[i + j*dim] += a[i + k*dim] * b[k + j*dim];
    }
  }
}

void MathUtils::MatrixVectorProduct4Inplace(double *matrix, double *vector)
{
  int i;
  double newVector[4];

  newVector[0] = matrix[0] * vector[0] + matrix[4] * vector[1] + matrix[8] * vector[2] + matrix[12] * vector[3];
  newVector[1] = matrix[1] * vector[0] + matrix[5] * vector[1] + matrix[9] * vector[2] + matrix[13] * vector[3];
  newVector[2] = matrix[2] * vector[0] + matrix[6] * vector[1] + matrix[10] * vector[2] + matrix[14] * vector[3];
  newVector[3] = matrix[3] * vector[0] + matrix[7] * vector[1] + matrix[11] * vector[2] + matrix[15] * vector[3];

  for (i=0;i<4;i++) vector[i] = newVector[i];

  double norm = 1.0/vector[3];
  for (i=0;i<4;i++) vector[i] *= norm ;
}

void MathUtils::MatrixVectorProduct4(double *matrix, double *vector, double *vectorOutput)
{
  int i;

  vectorOutput[0] = matrix[0] * vector[0] + matrix[4] * vector[1] + matrix[8] * vector[2] + matrix[12] * vector[3];
  vectorOutput[1] = matrix[1] * vector[0] + matrix[5] * vector[1] + matrix[9] * vector[2] + matrix[13] * vector[3];
  vectorOutput[2] = matrix[2] * vector[0] + matrix[6] * vector[1] + matrix[10] * vector[2] + matrix[14] * vector[3];
  vectorOutput[3] = matrix[3] * vector[0] + matrix[7] * vector[1] + matrix[11] * vector[2] + matrix[15] * vector[3];

  double norm = 1.0/vectorOutput[3];
  for (i=0;i<4;i++) vectorOutput[i] *= norm ;
}

void MathUtils::SquareMatrixProduct(float *c, const float *a, const float *b, int dim) // a*b = c
{
  int i, j, k;

  for (i = 0; i < dim; i++)
  {
    for (j = 0; j < dim; j++)
    {
      c[i + j*dim] = 0;
      for (k = 0; k < dim; k++) c[i + j*dim] += a[i + k*dim] * b[k + j*dim];
    }
  }
}

void MathUtils::MatrixVectorProduct(float **matrix, float *vector, float* output, int vectorSize)
{
  int i, j;
  for (i=0; i<vectorSize; i++)
  { output[i] = 0; for (j=0; j<vectorSize; j++) output[i] += matrix[j][i] * vector[j]; }
}

void MathUtils::MatrixVectorProduct4Inplace(float *matrix, float *vector)
{
  int i;
  float newVector[4];

  newVector[0] = matrix[0] * vector[0] + matrix[4] * vector[1] + matrix[8] * vector[2] + matrix[12] * vector[3];
  newVector[1] = matrix[1] * vector[0] + matrix[5] * vector[1] + matrix[9] * vector[2] + matrix[13] * vector[3];
  newVector[2] = matrix[2] * vector[0] + matrix[6] * vector[1] + matrix[10] * vector[2] + matrix[14] * vector[3];
  newVector[3] = matrix[3] * vector[0] + matrix[7] * vector[1] + matrix[11] * vector[2] + matrix[15] * vector[3];

  for (i=0;i<4;i++) vector[i] = newVector[i];

  float norm = 1.0f/vector[3];
  for (i=0;i<4;i++) vector[i] *= norm ;
}

void MathUtils::MatrixVectorProduct4(float *matrix, float *vector, double *vectorOutput)
{
  int i;

  vectorOutput[0] = matrix[0] * vector[0] + matrix[4] * vector[1] + matrix[8] * vector[2] + matrix[12] * vector[3];
  vectorOutput[1] = matrix[1] * vector[0] + matrix[5] * vector[1] + matrix[9] * vector[2] + matrix[13] * vector[3];
  vectorOutput[2] = matrix[2] * vector[0] + matrix[6] * vector[1] + matrix[10] * vector[2] + matrix[14] * vector[3];
  vectorOutput[3] = matrix[3] * vector[0] + matrix[7] * vector[1] + matrix[11] * vector[2] + matrix[15] * vector[3];

  double norm = 1.0f/vectorOutput[3];
  for (i=0;i<4;i++) vectorOutput[i] *= norm ;
}

void MathUtils::MatrixVectorProduct4(float *matrix, double *vector, float *vectorOutput)
{
  int i;

  vectorOutput[0] = (float) (matrix[0] * vector[0] + matrix[4] * vector[1] + matrix[8] * vector[2] + matrix[12] * vector[3]);
  vectorOutput[1] = (float) (matrix[1] * vector[0] + matrix[5] * vector[1] + matrix[9] * vector[2] + matrix[13] * vector[3]);
  vectorOutput[2] = (float) (matrix[2] * vector[0] + matrix[6] * vector[1] + matrix[10] * vector[2] + matrix[14] * vector[3]);
  vectorOutput[3] = (float) (matrix[3] * vector[0] + matrix[7] * vector[1] + matrix[11] * vector[2] + matrix[15] * vector[3]);

  float norm = 1.0f/vectorOutput[3];
  for (i=0;i<4;i++) vectorOutput[i] *= norm ;
}

void MathUtils::SquareMatrixProduct(long double *c, const long double *a, const long double *b, int dim)
{
  int i, j, k;

  for (i = 0; i < dim; i++)
  {
    for (j = 0; j < dim; j++)
    {
      c[i + j*dim] = 0;
      for (k = 0; k < dim; k++) c[i + j*dim] += a[i + k*dim] * b[k + j*dim];
    }
  }
}

void MathUtils::MatrixVectorProduct4Inplace(long double *matrix, long double *vector)
{
  int i;
  long double newVector[4];

  newVector[0] = matrix[0] * vector[0] + matrix[4] * vector[1] + matrix[8] * vector[2] + matrix[12] * vector[3];
  newVector[1] = matrix[1] * vector[0] + matrix[5] * vector[1] + matrix[9] * vector[2] + matrix[13] * vector[3];
  newVector[2] = matrix[2] * vector[0] + matrix[6] * vector[1] + matrix[10] * vector[2] + matrix[14] * vector[3];
  newVector[3] = matrix[3] * vector[0] + matrix[7] * vector[1] + matrix[11] * vector[2] + matrix[15] * vector[3];

  for (i=0;i<4;i++) vector[i] = newVector[i];

  long double norm = 1.0f/vector[3];
  for (i=0;i<4;i++) vector[i] *= norm ;
}

void MathUtils::MatrixVectorProduct4(long double *matrix, long double *vector, long double *vectorOutput)
{
  int i;

  vectorOutput[0] = matrix[0] * vector[0] + matrix[4] * vector[1] + matrix[8] * vector[2] + matrix[12] * vector[3];
  vectorOutput[1] = matrix[1] * vector[0] + matrix[5] * vector[1] + matrix[9] * vector[2] + matrix[13] * vector[3];
  vectorOutput[2] = matrix[2] * vector[0] + matrix[6] * vector[1] + matrix[10] * vector[2] + matrix[14] * vector[3];
  vectorOutput[3] = matrix[3] * vector[0] + matrix[7] * vector[1] + matrix[11] * vector[2] + matrix[15] * vector[3];

  long double norm = 1.0f/vectorOutput[3];
  for (i=0;i<4;i++) vectorOutput[i] *= norm ;
}

void MathUtils::TransposeSquareMatrix(float *in, float *out, int size)
{
  int i, j;
  for (i=0; i<size; i++) for (j=0; j<size; j++)
    out[i + j * size] = in[j + i * size];
}

void MathUtils::InvertMatrix4Pose(float *out, float *in)
{
  int size = 4;
  int i, j;
  for (i=0; i<3; i++) for (j=0; j<3; j++) out[i + j * size] = in[j + i * size];
  j=3; for (i=0; i<3; i++) out[i + j * size] = -in[i+j*size];
  i=3; for (j=0; j<3; j++) out[i + j * size] = 0.f;
  out[15] = 1.0f;
}

//ftp://download.intel.com/design/PentiumIII/sml/24504301.pdf
void MathUtils::InvertMatrix4(float *dst, float *mat)
{
  float    tmp[12];
  float    src[16];
  float    det;

  for (int i = 0; i < 4; i++) {
    src[i]        = mat[i*4];
    src[i + 4]    = mat[i*4 + 1];
    src[i + 8]    = mat[i*4 + 2];
    src[i + 12]   = mat[i*4 + 3];
  }

  tmp[0]  = src[10] * src[15];
  tmp[1]  = src[11] * src[14];
  tmp[2]  = src[9]  * src[15];
  tmp[3]  = src[11] * src[13];
  tmp[4]  = src[9]  * src[14];
  tmp[5]  = src[10] * src[13];
  tmp[6]  = src[8]  * src[15];
  tmp[7]  = src[11] * src[12];
  tmp[8]  = src[8]  * src[14];
  tmp[9]  = src[10] * src[12];
  tmp[10] = src[8]  * src[13];
  tmp[11] = src[9]  * src[12];

  dst[0]  = tmp[0]*src[5] + tmp[3]*src[6] + tmp[4]*src[7];
  dst[0] -= tmp[1]*src[5] + tmp[2]*src[6] + tmp[5]*src[7];
  dst[1]  = tmp[1]*src[4] + tmp[6]*src[6] + tmp[9]*src[7];
  dst[1] -= tmp[0]*src[4] + tmp[7]*src[6] + tmp[8]*src[7];
  dst[2]  = tmp[2]*src[4] + tmp[7]*src[5] + tmp[10]*src[7];
  dst[2] -= tmp[3]*src[4] + tmp[6]*src[5] + tmp[11]*src[7];
  dst[3]  = tmp[5]*src[4] + tmp[8]*src[5] + tmp[11]*src[6];
  dst[3] -= tmp[4]*src[4] + tmp[9]*src[5] + tmp[10]*src[6];
  dst[4]  = tmp[1]*src[1] + tmp[2]*src[2] + tmp[5]*src[3];
  dst[4] -= tmp[0]*src[1] + tmp[3]*src[2] + tmp[4]*src[3];
  dst[5]  = tmp[0]*src[0] + tmp[7]*src[2] + tmp[8]*src[3];
  dst[5] -= tmp[1]*src[0] + tmp[6]*src[2] + tmp[9]*src[3];
  dst[6]  = tmp[3]*src[0] + tmp[6]*src[1] + tmp[11]*src[3];
  dst[6] -= tmp[2]*src[0] + tmp[7]*src[1] + tmp[10]*src[3];
  dst[7]  = tmp[4]*src[0] + tmp[9]*src[1] + tmp[10]*src[2];
  dst[7] -= tmp[5]*src[0] + tmp[8]*src[1] + tmp[11]*src[2];

  tmp[0]  = src[2]*src[7];
  tmp[1]  = src[3]*src[6];
  tmp[2]  = src[1]*src[7];
  tmp[3]  = src[3]*src[5];
  tmp[4]  = src[1]*src[6];
  tmp[5]  = src[2]*src[5];
  tmp[6]  = src[0]*src[7];
  tmp[7]  = src[3]*src[4];
  tmp[8]  = src[0]*src[6];
  tmp[9]  = src[2]*src[4];
  tmp[10] = src[0]*src[5];
  tmp[11] = src[1]*src[4];

  dst[8]  = tmp[0]*src[13] + tmp[3]*src[14] + tmp[4]*src[15];
  dst[8] -= tmp[1]*src[13] + tmp[2]*src[14] + tmp[5]*src[15];
  dst[9]  = tmp[1]*src[12] + tmp[6]*src[14] + tmp[9]*src[15];
  dst[9] -= tmp[0]*src[12] + tmp[7]*src[14] + tmp[8]*src[15];
  dst[10] = tmp[2]*src[12] + tmp[7]*src[13] + tmp[10]*src[15];
  dst[10]-= tmp[3]*src[12] + tmp[6]*src[13] + tmp[11]*src[15];
  dst[11] = tmp[5]*src[12] + tmp[8]*src[13] + tmp[11]*src[14];
  dst[11]-= tmp[4]*src[12] + tmp[9]*src[13] + tmp[10]*src[14];
  dst[12] = tmp[2]*src[10] + tmp[5]*src[11] + tmp[1]*src[9];
  dst[12]-= tmp[4]*src[11] + tmp[0]*src[9] + tmp[3]*src[10];
  dst[13] = tmp[8]*src[11] + tmp[0]*src[8] + tmp[7]*src[10];
  dst[13]-= tmp[6]*src[10] + tmp[9]*src[11] + tmp[1]*src[8];
  dst[14] = tmp[6]*src[9] + tmp[11]*src[11] + tmp[3]*src[8];
  dst[14]-= tmp[10]*src[11] + tmp[2]*src[8] + tmp[7]*src[9];
  dst[15] = tmp[10]*src[10] + tmp[4]*src[8] + tmp[9]*src[9];
  dst[15]-= tmp[8]*src[9] + tmp[11]*src[10] + tmp[5]*src[8];

  det=src[0]*dst[0]+src[1]*dst[1]+src[2]*dst[2]+src[3]*dst[3];

  det = 1.f/det;
  for (int j = 0; j < 16; j++) dst[j] *= det;
}

float MathUtils::MatrixDeterminant3(float a[3][3])
{
  return a[0][0]*(a[1][1]*a[2][2]-a[2][1]*a[1][2])-a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])+a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]);
}

void MathUtils::InvertMatrix3(float in[3][3], float a[3][3])
{
  float det = a[0][0]*(a[1][1]*a[2][2]-a[2][1]*a[1][2])-a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])+a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]);

  in[0][0]=(a[1][1]*a[2][2]-a[2][1]*a[1][2])/det;
  in[0][1]=-(a[1][0]*a[2][2]-a[1][2]*a[2][0])/det;
  in[0][2]=(a[1][0]*a[2][1]-a[2][0]*a[1][1])/det;
  in[1][0]=-(a[0][1]*a[2][2]-a[0][2]*a[2][1])/det;
  in[1][1]=(a[0][0]*a[2][2]-a[0][2]*a[2][0])/det;
  in[1][2]=-(a[0][0]*a[2][1]-a[2][0]*a[0][1])/det;
  in[2][0]=(a[0][1]*a[1][2]-a[0][2]*a[1][1])/det;
  in[2][1]=-(a[0][0]*a[1][2]-a[1][0]*a[0][2])/det;
  in[2][2]=(a[0][0]*a[1][1]-a[1][0]*a[0][1])/det;
}

void MathUtils::InvertMatrix4(double *dst, double *mat)
{
  double    tmp[12];
  double    src[16];
  double    det;

  for (int i = 0; i < 4; i++) {
    src[i]        = mat[i*4];
    src[i + 4]    = mat[i*4 + 1];
    src[i + 8]    = mat[i*4 + 2];
    src[i + 12]   = mat[i*4 + 3];
  }

  tmp[0]  = src[10] * src[15];
  tmp[1]  = src[11] * src[14];
  tmp[2]  = src[9]  * src[15];
  tmp[3]  = src[11] * src[13];
  tmp[4]  = src[9]  * src[14];
  tmp[5]  = src[10] * src[13];
  tmp[6]  = src[8]  * src[15];
  tmp[7]  = src[11] * src[12];
  tmp[8]  = src[8]  * src[14];
  tmp[9]  = src[10] * src[12];
  tmp[10] = src[8]  * src[13];
  tmp[11] = src[9]  * src[12];

  dst[0]  = tmp[0]*src[5] + tmp[3]*src[6] + tmp[4]*src[7];
  dst[0] -= tmp[1]*src[5] + tmp[2]*src[6] + tmp[5]*src[7];
  dst[1]  = tmp[1]*src[4] + tmp[6]*src[6] + tmp[9]*src[7];
  dst[1] -= tmp[0]*src[4] + tmp[7]*src[6] + tmp[8]*src[7];
  dst[2]  = tmp[2]*src[4] + tmp[7]*src[5] + tmp[10]*src[7];
  dst[2] -= tmp[3]*src[4] + tmp[6]*src[5] + tmp[11]*src[7];
  dst[3]  = tmp[5]*src[4] + tmp[8]*src[5] + tmp[11]*src[6];
  dst[3] -= tmp[4]*src[4] + tmp[9]*src[5] + tmp[10]*src[6];
  dst[4]  = tmp[1]*src[1] + tmp[2]*src[2] + tmp[5]*src[3];
  dst[4] -= tmp[0]*src[1] + tmp[3]*src[2] + tmp[4]*src[3];
  dst[5]  = tmp[0]*src[0] + tmp[7]*src[2] + tmp[8]*src[3];
  dst[5] -= tmp[1]*src[0] + tmp[6]*src[2] + tmp[9]*src[3];
  dst[6]  = tmp[3]*src[0] + tmp[6]*src[1] + tmp[11]*src[3];
  dst[6] -= tmp[2]*src[0] + tmp[7]*src[1] + tmp[10]*src[3];
  dst[7]  = tmp[4]*src[0] + tmp[9]*src[1] + tmp[10]*src[2];
  dst[7] -= tmp[5]*src[0] + tmp[8]*src[1] + tmp[11]*src[2];

  tmp[0]  = src[2]*src[7];
  tmp[1]  = src[3]*src[6];
  tmp[2]  = src[1]*src[7];
  tmp[3]  = src[3]*src[5];
  tmp[4]  = src[1]*src[6];
  tmp[5]  = src[2]*src[5];
  tmp[6]  = src[0]*src[7];
  tmp[7]  = src[3]*src[4];
  tmp[8]  = src[0]*src[6];
  tmp[9]  = src[2]*src[4];
  tmp[10] = src[0]*src[5];
  tmp[11] = src[1]*src[4];

  dst[8]  = tmp[0]*src[13] + tmp[3]*src[14] + tmp[4]*src[15];
  dst[8] -= tmp[1]*src[13] + tmp[2]*src[14] + tmp[5]*src[15];
  dst[9]  = tmp[1]*src[12] + tmp[6]*src[14] + tmp[9]*src[15];
  dst[9] -= tmp[0]*src[12] + tmp[7]*src[14] + tmp[8]*src[15];
  dst[10] = tmp[2]*src[12] + tmp[7]*src[13] + tmp[10]*src[15];
  dst[10]-= tmp[3]*src[12] + tmp[6]*src[13] + tmp[11]*src[15];
  dst[11] = tmp[5]*src[12] + tmp[8]*src[13] + tmp[11]*src[14];
  dst[11]-= tmp[4]*src[12] + tmp[9]*src[13] + tmp[10]*src[14];
  dst[12] = tmp[2]*src[10] + tmp[5]*src[11] + tmp[1]*src[9];
  dst[12]-= tmp[4]*src[11] + tmp[0]*src[9] + tmp[3]*src[10];
  dst[13] = tmp[8]*src[11] + tmp[0]*src[8] + tmp[7]*src[10];
  dst[13]-= tmp[6]*src[10] + tmp[9]*src[11] + tmp[1]*src[8];
  dst[14] = tmp[6]*src[9] + tmp[11]*src[11] + tmp[3]*src[8];
  dst[14]-= tmp[10]*src[11] + tmp[2]*src[8] + tmp[7]*src[9];
  dst[15] = tmp[10]*src[10] + tmp[4]*src[8] + tmp[9]*src[9];
  dst[15]-= tmp[8]*src[9] + tmp[11]*src[10] + tmp[5]*src[8];

  det=src[0]*dst[0]+src[1]*dst[1]+src[2]*dst[2]+src[3]*dst[3];

  det = static_cast<double>(1)/det;
  for (int j = 0; j < 16; j++) dst[j] *= det;
}

void MathUtils::InvertMatrix4(long double *dst, long double *mat)
{
  long double    tmp[12];
  long double    src[16];
  long double    det;

  for (int i = 0; i < 4; i++) {
    src[i]        = mat[i*4];
    src[i + 4]    = mat[i*4 + 1];
    src[i + 8]    = mat[i*4 + 2];
    src[i + 12]   = mat[i*4 + 3];
  }

  tmp[0]  = src[10] * src[15];
  tmp[1]  = src[11] * src[14];
  tmp[2]  = src[9]  * src[15];
  tmp[3]  = src[11] * src[13];
  tmp[4]  = src[9]  * src[14];
  tmp[5]  = src[10] * src[13];
  tmp[6]  = src[8]  * src[15];
  tmp[7]  = src[11] * src[12];
  tmp[8]  = src[8]  * src[14];
  tmp[9]  = src[10] * src[12];
  tmp[10] = src[8]  * src[13];
  tmp[11] = src[9]  * src[12];

  dst[0]  = tmp[0]*src[5] + tmp[3]*src[6] + tmp[4]*src[7];
  dst[0] -= tmp[1]*src[5] + tmp[2]*src[6] + tmp[5]*src[7];
  dst[1]  = tmp[1]*src[4] + tmp[6]*src[6] + tmp[9]*src[7];
  dst[1] -= tmp[0]*src[4] + tmp[7]*src[6] + tmp[8]*src[7];
  dst[2]  = tmp[2]*src[4] + tmp[7]*src[5] + tmp[10]*src[7];
  dst[2] -= tmp[3]*src[4] + tmp[6]*src[5] + tmp[11]*src[7];
  dst[3]  = tmp[5]*src[4] + tmp[8]*src[5] + tmp[11]*src[6];
  dst[3] -= tmp[4]*src[4] + tmp[9]*src[5] + tmp[10]*src[6];
  dst[4]  = tmp[1]*src[1] + tmp[2]*src[2] + tmp[5]*src[3];
  dst[4] -= tmp[0]*src[1] + tmp[3]*src[2] + tmp[4]*src[3];
  dst[5]  = tmp[0]*src[0] + tmp[7]*src[2] + tmp[8]*src[3];
  dst[5] -= tmp[1]*src[0] + tmp[6]*src[2] + tmp[9]*src[3];
  dst[6]  = tmp[3]*src[0] + tmp[6]*src[1] + tmp[11]*src[3];
  dst[6] -= tmp[2]*src[0] + tmp[7]*src[1] + tmp[10]*src[3];
  dst[7]  = tmp[4]*src[0] + tmp[9]*src[1] + tmp[10]*src[2];
  dst[7] -= tmp[5]*src[0] + tmp[8]*src[1] + tmp[11]*src[2];

  tmp[0]  = src[2]*src[7];
  tmp[1]  = src[3]*src[6];
  tmp[2]  = src[1]*src[7];
  tmp[3]  = src[3]*src[5];
  tmp[4]  = src[1]*src[6];
  tmp[5]  = src[2]*src[5];
  tmp[6]  = src[0]*src[7];
  tmp[7]  = src[3]*src[4];
  tmp[8]  = src[0]*src[6];
  tmp[9]  = src[2]*src[4];
  tmp[10] = src[0]*src[5];
  tmp[11] = src[1]*src[4];

  dst[8]  = tmp[0]*src[13] + tmp[3]*src[14] + tmp[4]*src[15];
  dst[8] -= tmp[1]*src[13] + tmp[2]*src[14] + tmp[5]*src[15];
  dst[9]  = tmp[1]*src[12] + tmp[6]*src[14] + tmp[9]*src[15];
  dst[9] -= tmp[0]*src[12] + tmp[7]*src[14] + tmp[8]*src[15];
  dst[10] = tmp[2]*src[12] + tmp[7]*src[13] + tmp[10]*src[15];
  dst[10]-= tmp[3]*src[12] + tmp[6]*src[13] + tmp[11]*src[15];
  dst[11] = tmp[5]*src[12] + tmp[8]*src[13] + tmp[11]*src[14];
  dst[11]-= tmp[4]*src[12] + tmp[9]*src[13] + tmp[10]*src[14];
  dst[12] = tmp[2]*src[10] + tmp[5]*src[11] + tmp[1]*src[9];
  dst[12]-= tmp[4]*src[11] + tmp[0]*src[9] + tmp[3]*src[10];
  dst[13] = tmp[8]*src[11] + tmp[0]*src[8] + tmp[7]*src[10];
  dst[13]-= tmp[6]*src[10] + tmp[9]*src[11] + tmp[1]*src[8];
  dst[14] = tmp[6]*src[9] + tmp[11]*src[11] + tmp[3]*src[8];
  dst[14]-= tmp[10]*src[11] + tmp[2]*src[8] + tmp[7]*src[9];
  dst[15] = tmp[10]*src[10] + tmp[4]*src[8] + tmp[9]*src[9];
  dst[15]-= tmp[8]*src[9] + tmp[11]*src[10] + tmp[5]*src[8];

  det=src[0]*dst[0]+src[1]*dst[1]+src[2]*dst[2]+src[3]*dst[3];

  det = static_cast<long double>(1)/det;
  for (int j = 0; j < 16; j++) dst[j] *= det;
}

void MathUtils::ConvertArray(double *dst, float* src, int dim)
{
  int i;
  for (i=0; i < dim; i++)
    dst[i] = src[i];
}
void MathUtils::ConvertArray(float *dst, double* src, int dim)
{
  int i;
  for (i=0; i < dim; i++)
    dst[i] = (float) src[i];
}

int MathUtils::GetMinor(float **src, float **dest, int row, int col, int order)
{
  int colCount=0,rowCount=0;

  for(int i = 0; i < order; i++ )
  {
    if( i != row )
    {
      colCount = 0;
      for(int j = 0; j < order; j++ ) { if( j != col ) { dest[rowCount][colCount] = src[i][j]; colCount++; } }
      rowCount++;
    }
  }

  return 1;
}

double MathUtils::CalcDeterminant( float **mat, int order)
{
  if( order == 1 ) return mat[0][0];

  double det = 0;

  float **minor;
  minor = new float*[order-1];
  for(int i=0;i<order-1;i++) minor[i] = new float[order-1];

  for(int i = 0; i < order; i++ )
  {
    GetMinor( mat, minor, 0, i , order);
    det += pow( -1.0, i ) * double(mat[0][i]) * CalcDeterminant( minor,order-1 );
  }

  for(int i=0;i<order-1;i++) delete [] minor[i]; delete [] minor;

  return det;
}


void MathUtils::InvertMatrix(float **Y, int order, float **A)
{
  double det = 1.0/CalcDeterminant(A,order);

  float *temp = new float[(order-1)*(order-1)];
  float **minor = new float*[order-1];
  for(int i=0;i<order-1;i++) minor[i] = temp+(i*(order-1));

  for(int j=0;j<order;j++)
  {
    for(int i=0;i<order;i++)
    {
      GetMinor(A,minor,j,i,order);
      Y[i][j] = float( det*CalcDeterminant(minor,order-1) );
      if( (i+j)%2 == 1) Y[i][j] = -Y[i][j];
    }
  }

  delete [] minor[0];
  delete [] minor;
}

void MathUtils::ReadAndAllocateHeaviside(int heavisideSize, char* fileName)
{
  this->heavisideSize = heavisideSize;
  heavisideFunction = new float[heavisideSize];
  FileUtils::Instance()->ReadFromFile(heavisideFunction, heavisideSize, fileName);
}

void MathUtils::DeallocateHeaviside()
{
  delete heavisideFunction;
}
