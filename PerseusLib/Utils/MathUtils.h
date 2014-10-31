#pragma once

#include <PerseusLib/Others/PerseusLibDefines.h>
#include <string>

#define MATH_USE_OPENMP 1
//#define MATH_USE_DIRECTX 2
//#define MATH_USE_VNL 3

namespace PerseusLib
{
namespace Utils
{
class MathUtils
{
private:
  static MathUtils* instance;
public:

  int heavisideSize;
  float *heavisideFunction;

  static MathUtils* Instance(void) {
    if (instance == NULL) instance = new MathUtils();
    return instance;
  }

  void ReadAndAllocateHeaviside(int heavisideSize, char* fileName);
  void DeallocateHeaviside();

  void MatrixVectorProduct(float **matrix, float *vector, float* output, int vectorSize);

  void SquareMatrixProduct(double *c, const double *a, const double *b, int dim);
  void SquareMatrixProduct(float *c, const float *a, const float *b, int dim);
  void SquareMatrixProduct(long double *c, const long double *a, const long double *b, int dim);

  void MatrixVectorProduct4Inplace(double *matrix, double *vector);
  void MatrixVectorProduct4Inplace(float *matrix, float *vector);
  void MatrixVectorProduct4Inplace(long double *matrix, long double *vector);

  void InvertMatrix4Pose(float *out, float *in);
  void TransposeSquareMatrix(float *in, float *out, int size);

  void MatrixVectorProduct4(double *matrix, double *vector, double *vectorOutput);
  void MatrixVectorProduct4(float *matrix, float *vector, float *vectorOutput) {
    vectorOutput[0] = matrix[0] * vector[0] + matrix[4] * vector[1] + matrix[8] * vector[2] + matrix[12] * vector[3];
    vectorOutput[1] = matrix[1] * vector[0] + matrix[5] * vector[1] + matrix[9] * vector[2] + matrix[13] * vector[3];
    vectorOutput[2] = matrix[2] * vector[0] + matrix[6] * vector[1] + matrix[10] * vector[2] + matrix[14] * vector[3];
    vectorOutput[3] = matrix[3] * vector[0] + matrix[7] * vector[1] + matrix[11] * vector[2] + matrix[15] * vector[3];

    float norm = 1.0f/vectorOutput[3];
    for (int i = 0; i < 4; i++) vectorOutput[i] *= norm;
  }
  void MatrixVectorProduct4(long double *matrix, long double *vector, long double *vectorOutput);

  void MatrixVectorProduct4(float *matrix, double *vector, float *vectorOutput);
  void MatrixVectorProduct4(float *matrix, float *vector, double *vectorOutput);

  void InvertMatrix4(float *dst, float *mat);
  void InvertMatrix4(double *dst, double *mat);
  void InvertMatrix4(long double *dst, long double *mat);

  int GetMinor(float **src, float **dest, int row, int col, int order);
  double CalcDeterminant( float **mat, int order);
  void InvertMatrix(float **Y, int order, float **A);

  void ConvertArray(double *dst, float* src, int dim);
  void ConvertArray(float *dst, double* src, int dim);

  float MatrixDeterminant3(float a[3][3]);
  void InvertMatrix3(float in[3][3], float a[3][3]);

  MathUtils(void);
  ~MathUtils(void);
};
}
}
