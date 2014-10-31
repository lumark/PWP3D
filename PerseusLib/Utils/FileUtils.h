#pragma once

//#include <io.h>
#include <string.h>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <vector>
#include <list>
#include <PerseusLib/Others/PerseusLibDefines.h>

namespace PerseusLib
{
namespace Utils
{
class FileUtils
{
private:
  static FileUtils* instance;
public:
  static FileUtils* Instance(void);

  char* TextFileRead(char *fn);
  int TextFileWrite(char *fn, char *s);

  void WriteToFile(double *matrix, int m, int n, char *fileName);
  void WriteToFile(double *matrix, int m, int n, double* vector, int v, char *fileName);
  void WriteToFile(std::vector<std::vector<double>> matrix, std::string name, std::string fileName);
  void WriteToFile(double *matrix, int m, int n, std::string name, char *fileName);
  void WriteToFile(double *vector, int n, std::string name, char *fileName);

  void WriteToFile(float *matrix, int m, int n, char *fileName);
  void WriteToFile(float *matrix, int m, int n, float* vector, int v, char *fileName);
  void WriteToFile(float *matrix, int m, int n, std::string name, char *fileName);
  void WriteToFile(std::vector<std::vector<float>> matrix, std::string name, std::string fileName);
  void WriteToFile(float *vector, int n, std::string name, char *fileName);

  void WriteToFile(float **matrix, int m, int n, char* objectName, char* fileName);

  void WriteToFile(int *vector, int n, std::string name, char *fileName);
  void WriteToFile(unsigned int *vector, int n, std::string name, char *fileName);
  void WriteToFile(unsigned char *vector, int n, std::string name, char *fileName);

  void WriteToFile(std::vector<std::vector<int>> matrix, std::string name, std::string fileName);

  void ReadFromFile(double *matrix, int &m, int &n, char *fileName);
  void ReadFromFile(double *matrix, int &m, int &n, double* vector, int &v, char *fileName);

  void ReadFromFile(float *matrix, int &m, int &n, char *fileName);
  void ReadFromFile(float *matrix, int &m, int &n, float* vector, int &v, char *fileName);

  void ReadFromFile(float *vector, int m, std::string fileName);

  FileUtils(void);
  ~FileUtils(void);
};
}
}
