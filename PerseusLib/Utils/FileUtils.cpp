#include "FileUtils.h"

using namespace PerseusLib::Utils;

FileUtils* FileUtils::instance;

FileUtils::FileUtils(void)
{
}

FileUtils::~FileUtils(void)
{
}

FileUtils* FileUtils::Instance(void)
{
  if (instance == NULL)
    instance = new FileUtils();

  return instance;
}

char* FileUtils::TextFileRead(char *fn)
{
  FILE *fp;
  char *content = NULL;

  int f, count;

  f = open(fn, O_RDONLY);

  count = lseek(f, 0, SEEK_END);

  close(f);

  if (fn != NULL)
  {
    fp = fopen(fn,"rt");

    if (fp != NULL)
    {
      if (count > 0)
      {
        content = (char *)malloc(sizeof(char) * (count+1));
        count = (int)fread(content,(size_t)sizeof(char),(size_t)count,fp);
        content[count] = '\0';
      }
      fclose(fp);
    }
  }
  return content;
}            

int FileUtils::TextFileWrite(char *fn, char *s)
{
  FILE *fp;
  int status = 0;

  if (fn != NULL)
  {
    fp = fopen(fn,"w");

    if (fp != NULL)
    {
      if (fwrite(s,sizeof(char),strlen(s),fp) == strlen(s))
        status = 1;
      fclose(fp);
    }
  }
  return(status);
} 

void FileUtils::WriteToFile(float **matrix, int m, int n, char* objectName, char* fileName)
{
  FILE *f = fopen(fileName, "w+");
  int i, j;
  fprintf(f, "%s_width=%d\n", objectName, m);
  fprintf(f, "%s_height=%d\n", objectName, n);
  fprintf(f, "%s=[\n",objectName);
  for (i=0; i<m; i++)
  {
    for (j=0; j<n; j++) fprintf(f, "%lf ", matrix[i][j]);
    if (i != m-1) fprintf(f, ";");
  }
  fprintf(f, "];\n");
  fclose(f);
}

void FileUtils::WriteToFile(double *matrix, int m, int n, char *fileName)
{
  FILE *f = fopen(fileName, "w+");
  int i, j;
  fprintf(f, "% PerseusLib 2D double array file\n");
  fprintf(f, "m=%d\n", m);
  fprintf(f, "n=%d\n", n);
  for (i=0; i<m; i++)
  {
    for (j=0; j<n; j++)
      fprintf(f, "%lf ", matrix[j + i*m]);
    fprintf(f, "\n");
  }
  fclose(f);
}

void FileUtils::ReadFromFile(double *matrix, int &m, int &n, char *fileName)
{
  FILE *f = fopen(fileName, "r+");
  int i, j;
  fscanf(f, "%s\n");
  fscanf(f, "m=%d\n", &m);
  fscanf(f, "n=%d\n", &n);
  matrix = new double[m * n];
  for (i=0; i<m; i++)
  {
    for (j=0; j<n; j++)
      fscanf(f, "%lf ", matrix[j + i*m]);
  }
  fclose(f);
}

void FileUtils::WriteToFile(float *matrix, int m, int n, char *fileName)
{
  FILE *f = fopen(fileName, "w+");
  int i, j;
  fprintf(f, "% PerseusLib 2D double array file\n");
  fprintf(f, "m=%d\n", m);
  fprintf(f, "n=%d\n", n);
  for (i=0; i<m; i++)
  {
    for (j=0; j<n; j++)
      fprintf(f, "%f ", matrix[j + i*m]);
    fprintf(f, "\n");
  }
  fclose(f);
}

void FileUtils::ReadFromFile(float *matrix, int &m, int &n, char *fileName)
{
  FILE *f = fopen(fileName, "r+");
  int i, j;
  fscanf(f, "%s\n");
  fscanf(f, "m=%d\n", &m);
  fscanf(f, "n=%d\n", &n);
  matrix = new float[m * n];
  for (i=0; i<m; i++)
  {
    for (j=0; j<n; j++)
      fscanf(f, "%f ", matrix[j + i*m]);
  }
  fclose(f);
}

void FileUtils::WriteToFile(double *matrix, int m, int n, double* vector, int v, char *fileName)
{
  FILE *f = fopen(fileName, "w+");
  int i, j;
  fprintf(f, "% PerseusLib 2D double array file\n");
  fprintf(f, "m=%d\n", m);
  fprintf(f, "n=%d\n", n);
  for (i=0; i<m; i++)
  {
    for (j=0; j<n; j++)
      fprintf(f, "%lf ", matrix[j + i*n]);
    fprintf(f, ";\n");
  }
  fprintf(f, "v=%d\n", v);
  for (i=0; i<v; i++)
    fprintf(f, "%lf ", vector[i]);
  fclose(f);
}

void FileUtils::ReadFromFile(double *matrix, int &m, int &n, double* vector, int &v, char *fileName)
{
  FILE *f = fopen(fileName, "r+");
  int i, j;
  fscanf(f, "%s\n");
  fscanf(f, "m=%d\n", &m);
  fscanf(f, "n=%d\n", &n);
  matrix = new double[m * n];
  for (i=0; i<m; i++)
  {
    for (j=0; j<n; j++)
      fscanf(f, "%lf ", matrix[j + i*m]);
  }
  fscanf(f, "v=%d\n", &v);
  vector = new double[v];
  for (i=0; i<v; i++)
    fscanf(f, "%lf ", vector[i]);
  fclose(f);
}

void FileUtils::WriteToFile(float *matrix, int m, int n, float* vector, int v, char *fileName)
{
  FILE *f = fopen(fileName, "w+");
  int i, j;
  fprintf(f, "% PerseusLib 2D double array file\n");
  fprintf(f, "m=%d\n", m);
  fprintf(f, "n=%d\n", n);
  for (i=0; i<m; i++)
  {
    for (j=0; j<n; j++)
      fprintf(f, "%f ", matrix[j + i*m]);
    fprintf(f, ";\n");
  }
  fprintf(f, "v=%d\n", v);
  for (i=0; i<v; i++)
    fprintf(f, "%f ", vector[i]);
  fclose(f);
}

void FileUtils::ReadFromFile(float *matrix, int &m, int &n, float* vector, int &v, char *fileName)
{
  FILE *f = fopen(fileName, "r+");
  int i, j;
  fscanf(f, "%s\n");
  fscanf(f, "m=%d\n", &m);
  fscanf(f, "n=%d\n", &n);
  matrix = new float[m * n];
  for (i=0; i<m; i++)
  {
    for (j=0; j<n; j++)
      fscanf(f, "%f ", matrix[j + i*m]);
  }
  fscanf(f, "v=%d\n", &v);
  vector = new float[v];
  for (i=0; i<v; i++)
    fscanf(f, "%f ", vector[i]);
  fclose(f);
}

void FileUtils::WriteToFile(std::vector<std::vector<double>> matrix, std::string name, std::string fileName)
{
  FILE *f = fopen(fileName.data(), "w+");
  int i,j;
  int m = (int)matrix.size();
  int n = (int)matrix[0].size();
  fprintf(f, "height_%s=%d;\n", name.data(), static_cast<int>(matrix.size()));
  fprintf(f, "width_%s=%d;\n", name.data(), static_cast<int>(matrix[0].size()));

  fprintf(f, "%s=[\n", name.data());
  for (i=0; i<m; i++)
  {
    for (j=0; j<n; j++)
      fprintf(f, "%lf ", matrix[i][j]);
    if (i < m-1)
      fprintf(f, ";\n");
    else
      fprintf(f, "\n");
  }
  fprintf(f, "];");

  fclose(f);
}

void FileUtils::WriteToFile(std::vector<std::vector<float>> matrix, std::string name, std::string fileName)
{
  FILE *f = fopen(fileName.data(), "w+");
  int i,j;
  int m = (int)matrix.size();
  int n = (int)matrix[0].size();
  fprintf(f, "height_%s=%d;\n", name.data(), static_cast<int>(matrix.size()) );
  fprintf(f, "width_%s=%d;\n", name.data(), static_cast<int>(matrix[0].size()) );

  fprintf(f, "%s=[\n", name.data());
  for (i=0; i<m; i++)
  {
    for (j=0; j<n; j++)
      fprintf(f, "%f ", matrix[i][j]);
    if (i < m-1)
      fprintf(f, ";\n");
    else
      fprintf(f, "\n");
  }
  fprintf(f, "];");

  fclose(f);
}
void FileUtils::WriteToFile(std::vector<std::vector<int>> matrix, std::string name, std::string fileName)
{
  FILE *f = fopen(fileName.data(), "w+");
  int i,j;
  int m = (int)matrix.size();
  int n = (int)matrix[0].size();
  fprintf(f, "height_%s=%d;\n", name.data(), static_cast<int>(matrix.size()) );
  fprintf(f, "width_%s=%d;\n", name.data(), static_cast<int>(matrix[0].size()) );

  fprintf(f, "%s=[\n", name.data());
  for (i=0; i<m; i++)
  {
    for (j=0; j<n; j++)
      fprintf(f, "%d ", matrix[i][j]);
    if (i < m-1)
      fprintf(f, ";\n");
    else
      fprintf(f, "\n");
  }
  fprintf(f, "];");

  fclose(f);
}

void FileUtils::WriteToFile(float *matrix, int m, int n, std::string name, char *fileName)
{
  FILE *f = fopen(fileName, "w+");
  int i, j;
  fprintf(f, "width_%s=%d;\n", name.data(), m);
  fprintf(f, "height_%s=%d;\n", name.data(), n);
  fprintf(f, "%s=[\n", name.data());
  for (i=0; i<m; i++)
  {
    for (j=0; j<n; j++)
      fprintf(f, "%f ", matrix[j + i*m]);
    if (i < m-1)
      fprintf(f, ";\n");
    else
      fprintf(f, "\n");
  }
  fprintf(f, "];\n");
  fprintf(f, "%s = %s';\n", name.data(), name.data());
  fclose(f);
}

void FileUtils::WriteToFile(double *matrix, int m, int n, std::string name, char *fileName)
{
  FILE *f = fopen(fileName, "w+");
  int i, j;
  fprintf(f, "width_%s=%d;\n", name.data(), m);
  fprintf(f, "height_%s=%d;\n", name.data(), n);
  fprintf(f, "%s=[\n", name.data());
  for (i=0; i<m; i++)
  {
    for (j=0; j<n; j++)
      fprintf(f, "%lf ", matrix[j + i*m]);
    if (i < m-1)
      fprintf(f, ";\n");
    else
      fprintf(f, "\n");
  }
  fprintf(f, "];\n");
  fprintf(f, "%s = %s';\n", name.data(), name.data());
  fclose(f);
}

void FileUtils::WriteToFile(float *vector, int n, std::string name, char *fileName)
{
  FILE *f = fopen(fileName, "w+");
  int j;
  fprintf(f, "width_%s=%d;\n", name.data(), n);
  fprintf(f, "%s=[", name.data());

  for (j=0; j<n; j++)
  {
    fprintf(f, "%f ", vector[j]);
    if (j != n-1)
      fprintf(f, " ");
  }

  fprintf(f, "];");
  fclose(f);
}

void FileUtils::WriteToFile(unsigned char *vector, int n, std::string name, char *fileName)
{
  FILE *f = fopen(fileName, "w+");
  int j;
  fprintf(f, "width_%s=%d;\n", name.data(), n);
  fprintf(f, "%s=[", name.data());

  for (j=0; j<n; j++)
  {
    fprintf(f, "%d ", int(vector[j]));
    if (j != n-1)
      fprintf(f, " ");
  }

  fprintf(f, "];");
  fclose(f);
}

void FileUtils::WriteToFile(int *vector, int n, std::string name, char *fileName)
{
  FILE *f = fopen(fileName, "w+");
  int j;
  fprintf(f, "width_%s=%d;\n", name.data(), n);
  fprintf(f, "%s=[", name.data());

  for (j=0; j<n; j++)
  {
    fprintf(f, "%d ", vector[j]);
    if (j != n-1)
      fprintf(f, " ");
  }

  fprintf(f, "];");
  fclose(f);
}
void FileUtils::WriteToFile(unsigned int *vector, int n, std::string name, char *fileName)
{
  FILE *f = fopen(fileName, "w+");
  int j;
  fprintf(f, "width_%s=%d;\n", name.data(), n);
  fprintf(f, "%s=[", name.data());

  for (j=0; j<n; j++)
  {
    fprintf(f, "%d ", vector[j]);
    if (j != n-1)
      fprintf(f, " ");
  }

  fprintf(f, "];");
  fclose(f);
}
void FileUtils::WriteToFile(double *vector, int n, std::string name, char *fileName)
{
  FILE *f = fopen(fileName, "w+");
  int j;
  fprintf(f, "width_%s=%d;\n", name.data(), n);
  fprintf(f, "%s=[", name.data());

  for (j=0; j<n; j++)
  {
    fprintf(f, "%lf ", vector[j]);
    if (j != n-1)
      fprintf(f, " ");
  }

  fprintf(f, "];");
  fclose(f);
}

void FileUtils::ReadFromFile(float *vector, int m, std::string fileName)
{
  FILE *f = fopen(fileName.data(), "r");
  int i;
  for (i=0; i<m; i++)
    fscanf(f, "%f", &vector[i]);
  fclose(f);
}
