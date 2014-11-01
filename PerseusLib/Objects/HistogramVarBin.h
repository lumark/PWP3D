#pragma once

#include <string.h>
#include <PerseusLib/Others/PerseusLibDefines.h>
#include <PerseusLib/CUDA/CUDADefines.h>

namespace PerseusLib
{
namespace Objects
{
class HistogramVarBin
{
private:
public:
  int fullHistSize;
  int *histOffsets;

  float mergeAlphaForeground;
  float mergeAlphaBackground;

  float2* normalised;
  float2* notnormalised;

  float2* normalisedGPU;

  int *noBins, *factor, noHistograms;
  bool isAllocated, alreadyInitialised;

  float totalForegroundPixels, totalBackgroundPixels;
  float etaF, etaB;

  HistogramVarBin()
  {
    isAllocated = false;
  }

  void Set(int noHistograms, int *noBins)
  {
    if(noHistograms <= 0)
    {
      printf("fatal error! noHistograms must >0\n");
      exit(-1);
    }

    if (!isAllocated)
    {
      this->isAllocated = true;
      this->noHistograms = noHistograms;

      fullHistSize = 0;
      this->noBins = new int[noHistograms];
      this->histOffsets = new int[noHistograms];

      for (int i=0; i<noHistograms; i++)
      {
        this->noBins[i] = noBins[i];
        histOffsets[i] = fullHistSize;
        fullHistSize += noBins[i] * noBins[i] * noBins[i];
      }

      normalised = new float2[fullHistSize];
      notnormalised = new float2[fullHistSize];

      cudaMalloc((void**)&normalisedGPU, fullHistSize * sizeof(float2));

      // TODO MAKE NICER
      factor = new int[noHistograms];
      factor[0] = 5;
      factor[1] = 4;
      factor[2] = 3;
      factor[3] = 2;

      this->Clear();
    }
    else this->Clear();
  }

  void GetValue(float* foreground, float *background, int r, int g, int b, int x, int y)
  {
    int ru, gu, bu, pidx, currentHistogram;

    int greyVal = int(float(r) * 0.3f + float(g) * 0.59f + float(b) * 0.11f);

    currentHistogram = 0;
    if (greyVal < 128) currentHistogram = 3;
    else if (greyVal < 192) currentHistogram = 2;
    else if (greyVal < 224) currentHistogram = 1;

    //currentHistogram = 2;

    ru = (r >> factor[currentHistogram]) & (noBins[currentHistogram] - 1);
    gu = (g >> factor[currentHistogram]) & (noBins[currentHistogram] - 1);
    bu = (b >> factor[currentHistogram]) & (noBins[currentHistogram] - 1);
    pidx = (ru + gu * noBins[currentHistogram]) * noBins[currentHistogram] + bu;

    *foreground = normalised[histOffsets[currentHistogram] + pidx].x;
    *background = normalised[histOffsets[currentHistogram] + pidx].y;
  }

  void AddPoint(float foreground, float background, int r, int g, int b, int x, int y)
  {
    int i, ru, gu, bu, pidx;

    for (i=0; i<noHistograms; i++)
    {
      ru = (r >> factor[i]) & (noBins[i] - 1);
      gu = (g >> factor[i]) & (noBins[i] - 1);
      bu = (b >> factor[i]) & (noBins[i] - 1);
      pidx = (ru + gu * noBins[i]) * noBins[i] + bu;

      notnormalised[histOffsets[i] + pidx].x += foreground;
      notnormalised[histOffsets[i] + pidx].y += background;
    }

    totalForegroundPixels += foreground;
    totalBackgroundPixels += background;

    if (!alreadyInitialised)
    {
      etaF += foreground;
      etaB += background;
    }
  }

  void Clear()
  {
    totalForegroundPixels = 0;
    totalBackgroundPixels = 0;
    etaF = 0;
    etaB = 0;

    memset(normalised, 0, fullHistSize * 2 * sizeof(float));
    memset(notnormalised, 0, fullHistSize * 2 * sizeof(float));

    alreadyInitialised = false;
  }

  void ClearNormalised()
  {
    memset(normalised, 0, fullHistSize * 2 * sizeof(float));
    alreadyInitialised = false;

    totalForegroundPixels = 0;
    totalBackgroundPixels = 0;
    etaF = 0;
    etaB = 0;
  }

  void ClearNotNormalised()
  {
    memset(notnormalised, 0, fullHistSize * 2 * sizeof(float));

    totalForegroundPixels = 0;
    totalBackgroundPixels = 0;
    etaF = 0;
    etaB = 0;
  }

  void ClearNotNormalisedPartial()
  {
    memset(notnormalised, 0, fullHistSize * 2 * sizeof(float));

    totalForegroundPixels = 0;
    totalBackgroundPixels = 0;
  }

  void UpdateGPUFromCPU()
  {
    cudaMemcpy(normalisedGPU, normalised, sizeof(float2) * fullHistSize, cudaMemcpyHostToDevice);
  }

  void Free()
  {
    if (this->isAllocated)
    {
      delete normalised;
      delete notnormalised;

      cudaFree(normalisedGPU);
    }

    this->isAllocated = false;
  }

  ~HistogramVarBin() { this->Free(); }
};
}
}
