#pragma once

#include <PerseusLib/Others/PerseusLibDefines.h>
#include <PerseusLib/Utils/ImageUtils.h>

using namespace PerseusLib::Utils;

namespace PerseusLib
{
namespace Objects
{
class ImageRender
{
public:
  bool isAllocated;

  int width, height;

  ImageUChar *imageFill;
  ImageUInt *imageZBuffer;
  ImageUInt *imageZBufferInverse;
  ImageUChar *imageObjects;

  void Clear()
  {
    imageFill->Clear();
  }

  void ClearZBuffer()
  {
    imageObjects->Clear();
    imageZBuffer->Clear(MAX_INT);
    imageZBufferInverse->Clear();
  }

  ImageRender(int width, int height, bool useCudaAlloc)
  {
    this->width = width;
    this->height = height;

    this->imageFill = new ImageUChar(width, height, useCudaAlloc);
    this->imageZBuffer = new ImageUInt(width, height, useCudaAlloc);
    this->imageZBufferInverse = new ImageUInt(width, height, useCudaAlloc);
    this->imageObjects = new ImageUChar(width, height, useCudaAlloc);

    isAllocated = true;
  }

  void Free()
  {
    if (isAllocated)
    {
      imageFill->Free();
      imageZBuffer->Free();
      imageZBufferInverse->Free();
      imageObjects->Free();
    }

    isAllocated = false;
  }

  ~ImageRender() { this->Free(); }
};
}
}
