#pragma once

#include <PerseusLib/Utils/ImageUtils.h>

#include <PerseusLib/Primitives/Vector2D.h>

#include <PerseusLib/Others/PerseusLibDefines.h>

using namespace PerseusLib::Primitives;
using namespace PerseusLib::Utils;

namespace Renderer
{
namespace Engine
{
class DrawingPrimitives
{
  static DrawingPrimitives* instance;
public:
  static DrawingPrimitives* Instance(void) {
    if (instance == NULL) instance = new DrawingPrimitives();
    return instance;
  }

  int sgn(int num) { if (num > 0) return(1); else if (num < 0) return(-1); else return(0); }

  void DrawLine(ImageUChar *image, int x1, int y1, int x2, int y2, VBYTE color);

  DrawingPrimitives(void);
  ~DrawingPrimitives(void);
};
}
}
