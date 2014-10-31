#pragma once

#include <stdlib.h>
#include <string.h>

namespace PerseusLib
{
namespace Primitives
{
class PixelUCHAR4
{
public:
  unsigned char x, y, z, w;

  void SetFrom(unsigned char x, unsigned char y, unsigned char z, unsigned char w)
  { this->x = x; this->y = y; this->z = z; this->w = w; }

  void SetFrom(unsigned char x, unsigned char y, unsigned char z)
  { this->SetFrom(x, y, z, 255); }

  void SetFrom(unsigned char *xyzw, int size)
  { if (size == 4) this->SetFrom(xyzw[0], xyzw[1], xyzw[2], xyzw[3]); else this->SetFrom(xyzw[0], xyzw[1], xyzw[2]); }

  //this is an ugly hack used to get the memset working :(
  operator int() {return x;}

  PixelUCHAR4(unsigned char value) { this->SetFrom(value,value,value,value); }
  PixelUCHAR4(unsigned char x, unsigned char y, unsigned char z, unsigned char w) { this->SetFrom(x,y,z,w); }
  PixelUCHAR4() { this->SetFrom(0,0,0,0); }
};
}
}
