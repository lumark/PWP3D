// Extremely Fast Line Algorithm Var E (Addition Fixed Point PreCalc)
// Copyright 2001-2, By Po-Han Lin


// Freely useable in non-commercial applications as long as credits 
// to Po-Han Lin and link to http://www.edepot.com is provided in 
// source code and can been seen in compiled executable.  
// Commercial applications please inquire about licensing the algorithms.
//
// Lastest version at http://www.edepot.com/phl.html
// This version is for standard displays (up to 65536x65536)
// For small display version (256x256) visit http://www.edepot.com/lineex.html

#include "DrawingPrimitives.h"
#include <math.h>

using namespace Renderer::Engine;

DrawingPrimitives* DrawingPrimitives::instance;

DrawingPrimitives::DrawingPrimitives(void)
{
}

DrawingPrimitives::~DrawingPrimitives(void)
{
}

void DrawingPrimitives::DrawLine(ImageUChar *image, int x, int y, int x2, int y2, VBYTE color)
{
  int sx, sy;
  VBOOL yLonger=false;
  int shortLen=y2-y;
  int longLen=x2-x;
  if (abs(shortLen)>abs(longLen))
  {
    int swap=shortLen;
    shortLen=longLen;
    longLen=swap;
    yLonger=true;
  }
  int decInc;
  if (longLen==0) decInc=0;
  else decInc = (shortLen << 16) / longLen;

  if (yLonger)
  {
    if (longLen>0)
    {
      longLen+=y;
      for (int j=0x8000+(x<<16);y<=longLen;++y)
      {
        sx = CLAMP(j >> 16, 0, image->width-1);
        sy = CLAMP(y, 0, image->height-1);

        GETPIXEL(image,sx,sy) = color;
        j+=decInc;
      }
      return;
    }
    longLen+=y;
    for (int j=0x8000+(x<<16);y>=longLen;--y)
    {
      sx = CLAMP(j >> 16, 0, image->width-1);
      sy = CLAMP(y, 0, image->height-1);

      GETPIXEL(image,sx,sy) = color;
      j-=decInc;
    }
    return;
  }

  if (longLen>0)
  {
    longLen+=x;
    for (int j=0x8000+(y<<16);x<=longLen;++x)
    {
      sx = CLAMP(x, 0, image->width-1);
      sy = CLAMP(j>>16, 0, image->height-1);

      GETPIXEL(image,sx,sy) = color;
      j+=decInc;
    }
    return;
  }
  longLen+=x;
  for (int j=0x8000+(y<<16);x>=longLen;--x)
  {
    sx = CLAMP(x, 0, image->width-1);
    sy = CLAMP(j>>16, 0, image->height-1);

    GETPIXEL(image,sx,sy) = color;
    j-=decInc;
  }
}
