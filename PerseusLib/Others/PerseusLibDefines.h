#pragma once

#ifndef SUPPORTED_GETIMAGE_TYPES
#define SUPPORTED_GETIMAGE_TYPES
enum GetImageType
{
	GETIMAGE_WIREFRAME,
	GETIMAGE_FILL,
	GETIMAGE_ORIGINAL,
	GETIMAGE_POSTERIORS,
	GETIMAGE_OBJECTS,
	GETIMAGE_SIHLUETTE,
	GETIMAGE_DT,
	GETIMAGE_PROXIMITY
};
#endif

#ifndef COLUMN
#define COLUMN(columnId, index, width) ((columnId) + ((width) * (index)))
#endif

#ifndef ROW
#define ROW(rowId, index, width) (((rowId) * (width)) + (index))
#endif

#ifndef PIXELMAT
#define PIXELMAT(image, i, j, stride) image[(int)(i)+(int)(j)*(int)(stride)]
#endif

#ifndef PIXELMATINDEX
#define PIXELMATINDEX(i, j, stride) (int)(i)+(int)(j)*(int)(stride)
#endif

#ifndef PIXELMONOATF
#define PIXELMONOATF(image, i, j, stride) *((image) + ((int)j)*((int)stride)*sizeof(VFLOAT) + ((int)i))
#endif

#ifndef GETPIXEL
#define GETPIXEL(image, x, y) PIXELMAT(image->pixels, x, y, image->width) 
#endif

#ifndef GETZBUFFER
#define GETZBUFFER(image, x, y) PIXELMAT(image->zbuffer, x, y, image->width)
#endif

//math
#ifndef MIN
#define MIN(a,b) ((a < b) ? a : b)
#endif

#ifndef MAX
#define MAX(a,b) ((a < b) ? b : a)
#endif

#ifndef ABS
#define ABS(a) ((a < 0) ? -a : a)
#endif

#ifndef CLAMP
#define CLAMP(x,a,b) MAX((a), MIN((b), (x)))
#endif

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

#ifndef DEGTORAD
#define DEGTORAD 0.017453292519943295769236907684886
#endif

#ifndef INF
#define INF 0xFFFF
#endif

#ifndef INF_FLOAT
#define INF_FLOAT 0xFFFFFFFF
#endif

#ifndef INF_INT
#define INF_INT 0xFFFF
#endif

//conversions
#ifndef RGBTOINT
#define RGBTOINT(r,g,b,out) out = (int(b)<<16) | (int(g)<<8) | int(r);
#endif

#ifndef INTTORGB
#define INTTORGB(r,g,b,in) r = in&0xff; g = (in>>8)&0xff; b = (in>>16)&0xff;
#endif

#ifndef PIXELTOINT
#define PIXELTOINT(pixel,out) RGBTOINT(pixel.r, pixel.g, pixel.b, out);
#endif

#ifndef INTTOPIXEL
#define INTTOPIXEL(pixel,in) INTTORGB(pixel.r, pixel.g, pixel.b, in);
#endif

#ifndef MAX_INT
#define MAX_INT 4294967295
#endif

//debug
#ifndef DEBUGBREAK
#define DEBUGBREAK \
{ \
	int ryifrklaeybfcklarybckyar=0; \
	ryifrklaeybfcklarybckyar++; \
}
#endif

//types

#ifndef VFLOAT
#define VFLOAT float
#endif

#ifndef VFLOAT2_def
#define VFLOAT2_def
struct VFLOAT2
{
	VFLOAT x;
	VFLOAT y;
};
#endif

#ifndef VINT
#define VINT int
#endif

#ifndef VUINT
#define VUINT unsigned int
#endif

#ifndef VBYTE
#define VBYTE unsigned char
#endif

#ifndef VBOOL
#define VBOOL bool
#endif

#ifndef NULL
#define NULL 0
#endif

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#ifndef VECTOR3DA
#define VECTOR3DA Vector3D<VFLOAT>
#endif

#ifndef VECTOR3DI
#define VECTOR3DI Vector3D<VINT>
#endif

#ifndef VECTOR2DA
#define VECTOR2DA Vector2D<VFLOAT>
#endif

#ifndef VECTOR4DA
#define VECTOR4DA Vector4D<VFLOAT>
#endif

#ifndef RENDERER_ZBUFFER_CONDITION
#define RENDERER_ZBUFFER_CONDITION Sz < imageFill->zbuffer[index]
#endif

#ifndef RENDERER_ZBUFFERINVERSE_CONDITION
#define RENDERER_ZBUFFERINVERSE_CONDITION Sz > imageFill->zbufferInverse[index]
#endif

#ifndef DRAWLINE
#define DRAWLINE(image,x1,y1,x2,y2,color) DrawingPrimitives::Instance()->DrawLine(image, (VINT) x1, (VINT) y1, (VINT) x2, (VINT) y2, color)
#endif

#ifndef DRAWLINEZ
#define DRAWLINEZ(image,x1,y1,z1,x2,y2,z2,meshid,color) DrawingPrimitives::Instance()->DrawLineZ(image, x1, y1, z1, x2, y2, z2, meshid, color)
#endif

#ifndef ITERATION_TARGET
#define ITERATION_TARGET
enum IterationTarget { ITERATIONTARGET_TRANSLATION, ITERATIONTARGET_ROTATION, ITERATIONTARGET_BOTH };
#endif

#ifndef ImageFloat
#define ImageFloat ImagePerseus<float>
#endif

#ifndef ImageUInt
#define ImageUInt ImagePerseus<unsigned int>
#endif

#ifndef ImageInt
#define ImageInt ImagePerseus<int>
#endif

#ifndef ImageUChar
#define ImageUChar ImagePerseus<unsigned char>
#endif

#ifndef ImageUChar4
#define ImageUChar4 ImagePerseus<PixelUCHAR4>
#endif

#ifndef ImageBool
#define ImageBool ImagePerseus<bool>
#endif