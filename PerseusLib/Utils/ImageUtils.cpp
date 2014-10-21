#include <PerseusLib/Utils/ImageUtils.h>

#include <FreeImage.h>
#include <math.h>

using namespace PerseusLib::Utils;
#include <FreeImage.h>

ImageUtils* ImageUtils::instance;

ImageUtils::ImageUtils(void)
{
}

ImageUtils::~ImageUtils(void)
{
}

void ImageUtils::SaveImageToFile(ImageUChar4* image, char* fileName)
{
	ImageUChar4* newImage = new ImageUChar4(image->width, image->height, false);
	this->Copy(image, newImage);

	FIBITMAP *bmpO = FreeImage_ConvertFromRawBits((unsigned char*)newImage->pixels, newImage->width, newImage->height, newImage->width * 4, 32, 0, 0, 0, false);

	FreeImage_FlipVertical(bmpO);

	FIBITMAP *bmp = FreeImage_ConvertTo24Bits(bmpO);

	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
	fif = FreeImage_GetFileType(fileName, 0);
	if(fif == FIF_UNKNOWN)  { fif = FreeImage_GetFIFFromFilename(fileName); }
	if((fif != FIF_UNKNOWN) && FreeImage_FIFSupportsReading(fif)) { FreeImage_Save(fif, bmp, fileName, 0); }

	FreeImage_Unload(bmp);
	FreeImage_Unload(bmpO);
	delete newImage;
}

void ImageUtils::FlipColours(ImageUChar4 *image)
{
	int i;

	PixelUCHAR4 pixel;

	for (i=0; i<image->width * image->height; i++)
	{
		pixel = image->pixels[i];
		image->pixels[i].x = pixel.z;
		image->pixels[i].y = pixel.y;
		image->pixels[i].z = pixel.x;
	}
}

void ImageUtils::LoadImageFromFile(ImageUChar4* image, char* fileName)
{
	bool bLoaded = false;
	int bpp;
	FIBITMAP *bmp = 0;
	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
	fif = FreeImage_GetFileType(fileName);

	if (fif == FIF_UNKNOWN) { fif = FreeImage_GetFIFFromFilename(fileName); }

	if (fif != FIF_UNKNOWN && FreeImage_FIFSupportsReading(fif))
	{
		bmp = FreeImage_Load(fif, fileName, 0);
		bLoaded = true; if (bmp == NULL) bLoaded = false;
	}

	if (bLoaded)
	{
		FreeImage_FlipVertical(bmp);

		bpp = FreeImage_GetBPP(bmp);
		switch (bpp)
		{
		case 32:
			break;
		default:
			FIBITMAP *bmpTemp = FreeImage_ConvertTo32Bits(bmp);
			if (bmp != NULL) FreeImage_Unload(bmp);
			bmp = bmpTemp;
			bpp = FreeImage_GetBPP(bmp);
			break;
		}

		memcpy(image->pixels, FreeImage_GetBits(bmp),  sizeof(unsigned char) * 4 * image->width * image->height);

		FreeImage_Unload(bmp);
	}
}

void ImageUtils::LoadImageFromFile(ImageUChar* image, char* fileName, int fixedValue)
{
	ImageUChar4* newImage = new ImageUChar4(image->width, image->height, false);
	this->LoadImageFromFile(newImage, fileName);
	this->Copy(newImage, image, fixedValue);
	delete newImage;
}

void ImageUtils::Copy(ImageUChar4 *src, ImageUChar4* dst) { memcpy(dst->pixels, src->pixels, src->width * src->height * sizeof(PixelUCHAR4)); }

void ImageUtils::Copy(ImageUChar *src, ImageUChar4* dst)
{
	for (int i=0; i<src->width*src->height; i++)
	{ dst->pixels[i].x = src->pixels[i]; dst->pixels[i].y = src->pixels[i]; dst->pixels[i].z = src->pixels[i]; dst->pixels[i].w = 255; }
}
void ImageUtils::Copy(ImageUChar *src, ImageUChar* dst) { memcpy(dst->pixels, src->pixels, src->width * src->height * sizeof(unsigned char)); }
void ImageUtils::Copy(ImageUChar4 *src, ImageUChar* dst, int fixedValue)
{
	if (fixedValue > 0)
	{
		for (int i=0; i<src->width*src->height; i++)
		{ dst->pixels[i] = (src->pixels[i].x + src->pixels[i].y + src->pixels[i].x) > 0 ? fixedValue : 0; }
	}
	else
	{
		for (int i=0; i<src->width*src->height; i++)
		{ dst->pixels[i] = (unsigned char) (0.2989f * float(src->pixels[i].x) + 0.5870f * float(src->pixels[i].y) + 0.1140f * float(src->pixels[i].z)); }
	}
}


void ImageUtils::Overlay(ImageUChar* srcGrey, ImageUChar4 *destRGB, int destB, int destG, int destR)
{
	int idx;

	for (idx = 0; idx < srcGrey->width * srcGrey->height; idx++)
	{
		if (srcGrey->pixels[idx] > 0)
		{
      destRGB->pixels[idx].x = (unsigned char)(float(destR) * (float(srcGrey->pixels[idx]) / 255.0f));
      destRGB->pixels[idx].y = (unsigned char)(float(destG) * (float(srcGrey->pixels[idx]) / 255.0f));
      destRGB->pixels[idx].z = (unsigned char)(float(destB) * (float(srcGrey->pixels[idx]) / 255.0f));
			destRGB->pixels[idx].w = 255;
		}
	}
}

void ImageUtils::ScaleToGray(ImageFloat *src, ImageUChar4* dest)
{
	int idx;

	float *source = src->pixels;

	dest->Clear();

	float lims[2], out_lims[2], scale;

	lims[0] = 100000.0f; lims[1]= -100000.0f; out_lims[0] = 0.0f; out_lims[1] = 1.0f;

	//for (idx = 0; idx < dest->width * dest->height; idx++) if (source[idx] < 0) source[idx] = 0;

	for (idx = 0; idx < dest->width * dest->height; idx++)
	{
		if (source[idx] < lims[0]) lims[0] = source[idx];
		if (source[idx] > lims[1]) lims[1] = source[idx];
	}

	scale = (out_lims[1] - out_lims[0]) / (lims[1] - lims[0]);

	if (lims[0] == lims[1] || out_lims[0] == out_lims[1]) 
	{ 
		for (idx = 0; idx < dest->width * dest->height; idx++) source[idx] = 0; 
		return;
	}

	if (lims[0] != 0) for (idx = 0; idx < dest->width * dest->height; idx++) source[idx] = source[idx] - lims[0];
	if (scale != 1.0f) for (idx = 0; idx < dest->width * dest->height; idx++) source[idx] = source[idx] * scale;
	if (out_lims[0] != 0) for (idx = 0; idx < dest->width * dest->height; idx++) source[idx] = source[idx] + out_lims[0];

	for (idx = 0; idx < dest->width * dest->height; idx++)
	{
		dest->pixels[idx].x = (unsigned char) (source[idx] * 255.0f);
		dest->pixels[idx].y = (unsigned char) (source[idx] * 255.0f);
		dest->pixels[idx].z = (unsigned char) (source[idx] * 255.0f);
		dest->pixels[idx].w = 255;
	}
}
