#include "CUDAUtils.h"

//Round a / b to nearest higher integer value
int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }
//Round a / b to nearest lower integer value
int iDivDown(int a, int b) { return a / b; }
//Align a to nearest higher multiple of b
int iAlignUp(int a, int b) { return (a % b != 0) ?  (a - a % b + b) : a; }
//Align a to nearest lower multiple of b
int iAlignDown(int a, int b)  {return a - a % b; }

//__global__ void combineRenderedWithRegistered3sm(CUDA_PIXEL *output, CUDA_PIXEL *renderedImage, int width, int height)
//{
//	int offsetBlock = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * width;
//	int offset = offsetBlock + threadIdx.x + threadIdx.y * width;
//
//	CUDA_PIXEL a;
//	CUDA_PIXEL* pixelOutput = output + offset * 3;
//	CUDA_PIXEL* pixelRendered = renderedImage + offset;
//
//	a = pixelRendered[0];
//	if (a != 0)
//	{
//		//pixelOutput[0] = 254 - pixelOutput[0];
//		//pixelOutput[1] = 254 - pixelOutput[1];
//		//pixelOutput[2] = 254 - pixelOutput[2];
//
//		//pixelOutput[0] = pixelRendered[0] * 0; //was * 2 for red
//		//pixelOutput[1] = pixelRendered[0] * 0;
//		//pixelOutput[2] = pixelRendered[0] * 0;
//
//		pixelOutput[0] = 255;
//		pixelOutput[1] = 0;
//		pixelOutput[2] = 0;
//	}
//}
//
//__host__ void CombineRenderedWithRegistered3(CUDA_PIXEL *output, CUDA_PIXEL *renderedImage, int width, int height)
//{
//	dim3 threads_in_block(16,16);
//	dim3 blocks(iDivUp(width,16), iDivUp(height,16));
//	combineRenderedWithRegistered3sm<<<blocks, threads_in_block>>>(output, renderedImage, width, height);
//}
//__host__ void MakeGrayScale(int* in_data, CUDA_PIXEL* grayImage, int height, int width)
//{
//	makeGrayScale<<<1, height>>>(in_data, grayImage, width);
//}
//
//__host__ void GrayToOutput(CUDA_PIXEL* grayImage, int* out_data, int height, int width)
//{
//	grayToOutput<<<1, height>>>(grayImage, out_data, width);
//}
//
//__host__ void GetNormalizedImageParamatersCenter(int* roi1, int* roi2, int &normalizedImageWidth, int &normalizedImageHeight)
//{
//	normalizedImageWidth = max(roi1[4], roi2[4]);
//	normalizedImageHeight = max(roi1[5], roi2[5]);
//}
//__host__ void GetNormalizedImageParamaters(int* roi1, int* roi2, int *roiNormalizedOriginal, int *roiNormalizedNew, int &normalizedImageWidth, int &normalizedImageHeight)
//{
//	roiNormalizedOriginal[0] = min(roi1[0], roi2[0]);
//	roiNormalizedOriginal[1] = min(roi1[1], roi2[1]);
//	roiNormalizedOriginal[2] = max(roi1[2], roi2[2]);
//	roiNormalizedOriginal[3] = max(roi1[3], roi2[3]);
//
//	roiNormalizedOriginal[4] = roiNormalizedOriginal[2] - roiNormalizedOriginal[0];
//	roiNormalizedOriginal[5] = roiNormalizedOriginal[3] - roiNormalizedOriginal[1];
//
//	normalizedImageWidth = roiNormalizedOriginal[4];
//	normalizedImageHeight = roiNormalizedOriginal[5];
//
//	roiNormalizedNew[0] = 0;
//	roiNormalizedNew[1] = 0;
//	roiNormalizedNew[2] = normalizedImageWidth;
//	roiNormalizedNew[3] = normalizedImageHeight;
//	roiNormalizedNew[4] = normalizedImageWidth;
//	roiNormalizedNew[5] = normalizedImageHeight;
//}
//
//__host__ void GetNormalizedRoi(int* roiOriginal, int* roiNormalized, int &nw, int &nh)
//{
//	roiNormalized[0] = 0;
//	roiNormalized[1] = 0;
//	roiNormalized[2] = roiOriginal[4];
//	roiNormalized[3] = roiOriginal[5];
//	roiNormalized[4] = roiOriginal[4];
//	roiNormalized[5] = roiOriginal[5];
//
//	nw = roiNormalized[4];
//	nh = roiNormalized[5];
//}
//
//__host__ void GetCenteredRoi(int* roiOriginal, int normalizedImageWidth, int normalizedImageHeight, int *roiNormal)
//{
//	int oldImageHeight, oldImageWidth;
//
//	oldImageWidth = roiOriginal[4];
//	oldImageHeight = roiOriginal[5];
//
//	roiNormal[0] = (normalizedImageWidth - oldImageWidth) / 2;
//	roiNormal[1] = (normalizedImageHeight - oldImageHeight) / 2;
//	roiNormal[2] = (normalizedImageWidth + oldImageWidth) / 2;
//	roiNormal[3] = (normalizedImageHeight + oldImageHeight) / 2;
//	roiNormal[4] = roiOriginal[4];
//	roiNormal[5] = roiOriginal[5];
//}
//
//__host__ void NormalizeWithRoi(CUDA_PIXEL* originalImage, int* originalImageRoi, int originalImageWidth, int originalImageHeight, 
//								 int* normalizedRoi, int normalizedImageWidth, int normalizedImageHeight, CUDA_PIXEL* normalizedImage)
//{
//	cutilSafeCall(cudaMemset(normalizedImage, 0, normalizedImageWidth * normalizedImageHeight));
//
//	normalizeWithRoi<<<1, originalImageHeight>>>(originalImage, 
//		originalImageRoi[0], originalImageRoi[1], originalImageRoi[3],
//		originalImageWidth, originalImageHeight, 
//		normalizedImage, 
//		normalizedRoi[0], normalizedRoi[1],
//		normalizedImageWidth, normalizedImageHeight);
//}
//
//__host__ void Add(CUDA_PIXEL* image1, CUDA_PIXEL* image2, CUDA_PIXEL *imageSum, int width, int height)
//{
//	add<<<1, height>>>(image1, image2, imageSum, width, height);
//}
//
//__host__ void Sub(CUDA_PIXEL* image1, CUDA_PIXEL* image2, CUDA_PIXEL *imageDiff, int width, int height)
//{
//	sub<<<1, height>>>(image1, image2, imageDiff, width, height);
//}
//
//__host__ void AddDT(CUDA_FLOAT* image1, CUDA_FLOAT* image2, CUDA_FLOAT* imageSum, int width, int height)
//{
//	addDT<<<1, height>>>(image1, image2, imageSum, width, height);
//}
//
//__host__ void SubDT(CUDA_FLOAT* image1, CUDA_FLOAT* image2, CUDA_FLOAT* imageDiff, int width, int height)
//{
//	subDT<<<1, height>>>(image1, image2, imageDiff, width, height);
//}
//
//__global__ void add(CUDA_PIXEL* image1, CUDA_PIXEL* image2, CUDA_PIXEL* imageSum, int width, int height)
//{
//	int i;
//
//	CUDA_PIXEL* currentRowImage1 = image1 + threadIdx.x * width;
//	CUDA_PIXEL* currentRowImage2 = image2 + threadIdx.x * width;
//	CUDA_PIXEL* currentRowImageSum = imageSum + threadIdx.x * width;
//
//	for (i=0;i<width;i++) currentRowImageSum[i] = currentRowImage1[i] + currentRowImage2[i];
//}
//
//__global__ void sub(CUDA_PIXEL* image1, CUDA_PIXEL* image2, CUDA_PIXEL* imageDiff, int width, int height)
//{
//	int i;
//
//	CUDA_PIXEL* currentRowImage1 = image1 + threadIdx.x * width;
//	CUDA_PIXEL* currentRowImage2 = image2 + threadIdx.x * width;
//	CUDA_PIXEL* currentRowImageDiff = imageDiff + threadIdx.x * width;
//
//	for (i=0;i<width;i++) currentRowImageDiff[i] = currentRowImage1[i] - currentRowImage2[i];
//}
//
//__global__ void addDT(CUDA_FLOAT* image1, CUDA_FLOAT* image2, CUDA_FLOAT* imageSum, int width, int height)
//{
//	int i;
//
//	CUDA_FLOAT* currentRowImage1 = image1 + threadIdx.x * width;
//	CUDA_FLOAT* currentRowImage2 = image2 + threadIdx.x * width;
//	CUDA_FLOAT* currentRowImageSum = imageSum + threadIdx.x * width;
//
//	for (i=0;i<width;i++) currentRowImageSum[i] = currentRowImage1[i] + currentRowImage2[i];
//}
//
//__global__ void subDT(CUDA_FLOAT* image1, CUDA_FLOAT* image2, CUDA_FLOAT* imageDiff, int width, int height)
//{
//	int i;
//
//	CUDA_FLOAT* currentRowImage1 = image1 + threadIdx.x * width;
//	CUDA_FLOAT* currentRowImage2 = image2 + threadIdx.x * width;
//	CUDA_FLOAT* currentRowImageDiff = imageDiff + threadIdx.x * width;
//
//	for (i=0;i<width;i++) currentRowImageDiff[i] = currentRowImage1[i] - currentRowImage2[i];
//}
//
//__global__ void normalizeWithRoi(CUDA_PIXEL* originalImage, 
//								 int originalImageRoi0, int originalImageRoi1, int originalImageRoi3, 
//								 int originalImageWidth, int originalImageHeight,
//								 CUDA_PIXEL* normalizedImage, 
//								 int normalizedImageRoi0, int normalizedImageRoi1,
//								 int normalizedImageWidth, int normalizedImageHeight)
//{
//	int i, gotoWidth;
//
//	gotoWidth = MIN(normalizedImageWidth, originalImageWidth);
//
//	if (threadIdx.x >= originalImageRoi1 && threadIdx.x <= originalImageRoi3)
//	{
//		CUDA_PIXEL* currentRowOriginal = originalImage + threadIdx.x * originalImageWidth;
//		CUDA_PIXEL* currentRowNormalized = normalizedImage + (threadIdx.x - originalImageRoi1 + normalizedImageRoi1) * normalizedImageWidth;
//
//		for (i=0;i<gotoWidth;i++)
//			currentRowNormalized[i + normalizedImageRoi0] = currentRowOriginal[i + originalImageRoi0];
//	}
//}
//
//__global__ void makeGrayScale(int* in_data, CUDA_PIXEL* grayImage, int size)
//{
//	int i;
//	uchar4* in = (uchar4*) in_data;
//	uchar4* inRow = in + threadIdx.x * size;
//
//	CUDA_PIXEL* grayRow = grayImage + threadIdx.x * size;
//
//	for (i=0;i<size;i++)
//		grayRow[i] = (inRow[i].x + inRow[i].y + inRow[i].z)/3;
//}
//
//__global__ void grayToOutput(CUDA_PIXEL* grayImage, int* out_data, int size)
//{
//	int i=0;
//	PixelRGB pixel;
//	CUDA_PIXEL* grayRow = grayImage + threadIdx.x * size;
//	int* outRow = out_data + threadIdx.x * size;
//
//	for (i=0;i<size;i++)
//	{
//		pixel.r = grayRow[i];
//		pixel.g = grayRow[i];
//		pixel.b = grayRow[i];
//		PIXELTOINT(pixel, outRow[i]);
//	}
//}
//
//__host__ void CopyToOutputImageCentered(CUDA_PIXEL* output, CUDA_PIXEL* input, int outputWidth, int outputHeight, int inputWidth, int inputHeight)
//{
//	CUDA_PIXEL* forTest;
//	cutilSafeCall(cudaMalloc((void**)&forTest, outputWidth * outputHeight * sizeof(CUDA_PIXEL)));
//
//	int roiTest[6], roiTest1[6];
//
//	roiTest[0] = 0; roiTest[1] = 0; roiTest[2] = inputWidth;
//	roiTest[3] = inputHeight; roiTest[4] = inputWidth; roiTest[5] = inputHeight;
//
//	GetCenteredRoi(roiTest, outputWidth, outputHeight, roiTest1);
//
//	NormalizeWithRoi(input, roiTest, inputWidth, inputHeight, roiTest1, outputWidth, outputHeight, forTest);
//	cutilSafeCall(cudaMemcpy(output, forTest, outputWidth * outputHeight * sizeof(CUDA_PIXEL), cudaMemcpyDeviceToDevice));
//
//	cutilSafeCall(cudaFree(forTest));
//}
//
//__host__ void CopyToOutputImageOriginal(CUDA_PIXEL* output, CUDA_PIXEL* input, int *roiNormalizedOriginal, int* roiNormalizedNew,
//										int outputWidth, int outputHeight, int inputWidth, int inputHeight)
//{
//	CUDA_PIXEL* forTest;
//	cutilSafeCall(cudaMalloc((void**)&forTest, outputWidth * outputHeight * sizeof(CUDA_PIXEL)));
//
//	int roiTest[6], roiTest1[6];
//
//	roiTest[0] = 0; roiTest[1] = 0; roiTest[2] = inputWidth;
//	roiTest[3] = inputHeight; roiTest[4] = inputWidth; roiTest[5] = inputHeight;
//
//	GetCenteredRoi(roiTest, outputWidth, outputHeight, roiTest1);
//
//	NormalizeWithRoi(input, roiNormalizedNew, inputWidth, inputHeight,
//		roiNormalizedOriginal, outputWidth, outputHeight, forTest);
//
//	//NormalizeWithRoi(input, roiTest, inputWidth, inputHeight, roiTest1, outputWidth, outputHeight, forTest);
//	cutilSafeCall(cudaMemcpy(output, forTest, outputWidth * outputHeight * sizeof(CUDA_PIXEL), cudaMemcpyDeviceToDevice));
//
//	cutilSafeCall(cudaFree(forTest));
//}

//__device__ inline void atomicFloatAdd(float *address, float val)
//{
//	int tmp0 = *address;
//	int i_val = __float_as_int(val + __int_as_float(tmp0));
//	int tmp1;
//	// compare and swap v = (old == tmp0) ? i_val : old;
//	// returns old 
//	while( 
//		(tmp1 = atomicCAS((int *)address, tmp0, i_val)) != tmp0)
//	{
//		tmp0 = tmp1;
//		i_val = __float_as_int(val + __int_as_float(tmp1));
//	}
//} 