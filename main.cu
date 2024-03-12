#include <cuda.h>
#include <stdio.h>

#include "cdf.cuh"

#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>

#define checkCudaErrors(x)                                             \
  do {                                                                 \
	cudaError_t err = x;                                               \
	if (err != cudaSuccess) {                                          \
	  fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
			 cudaGetErrorString(err));                                 \
	  exit(EXIT_FAILURE);                                              \
	}                                                                  \
  } while (0)

// Computes the 2D CDF of a lat-long dome light map
// @param w, h, the width and height of the texture.
// @param src, the RGB-F32 2d texture buffer on device memory
// @param marginalCdf, a F32 2d texture where to store the marginal cdf over the rows
// @param conditionalCdf, a F32 1d texture where to store the conditional cdf
void makeCdf2d_rgb(uint32_t w, uint32_t h, const float* src,
				   // Results:
				   float* marginalCdf, float* conditionalCdf)
{
	constexpr int numAtomics = 1;
	int* counter = nullptr;
 
	// Note: this allocation should be reused and amortized, otherwise it'll cost more time
	//       than the kernel execution itself...
	checkCudaErrors(cudaMalloc(&counter, numAtomics * sizeof(int)));
	checkCudaErrors(cudaMemsetAsync(counter, 0, numAtomics * sizeof(float)));
 
	//GpuTimer gpuclock;
	{
		using Mode   = PrefixSumInclusive<float4>;
		using Loader = CdfLoaderRgb;
		using Remap  = Cdf2dSphericalCoordsJacobian;
 
		//gpuclock.Start();
 
		//int* counter = (int*)atomics;
		int numElements = divRoundUp(w, 4);
 
		// Determine the best block size for the given texture width, the bigger the better...
		// Choosing a block wider than needed will end in slower than ideal computation for
		// small textures.
		const int kBlockSize = (numElements <= 256 ?  256 :
							   (numElements <= 512 ?  512 :
													 1024));
 
		dim3 blockSize(kBlockSize, 1);
		dim3 numBlocks(1, h);
		int blocksPerRaw = divRoundUp(numElements, kBlockSize);

	#define LAUNCH_PARAMS numBlocks, blockSize, kBlockSize*sizeof(float)
		if (blocksPerRaw == 1)
		{   // From 1k up to 4k
			makeCdf2d_kernel<Mode, Loader, Remap,  1><<<LAUNCH_PARAMS>>>(numElements, h, src, marginalCdf, conditionalCdf, counter);
		}
		else if (blocksPerRaw == 2)
		{   // up to 8k
			makeCdf2d_kernel<Mode, Loader, Remap,  2><<<LAUNCH_PARAMS>>>(numElements, h, src, marginalCdf, conditionalCdf, counter);
		}
		else if (blocksPerRaw <= 4)
		{   // up to 16k
			makeCdf2d_kernel<Mode, Loader, Remap,  4><<<LAUNCH_PARAMS>>>(numElements, h, src, marginalCdf, conditionalCdf, counter);
		}
		else if (blocksPerRaw <= 8)
		{   // up to 32k
			makeCdf2d_kernel<Mode, Loader, Remap,  8><<<LAUNCH_PARAMS>>>(numElements, h, src, marginalCdf, conditionalCdf, counter);
		}
		else
		{
			fprintf(stderr, "Error: Light map resolution exceeds limit of 32k: [%u %u]\n", w, h);
		}
	#undef LAUNCH_PARAMS
	}
 
	checkCudaErrors(cudaFree(counter));
}

int main()
{
	char const * inputFilename = "../../data/autumn_field_4k.exr";
	float * inputImg; // width * height * RGBA
	int w;
	int h;
	char const * err = nullptr;
	int ret = LoadEXR(&inputImg, &w, &h, inputFilename, &err);

	if(ret != TINYEXR_SUCCESS)
	{
		if(err)
		{
			fprintf(stderr, "TINYEXR error: %s\n", err);
			FreeEXRErrorMessage(err);
		}
	}

	float * hInput = (float*)malloc(w * h * 3 * sizeof(float));
	{
		// Convert to RGB AOS
		#if 0
		for(int y = 0; y < h; ++y)
		{
			for(int x = 0; x < w; ++x)
			{
				// RRRRGGGGBBBB
			}
		}
		#endif

		for(int y = 0; y < h; ++y)
		{
			const int yInputOffset  = y * w * 4;
			const int yOutputOffset = y * w * 3;
			for(int x = 0; x < w; ++x)
			{
				const int xInputOffset  = x * 4;
				const int xOutputOffset = x * 3;
				const int inputPixelOffset  = yInputOffset  + xInputOffset;
				const int outputPixelOffset = yOutputOffset + xOutputOffset;
				hInput[outputPixelOffset + 0] = inputImg[inputPixelOffset + 0];
				hInput[outputPixelOffset + 1] = inputImg[inputPixelOffset + 1];
				hInput[outputPixelOffset + 2] = inputImg[inputPixelOffset + 2];
			}
		}
	}
	free(inputImg);

	const auto SaveAsEXR = [](float * data, int w, int h, int numComponents, char const * filename)
	{
		char const * err = nullptr;
		int ret = SaveEXR(data, w, h, numComponents, false, filename, &err);
		if(ret != TINYEXR_SUCCESS)
		{
			if(err)
			{
				fprintf(stderr, "SaveAsEXR error: %s\n", err);
				FreeEXRErrorMessage(err);
			}
		}
	};

	float *dInput, *dMarginalCdf, *dConditionalCdf;
	checkCudaErrors(cudaMalloc(&dInput, w * h * 3 * sizeof(float)));
	checkCudaErrors(cudaMalloc(&dMarginalCdf, w * h * sizeof(float)));
	checkCudaErrors(cudaMalloc(&dConditionalCdf, h * sizeof(float)));

	checkCudaErrors(cudaMemcpy(dInput, hInput, w * h * 3 * sizeof(float), cudaMemcpyHostToDevice));
	
	free(hInput);

	makeCdf2d_rgb(w, h, dInput, dMarginalCdf, dConditionalCdf);

	float * hMarginalCdf = (float*)malloc(w * h * sizeof(float));
	checkCudaErrors(cudaMemcpy(hMarginalCdf, dMarginalCdf, w * h * sizeof(float), cudaMemcpyDeviceToHost));
	SaveAsEXR(hMarginalCdf, w, h, 1, "marginal_cdf.exr");
	free(hMarginalCdf);

	float * hConditionalCdf = (float*)malloc(h * sizeof(float));
	checkCudaErrors(cudaMemcpy(hConditionalCdf, dConditionalCdf, h * sizeof(float), cudaMemcpyDeviceToHost));
	SaveAsEXR(hConditionalCdf, 1, h, 1, "conditional_cdf.exr");
	free(hConditionalCdf);

	return 0;
}