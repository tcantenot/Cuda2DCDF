#pragma once

#include <cuda.h>
#include <stdint.h>

// https://maxliani.wordpress.com/2024/03/09/about-fast-2d-cdf-construction/

#define PI 3.14159265359f

inline __device__ float4 & operator+=(float4 & a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a; }
inline __device__ float4 & operator+=(float4 & a, float k) { a.x += k; a.y += k; a.z += k; a.w += k; return a; }
inline __device__ float4 operator*(const float4& a, float k) { return make_float4(a.x * k, a.y * k, a.z * k, a.w * k); }
inline __device__ float4 & operator*=(float4 & a, float k) { a.x *= k; a.y *= k; a.z *= k; a.w *= k; return a; }

inline __device__ __host__ int divRoundUp(int x, int d)
{
	return (x + d - 1) / d;
}


template<typename Tn, typename T> __device__ inline T GetLastFieldOf(const Tn& val);
 
template<> __device__
inline float GetLastFieldOf<float, float>(const float& val)
{
	return val;
}
 
template<> __device__
inline float GetLastFieldOf<float4, float>(const float4& val)
{
	return val.w;
}
 
template<typename Tn> __device__ inline void PrefixSumComponents(Tn& val);
 
template<> __device__
inline void PrefixSumComponents<float>(float& val)
{}
 
template<> __device__
inline void PrefixSumComponents<float4>(float4& val)
{
	val.y += val.x;
	val.z += val.y;
	val.w += val.z;
}

template<typename T>
struct PrefixSumInclusive
{
	__device__
	PrefixSumInclusive(const T&) {}
 
	__device__
	void operator()(T&) const
	{
		// The algorithm natively compute the inclusive sum. Nothing to do here.
	}
};

template<typename T>
struct PrefixSumExclusive
{
	const T originalValue;
 
	__device__
	PrefixSumExclusive(const T& value) { originalValue = value; }
 
	__device__
	void operator()(T& value) const
	{
		// To obtain the exclusive prefix-sum from the inclusive one, we subtract the input value
		value -= originalValue;
	}
};
 
// Compute the parallel prefix-sum of a thread block
template<typename ScanMode, typename Tn, typename T> __device__ 
inline void PrefixSumBlock(Tn& val, volatile T* smem, T& total)
{
	const ScanMode scanMode(val);
 
	// Prepare the prefix sum within the n components in Tn. This let us compute the Hillis-Steele algorithm
	// only on the Nth elements and reduce the total amount of operations in shared memory.
	PrefixSumComponents(val);
 
	for (int i = 1; i < blockDim.x; i <<= 1)
	{
		smem[threadIdx.x] = GetLastFieldOf<Tn, T>(val);
		__syncthreads();
 
		// Add the value to the local variable
		if (threadIdx.x >= i)
		{
			const T offset = smem[threadIdx.x - i];
			val += offset;
		}
		__syncthreads();
	}
 
	// Apply prefix sum running total from the previous block
	val += total;
	__syncthreads();
 
	// Update the prefix sum running total using the last thread in the block
	if (threadIdx.x == blockDim.x - 1)
	{
		total = GetLastFieldOf<Tn, T>(val);
	}
	__syncthreads();
 
	// Apply the final transformation (i.e. subtract the initial value for exclusive prefix sums)
	scanMode(val);
}

// Example Remaps /////////////////////////////////////////////////
// No remap, this works for light maps for rect area lights.
struct Cdf2dIdentity
{
	__device__
	static void factorX(int index, int w, float4& val)
	{}
 
	__device__
	static void factorY(int index, int h, float4& val)
	{}
};
 
// The Jacobian for the change of variables from cartesian to spherical coordinates. This is used
// for dome lights latlong maps. No scaling is required within the rows, we only weight the
// rows themselves.
struct Cdf2dSphericalCoordsJacobian
{
	__device__
	static void factorX(int index, int w, float4& val)
	{}
 
	__device__
	static void factorY(int index, int h, float4& val)
	{
		// Each index represents 4 latitudes in spherical coordinates. The desired angles
		// are that at the center of those 4 pixels:
		// 
		//  Index: 0 -> .---.
		//              | x | <- +0.125
		//              |---|
		//              | y | <- +0.375
		//              |---|                      index + offset
		//              | z | <- +0.635      pi * ----------------
		//              |---|                            h
		//              | w | <- +0.875
		//         1 -> |---|
		//              | x | <- +0.125
		//         ...
		val.x *= sinf(float(PI) * (float(index) + 0.125f) / float(h));
		val.y *= sinf(float(PI) * (float(index) + 0.375f) / float(h));
		val.z *= sinf(float(PI) * (float(index) + 0.625f) / float(h));
		val.w *= sinf(float(PI) * (float(index) + 0.875f) / float(h));
	}
};
 

__device__ inline float luma(float r, float g, float b)
{
	return r * 0.2126f + g * 0.7152f + b * 0.0722f;
}
 
// Example Loaders /////////////////////////////////////////////////
struct CdfLoaderRgb
{
	__device__
	static void load(const void* input, uint32_t index, float4& val)
	{
		// Load 4 rgb texels in RGB AOS. This emits 3 128 bytes load.
		const float4 a = ((float4*)input)[index * 3 + 0];   // R
		const float4 b = ((float4*)input)[index * 3 + 1];   // G
		const float4 c = ((float4*)input)[index * 3 + 2];   // B
 
		// Shuffle the RGB values from AOS to 4-wide SOA, and
		// apply a clamp since negative values are not allowed.
		val.x = max(0.0f, luma(a.x, a.y, a.z));
		val.y = max(0.0f, luma(a.w, b.x, b.y));
		val.z = max(0.0f, luma(b.z, b.w, c.x));
		val.w = max(0.0f, luma(c.y, c.z, c.w));
	}
};

#if 0
// Example of a 64 bits packed RGB loader
struct CdfLoaderRgb64
{
	__device__
	static void load(const void* input, uint32_t index, float4& val)
	{
		struct __builtin_align__(32) Loader
		{
			rgb64_t a, b, c, d;
		};
 
		// Load 4 packed RGB values using 2 128-bits load instructions.
		Loader load4 = ((Loader*)input)[index];
 
		// Shuffle/unpack from AOS to 4-wide SOA
		float3 a, b, c, d;
		rgb64_unpack(load4.a, a.x, a.y, a.z);
		rgb64_unpack(load4.b, b.x, b.y, b.z);
		rgb64_unpack(load4.c, c.x, c.y, c.z);
		rgb64_unpack(load4.d, d.x, d.y, d.z);
 
		/ Clamp negative values
		val.x = max(0.0f, luma(a));
		val.y = max(0.0f, luma(b));
		val.z = max(0.0f, luma(c));
		val.w = max(0.0f, luma(d));
	}
};
#endif

// Create a table to sample from a discrete 2d distribution, such as that of a hdr light map.
// @param w, h, the texture width and height
// @param input, the input texture data
// @param cdf_x, the buffer where to store the marginal CDFs, the buffer size is
//        expected to hold as many elements as there are texels in the texture.
// @param cdf_y, the buffer where to store the marginal CDFs, the buffer size is
//        expected to hold as many elements as there are rows in the texture.
template<typename ScanMode, typename Loader, typename Remap, int N > __global__
void makeCdf2d_kernel(int w, int h, const void* input, float* cdf_x, float* cdf_y, int* counter)
{
	const int index_y = blockIdx.y;
 
	// Init the CDF running total in shared memory. This is to carry forward to prefix sum
	// across the loop iterations and store the row final sum at the end.
	__shared__ float total;
	if (threadIdx.x == 0)
	{
		total = 0;
	}
	__syncthreads();
 
	// Memory for the block-wise prefix scan. The size for it is specified in the kernels launch params.
	extern __shared__ float smemf[];
 
	// Step 1: compute the marginal sampling cumulative distribution over the rows in the image.
	{
		// Each block produces the CDF of a entire row and stores it into registers in valN.
		// This assumes 4 elements per thread and  blocks of 256, 1024 elements per iteration.
		// By doing this we load the values once, compute the CDF, normalize it and store the
		// result. 1 read, one write.
		// Don't consider this as a vector. The actual memory layout for this is:
		// - 4 consecutive elements per thread as the x, y, z, w components of the float4
		// - spread throught the number of threads per block across the SMs, 
		// - times the number of iterations N.
		float4 valN[N] = {};
 
		constexpr int kNumFields = sizeof(float4) / sizeof(float);
		const int numElements = divRoundUp(w, kNumFields);
 
		// Consume a row, produce the prefix sum.
		#pragma unroll
		for (int blockN = 0, index_x = (threadIdx.x + blockDim.x * blockIdx.x); blockN < N;
			++blockN, index_x += blockDim.x)
		{
			float4 val = make_float4(0, 0, 0, 0);
 
			bool inRange = (index_x < numElements);
			if (inRange)
			{
				// Load 4 texels
				Loader::load(input, index_x + index_y * numElements, val);
				Remap::factorX(index_x, numElements, val);
			}
 
			// Block-wise prefix scan
			PrefixSumBlock<ScanMode>(val, smemf, total);
 
			// Write the result to registers
			valN[blockN] = val;
		}
 
		// Normalize the row CDF. Write data from registers to the CDF
		float normalization = (total != 0.0f ? 1.0f / total : 0.0f);
		#pragma unroll
		for (int blockN = 0, index_x = (threadIdx.x + blockDim.x * blockIdx.x); blockN < N;
			++blockN, index_x += blockDim.x)
		{
			bool inRange = (index_x < numElements);
			if (inRange)
			{
				((float4*)cdf_x)[index_x + index_y * numElements] = valN[blockN] * normalization;
			}
		}
	}

	// Let the block store the row running total as the input value to compute the cdf_y (column)
	__shared__ bool done;
	if (threadIdx.x == 0)
	{
		cdf_y[index_y] = total;
		total = 0; //< reset the running total to use it for cdf_y
	
		// Are we done computing all rows CDFs? This atomic returns true only for the last block
		// completing the work.
		done = atomicAdd(counter, 1) == (h-1);
	}
	__syncthreads();
	
	// All blocks terminate here excepts for one
	if (!done) return;

	// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions
	__threadfence();

	// Step 2: compute the conditional cumulative sampling distribution (cdf_y) starting
	// from the rows running totals.
	{
		constexpr int kNumFields = sizeof(float4) / sizeof(float);
		const int numElements = divRoundUp(h, kNumFields);
 
		// Step 1, compute the conditianl CDF in place. This is very similar to step 1. However:
		// - Don't assume the vertical resolution is lower or equal the horizontal
		// - Instead of unrolling the loop and store intermediate values in registers, loop over
		//   whatever number of iterations there are.
		// - Store the non-normalized prefix-sum to cache, read it back in the normalization step.
		// The performance hit of this generaliztion is small only because step two executes as a
		// single block, which is a tiny fraction of the workload in step 1.
		const uint32_t numBlocks = divRoundUp(numElements, blockDim.x);
		for (int blockN = 0, index = threadIdx.x; blockN < numBlocks; ++blockN, index += blockDim.x)
		{
			float4 val = make_float4(0, 0, 0, 0);
 
			bool inRange = (index < numElements);
			if (inRange)
			{
				val = ((float4*)cdf_y)[index];
				Remap::factorY(index, numElements, val);
			}
 
			// Block-wise prefix sum
			PrefixSumBlock<ScanMode>(val, smemf, total);
 
			// Write the not-normalized result to the output buffer
			if (inRange) ((float4*)cdf_y)[index] = val;
		}
		__syncthreads();
 
		// Step 2, read back the cdf_y and normalize it in place.
		float normalization = (total != 0.0f ? 1.0f / total : 0.0f);
		for (int blockN = 0, index = threadIdx.x; blockN < numBlocks; ++blockN, index += blockDim.x)
		{
			bool inRange = (index < numElements);
			if (inRange)
			{
				((float4*)cdf_y)[index] *= normalization;
			}
		}
	}
}