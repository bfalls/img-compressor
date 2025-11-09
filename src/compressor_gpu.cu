#include <cuda_runtime.h>
#include <math_constants.h>   // CUDART_PI_F
#include <cmath>
#include "compressor.hpp"

// ------------------------------------------------------------
// Device constants
// ------------------------------------------------------------
__constant__ uint8_t d_qLuma[64];
__constant__ float   d_T[64];   // DCT matrix [u,x] (row-major)
__constant__ float   d_Tt[64];  // transpose [x,u]

//static int  g_quality = 90;
static bool g_inited = false;

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------
__device__ __forceinline__ uint8_t clamp_u8(float v) {
	v = fminf(255.f, fmaxf(0.f, v));
	return static_cast<uint8_t>(v + 0.5f);
}

// Build exact 8x8 DCT and its transpose
static void build_dct_mats(float T[64], float Tt[64]) {
	const float invSqrt8 = 1.0f / std::sqrt(8.0f);
	for (int u = 0; u < 8; ++u) {
		float alpha = (u == 0) ? invSqrt8 : std::sqrt(2.0f) * invSqrt8;
		for (int x = 0; x < 8; ++x) {
			float val = alpha * std::cos((float)((2 * x + 1) * u) * (CUDART_PI_F / 16.0f));
			T[u * 8 + x] = val;   // [u,x]
			Tt[x * 8 + u] = val;   // [x,u] = transpose
		}
	}
}

// ------------------------------------------------------------
// Kernel: 8x8 DCT -> quant/dequant -> IDCT
// Notes:
//  - Clamped edge loads (so partial tiles are defined)
//  - Bounds-checked stores
//  - Two correctness fixes:
//      * Forward col DCT uses T[v,x]  (d_T[tx*8 + k])   <-- FIXED HERE
//      * Row IDCT uses   T^T[y,k]    (d_Tt[ty*8 + k])  <-- FIXED BEFORE
// ------------------------------------------------------------
__global__ void k_dct8x8_quant_idct(const uint8_t* __restrict__ src,
	uint8_t* __restrict__ dst,
	int width, int height, int pitch)
{
	const int bx = blockIdx.x * 8;
	const int by = blockIdx.y * 8;
	const int tx = threadIdx.x;   // 0..7  -> x/column index
	const int ty = threadIdx.y;   // 0..7  -> y/row (or u in freq domain)

	__shared__ float s_in[8][9];   // +1 padding to avoid SMEM bank conflicts
	__shared__ float s_tmp[8][9];
	__shared__ float s_out[8][9];

	// ---- Load & center (clamped) ----
	int gx = bx + tx;
	int gy = by + ty;
	if (gx >= width)  gx = width - 1;
	if (gy >= height) gy = height - 1;
	s_in[ty][tx] = (float)src[gy * pitch + gx] - 128.f;
	__syncthreads();

	// ---- Row DCT: s_tmp[u,x] = sum_k T[u,k] * s_in[k,x] ----
	float sum = 0.f;
#pragma unroll
	for (int k = 0; k < 8; ++k) {
		sum += d_T[ty * 8 + k] * s_in[k][tx];   // T[u=ty,k] * in[k,x=tx]
	}
	s_tmp[ty][tx] = sum;
	__syncthreads();

	// ---- Col DCT: s_out[u,v] = sum_k s_tmp[u,k] * T[v,k] ----
	// FIX: use T[v,k] = d_T[tx*8 + k], not T^T[v,k].
	sum = 0.f;
#pragma unroll
	for (int k = 0; k < 8; ++k) {
		sum += s_tmp[ty][k] * d_T[tx * 8 + k];  // T[v=tx, x=k]
	}
	s_out[ty][tx] = sum;
	__syncthreads();

	// ---- Quantize / dequantize (JPEG-like) ----
	const int qidx = ty * 8 + tx;
	const float coeff = s_out[ty][tx];
	const int   q = __float2int_rn(coeff / (float)d_qLuma[qidx]);
	s_out[ty][tx] = (float)q * (float)d_qLuma[qidx];
	__syncthreads();

	// ---- Inverse col: s_tmp[u,x] = sum_k s_out[u,k] * T[k,x] ----
	sum = 0.f;
#pragma unroll
	for (int k = 0; k < 8; ++k) {
		sum += s_out[ty][k] * d_T[k * 8 + tx];  // T[k, x=tx]
	}
	s_tmp[ty][tx] = sum;
	__syncthreads();

	// ---- Inverse row: out[y,x] = sum_k T^T[y,k] * s_tmp[k,x] ----
	sum = 0.f;
#pragma unroll
	for (int k = 0; k < 8; ++k) {
		sum += d_Tt[ty * 8 + k] * s_tmp[k][tx]; // T^T[y=ty,k] * tmp[k,x=tx]
	}

	uint8_t outv = clamp_u8(sum + 128.f);

	// ---- Store (bounds-checked) ----
	int wx = bx + tx, wy = by + ty;
	if (wx < width && wy < height) {
		dst[wy * pitch + wx] = outv;
	}
}

// ------------------------------------------------------------
// Public API
// ------------------------------------------------------------
void init_compressor(int quality)
{
	//g_quality = quality;

	// 1) Upload scaled quant tables (host code sets DC step to 1)
	uint8_t qL[64], qC_unused[64];
	make_scaled_quant_tables(quality, qL, qC_unused);
	CUDA_CHECK(cudaMemcpyToSymbol(d_qLuma, qL, 64));

	// 2) Upload DCT matrices
	float T[64], Tt[64];
	build_dct_mats(T, Tt);
	CUDA_CHECK(cudaMemcpyToSymbol(d_T, T, sizeof(T)));
	CUDA_CHECK(cudaMemcpyToSymbol(d_Tt, Tt, sizeof(Tt)));

	g_inited = true;
}

void compress_channel_gpu(const uint8_t* src, uint8_t* dst, int width, int height)
{
	if (!g_inited) init_compressor(90);

	const int pitch = width;                       // tight packing
	dim3 block(8, 8);
	dim3 grid((width + 7) / 8, (height + 7) / 8);

	k_dct8x8_quant_idct << <grid, block >> > (src, dst, width, height, pitch);
	CUDA_CHECK(cudaGetLastError());
}

void compress_image_rgb_gpu(const ImageRGB& in, ImageRGB& out)
{
	if (out.width != in.width || out.height != in.height ||
		(int)out.r.size() != in.width * in.height ||
		(int)out.g.size() != in.width * in.height ||
		(int)out.b.size() != in.width * in.height)
	{
		throw std::runtime_error("compress_image_rgb_gpu: 'out' buffers must be pre-sized to match 'in'.");
	}

	const size_t N = (size_t)in.width * in.height;

	// Device buffers
	uint8_t* d_r_src = nullptr, * d_g_src = nullptr, * d_b_src = nullptr;
	uint8_t* d_r_dst = nullptr, * d_g_dst = nullptr, * d_b_dst = nullptr;
	CUDA_CHECK(cudaMalloc(&d_r_src, N));
	CUDA_CHECK(cudaMalloc(&d_g_src, N));
	CUDA_CHECK(cudaMalloc(&d_b_src, N));
	CUDA_CHECK(cudaMalloc(&d_r_dst, N));
	CUDA_CHECK(cudaMalloc(&d_g_dst, N));
	CUDA_CHECK(cudaMalloc(&d_b_dst, N));

	CUDA_CHECK(cudaMemcpy(d_r_src, in.r.data(), N, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_g_src, in.g.data(), N, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_b_src, in.b.data(), N, cudaMemcpyHostToDevice));

	// Process per channel
	compress_channel_gpu(d_r_src, d_r_dst, in.width, in.height);
	compress_channel_gpu(d_g_src, d_g_dst, in.width, in.height);
	compress_channel_gpu(d_b_src, d_b_dst, in.width, in.height);

	CUDA_CHECK(cudaMemcpy(out.r.data(), d_r_dst, N, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(out.g.data(), d_g_dst, N, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(out.b.data(), d_b_dst, N, cudaMemcpyDeviceToHost));

	cudaFree(d_r_src); cudaFree(d_g_src); cudaFree(d_b_src);
	cudaFree(d_r_dst); cudaFree(d_g_dst); cudaFree(d_b_dst);
}
