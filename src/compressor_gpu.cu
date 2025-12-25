#include <cuda_runtime.h>
#include <math_constants.h>   // CUDART_PI_F
#include <algorithm>
#include <cmath>
#include <vector>

#include "compressor.hpp"
#include "saliency.hpp"

/*
Device-resident state is kept in constant memory because the DCT basis and
quant tables are tiny, immutable during a run, and broadcast to all threads.
Constant memory delivers good cache behavior for these 8x8 kernels.
*/
__constant__ uint8_t d_qLuma[64];
__constant__ float   d_T[64];   // DCT matrix [u,x] (row-major)
__constant__ float   d_Tt[64];  // transpose [x,u]

static bool g_inited = false;

/*
Rounding and saturating to 8-bit here mirrors JPEG's final clamping behavior
and prevents device/host drift from float-to-int conversions.
*/
__device__ __forceinline__ uint8_t clamp_u8(float v) {
	v = fminf(255.f, fmaxf(0.f, v));
	return static_cast<uint8_t>(v + 0.5f);
}

static inline float clampf(float v, float lo, float hi) {
	return v < lo ? lo : (v > hi ? hi : v);
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

/*
Kernel: 8x8 DCT -> quant/dequant -> IDCT

Tile size matches JPEG's natural block size. Padding shared-memory rows by
+1 mitigates bank conflicts on older architectures. Edges are handled by
clamped loads instead of divergence-heavy conditionals so partial tiles
remain defined without branching.

The quantize/dequantize step sits between forward and inverse transforms to
emulate JPEG's energy shaping. Dequantizing in-kernel ensures the output
is directly comparable as pixels, without requiring a separate pass.

Two fragile indexing points are explicitly documented:
- Column DCT uses T[v,k] (not T^T), which is easy to misindex when
  reusing the row routine.
- Row IDCT uses T^T[y,k], keeping the inverse transform consistent with
  the precomputed transpose.

The kernel is intentionally side-effect free: all parameters
are passed in, and temporary storage stays in shared memory. Kernel is
predictable under profiling and simplifies future tiling work.
*/
__global__ void k_dct8x8_quant_idct(const uint8_t* __restrict__ src,
	uint8_t* __restrict__ dst,
	int width, int height, int pitch,
	const float* block_scales)
{
	const int bx = blockIdx.x * 8;
	const int by = blockIdx.y * 8;
	const int tx = threadIdx.x; // 0..7  -> x/column index
	const int ty = threadIdx.y; // 0..7  -> y/row (or u in freq domain)

	float block_scale = 1.0f;
	if (block_scales) {
		const int block_idx = blockIdx.y * gridDim.x + blockIdx.x;
		block_scale = block_scales[block_idx];
	}

	__shared__ float s_in[8][9]; // +1 padding to avoid SMEM bank conflicts
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

	if (block_scale != 1.0f) {
		// Per-block coefficient scaling (quality map) prior to quantization.
		s_out[ty][tx] *= block_scale;
	}
	__syncthreads();

	/*
	Quantization is applied and immediately inverted to simulate JPEG's
	coefficient coarsening without introducing a bitstream. This preserves
	the ability to compare pixel output directly while still exercising the
	quality setting and energy compaction trade-offs.
	*/
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

// Public API

/*
Initialization concentrates all quality-dependent setup (quant tables) and
static math (DCT bases) into a single call. Uploading once into constant
memory avoids repeated PCIe traffic.
*/
void init_compressor(int quality)
{
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

/*
Per-channel entry point exists so callers can mix-and-match scheduling,
benchmark just luma, or wire this into more elaborate pipelines (e.g., chroma
subsampling) without changing the kernel. The grid maps 1:1 onto 8x8 tiles,
which keeps the math simple.
*/
void compress_channel_gpu(const uint8_t* src, uint8_t* dst, int width, int height,
	const float* inv_block_scales)
{
	if (!g_inited) init_compressor(90);

	const int pitch = width;                       // tight packing
	dim3 block(8, 8);
	dim3 grid((width + 7) / 8, (height + 7) / 8);

	k_dct8x8_quant_idct << <grid, block >> > (src, dst, width, height, pitch, inv_block_scales);
	CUDA_CHECK(cudaGetLastError());
}

/*
RGB wrapper keeps ownership and sizing responsibilities explicit. The
outputs must be pre-sized to the input dimensions. Avoids hidden 
reallocations in perf-sensitive paths and makes buffer lifetimes
obvious to callers.

Each channel is transferred independently to align with the planar layout and
to keep the kernel interface minimal.
*/
void compress_image_rgb_gpu(const ImageRGB& in, ImageRGB& out,
	const QualityMapConfig& quality_map)
{
	if (out.width != in.width || out.height != in.height ||
		(int)out.r.size() != in.width * in.height ||
		(int)out.g.size() != in.width * in.height || 
		(int)out.b.size() != in.width * in.height)
	{
		throw std::runtime_error("compress_image_rgb_gpu: 'out' buffers must be pre-sized to match 'in'.");
	}

	const size_t N = (size_t)in.width * in.height;

	int blocks_x = (in.width + 7) / 8;
	std::vector<float> inv_scales_main;
	std::vector<float> inv_scales_chroma;

	auto compute_block_scales = [&](const ImageRGB& img) {
		if (!quality_map.enabled)
			return;

		SaliencyParams saliency_params;
		QualityMapDebugConfig debug_cfg;
		const QualityMapDebugConfig* debug_ptr = nullptr;
		if (!quality_map.debug_output_path.empty()) {
			debug_cfg.output_path = quality_map.debug_output_path;
			debug_ptr = &debug_cfg;
		}

		const BlockImportanceMap importance = compute_block_importance(img, saliency_params, debug_ptr);
		if (importance.values.empty())
			return;

		blocks_x = importance.blocks_x;
		const int blocks_y = importance.blocks_y;
		const size_t count = importance.values.size();
		inv_scales_main.resize(count);
		inv_scales_chroma.resize(count);

		float min_scale = quality_map.min_scale;
		float max_scale = quality_map.max_scale;
		if (min_scale > max_scale)
			std::swap(min_scale, max_scale);
		const float strength = clampf(quality_map.strength, 0.0f, 1.0f);
		const float min_allowed = 1e-6f;
		min_scale = std::max(min_scale, min_allowed);
		max_scale = std::max(max_scale, min_scale);

		for (int by = 0; by < blocks_y; ++by) {
			for (int bx = 0; bx < blocks_x; ++bx) {
				const size_t idx = (size_t)by * blocks_x + bx;
				const float s = clampf(importance.values[idx], 0.0f, 1.0f);
				const float base = max_scale + (min_scale - max_scale) * s;
				const float mixed = 1.0f + (base - 1.0f) * strength;
				const float m = clampf(mixed, min_scale, max_scale);
				const float inv = 1.0f / std::max(m, min_allowed);
				inv_scales_main[idx] = inv;
				inv_scales_chroma[idx] = 1.0f + (inv - 1.0f) * 0.5f;
			}
		}
	};

	compute_block_scales(in);

	float* d_block_main = nullptr;
	float* d_block_chroma = nullptr;
	if (!inv_scales_main.empty()) {
		const size_t bytes = inv_scales_main.size() * sizeof(float);
		CUDA_CHECK(cudaMalloc(&d_block_main, bytes));
		CUDA_CHECK(cudaMemcpy(d_block_main, inv_scales_main.data(), bytes, cudaMemcpyHostToDevice));
	}
	if (!inv_scales_chroma.empty()) {
		const size_t bytes = inv_scales_chroma.size() * sizeof(float);
		CUDA_CHECK(cudaMalloc(&d_block_chroma, bytes));
		CUDA_CHECK(cudaMemcpy(d_block_chroma, inv_scales_chroma.data(), bytes, cudaMemcpyHostToDevice));
	}

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
	compress_channel_gpu(d_r_src, d_r_dst, in.width, in.height, d_block_main);
	compress_channel_gpu(d_g_src, d_g_dst, in.width, in.height, d_block_chroma);
	compress_channel_gpu(d_b_src, d_b_dst, in.width, in.height, d_block_chroma);

	CUDA_CHECK(cudaMemcpy(out.r.data(), d_r_dst, N, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(out.g.data(), d_g_dst, N, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(out.b.data(), d_b_dst, N, cudaMemcpyDeviceToHost));

	cudaFree(d_r_src); cudaFree(d_g_src); cudaFree(d_b_src);
	cudaFree(d_r_dst); cudaFree(d_g_dst); cudaFree(d_b_dst);
	if (d_block_main) cudaFree(d_block_main);
	if (d_block_chroma) cudaFree(d_block_chroma);
}
