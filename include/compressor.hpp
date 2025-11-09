#pragma once
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <string>

// ----------------------
// Image type (unchanged)
// ----------------------
struct ImageRGB {
	int width = 0, height = 0;
	std::vector<uint8_t> r, g, b; // size = w*h for each
};

// --------------
// Error handling
// --------------
#ifdef _MSC_VER
#define NOMINMAX
#endif

#include <cuda_runtime.h>

#define CUDA_CHECK(stmt)                                                     \
    do {                                                                     \
        cudaError_t __err = (stmt);                                          \
        if (__err != cudaSuccess) {                                          \
            throw std::runtime_error(std::string("CUDA error: ") +           \
                                     cudaGetErrorString(__err) +             \
                                     " at " __FILE__ ":" + std::to_string(__LINE__)); \
        }                                                                    \
    } while (0)

// -------------------------------------------------------
// Quantization tables (host) and quality scaling utility
// -------------------------------------------------------
extern const uint8_t kBaseQ_Luma[64];
extern const uint8_t kBaseQ_Chroma[64];

// Scales the standard JPEG (IJG) base tables according to "quality" in [1..100].
void make_scaled_quant_tables(int quality,
	uint8_t out_luma[64],
	uint8_t out_chroma[64]);

// -------------------------------
// Public CUDA-side entry points
// -------------------------------

// Initialize compressor: uploads quant tables (derived from quality) and DCT matrices.
void init_compressor(int quality);

// Compress RGB image per channel using 8x8 DCT + quantization + IDCT.
// "out" must be pre-sized to (w*h) per plane; contents are overwritten.
void compress_image_rgb_gpu(const ImageRGB& in, ImageRGB& out);

// Optional utility if you want to benchmark channels separately.
void compress_channel_gpu(const uint8_t* src, uint8_t* dst, int width, int height);

// -------------------------------
// CPU reference (for --compare)
// -------------------------------
void compress_image_rgb_cpu(const ImageRGB& in, ImageRGB& out, int quality);

// Metrics helpers
double mse_plane(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b);
double psnr_from_mse(double mse, double peak = 255.0);
