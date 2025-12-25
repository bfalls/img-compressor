#pragma once

/*
This header defines the *stable surface area* shared by the GPU/CPU
compression backends and the small CLI front-end. The design goal is to
keep `main.cu` extremely simple.
*/

#include <cstdint>
#include <vector>
#include <stdexcept>
#include <string>

/*
 Store pixels as three independent vectors (R,G,B) rather than RGBRGB because:
   - CUDA kernels can process single channels with coalesced reads / writes
     and predictable strides (fewer cache misses vs AoS layouts).
   - The CPU reference path gets predictable cache lines when iterating
     by channel and block. This reduces surprise perf gaps due to
     accidental AoS striding.
   - The separation also mirrors JPEG's 8x8 block- and component-first
     pipeline (luma vs chroma), which simplifies reasoning and testing.
*/
struct ImageRGB {
	int width = 0, height = 0;
	std::vector<uint8_t> r, g, b; // size = w*h for each
};

struct QualityMapConfig {
    bool enabled = false;
    float strength = 0.6f;
    float min_scale = 0.7f;
    float max_scale = 1.6f;
    std::string debug_output_path; // directory or prefix for debug artifacts
};

#ifdef _MSC_VER
#define NOMINMAX
#endif

#include <cuda_runtime.h>

/*
 CUDA APIs fail loudly but late. CUDA_CHECK macro close to the declarations
 gives uniform behavior (source location + message) without spreading
 errors, handling patterns across multiple implementation files.
 */
#define CUDA_CHECK(stmt)                                                     \
    do {                                                                     \
        cudaError_t __err = (stmt);                                          \
        if (__err != cudaSuccess) {                                          \
            throw std::runtime_error(std::string("CUDA error: ") +           \
                                     cudaGetErrorString(__err) +             \
                                     " at " __FILE__ ":" + std::to_string(__LINE__)); \
        }                                                                    \
    } while (0)

// Quantization tables (host) and quality scaling utility
extern const uint8_t kBaseQ_Luma[64];
extern const uint8_t kBaseQ_Chroma[64];

// Scales the standard JPEG (IJG) base tables according to "quality" in [1..100].
void make_scaled_quant_tables(int quality,
	uint8_t out_luma[64],
	uint8_t out_chroma[64]);

/*
 Push quality and precomputed constants (e.g., quant tables, color
 conversion coefficients) into an explicit initialization step to make
 the hot paths (`compress_image_rgb_gpu/cpu`) deterministic and side-effect
 free. That keeps measurement honest: the timing you see for "compress"
 doesn't include one-time setup.
 */
void init_compressor(int quality);

/*
 Both backends read ImageRGB and write ImageRGB (planar). The *output* is
 conceptually "decoded" RGB from the compressed representation so we can:
 - Compare fidelity (MSE / PSNR) between CPU / GPU on the exact same buffers.
 - Swap a writer (e.g., libjpeg - turbo vs something else) without changing
   the compression surface at all.
 */

// Compress RGB image per channel using 8x8 DCT + quantization + IDCT.
// "out" must be pre-sized to (w*h) per plane; contents are overwritten.
void compress_image_rgb_gpu(const ImageRGB& in, ImageRGB& out,
    const QualityMapConfig& quality_map = {});

// Optional utility if you want to benchmark channels separately.
void compress_channel_gpu(const uint8_t* src, uint8_t* dst, int width, int height,
    const float* inv_block_scales = nullptr);

// CPU reference (for --compare)
void compress_image_rgb_cpu(const ImageRGB& in, ImageRGB& out, int quality,
    const QualityMapConfig& quality_map = {});

// Metrics helpers
double mse_plane(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b);
double psnr_from_mse(double mse, double peak = 255.0);
