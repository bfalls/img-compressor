#include "compressor.hpp"
#include "saliency.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

/*
Quant tables are fixed to the IJG (libjpeg) baselines so quality scaling
behaves like familiar JPEG encoders. Sticking to the canonical values makes
the output comparable across implementations and keeps tuning predictable.
*/
const uint8_t kBaseQ_Luma[64] = {
  16,11,10,16,24,40,51,61,
  12,12,14,19,26,58,60,55,
  14,13,16,24,40,57,69,56,
  14,17,22,29,51,87,80,62,
  18,22,37,56,68,109,103,77,
  24,35,55,64,81,104,113,92,
  49,64,78,87,103,121,120,101,
  72,92,95,98,112,100,103,99
};

const uint8_t kBaseQ_Chroma[64] = {
  17,18,24,47,99,99,99,99,
  18,21,26,66,99,99,99,99,
  24,26,56,99,99,99,99,99,
  47,66,99,99,99,99,99,99,
  99,99,99,99,99,99,99,99,
  99,99,99,99,99,99,99,99,
  99,99,99,99,99,99,99,99,
  99,99,99,99,99,99,99,99
};

static inline int clampi(int v, int lo, int hi) {
	return v < lo ? lo : (v > hi ? hi : v);
}

static inline float clampf(float v, float lo, float hi) {
	return v < lo ? lo : (v > hi ? hi : v);
}

// IJG rule, then make DC = 1 to avoid cross-block steps.
// (At high Q, also cap a few lowest ACs to small values.)
void make_scaled_quant_tables(int quality,
	uint8_t out_luma[64],
	uint8_t out_chroma[64]) {
	quality = clampi(quality, 1, 100);
	int scale = (quality < 50) ? (5000 / quality) : (200 - quality * 2);

	auto scale_one = [scale](const uint8_t* in, uint8_t* out) {
		for (int i = 0; i < 64; ++i) {
			int v = (in[i] * scale + 50) / 100;  // integer round
			out[i] = static_cast<uint8_t>(clampi(v, 1, 255));
		}
		};

	scale_one(kBaseQ_Luma, out_luma);
	scale_one(kBaseQ_Chroma, out_chroma);

	// Critical tweak: never quantize DC (index 0)
	out_luma[0] = 1;
	out_chroma[0] = 1;

	// Optional: at high qualities reduce first row/col AC steps a bit
	if (quality >= 80) {
		// indices near DC: (0,1)..(0,3) and (1,0)..(3,0)
		const int idxs[] = { 1,2,3, 8,16,24 };
		for (int k : idxs) {
			out_luma[k] = (uint8_t)clampi(out_luma[k], 1, 2);
			out_chroma[k] = (uint8_t)clampi(out_chroma[k], 1, 2);
		}
	}
}

namespace {

	inline uint8_t clamp_u8_cpu(float v) {
		if (v < 0.f) v = 0.f;
		else if (v > 255.f) v = 255.f;
		return static_cast<uint8_t>(v + 0.5f);
	}

	/*
	Using the analytical DCT-II basis makes CPU and GPU paths comparable
	and keeps quality changes traceable to quantization rather than
	transform differences. Precomputing
	T and T^T once per call avoids clutter in the inner loops.
	*/
	void build_dct_mats_cpu(float T[64], float Tt[64]) {
		constexpr double PI = 3.14159265358979323846;
		const float invSqrt8 = 1.0f / std::sqrt(8.0f);
		for (int u = 0; u < 8; ++u) {
			float alpha = (u == 0) ? invSqrt8 : std::sqrt(2.0f) * invSqrt8;
			for (int x = 0; x < 8; ++x) {
				float val = static_cast<float>(alpha * std::cos(((2.0 * x + 1.0) * u) * (PI / 16.0)));
				T[u * 8 + x] = val;
				Tt[x * 8 + u] = val;
			}
		}
	}

	inline int idx(int x, int y, int w) { return y * w + x; }

	/*
	Per-channel function matches the GPU structure closely. Blocked 8x8
	forward DCT, in-place quant/dequant, inverse DCT, with clamped edges for
	partial tiles. The goal is not ultimate CPU performance but a readable,
	deterministic reference that is easy to compare against device output and
	simple to instrument for accuracy checks.
	*/
	void compress_channel_cpu(const uint8_t* src, uint8_t* dst, int width, int height,
		const uint8_t qtbl[64], const std::vector<float>* inv_block_scales, int blocks_x) {
		float T[64], Tt[64];
		build_dct_mats_cpu(T, Tt);

		for (int by = 0; by < height; by += 8) {
			for (int bx = 0; bx < width; bx += 8) {
				float block[8][8], tmp[8][8], coeff[8][8], tmp2[8][8];
				float block_scale = 1.0f;
				if (inv_block_scales && !inv_block_scales->empty()) {
					const int bidx = (by / 8) * blocks_x + (bx / 8);
					if (bidx >= 0 && bidx < (int)inv_block_scales->size())
						block_scale = (*inv_block_scales)[bidx];
				}

				// load/center (clamped)
				for (int ty = 0; ty < 8; ++ty) {
					for (int tx = 0; tx < 8; ++tx) {
						int x = std::min(bx + tx, width - 1);
						int y = std::min(by + ty, height - 1);
						block[ty][tx] = float(src[idx(x, y, width)]) - 128.0f;
					}
				}

				// row DCT
				for (int u = 0; u < 8; ++u)
					for (int x = 0; x < 8; ++x) {
						float s = 0.f;
						for (int k = 0; k < 8; ++k) s += T[u * 8 + k] * block[k][x];
						tmp[u][x] = s;
					}
				// col DCT
				for (int u = 0; u < 8; ++u)
					for (int v = 0; v < 8; ++v) {
						float s = 0.f;
						for (int k = 0; k < 8; ++k) s += tmp[u][k] * T[v * 8 + k];
						coeff[u][v] = s;
					}

				// Apply per-block coefficient scaling before quantization.
				// Scaling coefficients (instead of per-block quant tables) keeps the
				// bitstream standards-compliant while biasing energy allocation.
				if (block_scale != 1.0f) {
					for (int u = 0; u < 8; ++u)
						for (int v = 0; v < 8; ++v)
							coeff[u][v] *= block_scale;
				}

				// quant/dequant (DC untouched because qtbl[0]==1)
				for (int u = 0; u < 8; ++u)
					for (int v = 0; v < 8; ++v) {
						int qidx = u * 8 + v;
						float c = coeff[u][v];
						int   q = (int)std::lrint(c / float(qtbl[qidx]));
						coeff[u][v] = float(q) * float(qtbl[qidx]);
					}

				// col IDCT
				for (int u = 0; u < 8; ++u)
					for (int x = 0; x < 8; ++x) {
						float s = 0.f;
						for (int k = 0; k < 8; ++k) s += coeff[u][k] * T[k * 8 + x];
						tmp2[u][x] = s;
					}
				// row IDCT + store
				for (int y8 = 0; y8 < 8; ++y8)
					for (int x8 = 0; x8 < 8; ++x8) {
						float s = 0.f;
						for (int k = 0; k < 8; ++k) s += Tt[y8 * 8 + k] * tmp2[k][x8];
						int x = bx + x8, y = by + y8;
						if (x < width && y < height)
							dst[idx(x, y, width)] = clamp_u8_cpu(s + 128.0f);
					}
			}
		}
	}

} // namespace

void compress_image_rgb_cpu(const ImageRGB& in, ImageRGB& out, int quality,
	const QualityMapConfig& quality_map) {
	if (out.width != in.width || out.height != in.height ||
		(int)out.r.size() != in.width * in.height ||
		(int)out.g.size() != in.width * in.height ||
		(int)out.b.size() != in.width * in.height)
		throw std::runtime_error("compress_image_rgb_cpu: 'out' mis-sized.");

	int blocks_x = 0;
	std::vector<float> inv_scales_main;
	std::vector<float> inv_scales_chroma;

	auto compute_block_scales = [&](const ImageRGB& img) {
		blocks_x = (img.width + 7) / 8;
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
				const float base = max_scale + (min_scale - max_scale) * s; // lerp(max_scale, min_scale, s)
				const float mixed = 1.0f + (base - 1.0f) * strength;
				const float m = clampf(mixed, min_scale, max_scale);
				const float inv = 1.0f / std::max(m, min_allowed);
				inv_scales_main[idx] = inv;
				inv_scales_chroma[idx] = 1.0f + (inv - 1.0f) * 0.5f; // milder effect for chroma
			}
		}
	};

	compute_block_scales(in);

	uint8_t qL[64], qC_unused[64];
	make_scaled_quant_tables(quality, qL, qC_unused);

	const std::vector<float>* map_main = inv_scales_main.empty() ? nullptr : &inv_scales_main;
	const std::vector<float>* map_chroma = inv_scales_chroma.empty() ? nullptr : &inv_scales_chroma;
	compress_channel_cpu(in.r.data(), out.r.data(), in.width, in.height, qL, map_main, blocks_x);
	compress_channel_cpu(in.g.data(), out.g.data(), in.width, in.height, qL, map_chroma, blocks_x);
	compress_channel_cpu(in.b.data(), out.b.data(), in.width, in.height, qL, map_chroma, blocks_x);
}

// Metrics
double mse_plane(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
	if (a.size() != b.size()) throw std::runtime_error("mse_plane: size mismatch");
	if (a.empty()) return 0.0;
	double acc = 0.0;
	const size_t N = a.size();
	for (size_t i = 0; i < N; ++i) {
		double d = double(a[i]) - double(b[i]);
		acc += d * d;
	}
	return acc / double(N);
}

double psnr_from_mse(double mse, double peak) {
	if (mse <= 1e-12) return 99.0;
	return 10.0 * std::log10((peak * peak) / mse);
}
