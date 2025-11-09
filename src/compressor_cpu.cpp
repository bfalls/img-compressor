#include "compressor.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

// ------------------------------
// IJG base quantization matrices
// ------------------------------
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

// ------------------------------
// CPU reference (unchanged API)
// ------------------------------
namespace {

	inline uint8_t clamp_u8_cpu(float v) {
		if (v < 0.f) v = 0.f;
		else if (v > 255.f) v = 255.f;
		return static_cast<uint8_t>(v + 0.5f);
	}

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

	void compress_channel_cpu(const uint8_t* src, uint8_t* dst, int width, int height, const uint8_t qtbl[64]) {
		float T[64], Tt[64];
		build_dct_mats_cpu(T, Tt);

		for (int by = 0; by < height; by += 8) {
			for (int bx = 0; bx < width; bx += 8) {
				float block[8][8], tmp[8][8], coeff[8][8], tmp2[8][8];

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
						for (int k = 0; k < 8; ++k) s += tmp[u][k] * Tt[v * 8 + k];
						coeff[u][v] = s;
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
						for (int k = 0; k < 8; ++k) s += Tt[k * 8 + y8] * tmp2[k][x8];
						int x = bx + x8, y = by + y8;
						if (x < width && y < height)
							dst[idx(x, y, width)] = clamp_u8_cpu(s + 128.0f);
					}
			}
		}
	}

} // namespace

void compress_image_rgb_cpu(const ImageRGB& in, ImageRGB& out, int quality) {
	if (out.width != in.width || out.height != in.height ||
		(int)out.r.size() != in.width * in.height ||
		(int)out.g.size() != in.width * in.height ||
		(int)out.b.size() != in.width * in.height)
		throw std::runtime_error("compress_image_rgb_cpu: 'out' mis-sized.");

	uint8_t qL[64], qC_unused[64];
	make_scaled_quant_tables(quality, qL, qC_unused);

	compress_channel_cpu(in.r.data(), out.r.data(), in.width, in.height, qL);
	compress_channel_cpu(in.g.data(), out.g.data(), in.width, in.height, qL);
	compress_channel_cpu(in.b.data(), out.b.data(), in.width, in.height, qL);
}

// ------------------------------
// Metrics
// ------------------------------
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
