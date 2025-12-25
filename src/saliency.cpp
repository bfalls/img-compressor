#include "saliency.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

namespace {

inline int clamp_int(int v, int lo, int hi) {
        return (v < lo) ? lo : (v > hi ? hi : v);
}

inline float clamp01(float v) {
        if (v < 0.0f) return 0.0f;
        if (v > 1.0f) return 1.0f;
        return v;
}

static float luma_from_rgb(uint8_t r, uint8_t g, uint8_t b) {
        return 0.299f * float(r) + 0.587f * float(g) + 0.114f * float(b);
}

static float percentile(std::vector<float> values, float pct) {
        if (values.empty()) return 0.0f;
        pct = clamp01(pct / 100.0f);
        const size_t idx = static_cast<size_t>(pct * float(values.size() - 1));
        std::nth_element(values.begin(), values.begin() + idx, values.end());
        return values[idx];
}

static std::string join_path(const std::string& base, const std::string& leaf) {
        if (base.empty()) return leaf;
        const char last = base.back();
        if (last == '/' || last == '\\') return base + leaf;
        return base + "/" + leaf;
}

static void write_heatmap_pgm(const std::string& path, const std::vector<float>& block_vals,
        int blocks_x, int blocks_y, int width, int height) {
        std::vector<uint8_t> pixels((size_t)width * height);
        for (int y = 0; y < height; ++y) {
                const int by = clamp_int(y / 8, 0, blocks_y - 1);
                for (int x = 0; x < width; ++x) {
                        const int bx = clamp_int(x / 8, 0, blocks_x - 1);
                        float v = block_vals[by * blocks_x + bx];
                        uint8_t g = static_cast<uint8_t>(clamp01(v) * 255.0f + 0.5f);
                        pixels[(size_t)y * width + x] = g;
                }
        }

        std::ofstream f(path, std::ios::binary);
        if (!f) return;
        f << "P5\n" << width << " " << height << "\n255\n";
        f.write(reinterpret_cast<const char*>(pixels.data()), pixels.size());
}

static void write_block_csv(const std::string& path, const std::vector<float>& block_vals,
        int blocks_x, int blocks_y) {
        std::ofstream f(path);
        if (!f) return;
        f << "x,y,s\n";
        f << std::fixed << std::setprecision(6);
        for (int by = 0; by < blocks_y; ++by) {
                for (int bx = 0; bx < blocks_x; ++bx) {
                        float v = clamp01(block_vals[by * blocks_x + bx]);
                        f << bx << "," << by << "," << v << "\n";
                }
        }
}

} // namespace

BlockImportanceMap compute_block_importance(const ImageRGB& img,
        const SaliencyParams& params,
        const QualityMapDebugConfig* debug) {
        const int width = img.width;
        const int height = img.height;
        if (width <= 0 || height <= 0)
                return {};

        const size_t N = static_cast<size_t>(width) * height;
        std::vector<float> luma(N);
        for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                        const size_t idx = static_cast<size_t>(y) * width + x;
                        luma[idx] = luma_from_rgb(img.r[idx], img.g[idx], img.b[idx]);
                }
        }

        // 3x3 box blur for local contrast baseline
        std::vector<float> luma_blur(N, 0.0f);
        for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                        float acc = 0.0f;
                        int count = 0;
                        for (int dy = -1; dy <= 1; ++dy) {
                                for (int dx = -1; dx <= 1; ++dx) {
                                        const int xx = clamp_int(x + dx, 0, width - 1);
                                        const int yy = clamp_int(y + dy, 0, height - 1);
                                        acc += luma[(size_t)yy * width + xx];
                                        ++count;
                                }
                        }
                        luma_blur[(size_t)y * width + x] = acc / float(count);
                }
        }

        // Sobel edge magnitude (|Gx| + |Gy|)
        static const int kGx[3][3] = {
                { -1, 0, 1 },
                { -2, 0, 2 },
                { -1, 0, 1 }
        };
        static const int kGy[3][3] = {
                { -1, -2, -1 },
                {  0,  0,  0 },
                {  1,  2,  1 }
        };

        std::vector<float> raw_importance(N, 0.0f);
        for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                        float gx = 0.0f, gy = 0.0f;
                        for (int dy = -1; dy <= 1; ++dy) {
                                const int yy = clamp_int(y + dy, 0, height - 1);
                                for (int dx = -1; dx <= 1; ++dx) {
                                        const int xx = clamp_int(x + dx, 0, width - 1);
                                        const float v = luma[(size_t)yy * width + xx];
                                        gx += float(kGx[dy + 1][dx + 1]) * v;
                                        gy += float(kGy[dy + 1][dx + 1]) * v;
                                }
                        }
                        const float edge_mag = std::fabs(gx) + std::fabs(gy);
                        const float contrast = std::fabs(luma[(size_t)y * width + x] - luma_blur[(size_t)y * width + x]);
                        raw_importance[(size_t)y * width + x] = params.edge_weight * edge_mag + params.contrast_weight * contrast;
                }
        }

        // Aggregate into 8x8 blocks
        const int blocks_x = (width + 7) / 8;
        const int blocks_y = (height + 7) / 8;
        std::vector<float> block_raw((size_t)blocks_x * blocks_y, 0.0f);
        std::vector<int> block_counts((size_t)blocks_x * blocks_y, 0);
        for (int y = 0; y < height; ++y) {
                const int by = y / 8;
                for (int x = 0; x < width; ++x) {
                        const int bx = x / 8;
                        const size_t bidx = (size_t)by * blocks_x + bx;
                        block_raw[bidx] += raw_importance[(size_t)y * width + x];
                        ++block_counts[bidx];
                }
        }
        for (size_t i = 0; i < block_raw.size(); ++i) {
                if (block_counts[i] > 0)
                        block_raw[i] /= float(block_counts[i]);
        }

        // Robust normalization using p10/p90 percentiles
        const float p10 = percentile(block_raw, 10.0f);
        const float p90 = percentile(block_raw, 90.0f);
        const float denom = std::max(1e-6f, p90 - p10);

        BlockImportanceMap result;
        result.blocks_x = blocks_x;
        result.blocks_y = blocks_y;
        result.values.resize(block_raw.size());
        for (size_t i = 0; i < block_raw.size(); ++i) {
                float s = (block_raw[i] - p10) / denom;
                result.values[i] = clamp01(s);
        }

        if (debug && !debug->output_path.empty()) {
                if (debug->write_heatmap) {
                        const std::string heatmap_path = join_path(debug->output_path, "importance_heatmap.pgm");
                        write_heatmap_pgm(heatmap_path, result.values, blocks_x, blocks_y, width, height);
                }
                if (debug->write_block_csv) {
                        const std::string csv_path = join_path(debug->output_path, "block_map.csv");
                        write_block_csv(csv_path, result.values, blocks_x, blocks_y);
                }
        }

        return result;
}

