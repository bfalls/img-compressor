#pragma once

#include <string>
#include <vector>

#include "compressor.hpp"

// Stores per-block importance scores in raster order (block-y major).
struct BlockImportanceMap {
        int blocks_x = 0;
        int blocks_y = 0;
        std::vector<float> values; // size = blocks_x * blocks_y
};

struct SaliencyParams {
        float edge_weight = 1.0f;
        float contrast_weight = 0.3f;
};

// Optional debug output controls for writing intermediate visualizations.
struct QualityMapDebugConfig {
        std::string output_path;     // directory or prefix for debug artifacts
        bool write_heatmap = true;   // write importance_heatmap.pgm
        bool write_block_csv = true; // write block_map.csv
};

// Compute per-8x8 block importance scores from an RGB image using Sobel edge
// magnitude and local contrast on the luma plane. The returned scores are
// normalized to [0,1] using robust percentiles (p10/p90).
BlockImportanceMap compute_block_importance(const ImageRGB& img,
        const SaliencyParams& params = {},
        const QualityMapDebugConfig* debug = nullptr);

