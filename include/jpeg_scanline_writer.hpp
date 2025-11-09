#pragma once
#include <string>
#include "compressor.hpp"  // brings in the existing ImageRGB {width,height,r,g,b}

void jpeg_write_rgb_scanlines(const ImageRGB& img, const std::string& out_path, int quality);
