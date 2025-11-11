#pragma once
#include <string>
#include "compressor.hpp"  // needed for ImageRGB {width,height,r,g,b}

/*
This isolates the JPEG writing surface from the compression logic.
Keeping the writer separate makes it easy to swap out the underlying JPEG
implementation.

jpeg_write_rgb_scanlines accepts a fully expanded RGB image and writes it to 
disk using the requested quality setting. It performs no color conversion, 
memory management, or compression setup, just final encoding.
*/
void jpeg_write_rgb_scanlines(const ImageRGB& img, const std::string& out_path, int quality);
