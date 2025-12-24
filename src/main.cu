#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>
#include <stdexcept>

/*
stb_image is used for image decoding to keep the project self-contained.
Decoding happens on the CPU to simplify the pipeline - GPU work is reserved
exclusively for the compression stage. Code can run on systems without 
a CUDA device.
*/
#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb_image.h"

#include "compressor.hpp"
#include "jpeg_scanline_writer.hpp"

/*
Converts RGB interleaved pixels into planar format.
Performing this conversion once up front simplifies all downstream processing.
The compression pipeline expects planar data because:

 - GPU kernels achieve coalesced memory access when reading one channel at a time.
 - The CPU path benefits from better cache locality by processing contiguous planes.
 - The format aligns with JPEG's natural separation of color components (luma/chroma),
   reducing complexity when comparing or validating output across implementations.
*/
static void split_rgb_planar(const unsigned char* rgb, int w, int h, ImageRGB& img) {
	img.width = w; img.height = h;
	img.r.resize((size_t)w * h); img.g.resize((size_t)w * h); img.b.resize((size_t)w * h);
	for (int i = 0; i < w * h; ++i) {
		img.r[i] = rgb[3 * i + 0]; img.g[i] = rgb[3 * i + 1]; img.b[i] = rgb[3 * i + 2];
	}
}

static void usage() {
	std::fprintf(stderr,
		"Usage: img-compressor --input <in.(png|jpg)> --output <out.jpg> "
		"[--quality Q] [--compare] [--quality-map] [--quality-map-strength S]\n"
		"                 [--quality-map-min-scale M] [--quality-map-max-scale M]\n"
		"                 [--quality-map-debug <dir>]\n"
		"  Default backend: GPU coefficients -> JPEG (4:2:0)\n"
		"  --legacy : use previous GPU pixel path + stb writer\n");
}

// helper: turn "C:\\path\\out.jpg" into "C:\\path\\out-gpu.jpg" / "out-cpu.jpg"
static std::string with_suffix_before_ext(const std::string& path, const char* suffix) {
	size_t dot = path.find_last_of('.');
	if (dot == std::string::npos || dot == 0 || dot == path.size() - 1) {
		return path + suffix; // no extension found; just append
	}
	return path.substr(0, dot) + suffix + path.substr(dot);
}

static bool has_cuda_device()
{
	int count = 0;
	cudaError_t e = cudaGetDeviceCount(&count);
	if (e != cudaSuccess || count <= 0)
		return false;
	// also try selecting device 0 to ensure it's usable
	if (cudaSetDevice(0) != cudaSuccess)
		return false;
	return true;
}
/*
Decode once on CPU, compress on GPU (if available) and/or CPU.
Compare GPU vs CPU output if requested.
*/
int main(int argc, char** argv) {
	std::string in_path, out_path;
	int quality = 90;
	bool do_compare = false;
	QualityMapConfig quality_map;

	for (int i = 1; i < argc; ++i) {
		std::string a = argv[i];
		if (a == "--input" && i + 1 < argc) in_path = argv[++i];
		else if (a == "--output" && i + 1 < argc) out_path = argv[++i];
		else if (a == "--quality" && i + 1 < argc) quality = std::atoi(argv[++i]);
		else if (a == "--compare") do_compare = true;
		else if (a == "--quality-map") quality_map.enabled = true;
		else if (a == "--quality-map-strength" && i + 1 < argc) quality_map.strength = std::strtof(argv[++i], nullptr);
		else if (a == "--quality-map-min-scale" && i + 1 < argc) quality_map.min_scale = std::strtof(argv[++i], nullptr);
		else if (a == "--quality-map-max-scale" && i + 1 < argc) quality_map.max_scale = std::strtof(argv[++i], nullptr);
		else if (a == "--quality-map-debug" && i + 1 < argc) quality_map.debug_output_path = argv[++i];
		else { usage(); return 1; }
	}
	if (in_path.empty() || out_path.empty()) { usage(); return 1; }
	if (quality < 1) quality = 1;
	if (quality > 100) quality = 100;

	// Load input image on CPU (stb_image handles PNG, JPEG, etc.)
	int w = 0, h = 0, c = 0;
	unsigned char* rgb = stbi_load(in_path.c_str(), &w, &h, &c, 3);
	if (!rgb) { std::fprintf(stderr, "Failed to load %s\n", in_path.c_str()); return 1; }

	/*
	The decoded buffer is converted once into planar format for both
	CPU and GPU paths.
	*/
	ImageRGB img;
	split_rgb_planar(rgb, w, h, img);

	const bool gpu_ok = has_cuda_device();

	try {
		if (gpu_ok) {
			const std::string out_gpu_path = with_suffix_before_ext(out_path, "-gpu");
			auto t0 = std::chrono::high_resolution_clock::now();

			// initialize compressor (e.g. quantization tables, constants, device buffers) for given quality
			init_compressor(quality);

			ImageRGB out_gpu;
			out_gpu.width = w; out_gpu.height = h;
			out_gpu.r.resize((size_t)w * h);
			out_gpu.g.resize((size_t)w * h);
			out_gpu.b.resize((size_t)w * h);

			compress_image_rgb_gpu(img, out_gpu, quality_map);  // CUDA pipeline to RGB
			CUDA_CHECK(cudaDeviceSynchronize());

			jpeg_write_rgb_scanlines(out_gpu, out_gpu_path, quality);
			auto t1 = std::chrono::high_resolution_clock::now();
			double ms_gpu = std::chrono::duration<double, std::milli>(t1 - t0).count();
			std::printf("[GPU] wrote %s in %.3f ms\n", out_gpu_path.c_str(), ms_gpu);
		}

		if (do_compare || !gpu_ok) {
			const std::string out_cpu_path = with_suffix_before_ext(out_path, "-cpu");
			auto t0 = std::chrono::high_resolution_clock::now();
			ImageRGB out_cpu;
			out_cpu.width = w; out_cpu.height = h;
			out_cpu.r.resize((size_t)w * h);
			out_cpu.g.resize((size_t)w * h);
			out_cpu.b.resize((size_t)w * h);

			compress_image_rgb_cpu(img, out_cpu, quality, quality_map);
			jpeg_write_rgb_scanlines(out_cpu, out_cpu_path, quality);
			auto t1 = std::chrono::high_resolution_clock::now();
			double ms_cpu = std::chrono::duration<double, std::milli>(t1 - t0).count();
			std::printf("[CPU] wrote %s in %.3f ms\n", out_cpu_path.c_str(), ms_cpu);
		}
	}
	catch (const std::exception& e) {
		std::fprintf(stderr, "Error: %s\n", e.what());
		stbi_image_free(rgb);
		return 1;
	}

	stbi_image_free(rgb);

	return 0;
}
