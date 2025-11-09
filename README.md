# img-compressor

A high-performance GPU-accelerated JPEG compressor written in C++ and CUDA.  
The tool supports both CPU and GPU paths for comparing compression performance and output quality.

This project demonstrates efficient RGB image compression using CUDA kernels and a custom JPEG scanline writer,  
with optional CPU comparison for benchmarking and verification.

---

## Features

- GPU-accelerated image compression (CUDA)
- Optional CPU comparison mode (`--compare`)
- Quality control with `--quality` parameter
- Simple CLI tool

---

## Requirements

- **Visual Studio 2022** with Desktop Development for C++
- **CUDA Toolkit 12.9** or later
- **vcpkg** (for dependency management)

---

## Build Instructions (Visual Studio)

1. Clone this repository:
   ```bash
   git clone https://github.com/bfalls/img-compressor.git
   cd img-compressor
   ```

2. Make sure you have vcpkg installed and integrated with Visual Studio:
  ```bash
  C:\vcpkg\vcpkg integrate install
  ```

3. Open img-compressor.sln in Visual Studio.

4. Select the desired configuration:
   Debug x64 (default for development)
   Release x64 (optimized for speed)

5. Build the project (Ctrl+Shift+B or Build -> Build Solution).

6. A sample image is included under tests\data\img-test.png.
   Run the compressor from the project root after building:
   ```bash
   img-compressor --input tests/data/img-test.png --output tests/artifacts/out.jpg --quality 85 --compare
   ```

   If you don't have a GPU it will just use the CPU path.

