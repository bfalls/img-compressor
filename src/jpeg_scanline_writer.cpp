// src/jpeg_scanline_writer.cpp
#include "jpeg_scanline_writer.hpp"
#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <csetjmp>
#include <vector>
#include <cstdint>

extern "C" {
#include <jpeglib.h>
}

struct my_error_mgr {
    jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
    char msg[JMSG_LENGTH_MAX];
};

static void my_output_message(j_common_ptr cinfo) {
    my_error_mgr* myerr = (my_error_mgr*)cinfo->err;
    (*cinfo->err->format_message)(cinfo, myerr->msg);
    std::fprintf(stderr, "[libjpeg] %s\n", myerr->msg);
    std::fflush(stderr);
}

static void my_error_exit(j_common_ptr cinfo) {
    my_error_mgr* myerr = (my_error_mgr*)cinfo->err;
    (*cinfo->err->output_message)(cinfo);
    longjmp(myerr->setjmp_buffer, 1);
}

static inline void interleave_row(const ImageRGB& img, int y, std::vector<unsigned char>& row) {
    const int w = img.width;
    row.resize((size_t)w * 3);
    const uint8_t* pr = img.r.data() + (size_t)y * w;
    const uint8_t* pg = img.g.data() + (size_t)y * w;
    const uint8_t* pb = img.b.data() + (size_t)y * w;
    unsigned char* p = row.data();
    for (int x = 0; x < w; ++x) {
        *p++ = pr[x];
        *p++ = pg[x];
        *p++ = pb[x];
    }
}

void jpeg_write_rgb_scanlines(const ImageRGB& img, const std::string& out_path, int quality)
{
    if (img.width <= 0 || img.height <= 0)
        throw std::runtime_error("jpeg_write_rgb_scanlines: bad size");
    if ((int)img.r.size() != img.width * img.height ||
        (int)img.g.size() != img.width * img.height ||
        (int)img.b.size() != img.width * img.height)
        throw std::runtime_error("jpeg_write_rgb_scanlines: plane size mismatch");

    jpeg_compress_struct cinfo;
    my_error_mgr jerr;
    std::memset(&cinfo, 0, sizeof(cinfo));
    std::memset(&jerr, 0, sizeof(jerr));

    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit;
    jerr.pub.output_message = my_output_message;

    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_compress(&cinfo);
        std::remove(out_path.c_str());
        throw std::runtime_error(jerr.msg[0] ? jerr.msg : "libjpeg error");
    }

    jpeg_create_compress(&cinfo);

    FILE* fp = nullptr;
#if defined(_MSC_VER)
    if (fopen_s(&fp, out_path.c_str(), "wb") != 0 || !fp)
        throw std::runtime_error("could not open output file");
#else
    fp = std::fopen(out_path.c_str(), "wb");
    if (!fp) throw std::runtime_error("could not open output file");
#endif
    jpeg_stdio_dest(&cinfo, fp);

    cinfo.image_width = (JDIMENSION)img.width;
    cinfo.image_height = (JDIMENSION)img.height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;

    jpeg_set_defaults(&cinfo);
    if (quality < 1) quality = 1;
    if (quality > 100) quality = 100;
    jpeg_set_quality(&cinfo, quality, TRUE);
#if JPEG_LIB_VERSION >= 70
    cinfo.optimize_coding = TRUE;
#endif

    jpeg_start_compress(&cinfo, TRUE);

    std::vector<unsigned char> rowbuf;
    while (cinfo.next_scanline < cinfo.image_height) {
        interleave_row(img, (int)cinfo.next_scanline, rowbuf);
        JSAMPROW row_pointer = rowbuf.data();
        jpeg_write_scanlines(&cinfo, &row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    if (fp) std::fclose(fp);
}
