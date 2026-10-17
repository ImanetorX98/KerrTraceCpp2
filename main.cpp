// ============================================================
//  main.cpp — KNdS relativistic ray-tracer
//
//  Pipeline (two-phase separation):
//    Phase 1 – Geodesic tracing  → GeoPixel buffer (+ optional .kgeo file)
//    Phase 2 – Colorization      → RGB image (loads GeoPixel buffer)
//
//  Modes:
//    Single frame    ./kerr_tracer [--720p] [--theta 80] ...
//    Geo only        ./kerr_tracer --geo-only [--geo-file out/foo.kgeo]
//    Color only      ./kerr_tracer --color-only out/foo.kgeo [--exposure 1.5]
//    Animation       ./kerr_tracer --anim --frames 120 --orbits 1 --720p
// ============================================================
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "renderer/cli.hpp"

#ifndef KERRTRACE_NO_MAIN
int main(int argc, char** argv) {
    try { return run_cli(argc, argv); }
    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
#endif
