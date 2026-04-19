#pragma once
// ============================================================
//  metal_renderer.hpp — C++ interface to the Metal GPU backend
//
//  Include this header from C++ translation units.
//  The implementation is in metal_renderer.mm (Objective-C++).
//
//  Prerequisites (macOS only):
//    - Xcode command-line tools
//    - Link: -framework Metal -framework Foundation -framework CoreGraphics
// ============================================================
#include <cstdint>
#include <vector>

struct KNdSParams_C {
    float M, a, Q, Lambda;
    float r_horizon, r_isco, r_disk_out;
};

struct CameraParams_C {
    float r_obs, theta_obs, phi_obs, fov_h;
    int   width, height;
};

/// Renders the image on the default Metal GPU device.
/// Returns the RGBA pixel buffer (width × height × 4 bytes, ABGR order).
/// Throws std::runtime_error if Metal is unavailable.
std::vector<uint32_t> metal_render(
    const KNdSParams_C&  kp,
    const CameraParams_C& cp,
    const uint8_t* bg_rgb = nullptr, // RGB8 data, row-major
    int bg_w = 0,
    int bg_h = 0);
