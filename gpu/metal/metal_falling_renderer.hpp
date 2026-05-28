#pragma once
// metal_falling_renderer.hpp — C++ bridge for the falling-camera Metal GPU backend.
// Build only on macOS with -DUSE_METAL.
#include "falling_renderer.hpp"
#include <vector>

// ── FallingCameraParams_C ─────────────────────────────────────────────────────
// POD struct mirrored exactly in MSL as FallingCameraParams.
// All fields float/int. 16-byte aligned (pad at end).
struct FallingCameraParams_C {
    float e[4][4];          // camera tetrad e[a][mu], pre-computed on CPU (double→float)
    float x[4];             // camera GPG position (T,r,θ,φ)
    float M, a, Q, Lambda;  // BH parameters
    float r_in;             // disk inner radius (ISCO when FallingParams.r_disk_in<0)
    float r_out;            // disk outer radius
    float r_isco;           // ISCO pre-computed
    float r_escape;
    float r_singularity;
    float r_horizon;        // outer horizon r_+
    float disk_brightness;
    float fov_h;            // horizontal FOV in radians
    float h0;               // initial affine step (GPU path uses 0.05f)
    float r_switch_factor;  // pixels with r_min < factor*r_horizon → CPU re-trace
    int   max_steps;        // hard cap (GPU path: 20000)
    int   width, height;
    int   pad[3];           // pad to 16-byte boundary (148 + 12 = 160 bytes)
};

// Build FallingCameraParams_C from C++ types (casts double→float).
FallingCameraParams_C make_falling_metal_params(
    const FallingParams& fp,
    const CameraState&   cs,
    const double         e[4][4]);

// Pass 1: dispatch Metal kernel for all pixels.
// Fills rgb (W*H*3 bytes) and r_min (W*H floats, minimum r per pixel).
// Returns false if Metal is unavailable or initialisation fails.
bool metal_render_falling_frame(
    const FallingCameraParams_C& params,
    std::vector<uint8_t>&        rgb,
    std::vector<float>&          r_min);
