#pragma once
// ============================================================
//  tracer.cuh — CUDA ray-tracer interface
//
//  One CUDA thread = one pixel.
//  Requires CUDA ≥ 11 and a compute-capability ≥ 6.0 GPU.
// ============================================================
#include <cstdint>
#include <vector>
#include "../../render_data.hpp"

struct KNdSParams_CUDA {
    double M, a, Q, Lambda;
    double r_horizon, r_isco, r_disk_out;
};

struct CameraParams_CUDA {
    double r_obs, theta_obs, phi_obs, fov_h;
    int    width, height;
    int    chart; // 0 = BL, 1 = KS
    int    max_steps;
    int    intersection_mode; // 0 = linear, 1 = Hermite
    double step_init, tolerance;
    double pixel_offset_x, pixel_offset_y;
};

/// Trace geometry on CUDA; shading is shared with the CPU renderer.
std::vector<GeoPixel> cuda_trace(
    const KNdSParams_CUDA&  kp,
    const CameraParams_CUDA& cp,
    bool require_fp64);
