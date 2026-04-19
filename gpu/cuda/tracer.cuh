#pragma once
// ============================================================
//  tracer.cuh — CUDA ray-tracer interface
//
//  One CUDA thread = one pixel.
//  Requires CUDA ≥ 11 and a compute-capability ≥ 6.0 GPU.
// ============================================================
#include <cstdint>
#include <vector>

struct KNdSParams_CUDA {
    double M, a, Q, Lambda;
    double r_horizon, r_isco, r_disk_out;
};

struct CameraParams_CUDA {
    double r_obs, theta_obs, fov_h;
    int    width, height;
};

/// Launch the CUDA kernel and return the rendered RGBA buffer.
std::vector<uint32_t> cuda_render(
    const KNdSParams_CUDA&  kp,
    const CameraParams_CUDA& cp);
