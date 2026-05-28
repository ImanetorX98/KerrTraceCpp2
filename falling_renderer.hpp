#pragma once
#include "falling_camera.hpp"
#include <string>
#include <cstdint>

// ── FallingGeoPixel ───────────────────────────────────────────────────────────
struct FallingGeoPixel {
    uint8_t outcome;   // 0=escaped, 1=disk_hit, 2=singularity, 3=trapped
    float   r_hit;     // r at disk crossing or escape radius
    float   phi_hit;   // φ at disk crossing
    float   redshift;  // g-factor (filled in shading step)
    float   r_min;     // minimum r reached (for CPU refinement mask)
    float   theta_esc; // θ direction at escape (for skybox lookup)
    float   phi_esc;   // φ direction at escape
};

// ── Photon k^μ from pixel (px,py) ────────────────────────────────────────────
// Builds k^μ from local tetrad direction, then rescales k^T to enforce null.
void init_photon_k(const CameraState& cs,
                   const double e[4][4],   // tetrad e[a][mu]
                   const KNdSMetric& bh,
                   int px, int py, int W, int H, double fov_h,
                   double k[4]);           // output contravariant k^μ

// ── Backward null geodesic ────────────────────────────────────────────────────
FallingGeoPixel trace_photon_gpg(const double x0[4], const double k0[4],
                                  const KNdSMetric& bh,
                                  const FallingParams& fp,
                                  int max_steps = 50000,
                                  double h0     = 0.05,
                                  double tol    = 1e-7);

// ── Render one frame ──────────────────────────────────────────────────────────
// Writes PNG to out_path. Prints progress to stdout:
// "[frame NNNN/TTTT] PP% T.Ts elapsed"
void render_falling_frame(int frame_idx, int total_frames,
                           const FallingParams& fp,
                           const CameraState& cs_at_frame,
                           const std::string& out_path);
