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
    int   chart; // 0 = BL, 1 = KS
    int   solver_mode; // 0 = standard, 1 = semi-analytic, 2 = elliptic-closed
    int   integrator_mode; // 0 = RK4-doubling, 1 = DOPRI5
    int   use_bundles; // 0 = single ray, 1 = ray-bundle (GPU finite-difference bundle)
    int   metal_kernel_mode; // 0 = auto, 1 = unified(legacy), 2 = single, 3 = bundle
    int   intersection_mode; // 0 = linear, 1 = hermite
    int   elliptic_fallback_black; // 0 = normal fallback, 1 = fallback pixels forced to black
    int   anti_fireflies; // 0 = off, 1 = robust anti-fireflies filter (ray-bundle path)
    int   max_steps; // hard cap on adaptive integration iterations per ray
    float step_init; // initial affine step size
    float integrator_tol; // adaptive integrator tolerance
    float pixel_offset_x; // subpixel X offset in pixel units
    float pixel_offset_y; // subpixel Y offset in pixel units
    float exposure; // tonemap exposure
    float gamma; // tonemap gamma
    float disk_brightness; // common disk brightness multiplier
    float disk_opacity; // common disk opacity [0,1] shared by all palettes
    int   disk_palette; // 0=blackbody, 1=stratified, 2=interstellar
    int   disk_radial_profile; // 0=page_thorne, 1=physical_nt
    float interstellar_omega0;
    float interstellar_p;
    float interstellar_inner_falloff_scale;
    float interstellar_band_strength;
    float interstellar_band_frequency;
    float interstellar_band_warp;
    float interstellar_turbulence_strength;
    float interstellar_hdr_intensity;
    float interstellar_softness_in_scale;
    float interstellar_softness_out_scale;
    float interstellar_edge_transparency;
    float interstellar_outer_r;
    float interstellar_outer_g;
    float interstellar_outer_b;
    float interstellar_time;
    int   page_thorne_gaussian_taper; // 0 = off, 1 = on
    float page_thorne_taper_sigma_scale; // sigma = scale * r_isco
    float disk_flux_ref; // normalization reference for selected radial profile
    float inner_emission_floor; // optional normalized flux floor at ISCO (0=off)
    float inner_emission_floor_width; // fade width as fraction of (r_out-r_isco)
    int   enable_doppler; // 0 = disable Doppler boost, 1 = enable
    int   radial_term_zero_torque; // 0 = off, 1 = on
    int   radial_term_r3_decay; // 0 = off, 1 = on
    int   radial_term_relativistic; // 0 = off, 1 = on (physical NT only)
    int   radial_term_b_denom; // 0 = off, 1 = on (physical NT only)
    int   interstellar_inner_glow; // 0 = off (physical), 1 = artistic exponential decay
    int   interstellar_physical_profile; // 1 = Novikov-Thorne flux, 0 = artistic power law
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
