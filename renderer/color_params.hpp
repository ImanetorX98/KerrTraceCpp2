#pragma once
// Phase-2 colour controls: RGB, palettes, ColorParams.
//
// Part of the single-translation-unit renderer included by main.cpp;
// see docs/ARCHITETTURA.md for how the modules fit together.

#include "common.hpp"

// ── Colour ────────────────────────────────────────────────────
struct RGB { uint8_t r, g, b; };

// ── Colorization parameters (Phase 2 controls) ───────────────
enum class DiskPalette { BLACKBODY, STRATIFIED, INTERSTELLAR, INTERSTELLAR_NASA };
enum class DiskRadialProfile { PAGE_THORNE, PHYSICAL_NT };

struct ColorParams {
    double exposure    = 1.0;   // intensity multiplier before tonemapping
    double gamma       = 2.2;   // gamma correction exponent
    double temp_scale  = 1.0;   // disk blackbody temperature scale
    // A redshifted blackbody is still a blackbody, at T_obs = g * T_emit, so the
    // colour temperature should follow g all the way down. The old code floored g
    // at 0.2 before that multiplication, which rendered the innermost, most
    // redshifted ring orange instead of deep red. Off by default; opt in to
    // reproduce older frames.
    // Exposure gain for the NASA palette alone. Its levels were calibrated when
    // that palette dropped the relativistic beaming altogether; switching the
    // beaming on in v0.2.28 multiplied the disk by g^4, whose median over the
    // disk runs 0.28 (edge-on, near-extremal) to 0.43 (mild), and the palette
    // went dark. This gain restores the overall level so the beaming shows up as
    // contrast rather than as an overall dimming. It is an exposure calibration,
    // not a physical term: the ratio between any two pixels is untouched.
    // Set to 10 by preference: 5.5 restores the pre-beaming level exactly on an
    // edge-on near-extremal scene, and 10 goes brighter than that on purpose.
    // Nothing clips -- the Reinhard curve compresses the highlights -- but on a
    // mild-redshift view it lands well above the old level; use --disk-nasa-gain
    // to dial it back there.
    double nasa_gain = 10.0;
    bool   temp_redshift_clamp = false;
    double temp_redshift_floor = 0.2;
    double disk_brightness = 1.0; // common accretion-disk brightness multiplier
    double disk_opacity = 1.0; // common disk opacity [0,1], shared by all palettes
    DiskPalette palette    = DiskPalette::BLACKBODY;
    int    disk_rings      = 7;    // number of radial rings in stratified mode
    int    disk_sectors    = 14;   // number of azimuthal sectors in stratified mode
    double disk_sigma      = 0.5;  // Gaussian blend width (in cell units) in stratified mode
    double disk_hue_offset = 0.0;  // hue offset added to each cell [0,1)
    DiskRadialProfile radial_profile = DiskRadialProfile::PAGE_THORNE;
    bool   radial_term_zero_torque = true;   // (1 - sqrt(r_isco/r))
    bool   radial_term_r3_decay = true;      // 1/r^3
    bool   radial_term_relativistic = true;  // sqrt(C/A), Physical NT proxy only
    bool   radial_term_b_denom = true;       // 1/B, Physical NT proxy only
    bool   page_thorne_gaussian_taper = false; // optional ad-hoc inner taper near ISCO
    double page_thorne_taper_sigma_scale = 0.3; // sigma = scale * r_isco when taper is enabled
    bool   nt_gaussian_taper = false;       // same Gaussian taper applied to NT profile
    double nt_taper_sigma_scale = 0.8;      // wider default than PT (0.3)
    bool   planck_emission = false;          // weight intensity by Planck B(T,550nm)/B(T_ref,550nm)
    double inner_emission_floor = 0.0; // optional normalized flux floor at ISCO (0=off)
    double inner_emission_floor_width = 0.25; // floor fade width as fraction of (r_out-r_in)
    bool   doppler_enabled = true; // relativistic Doppler boosting toggle
    double doppler_exp     = 4.0;  // beaming exponent: 4=physical, <4=softer effect
    bool   zero_torque_taper = false; // multiply intensity by (1-sqrt(r_isco/r)): zero at ISCO, physical inner BC
    // Artificial: let the ray-bundle magnification modulate the disk luminance.
    // OFF by default because it is not physical for a resolved disk (see
    // magnif_luma_scale).  When on, the value is normalised frame-wide first.
    bool   bundle_magnif_luma = false;
    double bundle_magnif_max  = 5.0;   // ceiling of the 1/|det J| factor when on
    // Average the disk emission over the bundle's footprint instead of sampling
    // its centre. This is what the ray bundle is actually for -- DNGR §2.2(iii)
    // integrates the emission over the ellipse, and A.3.1 uses the beam shape to
    // adapt the resampling filter against moire. On by default when bundles are.
    bool   bundle_filter        = true;
    int    bundle_edge_grid     = 3;    // n x n beams per boundary pixel (A.3.1)
    int    bundle_filter_rings  = 2;    // sample rings inside the footprint
    double bundle_filter_sigma  = 0.5;  // Gaussian sigma in footprint radii

    // Interstellar cinematic mode (artist-friendly controls)
    double interstellar_omega0                = 1.0;
    double interstellar_p                     = 2.2;
    // Artistic exponential brightness decay, OFF by default: no thin-disk model
    // contains one. With scale 0.7*r_in and a near-extremal spin (r_isco~1.24M)
    // it decays over ~0.87M while the disk extends to tens of M, killing the
    // outer disk by ~1e7 where Novikov-Thorne asks for ~50x.
    // Radial profile: physical Novikov-Thorne flux by default. The artistic
    // power law (r/r_in)^-p is available but is not a disk model: the physical
    // flux is r^-3 (1 - sqrt(r_isco/r)), which rises from the inner edge before
    // decaying, and whose log-slope is bounded by 3.
    bool   interstellar_physical_profile      = true;
    bool   interstellar_inner_glow            = false;
    double interstellar_inner_falloff_scale   = 0.7;   // multiplied by r_in; only used when inner_glow is on
    double interstellar_band_strength         = 0.18;
    double interstellar_band_frequency        = 20.0;
    double interstellar_band_warp             = 2.0;
    double interstellar_turbulence_strength   = 1.0;
    double interstellar_hdr_intensity         = 4.0;
    double interstellar_softness_in_scale     = 0.08;  // multiplied by r_in
    double interstellar_softness_out_scale    = 0.15;  // multiplied by r_out
    double interstellar_edge_transparency   = 0.02;  // mask cutoff: low-opacity edge rays pass through
    double interstellar_outer_r               = 0.16;
    double interstellar_outer_g               = 0.045;
    double interstellar_outer_b               = 0.018;
    double interstellar_time                  = 0.0;
    double interstellar_inner_taper_scale     = 0.15; // Gaussian rise from inner edge: σ = scale * r_in
    // General radial ring taper: multiplies intensity by exp(-|r-r0|/sigma) [all palettes]
    bool   radial_ring_taper          = false;
    bool   radial_ring_taper_use_isco = false; // if true, r0 = r_ISCO at render time
    double radial_ring_taper_r0       = 6.0;   // center radius [M] (ignored when use_isco)
    double radial_ring_taper_sigma    = 2.0;   // falloff width [M]
};

// Every unsupported setting must select the reference pipeline explicitly.
// Keep the decision independent of the Metal build so it is covered by CPU CI.
