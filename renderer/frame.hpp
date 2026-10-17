#pragma once
// Per-frame camera/physics parameters and progress output.
//
// Part of the single-translation-unit renderer included by main.cpp;
// see docs/ARCHITETTURA.md for how the modules fit together.

#include "util.hpp"

// ── Progress bar ──────────────────────────────────────────────
static void print_progress(int done, int total, double elapsed) {
    const int BAR=40;
    double frac=(double)done/total;
    int filled=(int)(frac*BAR);
    std::cerr<<"\r[";
    for(int i=0;i<BAR;++i) std::cerr<<(i<filled?'#':'-');
    std::cerr<<"] "<<(int)(frac*100)<<"%";
    if(done>0&&elapsed>0.1){
        double eta=elapsed/frac-elapsed;
        std::cerr<<"  "<<std::fixed;
        std::cerr.precision(1);
        std::cerr<<elapsed<<"s elapsed, "<<eta<<"s ETA";
    }
    std::cerr<<"   "<<std::flush;
}

// ── Spacetimes ───────────────────────────────────────────────
// Which geometry a frame lives in. What differs between them beyond the tracer
// itself is answered by spacetime_traits, so callers ask "does it have GPU
// kernels?" rather than "is it the wormhole?".
enum class Spacetime : uint8_t { KNDS, DNEG_WORMHOLE };

struct SpacetimeTraits {
    const char* name;      ///< label in logs
    bool gpu_kernels;      ///< Metal/CUDA kernels exist for this geometry
    bool signed_radius;    ///< r_obs may be negative (observer on the far side)
};

static constexpr SpacetimeTraits spacetime_traits(Spacetime s) {
    return s == Spacetime::DNEG_WORMHOLE
        ? SpacetimeTraits{"wormhole", false, true}
        : SpacetimeTraits{"knds",     true,  false};
}

// ── Per-frame camera/physics parameters ───────────────────────
struct FrameParams {
    double a=0.7, theta=80.0, phi=0.0, r_obs=60.0, disk_out=12.0, fov=30.0;
    Spacetime spacetime = Spacetime::KNDS;
    // DNEG wormhole parameters (used when spacetime == DNEG_WORMHOLE)
    double wh_rho        = 1.0;   ///< throat areal radius ρ  [geometric units]
    double wh_a_tunnel   = 0.01;  ///< half-length of cylindrical tunnel a  (Interstellar default)
    double wh_M_lens     = 1.0;   ///< lensing parameter M
    double wh_look_deg   = 0.0;   ///< boresight rotation away from –ê_ℓ (0=toward wormhole, 90=perpendicular)
    double wh_sky_dist   = 1.0e4; ///< |ℓ| of the two celestial spheres (paper: ±∞)
};
