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
#include "camera.hpp"
#include "geodesic.hpp"
#include "knds_metric.hpp"
#include "ray_bundle.hpp"
#include "wormhole_metric.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <atomic>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(USE_METAL)
#  include "gpu/metal/metal_renderer.hpp"
#elif defined(USE_CUDA)
#  include "gpu/cuda/tracer.cuh"
#elif defined(_OPENMP)
#  include <omp.h>
#endif

#include <chrono>
static double get_time() {
#if defined(_OPENMP) && !defined(USE_METAL) && !defined(USE_CUDA)
    return omp_get_wtime();
#else
    using C = std::chrono::steady_clock;
    static auto t0 = C::now();
    return std::chrono::duration<double>(C::now()-t0).count();
#endif
}

template<class T> static T clamp(T v, T lo, T hi) {
    return v < lo ? lo : v > hi ? hi : v;
}

// ── Colour ────────────────────────────────────────────────────
struct RGB { uint8_t r, g, b; };

// ── Colorization parameters (Phase 2 controls) ───────────────
enum class DiskPalette { BLACKBODY, INTERSTELLAR };

struct ColorParams {
    double exposure    = 1.0;   // intensity multiplier before tonemapping
    double gamma       = 2.2;   // gamma correction exponent
    double temp_scale  = 1.0;   // disk blackbody temperature scale
    DiskPalette palette    = DiskPalette::BLACKBODY;
    int    disk_rings      = 7;    // number of radial rings in segmented mode
    int    disk_sectors    = 14;   // number of azimuthal sectors
    double disk_sigma      = 0.5;  // Gaussian blend width (in cell units)
    double disk_hue_offset = 0.0;  // hue offset added to each cell [0,1)
};

// ── Per-pixel geodesic result (Phase 1 output) ───────────────
struct GeoPixel {
    uint8_t outcome;    // 0 = escaped, 1 = disk_hit, 2 = horizon
    uint8_t _pad[3];    // _pad[0]: debug solver tag (EllipticFallbackReason) when --debug-elliptic
    float   r;          // BL radius at disk crossing (or final r)
    float   redshift;   // g = ν_obs/ν_em
    float   magnif;     // flux magnification (bundle mode; 1 in single-ray)
    float   phi_disk;   // BL azimuthal angle at disk crossing (0 if not disk hit)
    float   theta_esc;  // direction at escape (background lookup)
    float   phi_esc;
};
static_assert(sizeof(GeoPixel) == 28, "GeoPixel size mismatch");

// ── .kgeo file format ─────────────────────────────────────────
static const char   KGEO_MAGIC[4]  = {'K','G','E','O'};
static const uint32_t KGEO_VERSION = 1;

struct KGeoMeta {
    uint32_t W, H;
    double   M_bh, a_bh, Q_bh, Lam;
    double   r_isco, r_disk_in, r_disk_out;
    double   theta_obs, phi_obs, r_obs;
};

static void save_kgeo(const char* path,
                      const std::vector<GeoPixel>& geo,
                      const KGeoMeta& meta) {
    std::ofstream f(path, std::ios::binary);
    f.write(KGEO_MAGIC, 4);
    uint32_t ver = KGEO_VERSION;
    f.write(reinterpret_cast<const char*>(&ver), 4);
    f.write(reinterpret_cast<const char*>(&meta), sizeof(meta));
    f.write(reinterpret_cast<const char*>(geo.data()),
            (std::streamsize)(geo.size() * sizeof(GeoPixel)));
}

static bool load_kgeo(const char* path,
                      std::vector<GeoPixel>& geo,
                      KGeoMeta& meta) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[4]; f.read(magic, 4);
    if (std::memcmp(magic, KGEO_MAGIC, 4) != 0) {
        std::cerr << "load_kgeo: bad magic in " << path << "\n"; return false;
    }
    uint32_t ver; f.read(reinterpret_cast<char*>(&ver), 4);
    if (ver != KGEO_VERSION) {
        std::cerr << "load_kgeo: unsupported version " << ver << "\n"; return false;
    }
    f.read(reinterpret_cast<char*>(&meta), sizeof(meta));
    geo.resize(meta.W * meta.H);
    f.read(reinterpret_cast<char*>(geo.data()),
           (std::streamsize)(geo.size() * sizeof(GeoPixel)));
    return f.good();
}

// ── Background HDRI ───────────────────────────────────────────
struct BackgroundImage {
    int w = 0, h = 0;
    std::vector<uint8_t> px;

    bool load(const char* path) {
        int c;
        uint8_t* data = stbi_load(path, &w, &h, &c, 3);
        if (!data) return false;
        px.assign(data, data + (size_t)w*h*3);
        stbi_image_free(data);
        return true;
    }

    RGB sample(double theta, double phi) const {
        if (px.empty()) return {0,0,0};
        double pw = std::fmod(phi, 2.0*M_PI);
        if (pw < 0) pw += 2.0*M_PI;
        double uf = pw  / (2.0*M_PI) * (w-1);
        double vf = theta / M_PI     * (h-1);
        int u0=(int)uf, u1=(w > 1) ? ((u0 + 1) % w) : 0;
        int v0=(int)vf, v1=std::min(v0+1,h-1);
        double fu=uf-u0, fv=vf-v0;
        auto at=[&](int u,int v)->const uint8_t*{
            return px.data()+3*(clamp(v,0,h-1)*w+clamp(u,0,w-1));
        };
        auto lp=[](double a,double b,double t){return a+t*(b-a);};
        double R=lp(lp(at(u0,v0)[0],at(u1,v0)[0],fu),lp(at(u0,v1)[0],at(u1,v1)[0],fu),fv);
        double G=lp(lp(at(u0,v0)[1],at(u1,v0)[1],fu),lp(at(u0,v1)[1],at(u1,v1)[1],fu),fv);
        double B=lp(lp(at(u0,v0)[2],at(u1,v0)[2],fu),lp(at(u0,v1)[2],at(u1,v1)[2],fu),fv);
        return {(uint8_t)R,(uint8_t)G,(uint8_t)B};
    }
};

// ── Geodesic trace helpers ────────────────────────────────────
enum class Outcome { ESCAPED, DISK_HIT, HORIZON, ESCAPED_B };
enum class CoordinateChart { KS, BL };
enum class RaySolverMode { STANDARD, SEMI_ANALYTIC, ELLIPTIC_CLOSED };
enum class IntersectionMode { LINEAR, HERMITE };
enum class EllipticFallbackReason : uint8_t {
    NONE = 0,
    INIT_SEPARABLE_CONSTS,
    THETA_CROSSING_SOLVE,
    RADIAL_MAP_INIT,
    DIRECT_RADIAL_INVALID,
    DIRECT_OUTSIDE_DISK,
    COUNT
};
struct TraceResult {
    Outcome out; double r, redshift;
    double phi_disk=0.0;   // BL azimuthal angle at disk crossing (0 if not a disk hit)
    double theta_esc=0.0, phi_esc=0.0;
};

struct IntegratorControls {
    int    max_steps = 500000;
    double step_init = 1.0;
    double tol       = 1e-7;
};

static const char* solver_mode_name(RaySolverMode mode) {
    switch (mode) {
        case RaySolverMode::STANDARD:         return "standard";
        case RaySolverMode::SEMI_ANALYTIC:    return "semi-analytic";
        case RaySolverMode::ELLIPTIC_CLOSED:  return "elliptic-closed";
    }
    return "standard";
}

static const char* intersection_mode_name(IntersectionMode mode) {
    switch (mode) {
        case IntersectionMode::LINEAR:  return "linear";
        case IntersectionMode::HERMITE: return "hermite";
    }
    return "hermite";
}

static const char* elliptic_fallback_reason_name(EllipticFallbackReason reason) {
    switch (reason) {
        case EllipticFallbackReason::NONE:                  return "none";
        case EllipticFallbackReason::INIT_SEPARABLE_CONSTS: return "init-separable-consts";
        case EllipticFallbackReason::THETA_CROSSING_SOLVE:  return "theta-crossing-solve";
        case EllipticFallbackReason::RADIAL_MAP_INIT:       return "radial-map-init";
        case EllipticFallbackReason::DIRECT_RADIAL_INVALID: return "direct-radial-invalid";
        case EllipticFallbackReason::DIRECT_OUTSIDE_DISK:   return "direct-outside-disk";
        case EllipticFallbackReason::COUNT:                 break;
    }
    return "unknown";
}

struct SolverSelection {
    RaySolverMode effective = RaySolverMode::STANDARD;
    bool fallback = false;
    const char* reason = nullptr;
};

static SolverSelection select_solver_mode(
    RaySolverMode requested,
    bool use_bundles,
    CoordinateChart chart,
    double Q_bh,
    double Lam)
{
    SolverSelection sel{};
    sel.effective = requested;

    if (use_bundles && requested != RaySolverMode::STANDARD) {
        sel.effective = RaySolverMode::STANDARD;
        sel.fallback = true;
        sel.reason = "requested solver is not available with ray bundles";
        return sel;
    }

    if (requested == RaySolverMode::SEMI_ANALYTIC) {
        if (chart != CoordinateChart::BL) {
            sel.effective = RaySolverMode::STANDARD;
            sel.fallback = true;
            sel.reason = "semi-analytic solver currently supports BL chart only";
            return sel;
        }
        if (std::abs(Q_bh) > 1e-15 || std::abs(Lam) > 1e-15) {
            sel.effective = RaySolverMode::STANDARD;
            sel.fallback = true;
            sel.reason = "semi-analytic solver currently supports Kerr only (Q=0, Lambda=0)";
            return sel;
        }
    }

    if (requested == RaySolverMode::ELLIPTIC_CLOSED) {
        if (std::abs(Q_bh) > 1e-15 || std::abs(Lam) > 1e-15) {
            sel.effective = RaySolverMode::STANDARD;
            sel.fallback = true;
            sel.reason = "elliptic-closed solver currently supports Kerr only (Q=0, Lambda=0)";
            return sel;
        }
    }
    return sel;
}

#if defined(USE_METAL)
static bool metal_solver_supported(RaySolverMode mode, CoordinateChart chart, double Q_bh, double Lam) {
    if (mode == RaySolverMode::STANDARD) return true;
    if (mode == RaySolverMode::SEMI_ANALYTIC)
        return chart == CoordinateChart::BL && std::abs(Q_bh) <= 1e-15 && std::abs(Lam) <= 1e-15;
    if (mode == RaySolverMode::ELLIPTIC_CLOSED)
        return std::abs(Q_bh) <= 1e-15 && std::abs(Lam) <= 1e-15;
    return false;
}

static int metal_solver_mode_code(RaySolverMode mode) {
    switch (mode) {
        case RaySolverMode::STANDARD:        return 0;
        case RaySolverMode::SEMI_ANALYTIC:   return 1;
        case RaySolverMode::ELLIPTIC_CLOSED: return 2;
    }
    return 0;
}

static int intersection_mode_code(IntersectionMode mode) {
    return (mode == IntersectionMode::LINEAR) ? 0 : 1;
}

enum class MetalKernelMode {
    AUTO = 0,
    UNIFIED = 1, // legacy all-in-one kernel entrypoint
    SINGLE = 2,  // force single-ray entrypoint
    BUNDLE = 3   // force ray-bundle entrypoint
};

static int metal_kernel_mode_code(MetalKernelMode mode) {
    switch (mode) {
        case MetalKernelMode::AUTO:    return 0;
        case MetalKernelMode::UNIFIED: return 1;
        case MetalKernelMode::SINGLE:  return 2;
        case MetalKernelMode::BUNDLE:  return 3;
    }
    return 0;
}

static const char* metal_kernel_mode_name(MetalKernelMode mode) {
    switch (mode) {
        case MetalKernelMode::AUTO:    return "auto";
        case MetalKernelMode::UNIFIED: return "unified";
        case MetalKernelMode::SINGLE:  return "single";
        case MetalKernelMode::BUNDLE:  return "bundle";
    }
    return "auto";
}
#endif

static double disk_redshift(double r, double pt, double pphi, const KNdSMetric& g) {
    double Omega = g.keplerian_omega(r);
    double b     = -pphi / (-pt);
    double gLL[4][4]; g.covariant_BL(r, M_PI/2.0, gLL);
    double d2 = -(gLL[0][0]+2.0*gLL[0][3]*Omega+gLL[3][3]*Omega*Omega);
    if (d2 <= 0.0) return 1.0;
    // Numerical robustness: around caustics and grazing hits the Doppler denominator
    // can flip sign due to interpolation noise, producing spurious black seams.
    const double dv = std::abs(1.0 - Omega*b);
    if (dv < 1e-8) return 20.0;
    const double red = std::sqrt(d2) / dv;
    return std::isfinite(red) ? red : 1.0;
}

static TraceResult trace_single(GeodesicState s, const KNdSMetric& g,
                                double r_disk_in, double r_disk_out,
                                double r_escape,
                                Integrator intg=Integrator::RK4_DOUBLING,
                                const IntegratorControls& ctl = IntegratorControls{}) {
    double rh=g.r_horizon(), dlam=std::max(ctl.step_init, 1e-10);
    const double rh_cut = rh * 1.03;
    Vec4d fsal=Vec4d::nan_init();
    const int max_steps = std::max(1, ctl.max_steps);
    for (int it=0; it<max_steps; ++it) {
        const GeodesicState s_prev = s;
        double step_used = dlam;
        int rejects = 0;
        while (true) {
            step_used = dlam;
            if (adaptive_step(g, s, dlam, intg, fsal, ctl.tol)) break;
            if (!std::isfinite(dlam) || ++rejects > 64)
                return {Outcome::ESCAPED, s.r, 1.0, s.theta, s.phi};
        }

        double best_alpha = 2.0;
        enum class StepEvent { NONE, DISK, HORIZON, ESCAPE };
        StepEvent best_event = StepEvent::NONE;

        double disk_r_hit = 0.0;
        double disk_redshift_hit = 1.0;
        double disk_phi_hit = 0.0;

        const double q0 = s_prev.theta - M_PI/2.0;
        const double q1 = s.theta      - M_PI/2.0;
        const bool maybe_equator = sign_change(q0, q1) ||
                                   (std::min(std::abs(q0), std::abs(q1)) < 0.35);
        if (maybe_equator) {
            double dr0,dth0,dpr0,dpth0;
            double dr1,dth1,dpr1,dpth1;
            geodesic_rhs(g, s_prev.r, s_prev.theta, s_prev.pr, s_prev.ptheta,
                         s_prev.pt, s_prev.pphi, dr0,dth0,dpr0,dpth0);
            geodesic_rhs(g, s.r, s.theta, s.pr, s.ptheta,
                         s.pt, s.pphi, dr1,dth1,dpr1,dpth1);
            double alpha = 0.0;
            if (first_event_alpha_hermite(
                    s_prev.theta, s.theta, dth0, dth1, step_used, M_PI/2.0,
                    alpha, 8, 8)) {
                const double r_hit = hermite_interp_scalar(
                    s_prev.r, s.r, dr0, dr1, step_used, alpha);
                if (r_hit>=r_disk_in && r_hit<=r_disk_out) {
                    disk_r_hit = r_hit;
                    disk_redshift_hit = clamp(disk_redshift(r_hit, s.pt, s.pphi, g), 0.0, 20.0);
                    disk_phi_hit = s_prev.phi + alpha * (s.phi - s_prev.phi);
                    best_alpha = alpha;
                    best_event = StepEvent::DISK;
                }
            }
        }

        const bool horizon_cross = ((s_prev.r > rh_cut) && (s.r <= rh_cut)) || (s.r <= rh_cut);
        if (horizon_cross) {
            const double denom_h = s_prev.r - s.r;
            double alpha_h = (std::abs(denom_h) > 1e-12) ? ((s_prev.r - rh_cut) / denom_h) : 0.0;
            alpha_h = clamp(alpha_h, 0.0, 1.0);
            if (alpha_h < best_alpha) {
                best_alpha = alpha_h;
                best_event = StepEvent::HORIZON;
            }
        }

        const bool escape_cross = ((s_prev.r < r_escape) && (s.r >= r_escape)) || (s.r >= r_escape);
        if (escape_cross) {
            const double denom_e = s.r - s_prev.r;
            double alpha_e = (std::abs(denom_e) > 1e-12) ? ((r_escape - s_prev.r) / denom_e) : 1.0;
            alpha_e = clamp(alpha_e, 0.0, 1.0);
            if (alpha_e < best_alpha) {
                best_alpha = alpha_e;
                best_event = StepEvent::ESCAPE;
            }
        }

        if (best_event == StepEvent::DISK) {
            return {Outcome::DISK_HIT, disk_r_hit, disk_redshift_hit, disk_phi_hit};
        }
        if (best_event == StepEvent::HORIZON) {
            const double r_h = s_prev.r + best_alpha * (s.r - s_prev.r);
            return {Outcome::HORIZON, r_h, 0.0};
        }
        if (best_event == StepEvent::ESCAPE) {
            const double th_esc = s_prev.theta + best_alpha * (s.theta - s_prev.theta);
            const double ph_esc = s_prev.phi   + best_alpha * (s.phi   - s_prev.phi);
            return {Outcome::ESCAPED, r_escape, 1.0, th_esc, ph_esc};
        }
    }
    return {Outcome::ESCAPED, s.r, 1.0, s.theta, s.phi};
}

// ── Separable Kerr fast path (Q=0, Lambda=0, BL chart) ──────
// Uses first integrals from Hamilton-Jacobi separation:
//   R(r), Theta(theta), and dphi/dlambda from Carter constants.
// This is a semi-analytic step toward fully elliptic inversion.
struct SeparableKerrConsts {
    double M, a;
    double E;   // E = -p_t
    double Lz;  // p_phi
    double Qc;  // Carter constant
};

struct SeparableState {
    double r, theta, phi;
    int sgn_r;
    int sgn_th;
};

static bool init_separable_consts(const GeodesicState& s, const KNdSMetric& g,
                                  SeparableKerrConsts& c) {
    if (std::abs(g.Q) > 1e-15 || std::abs(g.Lambda) > 1e-15) return false;
    c.M  = g.M;
    c.a  = g.a;
    c.E  = -s.pt;
    c.Lz = s.pphi;
    if (!(std::isfinite(c.E) && std::isfinite(c.Lz) && c.E > 0.0)) return false;

    const double st  = std::sin(s.theta);
    const double ct  = std::cos(s.theta);
    const double st2 = std::max(st*st, 1e-14);
    c.Qc = s.ptheta*s.ptheta + ct*ct * (c.Lz*c.Lz/st2 - c.a*c.a*c.E*c.E);
    return std::isfinite(c.Qc);
}

static inline double kerr_sep_delta(const SeparableKerrConsts& c, double r) {
    return r*r - 2.0*c.M*r + c.a*c.a;
}

static inline double kerr_sep_R(const SeparableKerrConsts& c, double r) {
    const double r2 = r*r;
    const double P = c.E*(r2 + c.a*c.a) - c.a*c.Lz;
    const double K = c.Qc + (c.Lz - c.a*c.E)*(c.Lz - c.a*c.E);
    return P*P - kerr_sep_delta(c, r) * K;
}

static inline double kerr_sep_Theta(const SeparableKerrConsts& c, double theta) {
    const double st = std::sin(theta);
    const double ct = std::cos(theta);
    const double st2 = std::max(st*st, 1e-14);
    return c.Qc - ct*ct * (c.Lz*c.Lz/st2 - c.a*c.a*c.E*c.E);
}

static bool kerr_sep_rhs(const SeparableKerrConsts& c, const SeparableState& s,
                         double& dr, double& dth, double& dphi) {
    const double st = std::sin(s.theta);
    const double ct = std::cos(s.theta);
    const double st2 = std::max(st*st, 1e-14);
    const double sigma = std::max(s.r*s.r + c.a*c.a*ct*ct, 1e-14);
    const double delta = kerr_sep_delta(c, s.r);
    const double P = c.E*(s.r*s.r + c.a*c.a) - c.a*c.Lz;
    const double R = kerr_sep_R(c, s.r);
    const double Th = kerr_sep_Theta(c, s.theta);

    dr  = double((s.sgn_r >= 0) ? 1 : -1) * std::sqrt(std::max(R,  0.0)) / sigma;
    dth = double((s.sgn_th>= 0) ? 1 : -1) * std::sqrt(std::max(Th, 0.0)) / sigma;
    dphi = (c.Lz/st2 + c.a*(P/std::max(delta, 1e-14) - c.E)) / sigma;

    return std::isfinite(dr) && std::isfinite(dth) && std::isfinite(dphi);
}

static bool rk4_step_separable_kerr(const SeparableKerrConsts& c, SeparableState& s, double h) {
    double dr1,dth1,dphi1;
    double dr2,dth2,dphi2;
    double dr3,dth3,dphi3;
    double dr4,dth4,dphi4;
    const SeparableState s0 = s;

    if (!kerr_sep_rhs(c, s0, dr1, dth1, dphi1)) return false;

    SeparableState s2 = s0;
    s2.r     = s0.r     + 0.5*h*dr1;
    s2.theta = s0.theta + 0.5*h*dth1;
    s2.phi   = s0.phi   + 0.5*h*dphi1;
    if (!kerr_sep_rhs(c, s2, dr2, dth2, dphi2)) return false;

    SeparableState s3 = s0;
    s3.r     = s0.r     + 0.5*h*dr2;
    s3.theta = s0.theta + 0.5*h*dth2;
    s3.phi   = s0.phi   + 0.5*h*dphi2;
    if (!kerr_sep_rhs(c, s3, dr3, dth3, dphi3)) return false;

    SeparableState s4 = s0;
    s4.r     = s0.r     + h*dr3;
    s4.theta = s0.theta + h*dth3;
    s4.phi   = s0.phi   + h*dphi3;
    if (!kerr_sep_rhs(c, s4, dr4, dth4, dphi4)) return false;

    s.r     += h/6.0*(dr1   + 2.0*dr2   + 2.0*dr3   + dr4);
    s.theta += h/6.0*(dth1  + 2.0*dth2  + 2.0*dth3  + dth4);
    s.phi   += h/6.0*(dphi1 + 2.0*dphi2 + 2.0*dphi3 + dphi4);
    return std::isfinite(s.r) && std::isfinite(s.theta) && std::isfinite(s.phi);
}

static bool rk4_adaptive_separable_kerr(const SeparableKerrConsts& c, SeparableState& s,
                                        double& h, double tol = 1e-8) {
    const SeparableState s0 = s;

    SeparableState sA = s0;
    if (!rk4_step_separable_kerr(c, sA, h)) {
        h = std::max(h*0.5, 1e-10);
        return false;
    }

    SeparableState sB = s0;
    if (!rk4_step_separable_kerr(c, sB, 0.5*h) ||
        !rk4_step_separable_kerr(c, sB, 0.5*h)) {
        h = std::max(h*0.5, 1e-10);
        return false;
    }

    const double err = std::sqrt(
        (sA.r     - sB.r)     * (sA.r     - sB.r) +
        (sA.theta - sB.theta) * (sA.theta - sB.theta) +
        (sA.phi   - sB.phi)   * (sA.phi   - sB.phi)
    ) / 15.0;

    if (!std::isfinite(err)) {
        h = std::max(h*0.5, 1e-10);
        return false;
    }

    if (err < tol || h < 1e-10) {
        s = sB;
        const double sc = (err > 1e-14) ? 0.9 * std::pow(tol/err, 0.2) : 4.0;
        h = clamp(h*sc, 1e-10, 100.0);
        return true;
    }

    h = std::max(h * 0.9 * std::pow(tol/err, 0.25), 1e-10);
    return false;
}

static TraceResult trace_single_separable_kerr(GeodesicState s_bl, const KNdSMetric& g,
                                               double r_disk_in, double r_disk_out,
                                               double r_escape,
                                               const IntegratorControls& ctl = IntegratorControls{}) {
    SeparableKerrConsts c{};
    if (!init_separable_consts(s_bl, g, c))
        return trace_single(s_bl, g, r_disk_in, r_disk_out, r_escape, Integrator::RK4_DOUBLING, ctl);

    // Initial signs from BL canonical velocity directions.
    double dr0=0.0, dth0=0.0, dpr0=0.0, dpth0=0.0;
    geodesic_rhs(g, s_bl.r, s_bl.theta, s_bl.pr, s_bl.ptheta, s_bl.pt, s_bl.pphi,
                 dr0, dth0, dpr0, dpth0);
    SeparableState s{ s_bl.r, s_bl.theta, s_bl.phi,
                      (dr0 >= 0.0 ? 1 : -1),
                      (dth0 >= 0.0 ? 1 : -1) };

    const double rh = g.r_horizon();
    const double rh_cut = rh * 1.03;
    double h = std::max(ctl.step_init, 1e-10);
    double prev_Rpot = kerr_sep_R(c, s.r);
    double prev_Thpot = kerr_sep_Theta(c, s.theta);
    const double R_turn_eps = 1e-9;
    const double Th_turn_eps = 1e-11;

    const int max_steps = std::max(1, ctl.max_steps);
    for (int it = 0; it < max_steps; ++it) {
        const SeparableState s_prev = s;
        const double prev_Rpot_step = prev_Rpot;
        const double prev_Thpot_step = prev_Thpot;
        double step_used = h;
        int rejects = 0;
        while (true) {
            step_used = h;
            if (rk4_adaptive_separable_kerr(c, s, h, ctl.tol)) break;
            if (!std::isfinite(h) || ++rejects > 64)
                return trace_single(s_bl, g, r_disk_in, r_disk_out, r_escape, Integrator::RK4_DOUBLING, ctl);
        }

        // Keep theta in [0, pi].
        if (s.theta < 0.0) {
            s.theta = -s.theta;
            s.sgn_th *= -1;
        } else if (s.theta > M_PI) {
            s.theta = 2.0*M_PI - s.theta;
            s.sgn_th *= -1;
        }

        // Turning points in separated potentials: flip branch signs.
        const double Rnow = kerr_sep_R(c, s.r);
        const double Thnow = kerr_sep_Theta(c, s.theta);
        if (Rnow <= R_turn_eps && prev_Rpot_step > R_turn_eps) s.sgn_r *= -1;
        if (Thnow <= Th_turn_eps && prev_Thpot_step > Th_turn_eps) s.sgn_th *= -1;

        double best_alpha = 2.0;
        enum class StepEvent { NONE, DISK, HORIZON, ESCAPE };
        StepEvent best_event = StepEvent::NONE;
        double disk_r_hit = 0.0;
        double disk_red_hit = 1.0;
        double disk_phi_hit = 0.0;

        const double q0 = s_prev.theta - M_PI/2.0;
        const double q1 = s.theta      - M_PI/2.0;
        const bool maybe_equator = sign_change(q0, q1) ||
                                   (std::min(std::abs(q0), std::abs(q1)) < 0.35);
        if (maybe_equator) {
            double dr0=0.0, dth0=0.0, dphi0=0.0;
            double dr1=0.0, dth1=0.0, dphi1=0.0;
            if (!kerr_sep_rhs(c, s_prev, dr0, dth0, dphi0) ||
                !kerr_sep_rhs(c, s,      dr1, dth1, dphi1)) {
                return trace_single(s_bl, g, r_disk_in, r_disk_out, r_escape, Integrator::RK4_DOUBLING, ctl);
            }

            double alpha = 0.0;
            if (first_event_alpha_hermite(
                    s_prev.theta, s.theta, dth0, dth1, step_used, M_PI/2.0,
                    alpha, 8, 8)) {
                const double r_hit = hermite_interp_scalar(
                    s_prev.r, s.r, dr0, dr1, step_used, alpha);
                if (r_hit >= r_disk_in && r_hit <= r_disk_out) {
                    disk_r_hit = r_hit;
                    disk_red_hit = clamp(disk_redshift(r_hit, s_bl.pt, s_bl.pphi, g), 0.0, 20.0);
                    disk_phi_hit = s_prev.phi + alpha * (s.phi - s_prev.phi);
                    best_alpha = alpha;
                    best_event = StepEvent::DISK;
                }
            }
        }

        const bool horizon_cross = ((s_prev.r > rh_cut) && (s.r <= rh_cut)) || (s.r <= rh_cut);
        if (horizon_cross) {
            const double denom_h = s_prev.r - s.r;
            double alpha_h = (std::abs(denom_h) > 1e-12) ? ((s_prev.r - rh_cut) / denom_h) : 0.0;
            alpha_h = clamp(alpha_h, 0.0, 1.0);
            if (alpha_h < best_alpha) {
                best_alpha = alpha_h;
                best_event = StepEvent::HORIZON;
            }
        }

        const bool escape_cross = ((s_prev.r < r_escape) && (s.r >= r_escape)) || (s.r >= r_escape);
        if (escape_cross) {
            const double denom_e = s.r - s_prev.r;
            double alpha_e = (std::abs(denom_e) > 1e-12) ? ((r_escape - s_prev.r) / denom_e) : 1.0;
            alpha_e = clamp(alpha_e, 0.0, 1.0);
            if (alpha_e < best_alpha) {
                best_alpha = alpha_e;
                best_event = StepEvent::ESCAPE;
            }
        }

        if (best_event == StepEvent::DISK)
            return {Outcome::DISK_HIT, disk_r_hit, disk_red_hit, disk_phi_hit};
        if (best_event == StepEvent::HORIZON) {
            const double r_h = s_prev.r + best_alpha * (s.r - s_prev.r);
            return {Outcome::HORIZON, r_h, 0.0};
        }
        if (best_event == StepEvent::ESCAPE) {
            const double th_esc = s_prev.theta + best_alpha * (s.theta - s_prev.theta);
            const double ph_esc = s_prev.phi   + best_alpha * (s.phi   - s_prev.phi);
            return {Outcome::ESCAPED, r_escape, 1.0, th_esc, ph_esc};
        }
        prev_Rpot = Rnow;
        prev_Thpot = Thnow;
    }

    return {Outcome::ESCAPED, s.r, 1.0, s.theta, s.phi};
}

// ── Elliptic-closed helpers (Carlson RF + Jacobi inversion) ──
static double carlson_rf(double x, double y, double z) {
    if (!(x >= 0.0 && y >= 0.0 && z >= 0.0)) return std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
        return std::numeric_limits<double>::quiet_NaN();

    double xn = x, yn = y, zn = z;
    constexpr double ERR_TOL = 1e-10;
    for (int it = 0; it < 64; ++it) {
        const double mu = (xn + yn + zn) / 3.0;
        if (!(mu > 0.0) || !std::isfinite(mu))
            return std::numeric_limits<double>::quiet_NaN();

        const double X = 1.0 - xn / mu;
        const double Y = 1.0 - yn / mu;
        const double Z = 1.0 - zn / mu;
        const double eps = std::max({std::abs(X), std::abs(Y), std::abs(Z)});
        if (eps < ERR_TOL) {
            const double E2 = X*Y - Z*Z;
            const double E3 = X*Y*Z;
            const double corr =
                1.0
                - E2 / 10.0
                + E3 / 14.0
                + (E2*E2) / 24.0
                - (3.0 * E2 * E3) / 44.0;
            return corr / std::sqrt(mu);
        }

        const double sx = std::sqrt(xn);
        const double sy = std::sqrt(yn);
        const double sz = std::sqrt(zn);
        const double lam = sx*sy + sx*sz + sy*sz;
        xn = 0.25 * (xn + lam);
        yn = 0.25 * (yn + lam);
        zn = 0.25 * (zn + lam);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

static double ellint_F_incomplete(double phi, double m) {
    const double s = std::sin(phi);
    const double c = std::cos(phi);
    const double x = c*c;
    const double y = 1.0 - m*s*s;
    if (y <= 0.0) return std::numeric_limits<double>::quiet_NaN();
    const double rf = carlson_rf(x, y, 1.0);
    return s * rf;
}

static double ellint_K_complete(double m) {
    if (!(m >= 0.0 && m < 1.0)) return std::numeric_limits<double>::quiet_NaN();
    return carlson_rf(0.0, 1.0 - m, 1.0);
}

static double jacobi_sn2_from_u(double u, double m, double Kc) {
    if (!(m >= 0.0 && m < 1.0) || !(Kc > 0.0) || !std::isfinite(u))
        return std::numeric_limits<double>::quiet_NaN();

    double ur = std::fmod(u, 2.0 * Kc);
    if (ur < 0.0) ur += 2.0 * Kc;
    if (ur > Kc) ur = 2.0 * Kc - ur; // sn^2 symmetry on [0, 2K]

    double lo = 0.0, hi = 0.5 * M_PI;
    for (int it = 0; it < 64; ++it) {
        const double mid = 0.5 * (lo + hi);
        const double Fm = ellint_F_incomplete(mid, m);
        if (!std::isfinite(Fm)) return std::numeric_limits<double>::quiet_NaN();
        if (Fm < ur) lo = mid;
        else         hi = mid;
    }
    const double s = std::sin(0.5 * (lo + hi));
    return s*s;
}

static std::complex<double> eval_monic_quartic(const std::array<double, 4>& b,
                                               const std::complex<double>& z) {
    // z^4 + b[0] z^3 + b[1] z^2 + b[2] z + b[3]
    return (((z + b[0]) * z + b[1]) * z + b[2]) * z + b[3];
}

static void quartic_roots_monic_complex(const std::array<double, 4>& b,
                                        std::array<std::complex<double>, 4>& roots_out) {
    constexpr double PI = 3.141592653589793238462643383279502884;
    const double radius = 1.0 + std::max({std::abs(b[0]), std::abs(b[1]), std::abs(b[2]), std::abs(b[3])});
    std::array<std::complex<double>, 4> z = {
        std::polar(radius, 0.0 * PI),
        std::polar(radius, 0.5 * PI),
        std::polar(radius, 1.0 * PI),
        std::polar(radius, 1.5 * PI)
    };

    for (int it = 0; it < 160; ++it) {
        double max_delta = 0.0;
        std::array<std::complex<double>, 4> zn = z;
        for (int i = 0; i < 4; ++i) {
            std::complex<double> den = 1.0;
            for (int j = 0; j < 4; ++j) {
                if (i == j) continue;
                den *= (z[i] - z[j]);
            }
            if (std::abs(den) < 1e-20) den = std::complex<double>(1e-20, 0.0);
            const std::complex<double> delta = eval_monic_quartic(b, z[i]) / den;
            zn[i] = z[i] - delta;
            max_delta = std::max(max_delta, std::abs(delta));
        }
        z = zn;
        if (max_delta < 1e-13) break;
    }
    roots_out = z;
}

static bool quartic_real_roots_monic(const std::array<double, 4>& b,
                                     std::array<double, 4>& roots_out) {
    std::array<std::complex<double>, 4> z{};
    quartic_roots_monic_complex(b, z);

    std::vector<double> rr;
    rr.reserve(4);
    for (const auto& zi : z) {
        const double tol = 1e-7 * std::max(1.0, std::abs(zi.real()));
        if (std::abs(zi.imag()) <= tol && std::isfinite(zi.real()))
            rr.push_back(zi.real());
    }
    if (rr.size() != 4) return false;

    std::sort(rr.begin(), rr.end(), std::greater<double>());
    for (int i = 0; i < 4; ++i) roots_out[i] = rr[(size_t)i];
    return true;
}

static bool jacobi_sn_cn_from_u(double u, double m, double Kc, double& sn, double& cn) {
    sn = std::numeric_limits<double>::quiet_NaN();
    cn = std::numeric_limits<double>::quiet_NaN();
    if (!(m >= 0.0 && m < 1.0) || !(Kc > 0.0) || !std::isfinite(u)) return false;

    const double twoK = 2.0 * Kc;
    long long n = static_cast<long long>(std::floor(u / twoK));
    double ur = u - static_cast<double>(n) * twoK;
    if (ur < 0.0) {
        ur += twoK;
        --n;
    }

    const bool second_half = (ur > Kc);
    const double target = second_half ? (twoK - ur) : ur; // in [0, K]

    double lo = 0.0;
    double hi = 0.5 * M_PI;
    for (int it = 0; it < 72; ++it) {
        const double mid = 0.5 * (lo + hi);
        const double Fm = ellint_F_incomplete(mid, m);
        if (!std::isfinite(Fm)) return false;
        if (Fm < target) lo = mid;
        else             hi = mid;
    }

    const double amp = 0.5 * (lo + hi);
    const double s = std::sin(amp);
    const double c = std::cos(amp);
    double sn_local = s;
    double cn_local = second_half ? -c : c;

    if ((n & 1LL) != 0) {
        sn_local = -sn_local;
        cn_local = -cn_local;
    }

    sn = sn_local;
    cn = cn_local;
    return std::isfinite(sn) && std::isfinite(cn);
}

static bool jacobi_sc_from_u(double u, double m, double Kc, double& sc) {
    sc = std::numeric_limits<double>::quiet_NaN();
    double sn = 0.0, cn = 0.0;
    if (!jacobi_sn_cn_from_u(u, m, Kc, sn, cn)) return false;
    if (std::abs(cn) < 1e-14) return false;
    sc = sn / cn;
    return std::isfinite(sc);
}

enum class EllipticRadialCase : uint8_t {
    REGION_I_II_REAL4 = 0,
    REGION_III_REAL2_COMPLEX2,
    REGION_IV_COMPLEX4,
};

struct EllipticRadialMap {
    EllipticRadialCase radial_case = EllipticRadialCase::REGION_I_II_REAL4;

    // Real-root ordering (descending, code convention) when available.
    double r1=0.0, r2=0.0, r3=0.0, r4=0.0;

    double m=0.0;      // Jacobi modulus squared (k^2)
    double Kc=0.0;     // complete elliptic integral K(m)
    double omega=0.0;  // linear phase frequency in Mino time
    double X0=0.0;     // absolute source phase magnitude

    // X(tau) = phase_sign * X0 + tau_sign * omega * tau
    int phase_sign=1;
    int tau_sign=1;

    // Region III parameters (two real + one complex-conjugate pair)
    double A=0.0, B=0.0;
    double r_lo=0.0, r_hi=0.0; // ascending real roots: r_lo < r_hi

    // Region IV parameters (two complex-conjugate pairs)
    double a2=0.0, b1=0.0, g0=0.0;
};

static double elliptic_radial_r_from_phase(const EllipticRadialMap& mp, double X) {
    if (mp.radial_case == EllipticRadialCase::REGION_I_II_REAL4) {
        const double sn2 = jacobi_sn2_from_u(X, mp.m, mp.Kc);
        if (!std::isfinite(sn2)) return std::numeric_limits<double>::quiet_NaN();

        // GL Eq. (B46) alternative form using GL_r1 (code_r4) in denominator factors.
        // This form has a larger valid sn^2 range before the Möbius singularity.
        const double r24 = mp.r2 - mp.r4;
        const double r14 = mp.r1 - mp.r4;
        const double den = r24 - r14 * sn2;
        if (std::abs(den) < 1e-14) return std::numeric_limits<double>::quiet_NaN();
        return (mp.r1 * r24 - mp.r2 * r14 * sn2) / den;
    }

    if (mp.radial_case == EllipticRadialCase::REGION_III_REAL2_COMPLEX2) {
        double sn = 0.0, cn = 0.0;
        if (!jacobi_sn_cn_from_u(X, mp.m, mp.Kc, sn, cn))
            return std::numeric_limits<double>::quiet_NaN();

        // GL Eq. (B75)
        const double num = (mp.B * mp.r_hi - mp.A * mp.r_lo)
                         + (mp.B * mp.r_hi + mp.A * mp.r_lo) * cn;
        const double den = (mp.B - mp.A) + (mp.B + mp.A) * cn;
        if (std::abs(den) < 1e-14) return std::numeric_limits<double>::quiet_NaN();
        return num / den;
    }

    // REGION_IV_COMPLEX4
    double sc = 0.0;
    if (!jacobi_sc_from_u(X, mp.m, mp.Kc, sc))
        return std::numeric_limits<double>::quiet_NaN();

    // GL Eq. (B109)
    const double den = 1.0 + mp.g0 * sc;
    if (std::abs(den) < 1e-14) return std::numeric_limits<double>::quiet_NaN();
    return -mp.a2 * ((mp.g0 - sc) / den) - mp.b1;
}

static bool init_elliptic_radial_map(const SeparableKerrConsts& c,
                                     double r0, double dr0,
                                     EllipticRadialMap& out) {
    const double Kpot = c.Qc + (c.Lz - c.a*c.E)*(c.Lz - c.a*c.E);
    const double A0 = c.E*c.a*c.a - c.a*c.Lz;
    const double c4 = c.E*c.E;
    if (!(c4 > 1e-18)) return false;
    const double c2 = 2.0*c.E*A0 - Kpot;
    const double c1 = 2.0*c.M*Kpot;
    const double c0 = A0*A0 - c.a*c.a*Kpot;

    const std::array<double, 4> b = {0.0, c2/c4, c1/c4, c0/c4}; // monic quartic

    std::array<std::complex<double>, 4> z{};
    quartic_roots_monic_complex(b, z);

    std::vector<double> real_roots;
    std::vector<std::complex<double>> complex_roots;
    real_roots.reserve(4);
    complex_roots.reserve(4);

    for (const auto& zi : z) {
        if (!std::isfinite(zi.real()) || !std::isfinite(zi.imag())) return false;
        const double tol_im = 1e-7 * std::max(1.0, std::abs(zi.real()));
        if (std::abs(zi.imag()) <= tol_im) real_roots.push_back(zi.real());
        else                               complex_roots.push_back(zi);
    }

    auto align_initial_radial_sign = [&](EllipticRadialMap& mp_local, bool flip_phase, bool flip_tau) {
        if (std::abs(dr0) < 1e-14) return;
        const double probe_tau = 1e-6;
        const double Xp = double(mp_local.phase_sign) * mp_local.X0
                        + double(mp_local.tau_sign) * mp_local.omega * probe_tau;
        const double Xm = double(mp_local.phase_sign) * mp_local.X0
                        - double(mp_local.tau_sign) * mp_local.omega * probe_tau;
        const double rp = elliptic_radial_r_from_phase(mp_local, Xp);
        const double rm = elliptic_radial_r_from_phase(mp_local, Xm);
        if (!std::isfinite(rp) || !std::isfinite(rm)) return;

        const bool want_out = (dr0 >= 0.0);
        const bool is_out = (rp >= rm);
        if (want_out == is_out) return;

        if (flip_phase) mp_local.phase_sign *= -1;
        if (flip_tau)   mp_local.tau_sign *= -1;
    };

    // Region I/II: four real roots.
    if (real_roots.size() == 4) {
        std::sort(real_roots.begin(), real_roots.end(), std::greater<double>());

        out = EllipticRadialMap{};
        out.radial_case = EllipticRadialCase::REGION_I_II_REAL4;
        out.r1 = real_roots[0];
        out.r2 = real_roots[1];
        out.r3 = real_roots[2];
        out.r4 = real_roots[3];

        const double den_omega = (out.r1 - out.r3) * (out.r2 - out.r4);
        if (!(den_omega > 0.0)) return false;
        out.omega = 0.5 * std::sqrt(den_omega);

        // Alternative-form modulus consistent with the GL_r1 (code_r4) Möbius parametrization.
        // k'^2 = (r3_GL-r2_GL)*(r4_GL-r1_GL) / ((r3_GL-r1_GL)*(r4_GL-r2_GL))
        // In code-descending (r3_GL=r2, r2_GL=r3, r4_GL=r1, r1_GL=r4):
        // k'^2 = (r2-r3)*(r1-r4) / ((r2-r4)*(r1-r3))
        const double k_num = (out.r2 - out.r3) * (out.r1 - out.r4);
        const double k_den = (out.r2 - out.r4) * (out.r1 - out.r3);
        if (!(k_den > 0.0)) return false;
        out.m = k_num / k_den;
        if (!(out.m >= 0.0 && out.m < 1.0)) return false;
        out.Kc = ellint_K_complete(out.m);
        if (!std::isfinite(out.Kc) || !(out.Kc > 0.0)) return false;

        // Source phase consistent with the alternative (r4/code) Möbius form:
        // sn^2(F0,k') = (r0-r4_GL)*(r3_GL-r1_GL) / ((r0-r3_GL)*(r4_GL-r1_GL))
        // In code-descending: (r0-r1)*(r2-r4) / ((r0-r2)*(r1-r4))
        const double den_obs = (r0 - out.r2);
        const double den_root = (out.r1 - out.r4);
        if (std::abs(den_obs) < 1e-14 || std::abs(den_root) < 1e-14) return false;
        const double u0_raw = ((r0 - out.r1) / den_obs) * ((out.r2 - out.r4) / den_root);
        if (!std::isfinite(u0_raw)) return false;
        if (u0_raw < -1e-6 || u0_raw > 1.0 + 1e-6) return false;
        const double u0 = clamp(u0_raw, 0.0, 1.0);

        out.X0 = ellint_F_incomplete(std::asin(std::sqrt(u0)), out.m);
        if (!std::isfinite(out.X0)) return false;

        // Step 4 (GL B119): alpha = -1 in case (1), +1 otherwise.
        const bool case1 = (r0 >= out.r3 && r0 <= out.r2);
        const int alpha = case1 ? -1 : +1;
        const int nu_r = (dr0 >= 0.0) ? +1 : -1;
        out.phase_sign = alpha * nu_r;
        out.tau_sign = +1;

        align_initial_radial_sign(out, /*flip_phase=*/true, /*flip_tau=*/false);
        return true;
    }

    // Region III: two real roots + one complex-conjugate pair.
    if (real_roots.size() == 2 && complex_roots.size() == 2) {
        std::sort(real_roots.begin(), real_roots.end()); // ascending GL notation
        const double r1 = real_roots[0];
        const double r2 = real_roots[1];

        const double b1 = 0.5 * (complex_roots[0].real() + complex_roots[1].real());
        const double a1 = 0.5 * (std::abs(complex_roots[0].imag()) + std::abs(complex_roots[1].imag()));
        if (!(a1 > 0.0) || !std::isfinite(a1) || !std::isfinite(b1)) return false;

        const double A = std::sqrt(a1*a1 + (b1 - r2)*(b1 - r2));
        const double B = std::sqrt(a1*a1 + (b1 - r1)*(b1 - r1));
        if (!(A > 0.0 && B > 0.0 && B > A)) return false;

        const double r21 = r2 - r1;
        const double k3 = (((A + B) * (A + B)) - r21 * r21) / (4.0 * A * B);
        if (!(k3 > 0.0 && k3 < 1.0)) return false;
        const double K3 = ellint_K_complete(k3);
        if (!std::isfinite(K3) || !(K3 > 0.0)) return false;

        const double x3 = (A*(r0 - r1) - B*(r0 - r2))
                        / (A*(r0 - r1) + B*(r0 - r2));
        if (!std::isfinite(x3)) return false;
        const double phi0 = std::acos(clamp(x3, -1.0, 1.0));
        const double F0 = ellint_F_incomplete(phi0, k3);
        if (!std::isfinite(F0)) return false;

        out = EllipticRadialMap{};
        out.radial_case = EllipticRadialCase::REGION_III_REAL2_COMPLEX2;
        out.r1 = r2; // outer real root (descending-like convenience)
        out.r2 = r1;
        out.r_lo = r1;
        out.r_hi = r2;
        out.A = A;
        out.B = B;
        out.m = k3;
        out.Kc = K3;
        out.omega = std::sqrt(A * B);
        out.X0 = F0;
        out.phase_sign = (dr0 >= 0.0) ? +1 : -1; // X3 = nu_r*F0 + omega*tau
        out.tau_sign = +1;

        align_initial_radial_sign(out, /*flip_phase=*/true, /*flip_tau=*/false);
        return true;
    }

    // Region IV: two complex-conjugate pairs.
    if (real_roots.empty() && complex_roots.size() == 4) {
        std::vector<std::complex<double>> pos_im, neg_im;
        for (const auto& zi : complex_roots) {
            if (zi.imag() >= 0.0) pos_im.push_back(zi);
            else                  neg_im.push_back(zi);
        }
        if (pos_im.size() != 2 || neg_im.size() != 2) return false;

        auto conj_dist = [](const std::complex<double>& a, const std::complex<double>& b) {
            return std::abs(a - std::conj(b));
        };
        const int j0 = (conj_dist(pos_im[0], neg_im[0]) <= conj_dist(pos_im[0], neg_im[1])) ? 0 : 1;

        auto pair_params = [](const std::complex<double>& zp, const std::complex<double>& zn) {
            const double b = 0.5 * (zp.real() + zn.real());
            const double a = 0.5 * (std::abs(zp.imag()) + std::abs(zn.imag()));
            return std::pair<double,double>{b, a};
        };

        auto [bA, aA] = pair_params(pos_im[0], neg_im[j0]);
        auto [bB, aB] = pair_params(pos_im[1], neg_im[1 - j0]);
        if (!(aA > 0.0 && aB > 0.0)) return false;

        // GL notation: b1 > 0 > b2 and a1 > a2 > 0.
        double b1 = bA, a1 = aA, b2 = bB, a2 = aB;
        if (bB > bA) {
            b1 = bB; a1 = aB;
            b2 = bA; a2 = aA;
        }
        if (!(b1 > b2)) return false;

        const double C = std::sqrt((a1 - a2)*(a1 - a2) + (b1 - b2)*(b1 - b2));
        const double D = std::sqrt((a1 + a2)*(a1 + a2) + (b1 - b2)*(b1 - b2));
        if (!(C > 0.0 && D > 0.0)) return false;

        const double k4 = (4.0 * C * D) / ((C + D) * (C + D));
        if (!(k4 > 0.0 && k4 < 1.0)) return false;
        const double K4 = ellint_K_complete(k4);
        if (!std::isfinite(K4) || !(K4 > 0.0)) return false;

        const double g_num = 4.0 * a2 * a2 - (C - D) * (C - D);
        const double g_den = (C + D) * (C + D) - 4.0 * a2 * a2;
        if (!(g_num > 0.0 && g_den > 0.0)) return false;
        const double g0 = std::sqrt(g_num / g_den);
        if (!(g0 > 0.0) || !std::isfinite(g0)) return false;

        const double x4 = (r0 - b2) / std::max(a2, 1e-15);
        const double phi0 = std::atan(x4) + std::atan(g0);
        const double F0 = ellint_F_incomplete(phi0, k4);
        if (!std::isfinite(F0)) return false;

        out = EllipticRadialMap{};
        out.radial_case = EllipticRadialCase::REGION_IV_COMPLEX4;
        out.m = k4;
        out.Kc = K4;
        out.omega = 0.5 * (C + D);
        out.X0 = F0;
        out.phase_sign = +1;
        out.tau_sign = (dr0 >= 0.0) ? +1 : -1; // X4 = F0 + nu_r*omega*tau
        out.a2 = a2;
        out.b1 = b1;
        out.g0 = g0;

        align_initial_radial_sign(out, /*flip_phase=*/false, /*flip_tau=*/true);
        return true;
    }

    return false;
}

static double separable_radial_potential(const SeparableKerrConsts& c, double r) {
    const double Kpot = c.Qc + (c.Lz - c.a*c.E) * (c.Lz - c.a*c.E);
    const double A0 = c.E * (r*r + c.a*c.a) - c.a * c.Lz;
    const double Delta = r*r - 2.0*c.M*r + c.a*c.a;
    return A0*A0 - Delta*Kpot;
}

static bool first_equator_crossing_mino_time(const SeparableKerrConsts& c,
                                             double theta0, double dtheta0,
                                             double& tau_first, double& tau_period) {
    const double A = c.a*c.a*c.E*c.E;
    if (!(A > 1e-15)) return false;

    const double B = A - c.Lz*c.Lz - c.Qc;
    const double disc = B*B + 4.0*A*c.Qc;
    if (!(disc >= 0.0)) return false;
    const double sq = std::sqrt(disc);
    const double u_plus  = (B + sq) / (2.0*A);
    const double u_minus = (B - sq) / (2.0*A);
    if (!(u_plus > 0.0 && u_minus < 0.0)) return false;

    // For the physically relevant M_- < 0 branch (Dexter & Agol 2009, Eq. 42):
    //   mu(tau) = mu_+ * cn(u, k),   k^2 = u_+ / (u_+ - u_-)
    // with u evolving linearly in Mino time.
    const double m = u_plus / (u_plus - u_minus);
    if (!(m >= 0.0 && m < 1.0)) return false;
    const double Kc = ellint_K_complete(m);
    if (!std::isfinite(Kc) || !(Kc > 0.0)) return false;

    const double omega = std::sqrt(A * (u_plus - u_minus));
    if (!(omega > 0.0) || !std::isfinite(omega)) return false;

    const double mu_plus = std::sqrt(u_plus);
    const double mu0 = std::cos(theta0);
    const double c0 = clamp(std::abs(mu0) / std::max(mu_plus, 1e-15), 0.0, 1.0);
    const double phi0 = std::acos(c0);
    const double u_base = ellint_F_incomplete(phi0, m);
    if (!std::isfinite(u_base)) return false;

    // Full period of cn in Mino-time units.
    tau_period = 4.0 * Kc / omega;
    if (!(tau_period > 0.0) || !std::isfinite(tau_period)) return false;

    // If theta moves toward equator, use +u_base; otherwise start from -u_base.
    // First equator crossing is cn(u)=0 => u = K.
    const bool toward_equator = (std::cos(theta0) * dtheta0) > 0.0;
    const double u0 = toward_equator ? u_base : -u_base;
    tau_first = (Kc - u0) / omega;
    if (tau_first < 0.0) tau_first += tau_period;
    return std::isfinite(tau_first);
}

static TraceResult trace_single_ks(GeodesicState s_bl, const KNdSMetric& g,
                                   double r_disk_in, double r_disk_out,
                                   double r_escape,
                                   Integrator intg,
                                   const IntegratorControls& ctl = IntegratorControls{});

static TraceResult trace_single_elliptic_closed(GeodesicState s_bl, const KNdSMetric& g,
                                                double r_disk_in, double r_disk_out,
                                                double r_escape,
                                                EllipticFallbackReason* fallback_reason = nullptr,
                                                CoordinateChart fallback_chart = CoordinateChart::BL,
                                                Integrator intg = Integrator::RK4_DOUBLING,
                                                const IntegratorControls& ctl = IntegratorControls{}) {
    if (fallback_reason) *fallback_reason = EllipticFallbackReason::NONE;
    auto fallback_trace = [&](EllipticFallbackReason reason) -> TraceResult {
        if (fallback_reason) *fallback_reason = reason;
        if (fallback_chart == CoordinateChart::KS)
            return trace_single_ks(s_bl, g, r_disk_in, r_disk_out, r_escape, intg, ctl);
        return trace_single(s_bl, g, r_disk_in, r_disk_out, r_escape, intg, ctl);
    };

    SeparableKerrConsts c{};
    if (!init_separable_consts(s_bl, g, c))
        return fallback_trace(EllipticFallbackReason::INIT_SEPARABLE_CONSTS);

    double dr0=0.0, dth0=0.0, dpr0=0.0, dpth0=0.0;
    geodesic_rhs(g, s_bl.r, s_bl.theta, s_bl.pr, s_bl.ptheta, s_bl.pt, s_bl.pphi,
                 dr0, dth0, dpr0, dpth0);

    double tau_first = 0.0, tau_period = 0.0;
    if (!first_equator_crossing_mino_time(c, s_bl.theta, dth0, tau_first, tau_period))
        return fallback_trace(EllipticFallbackReason::THETA_CROSSING_SOLVE);
    (void)tau_period;

    EllipticRadialMap mp{};
    if (!init_elliptic_radial_map(c, s_bl.r, dr0, mp))
        return fallback_trace(EllipticFallbackReason::RADIAL_MAP_INIT);

    const double rh_cut = g.r_horizon() * 1.03;
    const double X = double(mp.phase_sign) * mp.X0
                   + double(mp.tau_sign) * mp.omega * tau_first;
    const double r_now = elliptic_radial_r_from_phase(mp, X);
    if (!std::isfinite(r_now))
        return fallback_trace(EllipticFallbackReason::DIRECT_RADIAL_INVALID);

    const double R = separable_radial_potential(c, r_now);
    const double R_scale = std::max(1.0, std::abs(c.E*c.E*r_now*r_now*r_now*r_now));
    if (R < -1e-8 * R_scale)
        return fallback_trace(EllipticFallbackReason::DIRECT_RADIAL_INVALID);

    // For Region I/II and III, enforce the observer-accessible outer branch.
    // Region IV has no real turning points.
    if (mp.radial_case != EllipticRadialCase::REGION_IV_COMPLEX4 && r_now < mp.r1)
        return fallback_trace(EllipticFallbackReason::DIRECT_RADIAL_INVALID);

    // For Region III, the GL B75 radial formula evaluates r on the hypothetical
    // post-bounce leg (after the inner turning point r_hi, which lies inside the
    // outer horizon for near-extremal Kerr).  In reality those photons cross
    // theta=pi/2 *before* the horizon and hit the disk — the formula gives the
    // wrong r.  Fall back to numerical for any sub-horizon r_now to be safe.
    if (r_now < rh_cut)
        return fallback_trace(EllipticFallbackReason::DIRECT_RADIAL_INVALID);
    if (r_now > r_escape)
        return fallback_trace(EllipticFallbackReason::DIRECT_OUTSIDE_DISK);
    if (r_now >= r_disk_in && r_now <= r_disk_out) {
        // Horizon-precedence safety check:
        // if chart-native standard tracing says this ray falls into horizon
        // before any disk intersection, prefer horizon to avoid leakage
        // inside the black-hole silhouette.
        const TraceResult verify = (fallback_chart == CoordinateChart::KS)
            ? trace_single_ks(s_bl, g, r_disk_in, r_disk_out, r_escape, intg, ctl)
            : trace_single(s_bl, g, r_disk_in, r_disk_out, r_escape, intg, ctl);
        if (verify.out == Outcome::HORIZON)
            return verify;

        return {Outcome::DISK_HIT, r_now,
                clamp(disk_redshift(r_now, s_bl.pt, s_bl.pphi, g), 0.0, 20.0)};
    }
    return fallback_trace(EllipticFallbackReason::DIRECT_OUTSIDE_DISK);
}

// ── KS single-ray tracing (CPU) ───────────────────────────────
// Uses Kerr-Schild Cartesian coordinates (Lambda=0 chart) with
// numerical Hamiltonian gradients in (X,Y,Z).
struct KSState {
    double X, Y, Z;
    double pX, pY, pZ;
    double pT; // conserved (stationary metric)
};

static constexpr bool KS_INGOING = true;

static bool solve_3x3(const double A_in[3][3], const double b_in[3], double x[3]) {
    double A[3][4] = {
        {A_in[0][0], A_in[0][1], A_in[0][2], b_in[0]},
        {A_in[1][0], A_in[1][1], A_in[1][2], b_in[1]},
        {A_in[2][0], A_in[2][1], A_in[2][2], b_in[2]},
    };

    for (int col = 0; col < 3; ++col) {
        int piv = col;
        double best = std::abs(A[piv][col]);
        for (int r = col + 1; r < 3; ++r) {
            const double v = std::abs(A[r][col]);
            if (v > best) { best = v; piv = r; }
        }
        if (best < 1e-14) return false;
        if (piv != col) {
            for (int k = col; k < 4; ++k) std::swap(A[col][k], A[piv][k]);
        }

        const double inv = 1.0 / A[col][col];
        for (int k = col; k < 4; ++k) A[col][k] *= inv;
        for (int r = 0; r < 3; ++r) {
            if (r == col) continue;
            const double f = A[r][col];
            for (int k = col; k < 4; ++k) A[r][k] -= f * A[col][k];
        }
    }

    x[0] = A[0][3];
    x[1] = A[1][3];
    x[2] = A[2][3];
    return true;
}

static void jacobian_bl_to_ks(double r, double theta, double phi, double a_spin, double J[3][3]) {
    const double er  = 1e-6 * (std::abs(r) + 1.0);
    const double eth = 1e-6;
    const double eph = 1e-6;

    auto f = [&](double rr, double tt, double pp) {
        double X, Y, Z;
        KNdSMetric::BL_to_KS_spatial(rr, tt, pp, a_spin, X, Y, Z);
        return std::array<double, 3>{X, Y, Z};
    };

    const auto xr_p = f(r + er, theta, phi);
    const auto xr_m = f(r - er, theta, phi);
    const auto xt_p = f(r, theta + eth, phi);
    const auto xt_m = f(r, theta - eth, phi);
    const auto xp_p = f(r, theta, phi + eph);
    const auto xp_m = f(r, theta, phi - eph);

    for (int i = 0; i < 3; ++i) {
        J[i][0] = (xr_p[i] - xr_m[i]) / (2.0 * er);   // d(X,Y,Z)/dr
        J[i][1] = (xt_p[i] - xt_m[i]) / (2.0 * eth);  // d(X,Y,Z)/dtheta
        J[i][2] = (xp_p[i] - xp_m[i]) / (2.0 * eph);  // d(X,Y,Z)/dphi
    }
}

static bool bl_covector_to_ks(double r, double theta, double phi, double a_spin,
                              double pr, double ptheta, double pphi,
                              double& pX, double& pY, double& pZ) {
    double J[3][3];
    jacobian_bl_to_ks(r, theta, phi, a_spin, J);

    // p_q = J^T p_xyz
    double A[3][3];
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i)
            A[j][i] = J[i][j];
    const double b[3] = {pr, ptheta, pphi};
    double x[3];
    if (!solve_3x3(A, b, x)) return false;
    pX = x[0]; pY = x[1]; pZ = x[2];
    return true;
}

static std::array<double, 3> ks_covector_to_bl(double r, double theta, double phi, double a_spin,
                                                double pX, double pY, double pZ) {
    double J[3][3];
    jacobian_bl_to_ks(r, theta, phi, a_spin, J);
    std::array<double, 3> pbl{};
    const double pxyz[3] = {pX, pY, pZ};
    // p_q = J^T p_xyz
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i)
            pbl[j] += J[i][j] * pxyz[i];
    return pbl;
}

static bool init_ks_state(const GeodesicState& s_bl, const KNdSMetric& g, KSState& s_ks) {
    if (std::abs(g.Lambda) > 1e-15) return false;

    KNdSMetric::BL_to_KS_spatial(s_bl.r, s_bl.theta, s_bl.phi, g.a, s_ks.X, s_ks.Y, s_ks.Z);
    if (!bl_covector_to_ks(s_bl.r, s_bl.theta, s_bl.phi, g.a,
                           s_bl.pr, s_bl.ptheta, s_bl.pphi,
                           s_ks.pX, s_ks.pY, s_ks.pZ)) {
        return false;
    }

    double gUU[4][4];
    g.contravariant_KS(0.0, s_ks.X, s_ks.Y, s_ks.Z, KS_INGOING, gUU);
    const double A = gUU[0][0];
    const double B = 2.0 * (gUU[0][1] * s_ks.pX + gUU[0][2] * s_ks.pY + gUU[0][3] * s_ks.pZ);
    const double C =
        gUU[1][1] * s_ks.pX * s_ks.pX + gUU[2][2] * s_ks.pY * s_ks.pY + gUU[3][3] * s_ks.pZ * s_ks.pZ +
        2.0 * gUU[1][2] * s_ks.pX * s_ks.pY + 2.0 * gUU[1][3] * s_ks.pX * s_ks.pZ + 2.0 * gUU[2][3] * s_ks.pY * s_ks.pZ;
    const double disc = B * B - 4.0 * A * C;
    if (disc < 0.0 || std::abs(A) < 1e-15) return false;

    const double sq = std::sqrt(disc);
    const double pT1 = (-B - sq) / (2.0 * A);
    const double pT2 = (-B + sq) / (2.0 * A);
    s_ks.pT = (pT1 < 0.0) ? pT1 : pT2;
    if (s_ks.pT > 0.0) s_ks.pT = std::min(pT1, pT2);
    return std::isfinite(s_ks.pT);
}

static void geodesic_rhs_ks(const KNdSMetric& g, const KSState& s,
                            double& dX, double& dY, double& dZ,
                            double& dpX, double& dpY, double& dpZ) {
    double gUU[4][4];
    g.contravariant_KS(0.0, s.X, s.Y, s.Z, KS_INGOING, gUU);

    dX = gUU[1][0] * s.pT + gUU[1][1] * s.pX + gUU[1][2] * s.pY + gUU[1][3] * s.pZ;
    dY = gUU[2][0] * s.pT + gUU[2][1] * s.pX + gUU[2][2] * s.pY + gUU[2][3] * s.pZ;
    dZ = gUU[3][0] * s.pT + gUU[3][1] * s.pX + gUU[3][2] * s.pY + gUU[3][3] * s.pZ;

    auto Hxyz = [&](double X, double Y, double Z) {
        double p[4] = {s.pT, s.pX, s.pY, s.pZ};
        return g.hamiltonian_KS(0.0, X, Y, Z, p, KS_INGOING);
    };

    const double eX = 1e-5 * (std::abs(s.X) + 0.1);
    const double eY = 1e-5 * (std::abs(s.Y) + 0.1);
    const double eZ = 1e-5 * (std::abs(s.Z) + 0.1);

    dpX = -(Hxyz(s.X + eX, s.Y, s.Z) - Hxyz(s.X - eX, s.Y, s.Z)) / (2.0 * eX);
    dpY = -(Hxyz(s.X, s.Y + eY, s.Z) - Hxyz(s.X, s.Y - eY, s.Z)) / (2.0 * eY);
    dpZ = -(Hxyz(s.X, s.Y, s.Z + eZ) - Hxyz(s.X, s.Y, s.Z - eZ)) / (2.0 * eZ);
}

static void rk4_step_ks(const KNdSMetric& g, KSState& s, double dlam) {
    double dX1, dY1, dZ1, dpX1, dpY1, dpZ1;
    double dX2, dY2, dZ2, dpX2, dpY2, dpZ2;
    double dX3, dY3, dZ3, dpX3, dpY3, dpZ3;
    double dX4, dY4, dZ4, dpX4, dpY4, dpZ4;

    const KSState s0 = s;

    geodesic_rhs_ks(g, s0, dX1, dY1, dZ1, dpX1, dpY1, dpZ1);
    KSState s2{s0.X + 0.5 * dlam * dX1, s0.Y + 0.5 * dlam * dY1, s0.Z + 0.5 * dlam * dZ1,
               s0.pX + 0.5 * dlam * dpX1, s0.pY + 0.5 * dlam * dpY1, s0.pZ + 0.5 * dlam * dpZ1,
               s0.pT};
    geodesic_rhs_ks(g, s2, dX2, dY2, dZ2, dpX2, dpY2, dpZ2);

    KSState s3{s0.X + 0.5 * dlam * dX2, s0.Y + 0.5 * dlam * dY2, s0.Z + 0.5 * dlam * dZ2,
               s0.pX + 0.5 * dlam * dpX2, s0.pY + 0.5 * dlam * dpY2, s0.pZ + 0.5 * dlam * dpZ2,
               s0.pT};
    geodesic_rhs_ks(g, s3, dX3, dY3, dZ3, dpX3, dpY3, dpZ3);

    KSState s4{s0.X + dlam * dX3, s0.Y + dlam * dY3, s0.Z + dlam * dZ3,
               s0.pX + dlam * dpX3, s0.pY + dlam * dpY3, s0.pZ + dlam * dpZ3,
               s0.pT};
    geodesic_rhs_ks(g, s4, dX4, dY4, dZ4, dpX4, dpY4, dpZ4);

    s.X  += dlam / 6.0 * (dX1 + 2.0 * dX2 + 2.0 * dX3 + dX4);
    s.Y  += dlam / 6.0 * (dY1 + 2.0 * dY2 + 2.0 * dY3 + dY4);
    s.Z  += dlam / 6.0 * (dZ1 + 2.0 * dZ2 + 2.0 * dZ3 + dZ4);
    s.pX += dlam / 6.0 * (dpX1 + 2.0 * dpX2 + 2.0 * dpX3 + dpX4);
    s.pY += dlam / 6.0 * (dpY1 + 2.0 * dpY2 + 2.0 * dpY3 + dpY4);
    s.pZ += dlam / 6.0 * (dpZ1 + 2.0 * dpZ2 + 2.0 * dpZ3 + dpZ4);
}

static bool rk4_adaptive_ks(const KNdSMetric& g, KSState& s, double& dlam, double tol = 1e-7) {
    const KSState s0 = s;
    KSState sA = s0;
    rk4_step_ks(g, sA, dlam);
    KSState sB = s0;
    rk4_step_ks(g, sB, 0.5 * dlam);
    rk4_step_ks(g, sB, 0.5 * dlam);

    const double err = std::sqrt(
        (sA.X  - sB.X)  * (sA.X  - sB.X)  +
        (sA.Y  - sB.Y)  * (sA.Y  - sB.Y)  +
        (sA.Z  - sB.Z)  * (sA.Z  - sB.Z)  +
        (sA.pX - sB.pX) * (sA.pX - sB.pX) +
        (sA.pY - sB.pY) * (sA.pY - sB.pY) +
        (sA.pZ - sB.pZ) * (sA.pZ - sB.pZ)) / 15.0;

    if (!std::isfinite(err)) {
        s = s0;
        dlam = (std::isfinite(dlam) && dlam > 1e-10) ? dlam * 0.5 : 1e-6;
        if (dlam < 1e-10) dlam = 1e-10;
        return false;
    }

    const bool accepted = (err < tol || dlam < 1e-10);
    if (accepted) {
        s = sB;
        const double scale = (err > 1e-14) ? 0.9 * std::pow(tol / err, 0.2) : 4.0;
        double hnew = dlam * scale;
        if (!std::isfinite(hnew)) hnew = dlam;
        dlam = clamp(hnew, 1e-10, 100.0);
    } else {
        s = s0;
        const double half = dlam * 0.5;
        double hnew = dlam * 0.9 * std::pow(tol / err, 0.25);
        if (!std::isfinite(hnew)) hnew = half;
        dlam = hnew;
        if (dlam > half) dlam = half;
        if (dlam < 1e-10) dlam = 1e-10;
    }
    return accepted;
}

static TraceResult trace_single_ks(GeodesicState s_bl, const KNdSMetric& g,
                                   double r_disk_in, double r_disk_out,
                                   double r_escape,
                                   Integrator intg,
                                   const IntegratorControls& ctl) {
    KSState s{};
    if (!init_ks_state(s_bl, g, s))
        return trace_single(s_bl, g, r_disk_in, r_disk_out, r_escape, intg, ctl);

    if (intg == Integrator::DOPRI5) {
        static bool warned_dopri5_ks = false;
        if (!warned_dopri5_ks) {
            std::cerr << "Info: KS chart currently uses RK4-doubling; ignoring --dopri5.\n";
            warned_dopri5_ks = true;
        }
    }

    const double rh = g.r_horizon();
    const double rh_cut = rh * 1.03;
    double dlam = std::max(ctl.step_init, 1e-10);
    double prev_r_ks = KNdSMetric::r_KS(s.X, s.Y, s.Z, g.a);

    const int max_steps = std::max(1, ctl.max_steps);
    for (int it = 0; it < max_steps; ++it) {
        const KSState s_prev = s;
        double step_used = dlam;
        int rejects = 0;
        while (true) {
            step_used = dlam;
            if (rk4_adaptive_ks(g, s, dlam, ctl.tol)) break;
            if (!std::isfinite(dlam) || ++rejects > 64) {
                double r_tmp, th_tmp, ph_tmp;
                KNdSMetric::KS_to_BL_spatial(s.X, s.Y, s.Z, g.a, r_tmp, th_tmp, ph_tmp);
                return {Outcome::ESCAPED, r_tmp, 1.0, th_tmp, ph_tmp};
            }
        }

        const double r_now = KNdSMetric::r_KS(s.X, s.Y, s.Z, g.a);
        double best_alpha = 2.0;
        enum class StepEvent { NONE, DISK, HORIZON, ESCAPE };
        StepEvent best_event = StepEvent::NONE;
        double disk_r_hit = 0.0;
        double disk_red_hit = 1.0;
        double disk_phi_hit = 0.0;

        const bool maybe_equator = sign_change(s_prev.Z, s.Z) ||
                                   (std::min(std::abs(s_prev.Z), std::abs(s.Z)) < 0.35);
        if (maybe_equator) {
            double dX0=0.0, dY0=0.0, dZ0=0.0, dpX0=0.0, dpY0=0.0, dpZ0=0.0;
            double dX1=0.0, dY1=0.0, dZ1=0.0, dpX1=0.0, dpY1=0.0, dpZ1=0.0;
            geodesic_rhs_ks(g, s_prev, dX0, dY0, dZ0, dpX0, dpY0, dpZ0);
            geodesic_rhs_ks(g, s,      dX1, dY1, dZ1, dpX1, dpY1, dpZ1);

            double alpha = 0.0;
            bool has_equator_hit = first_event_alpha_hermite(
                s_prev.Z, s.Z, dZ0, dZ1, step_used, 0.0, alpha, 8, 8);
            if (!has_equator_hit && sign_change(s_prev.Z, s.Z)) {
                // Fallback: if Hermite misses a genuine sign-crossing,
                // recover with linear interpolation instead of losing the event.
                const double denom = s_prev.Z - s.Z;
                alpha = (std::abs(denom) > 1e-12) ? (s_prev.Z / denom) : 0.5;
                alpha = clamp(alpha, 0.0, 1.0);
                has_equator_hit = true;
            }
            if (has_equator_hit) {
                const double Xh = hermite_interp_scalar(s_prev.X,  s.X,  dX0,  dX1,  step_used, alpha);
                const double Yh = hermite_interp_scalar(s_prev.Y,  s.Y,  dY0,  dY1,  step_used, alpha);
                const double Zh = hermite_interp_scalar(s_prev.Z,  s.Z,  dZ0,  dZ1,  step_used, alpha);
                const double pXh = hermite_interp_scalar(s_prev.pX, s.pX, dpX0, dpX1, step_used, alpha);
                const double pYh = hermite_interp_scalar(s_prev.pY, s.pY, dpY0, dpY1, step_used, alpha);
                const double pZh = hermite_interp_scalar(s_prev.pZ, s.pZ, dpZ0, dpZ1, step_used, alpha);
                const double pTh = s_prev.pT;

                double r_hit, th_hit, ph_hit;
                KNdSMetric::KS_to_BL_spatial(Xh, Yh, Zh, g.a, r_hit, th_hit, ph_hit);
                if (r_hit >= r_disk_in && r_hit <= r_disk_out) {
                    const auto pbl = ks_covector_to_bl(r_hit, th_hit, ph_hit, g.a, pXh, pYh, pZh);
                    disk_r_hit  = r_hit;
                    disk_red_hit = clamp(disk_redshift(r_hit, pTh, pbl[2], g), 0.0, 20.0);
                    disk_phi_hit = ph_hit;
                    best_alpha  = alpha;
                    best_event  = StepEvent::DISK;
                }
            }
        }

        const bool horizon_cross = ((prev_r_ks > rh_cut) && (r_now <= rh_cut)) || (r_now <= rh_cut);
        if (horizon_cross) {
            const double denom_h = prev_r_ks - r_now;
            double alpha_h = (std::abs(denom_h) > 1e-12) ? ((prev_r_ks - rh_cut) / denom_h) : 0.0;
            alpha_h = clamp(alpha_h, 0.0, 1.0);
            if (alpha_h < best_alpha) {
                best_alpha = alpha_h;
                best_event = StepEvent::HORIZON;
            }
        }

        const bool escape_cross = ((prev_r_ks < r_escape) && (r_now >= r_escape)) || (r_now >= r_escape);
        if (escape_cross) {
            const double denom_e = r_now - prev_r_ks;
            double alpha_e = (std::abs(denom_e) > 1e-12) ? ((r_escape - prev_r_ks) / denom_e) : 1.0;
            alpha_e = clamp(alpha_e, 0.0, 1.0);
            if (alpha_e < best_alpha) {
                best_alpha = alpha_e;
                best_event = StepEvent::ESCAPE;
            }
        }

        if (best_event == StepEvent::DISK) {
            return {Outcome::DISK_HIT, disk_r_hit, disk_red_hit, disk_phi_hit};
        }
        if (best_event == StepEvent::HORIZON) {
            const double r_h = prev_r_ks + best_alpha * (r_now - prev_r_ks);
            return {Outcome::HORIZON, r_h, 0.0};
        }
        if (best_event == StepEvent::ESCAPE) {
            const double X_esc = s_prev.X + best_alpha * (s.X - s_prev.X);
            const double Y_esc = s_prev.Y + best_alpha * (s.Y - s_prev.Y);
            const double Z_esc = s_prev.Z + best_alpha * (s.Z - s_prev.Z);
            double r_esc, th_esc, ph_esc;
            KNdSMetric::KS_to_BL_spatial(X_esc, Y_esc, Z_esc, g.a, r_esc, th_esc, ph_esc);
            return {Outcome::ESCAPED, r_esc, 1.0, th_esc, ph_esc};
        }

        prev_r_ks = r_now;
    }

    double r_end, th_end, ph_end;
    KNdSMetric::KS_to_BL_spatial(s.X, s.Y, s.Z, g.a, r_end, th_end, ph_end);
    return {Outcome::ESCAPED, r_end, 1.0, th_end, ph_end};
}

// ── Colorization (Phase 2) ────────────────────────────────────
static RGB blackbody(double T) {
    T=clamp(T,800.0,4e4);
    double t=std::log10(T/800.0)/std::log10(4e4/800.0);
    double R,G,B;
    if(t<0.25){R=1;G=t/0.25*0.4;B=0;}
    else if(t<0.5){double f=(t-0.25)/0.25;R=1;G=0.4+f*0.4;B=f*0.3;}
    else if(t<0.75){double f=(t-0.5)/0.25;R=1;G=0.8+f*0.2;B=0.3+f*0.5;}
    else{double f=(t-0.75)/0.25;R=1-f*0.2;G=1;B=0.8+f*0.2;}
    return {(uint8_t)(clamp(R,0.0,1.0)*255),
            (uint8_t)(clamp(G,0.0,1.0)*255),
            (uint8_t)(clamp(B,0.0,1.0)*255)};
}

static double page_thorne(double r, double r_isco) {
    if (r <= r_isco) return 0.0;
    double x=std::sqrt(r_isco/r);
    return (1.0-x)/(r*r*r);
}

// Reinhard tonemapping with variable gamma
static double tonemap(double x, double exposure, double gamma) {
    x = x * exposure;
    x = x / (1.0 + x);
    return std::pow(clamp(x, 0.0, 1.0), 1.0/gamma);
}

// HSV → RGB (all values in [0,1])
static RGB hsv_to_rgb(double h, double s, double v) {
    h = h - std::floor(h);  // wrap to [0,1)
    const double hh = h * 6.0;
    const int    hi = (int)hh;
    const double f  = hh - hi;
    const double p  = v * (1.0 - s);
    const double q  = v * (1.0 - s * f);
    const double t  = v * (1.0 - s * (1.0 - f));
    double rr, gg, bb;
    switch (hi % 6) {
        case 0: rr=v; gg=t; bb=p; break;
        case 1: rr=q; gg=v; bb=p; break;
        case 2: rr=p; gg=v; bb=t; break;
        case 3: rr=p; gg=q; bb=v; break;
        case 4: rr=t; gg=p; bb=v; break;
        default:rr=v; gg=p; bb=q; break;
    }
    return {(uint8_t)(rr*255+0.5), (uint8_t)(gg*255+0.5), (uint8_t)(bb*255+0.5)};
}

// Deterministic per-cell hash in [0,1) using a sin-based scramble
static double cell_hash(int cell_id, double salt_a, double salt_b) {
    double v = std::sin((double)cell_id * salt_a + salt_b) * 43758.5453;
    return v - std::floor(v);
}

// Interstellar / accretion_warm disk coloring:
// n_rings concentric bands × n_sectors azimuthal slices, each assigned
// a deterministic warm color (red/orange/yellow, some transparent).
// Colors are Gaussian-blended across cell boundaries so there are no hard edges.
// Physical modulation (redshift, Page-Thorne emissivity, magnification) applied on top.
static RGB disk_colour_interstellar(double r, double phi,
                                    double red, double magnif,
                                    double r_in, double r_out,
                                    double M, double r_isco,
                                    const ColorParams& cp) {
    const int    NR    = cp.disk_rings   > 0 ? cp.disk_rings   : 1;
    const int    NS    = cp.disk_sectors > 0 ? cp.disk_sectors : 1;
    const double sigma = cp.disk_sigma   > 0 ? cp.disk_sigma   : 0.5;

    // Normalize position to cell-index space
    const double r_norm   = clamp((r - r_in) / (r_out - r_in), 0.0, 1.0);
    const double phi_norm = (phi - std::floor(phi / (2.0*M_PI)) * (2.0*M_PI)) / (2.0*M_PI); // [0,1)
    const double ring_pos = r_norm * NR;
    const double sec_pos  = phi_norm * NS;

    // Gaussian accumulation over nearby cells
    double acc_r = 0.0, acc_g = 0.0, acc_b = 0.0, w_total = 0.0;
    const double inv2sig2 = 1.0 / (2.0 * sigma * sigma);
    const int    SEARCH   = (int)std::ceil(3.0 * sigma) + 1;

    for (int ri = -SEARCH; ri <= NR + SEARCH; ++ri) {
        const double dr = ring_pos - (double)ri;
        if (dr * dr * inv2sig2 > 16.0) continue;  // early cull
        const int ri_clamped = ri < 0 ? 0 : ri >= NR ? NR-1 : ri;

        for (int si_raw = -SEARCH; si_raw <= NS + SEARCH; ++si_raw) {
            // Sector distance with wrap-around
            double ds = sec_pos - (double)si_raw;
            // Wrap ds to [-NS/2, NS/2]
            ds = ds - std::floor((ds + NS * 0.5) / NS) * NS;
            const double dist2 = dr * dr + ds * ds;
            const double w = std::exp(-dist2 * inv2sig2);
            if (w < 1e-4) continue;

            const int si_clamped = ((si_raw % NS) + NS) % NS;
            const int cell_id = ri_clamped * NS + si_clamped;

            // Three independent hashes per cell
            const double h1 = cell_hash(cell_id, 127.1, 311.7);
            const double h2 = cell_hash(cell_id, 269.5, 183.3);
            const double h3 = cell_hash(cell_id, 419.2, 371.9);

            // ~30% of cells are transparent (gaps in the disk)
            if (h3 < 0.30) continue;

            // Warm accretion palette: hue in [0, 0.14] → deep red through yellow
            const double hue = std::fmod(h1 * 0.14 + cp.disk_hue_offset, 1.0);
            const double sat = 0.70 + h2 * 0.30;
            const double val = 0.50 + h1 * 0.50;

            const RGB c = hsv_to_rgb(hue, sat, val);
            acc_r += w * c.r;
            acc_g += w * c.g;
            acc_b += w * c.b;
            w_total += w;
        }
    }

    if (w_total < 1e-6) return {0, 0, 0};  // transparent region

    // Physical intensity modulation (same as blackbody path)
    const double pt_norm = page_thorne(r, r_isco);
    const double r_peak  = 3.0 * r_isco;
    const double pt_peak = page_thorne(r_peak, r_isco);
    double I = (pt_peak > 0.0) ? pt_norm / pt_peak : 0.0;
    const double red_c = clamp(red, 0.1, 10.0);
    const double receding_lift = 1.0 + 0.85 * clamp(1.0 - red_c, 0.0, 1.0);
    I *= std::pow(red_c, 4.0) * receding_lift;
    I *= clamp(1.0 / magnif, 0.05, 5.0);

    const double norm = w_total * 255.0;
    return {
        (uint8_t)(tonemap(acc_r / norm * I, cp.exposure, cp.gamma) * 255),
        (uint8_t)(tonemap(acc_g / norm * I, cp.exposure, cp.gamma) * 255),
        (uint8_t)(tonemap(acc_b / norm * I, cp.exposure, cp.gamma) * 255)
    };
}

static RGB disk_colour(double r, double red, double magnif,
                       double M, double r_isco, const ColorParams& cp) {
    double T = 6500.0 * cp.temp_scale * std::sqrt(6.0*M/r) * clamp(red, 0.2, 5.0);
    double pt_norm = page_thorne(r, r_isco);
    double r_peak  = 3.0*r_isco;
    double pt_peak = page_thorne(r_peak, r_isco);
    double I = (pt_peak > 0.0) ? pt_norm/pt_peak : 0.0;
    const double red_c = clamp(red, 0.1, 10.0);
    const double receding_lift = 1.0 + 0.85 * clamp(1.0 - red_c, 0.0, 1.0);
    I *= std::pow(red_c, 4.0) * receding_lift;
    I *= clamp(1.0/magnif, 0.05, 5.0);
    auto c = blackbody(T);
    return {(uint8_t)(tonemap(c.r/255.0*I, cp.exposure, cp.gamma)*255),
            (uint8_t)(tonemap(c.g/255.0*I, cp.exposure, cp.gamma)*255),
            (uint8_t)(tonemap(c.b/255.0*I, cp.exposure, cp.gamma)*255)};
}

// Phase 2: GeoPixel buffer → RGB image
static std::vector<RGB> colorize_buffer(
    const std::vector<GeoPixel>& geo, int W, int H,
    const ColorParams& cp,
    const BackgroundImage& bg,
    double M_bh, double r_isco,
    double r_disk_in, double r_disk_out,
    bool debug_elliptic = false,
    const BackgroundImage* bg_b = nullptr)  // Universe B background (wormhole)
{
    std::vector<RGB> image(W*H, {0,0,0});

    if (debug_elliptic) {
        // Color by elliptic solver tag stored in _pad[0]:
        //   NONE(0)=green   INIT_SEPARABLE_CONSTS(1)=white   THETA_CROSSING_SOLVE(2)=cyan
        //   RADIAL_MAP_INIT(3)=red   DIRECT_RADIAL_INVALID(4)=yellow   DIRECT_OUTSIDE_DISK(5)=blue
        static const RGB debug_colors[] = {
            {  0, 220,   0},  // 0 NONE           = elliptic success  → green
            {255, 255, 255},  // 1 INIT_SEP_CONSTS                    → white
            {  0, 220, 220},  // 2 THETA_CROSSING                     → cyan
            {220,   0,   0},  // 3 RADIAL_MAP_INIT                    → red  (dominant)
            {220, 220,   0},  // 4 DIRECT_RADIAL_INVALID              → yellow
            { 60,  60, 220},  // 5 DIRECT_OUTSIDE_DISK                → blue
        };
        for (int i = 0; i < W*H; ++i) {
            const uint8_t tag = geo[i]._pad[0];
            image[i] = (tag < 6) ? debug_colors[tag] : RGB{128, 0, 128};
        }
        return image;
    }

    for (int i = 0; i < W*H; ++i) {
        const GeoPixel& p = geo[i];
        if (p.outcome == 1) {
            if (cp.palette == DiskPalette::INTERSTELLAR)
                image[i] = disk_colour_interstellar(p.r, p.phi_disk,
                                                    p.redshift, p.magnif,
                                                    r_disk_in, r_disk_out,
                                                    M_bh, r_isco, cp);
            else
                image[i] = disk_colour(p.r, p.redshift, p.magnif, M_bh, r_isco, cp);
        } else if (p.outcome == 3) {
            // Universe B — use secondary background if provided, else bg
            const BackgroundImage& bgB = (bg_b && !bg_b->px.empty()) ? *bg_b : bg;
            if (!bgB.px.empty())
                image[i] = bgB.sample(p.theta_esc, p.phi_esc);
        } else if (p.outcome == 0 && !bg.px.empty())
            image[i] = bg.sample(p.theta_esc, p.phi_esc);
    }
    return image;
}

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

// ── Per-frame camera/physics parameters ───────────────────────
struct FrameParams {
    double a=0.998, theta=80.0, phi=0.0, r_obs=40.0, disk_out=12.0, fov=30.0;
    // Wormhole mode (DNEG metric)
    bool   wormhole      = false;
    double wh_rho        = 1.0;   ///< throat areal radius ρ  [geometric units]
    double wh_a_tunnel   = 0.01;  ///< half-length of cylindrical tunnel a  (Interstellar default)
    double wh_M_lens     = 1.0;   ///< lensing parameter M
};

// ── Wormhole (DNEG) single-ray tracer ────────────────────────
// Returns outcome ESCAPED (Universe A, ℓ>0), ESCAPED_B (Universe B, ℓ<0).
// No disk and no horizon for the minimal first pass; disk can be layered later.
static TraceResult trace_single_wormhole(
    WormholeState s, const DnegParams& dp,
    double escape_radius,
    const IntegratorControls& ctl = IntegratorControls{})
{
    double h = ctl.step_init > 1e-10 ? ctl.step_init : 1e-10;
    const int max_steps = ctl.max_steps > 0 ? ctl.max_steps : 500000;
    for (int it = 0; it < max_steps; ++it) {
        const WormholeState s_prev = s;
        rk4_adaptive_wormhole(dp, s, h, ctl.tol);

        if (!std::isfinite(s.l) || !std::isfinite(s.theta))
            return {Outcome::ESCAPED, escape_radius, 1.0, 0.0, M_PI/2.0, 0.0};

        const double abs_prev = s_prev.l >= 0 ? s_prev.l : -s_prev.l;
        const double abs_now  = s.l      >= 0 ? s.l      : -s.l;
        if (abs_prev < escape_radius && abs_now >= escape_radius) {
            // interpolate crossing point
            const double denom = abs_now - abs_prev;
            const double alpha = (denom > 1e-12) ? (escape_radius - abs_prev) / denom : 1.0;
            const double th_esc = s_prev.theta + alpha * (s.theta - s_prev.theta);
            double ph_esc = s_prev.phi + alpha * (s.phi - s_prev.phi);
            // wrap phi to [0, 2π)
            ph_esc = std::fmod(ph_esc, 2.0*M_PI);
            if (ph_esc < 0.0) ph_esc += 2.0*M_PI;
            Outcome out = (s.l >= 0.0) ? Outcome::ESCAPED : Outcome::ESCAPED_B;
            return {out, abs_now, 1.0, 0.0, th_esc, ph_esc};
        }
    }
    // Timeout — treat as escaped in current universe
    double ph_esc = std::fmod(s.phi, 2.0*M_PI);
    if (ph_esc < 0.0) ph_esc += 2.0*M_PI;
    Outcome out = (s.l >= 0.0) ? Outcome::ESCAPED : Outcome::ESCAPED_B;
    return {out, std::abs(s.l), 1.0, 0.0, s.theta, ph_esc};
}

// ── Wormhole Phase 1 tracer (replaces trace_geodesics when fp.wormhole) ──
static std::vector<GeoPixel> trace_geodesics_wormhole(
    int W, int H,
    const FrameParams& fp,
    const IntegratorControls& ctl,
    double pixel_offset_x, double pixel_offset_y,
    KGeoMeta* meta_out)
{
    DnegParams dp{fp.wh_rho, fp.wh_a_tunnel, fp.wh_M_lens};
    WormholeCamera cam(fp.r_obs, fp.theta, fp.phi, fp.fov, W, H);
    const double escape_radius = fp.r_obs * 1.05;

    if (meta_out) {
        // Store wormhole geometry in KGeoMeta — reuse existing fields:
        //   M_bh    = M_lens,   a_bh = wh_rho,   Q_bh = wh_a_tunnel
        //   r_isco  = wh_rho  (throat = inner boundary)
        *meta_out = KGeoMeta{
            (uint32_t)W, (uint32_t)H,
            fp.wh_M_lens, fp.wh_rho, fp.wh_a_tunnel, 0.0,
            fp.wh_rho, fp.wh_rho, fp.disk_out,
            fp.theta, fp.phi, fp.r_obs
        };
    }

    std::vector<GeoPixel> geo(W*H);
    std::atomic<int> rows_done{0};
    const double t0_wh = get_time();
    std::mutex progress_mu;

    auto trace_row = [&](int py) {
        for (int px_ = 0; px_ < W; ++px_) {
            GeoPixel& pix = geo[py*W+px_];
            WormholeState s = cam.pixel_ray(px_, py, dp, pixel_offset_x, pixel_offset_y);
            TraceResult res = trace_single_wormhole(s, dp, escape_radius, ctl);

            pix.outcome   = (res.out == Outcome::ESCAPED_B) ? 3 : 0;
            pix.r         = (float)res.r;
            pix.redshift  = 1.0f;
            pix.magnif    = 1.0f;
            pix.phi_disk  = 0.0f;
            pix.theta_esc = (float)res.theta_esc;
            pix.phi_esc   = (float)res.phi_esc;
            pix._pad[0]   = pix._pad[1] = pix._pad[2] = 0;
        }
        int done = ++rows_done;
        if (done%4==0 || done==H) {
            std::lock_guard<std::mutex> lk(progress_mu);
            print_progress(done, H, get_time()-t0_wh);
        }
    };

#if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic, 4)
    for (int py = 0; py < H; ++py) trace_row(py);
#else
    const unsigned hw = std::thread::hardware_concurrency();
    const unsigned workers = (hw > 1u) ? hw : 1u;
    if (workers == 1 || H < 4) {
        for (int py = 0; py < H; ++py) trace_row(py);
    } else {
        std::atomic<int> next_row{0};
        std::vector<std::thread> pool;
        pool.reserve(workers);
        for (unsigned t = 0; t < workers; ++t) {
            pool.emplace_back([&]() {
                while (true) {
                    const int py = next_row.fetch_add(1);
                    if (py >= H) break;
                    trace_row(py);
                }
            });
        }
        for (auto& th : pool) th.join();
    }
#endif
    std::cerr << "\n";
    return geo;
}

// ── Phase 1: trace all geodesics → GeoPixel buffer ───────────
static std::vector<GeoPixel> trace_geodesics(
    int W, int H,
    const FrameParams& fp,
    bool use_bundles,
    RaySolverMode solver_mode,
    CoordinateChart chart,
    Integrator intg,
    const IntegratorControls& ctl,
    double pixel_offset_x,
    double pixel_offset_y,
    double M_bh, double Q_bh, double Lam,
    KGeoMeta* meta_out = nullptr,
    bool debug_elliptic = false)
{
    // Wormhole: delegate entirely to the DNEG integrator
    if (fp.wormhole)
        return trace_geodesics_wormhole(W, H, fp, ctl, pixel_offset_x, pixel_offset_y, meta_out);

    KNdSMetric g(M_bh, fp.a, Q_bh, Lam);
    CoordinateChart eff_chart = chart;
    if (eff_chart == CoordinateChart::KS && std::abs(Lam) > 1e-15) {
        static bool warned_lam_ks = false;
        if (!warned_lam_ks) {
            std::cerr << "Info: KS chart currently supports Lambda=0 only; falling back to BL.\n";
            warned_lam_ks = true;
        }
        eff_chart = CoordinateChart::BL;
    }
    if (use_bundles && eff_chart == CoordinateChart::KS) {
        static bool warned_bundle_ks = false;
        if (!warned_bundle_ks) {
            std::cerr << "Info: Ray bundles currently run in BL chart; falling back to BL for --bundles.\n";
            warned_bundle_ks = true;
        }
        eff_chart = CoordinateChart::BL;
    }

    const SolverSelection solver_sel = select_solver_mode(
        solver_mode, use_bundles, eff_chart, Q_bh, Lam);
    if (solver_sel.fallback) {
        static bool warned_solver_fallback = false;
        if (!warned_solver_fallback) {
            std::cerr << "Info: requested solver '" << solver_mode_name(solver_mode)
                      << "' not available here (" << solver_sel.reason
                      << "); using standard integrator.\n";
            warned_solver_fallback = true;
        }
    }
    if (solver_sel.effective == RaySolverMode::ELLIPTIC_CLOSED) {
        static bool warned_elliptic_mode = false;
        if (!warned_elliptic_mode) {
            std::cerr << "Info: elliptic-closed enabled (per-ray fallback to chart-native standard path when constraints are not met).\n";
            warned_elliptic_mode = true;
        }
    }
    const bool track_elliptic_fallbacks =
        (!use_bundles && solver_sel.effective == RaySolverMode::ELLIPTIC_CLOSED);
    constexpr size_t elliptic_reason_count =
        static_cast<size_t>(EllipticFallbackReason::COUNT);
    std::array<std::atomic<uint64_t>, elliptic_reason_count> elliptic_fallback_by_reason{};
    for (auto& c : elliptic_fallback_by_reason) c.store(0, std::memory_order_relaxed);
    std::atomic<uint64_t> elliptic_total_rays{0};
    std::atomic<uint64_t> elliptic_fallback_rays{0};

    const double r_isco   = g.r_isco();
    Camera cam(fp.r_obs, fp.theta, fp.phi, fp.fov, W, H);
    const double r_disk_in  = r_isco;
    const double r_disk_out = fp.disk_out;
    const double r_escape   = cam.r_obs * 1.05;

    if (meta_out) {
        *meta_out = KGeoMeta{
            (uint32_t)W, (uint32_t)H,
            M_bh, fp.a, Q_bh, Lam,
            r_isco, r_disk_in, r_disk_out,
            fp.theta, fp.phi, fp.r_obs
        };
    }

    std::vector<GeoPixel> geo(W*H);
    std::atomic<int> rows_done{0};
    const double t0_geo = get_time();
    std::mutex progress_mu;

    auto trace_row = [&](int py) {
        for (int px_=0; px_<W; ++px_) {
            GeoPixel& pix = geo[py*W+px_];
            if (use_bundles) {
                auto res = trace_bundle(px_, py, cam, g, r_disk_in, r_disk_out, r_escape,
                                        ctl.max_steps, ctl.step_init, ctl.tol,
                                        pixel_offset_x, pixel_offset_y);
                pix.outcome   = res.disk_hit ? 1 : 0;
                pix.r         = res.disk_hit ? (float)res.r_hit : 0.0f;
                pix.redshift  = (float)res.redshift;
                pix.magnif    = (float)res.magnif;
                pix.phi_disk  = (float)res.phi_disk;
                pix.theta_esc = (float)res.theta_esc;
                pix.phi_esc   = (float)res.phi_esc;
            } else {
                auto s   = cam.pixel_ray(px_, py, g, pixel_offset_x, pixel_offset_y);
                TraceResult res{};
                if (solver_sel.effective == RaySolverMode::ELLIPTIC_CLOSED) {
                    EllipticFallbackReason fb_reason = EllipticFallbackReason::NONE;
                    res = trace_single_elliptic_closed(
                        s, g, r_disk_in, r_disk_out, r_escape,
                        &fb_reason, eff_chart, intg, ctl);
                    if (debug_elliptic)
                        pix._pad[0] = static_cast<uint8_t>(fb_reason);
                    if (track_elliptic_fallbacks) {
                        elliptic_total_rays.fetch_add(1, std::memory_order_relaxed);
                        if (fb_reason != EllipticFallbackReason::NONE) {
                            elliptic_fallback_rays.fetch_add(1, std::memory_order_relaxed);
                            const size_t idx = static_cast<size_t>(fb_reason);
                            if (idx < elliptic_reason_count)
                                elliptic_fallback_by_reason[idx].fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                } else if (eff_chart == CoordinateChart::KS) {
                    res = trace_single_ks(s, g, r_disk_in, r_disk_out, r_escape, intg, ctl);
                } else if (solver_sel.effective == RaySolverMode::SEMI_ANALYTIC) {
                    res = trace_single_separable_kerr(s, g, r_disk_in, r_disk_out, r_escape, ctl);
                } else {
                    res = trace_single(s, g, r_disk_in, r_disk_out, r_escape, intg, ctl);
                }
                pix.outcome   = (res.out==Outcome::DISK_HIT)  ? 1
                              : (res.out==Outcome::HORIZON)    ? 2
                              : (res.out==Outcome::ESCAPED_B)  ? 3 : 0;
                pix.r         = (float)res.r;
                pix.redshift  = (float)res.redshift;
                pix.magnif    = 1.0f;
                pix.phi_disk  = (float)res.phi_disk;
                pix.theta_esc = (float)res.theta_esc;
                pix.phi_esc   = (float)res.phi_esc;
            }
            if (!debug_elliptic) pix._pad[0]=0;
            pix._pad[1]=pix._pad[2]=0;
        }
        int done = ++rows_done;
        if (done%4==0 || done==H) {
            std::lock_guard<std::mutex> lk(progress_mu);
            print_progress(done, H, get_time()-t0_geo);
        }
    };

#if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic, 4)
    for (int py=0; py<H; ++py) {
        trace_row(py);
    }
#else
    const unsigned hw = std::thread::hardware_concurrency();
    const unsigned workers = std::max(1u, hw);
    if (workers == 1 || H < 4) {
        for (int py = 0; py < H; ++py) trace_row(py);
    } else {
        std::atomic<int> next_row{0};
        std::vector<std::thread> pool;
        pool.reserve(workers);
        for (unsigned t = 0; t < workers; ++t) {
            pool.emplace_back([&]() {
                while (true) {
                    const int py = next_row.fetch_add(1);
                    if (py >= H) break;
                    trace_row(py);
                }
            });
        }
        for (auto& th : pool) th.join();
    }
#endif

    std::cerr << "\n";
    if (track_elliptic_fallbacks) {
        const uint64_t total = elliptic_total_rays.load(std::memory_order_relaxed);
        const uint64_t fb = elliptic_fallback_rays.load(std::memory_order_relaxed);
        if (total > 0) {
            const double pct = 100.0 * double(fb) / double(total);
            std::cerr << "Info: elliptic-closed fallback summary: "
                      << fb << "/" << total << " rays ("
                      << std::fixed << std::setprecision(2) << pct << "%)\n";
            if (fb > 0) {
                for (size_t i = 1; i < elliptic_reason_count; ++i) {
                    const uint64_t cnt = elliptic_fallback_by_reason[i].load(std::memory_order_relaxed);
                    if (cnt == 0) continue;
                    const auto reason = static_cast<EllipticFallbackReason>(i);
                    std::cerr << "  - " << elliptic_fallback_reason_name(reason)
                              << ": " << cnt << "\n";
                }
            }
        }
    }
    return geo;
}

// ── Full render (trace + colorize, for animation / single frame) ──
static std::vector<RGB> render_image(
    int W, int H,
    const FrameParams& fp,
    const BackgroundImage& bg,
    bool use_bundles,
    RaySolverMode solver_mode,
    CoordinateChart chart,
    Integrator intg,
    const IntegratorControls& ctl,
    int camera_spp,
    double M_bh, double Q_bh, double Lam,
    IntersectionMode intersection_mode = IntersectionMode::HERMITE,
    int metal_kernel_mode = 0,
    bool gpu_fp64 = false,
    bool elliptic_fallback_black = false,
    bool anti_fireflies = false,
    const ColorParams& cp = ColorParams{},
    std::vector<GeoPixel>* geo_out = nullptr,
    bool debug_elliptic = false,
    double pixel_offset_x = 0.0,
    double pixel_offset_y = 0.0,
    const BackgroundImage* bg_b = nullptr)  // Universe B background (wormhole)
{
    if (camera_spp > 1) {
        static bool warned_geo_with_multisample = false;
        if (geo_out && !warned_geo_with_multisample) {
            std::cerr << "Info: camera-spp>1 uses jittered multi-pass sampling and disables .kgeo export for this frame.\n";
            warned_geo_with_multisample = true;
        }

        const auto build_offsets = [](int spp) {
            std::vector<std::pair<double, double>> offsets;
            if (spp <= 1) {
                offsets.emplace_back(0.0, 0.0);
                return offsets;
            }
            if (spp == 2) {
                // Python KerrTrace "meridian_supersample"-style horizontal offsets.
                offsets.emplace_back(-0.35, 0.0);
                offsets.emplace_back(+0.35, 0.0);
                return offsets;
            }
            if ((spp % 2) == 1) offsets.emplace_back(0.0, 0.0);
            const std::pair<double, double> pair_seed[] = {
                {0.35, 0.00},
                {0.00, 0.35},
                {0.25, 0.25},
                {0.45, 0.20},
                {0.20, 0.45},
                {0.40, 0.00},
                {0.00, 0.40}
            };
            for (const auto& p : pair_seed) {
                if ((int)offsets.size() + 2 > spp) break;
                offsets.emplace_back(+p.first, +p.second);
                offsets.emplace_back(-p.first, -p.second);
            }
            if ((int)offsets.size() < spp) {
                constexpr double golden = 2.39996322972865332; // golden angle
                int n = 0;
                while ((int)offsets.size() < spp) {
                    const double r = 0.49 * std::sqrt((double)(n + 1) / (double)(spp + 1));
                    const double a = golden * (double)(n + 1);
                    const double x = r * std::cos(a);
                    const double y = r * std::sin(a);
                    offsets.emplace_back(x, y);
                    if ((int)offsets.size() < spp) offsets.emplace_back(-x, -y);
                    ++n;
                }
            }
            offsets.resize((size_t)spp);
            return offsets;
        };

        const auto to_linear = [](double c, double gamma) {
            const double x = clamp(c, 0.0, 1.0);
            return std::pow(x, gamma);
        };
        const auto to_display = [](double l, double gamma) {
            const double x = clamp(l, 0.0, 1.0);
            return std::pow(x, 1.0 / gamma);
        };

        const auto offsets = build_offsets(camera_spp);
        const double gamma_eff = std::max(cp.gamma, 1e-6);
        std::vector<double> accum((size_t)W * (size_t)H * 3ull, 0.0);
        for (size_t sidx = 0; sidx < offsets.size(); ++sidx) {
            const auto& off = offsets[sidx];
            auto pass = render_image(
                W, H, fp, bg, use_bundles, solver_mode, chart, intg,
                ctl, 1, M_bh, Q_bh, Lam,
                intersection_mode, metal_kernel_mode, gpu_fp64,
                elliptic_fallback_black, anti_fireflies, cp,
                nullptr, debug_elliptic,
                pixel_offset_x + off.first,
                pixel_offset_y + off.second,
                bg_b);
            for (int i = 0; i < W * H; ++i) {
                const size_t j = (size_t)i * 3ull;
                accum[j + 0] += to_linear((double)pass[i].r / 255.0, gamma_eff);
                accum[j + 1] += to_linear((double)pass[i].g / 255.0, gamma_eff);
                accum[j + 2] += to_linear((double)pass[i].b / 255.0, gamma_eff);
            }
        }
        const double inv_n = 1.0 / (double)offsets.size();
        std::vector<RGB> out(W * H, {0, 0, 0});
        for (int i = 0; i < W * H; ++i) {
            const size_t j = (size_t)i * 3ull;
            out[i].r = (uint8_t)std::lround(255.0 * to_display(accum[j + 0] * inv_n, gamma_eff));
            out[i].g = (uint8_t)std::lround(255.0 * to_display(accum[j + 1] * inv_n, gamma_eff));
            out[i].b = (uint8_t)std::lround(255.0 * to_display(accum[j + 2] * inv_n, gamma_eff));
        }
        return out;
    }

    (void)metal_kernel_mode;
    (void)intersection_mode;
    (void)gpu_fp64;
    (void)elliptic_fallback_black;
    (void)anti_fireflies;
#if defined(USE_METAL)
    const bool ks_chart_supported = (std::abs(Lam) <= 1e-15);
    const bool gpu_chart_ok = (chart == CoordinateChart::BL) ||
                              (chart == CoordinateChart::KS && ks_chart_supported);
    const SolverSelection gpu_solver_sel = select_solver_mode(
        solver_mode, use_bundles, chart, Q_bh, Lam);
    const RaySolverMode gpu_solver_mode = gpu_solver_sel.effective;
    const bool solver_gpu_supported = metal_solver_supported(gpu_solver_mode, chart, Q_bh, Lam);
    const bool bundle_gpu_supported =
        use_bundles &&
        gpu_chart_ok &&
        gpu_solver_mode == RaySolverMode::STANDARD;
    const bool single_ray_gpu_supported =
        !use_bundles &&
        gpu_chart_ok &&
        solver_gpu_supported;
    const bool base_can_use_gpu = bundle_gpu_supported || single_ray_gpu_supported;
    const bool metal_fp64_supported = false; // Current Metal kernel path is FP32 only.
    const bool can_use_gpu = base_can_use_gpu && (!gpu_fp64 || metal_fp64_supported) && !fp.wormhole;

    if (can_use_gpu) {
        KNdSMetric g(M_bh, fp.a, Q_bh, Lam);
        const double r_isco = g.r_isco();
        Camera cam(fp.r_obs, fp.theta, fp.phi, fp.fov, W, H);
        KNdSParams_C kpc{(float)M_bh,(float)fp.a,(float)Q_bh,(float)Lam,
                         (float)g.r_horizon(),(float)r_isco,(float)fp.disk_out};
        CameraParams_C cpc{(float)cam.r_obs,(float)cam.theta_obs,(float)cam.phi_obs,(float)cam.fov_h,
                           W,H,(chart==CoordinateChart::KS)?1:0,metal_solver_mode_code(gpu_solver_mode),
                           (intg==Integrator::DOPRI5)?1:0,
                           use_bundles ? 1 : 0,
                           metal_kernel_mode_code(static_cast<MetalKernelMode>(metal_kernel_mode)),
                           intersection_mode_code(intersection_mode),
                           elliptic_fallback_black ? 1 : 0,
                           anti_fireflies ? 1 : 0,
                           std::max(1, ctl.max_steps),
                           (float)std::max(ctl.step_init, 1e-6),
                           (float)std::max(ctl.tol, 1e-9),
                           (float)pixel_offset_x,
                           (float)pixel_offset_y};
        const uint8_t* bg_ptr = bg.px.empty() ? nullptr : bg.px.data();
        const int bg_w = bg.px.empty() ? 0 : bg.w;
        const int bg_h = bg.px.empty() ? 0 : bg.h;
        auto px32 = metal_render(kpc, cpc, bg_ptr, bg_w, bg_h);
        std::vector<RGB> image(W*H);
        for (int i=0;i<W*H;++i) {
            image[i].r=(px32[i])    &0xFF;
            image[i].g=(px32[i]>>8) &0xFF;
            image[i].b=(px32[i]>>16)&0xFF;
        }
        return image;
    }

    static bool warned_metal_fallback = false;
    static bool warned_metal_fp64_fallback = false;
    if (!warned_metal_fallback) {
        if (gpu_fp64 && base_can_use_gpu && !metal_fp64_supported) {
            if (!warned_metal_fp64_fallback) {
                std::cerr << "Info: --gpu-fp64 requested, but Metal GPU integration currently runs in FP32;"
                          << " using CPU fallback (double precision).\n";
                warned_metal_fp64_fallback = true;
            }
        } else if (use_bundles && !gpu_chart_ok && chart == CoordinateChart::KS)
            std::cerr << "Info: Metal ray bundles on KS currently support Lambda=0 only; using CPU fallback for --bundles.\n";
        else if (use_bundles && gpu_solver_mode != RaySolverMode::STANDARD)
            std::cerr << "Info: Metal ray bundles currently support standard solver only; using CPU fallback for --bundles.\n";
        else if (use_bundles)
            std::cerr << "Info: Metal ray bundles fallback to CPU for this configuration.\n";
        else if (!solver_gpu_supported)
            std::cerr << "Info: solver mode '" << solver_mode_name(gpu_solver_mode)
                      << "' on Metal is not available for this chart/metric; using CPU fallback.\n";
        else if (chart == CoordinateChart::KS && !ks_chart_supported)
            std::cerr << "Info: KS chart on GPU currently supports Lambda=0 only; using CPU fallback.\n";
        else
            std::cerr << "Info: Metal backend using CPU fallback.\n";
        warned_metal_fallback = true;
    }
    KGeoMeta meta;
    auto geo = trace_geodesics(W, H, fp, use_bundles, solver_mode, chart, intg, ctl,
                               pixel_offset_x, pixel_offset_y,
                               M_bh, Q_bh, Lam, &meta, debug_elliptic);
    if (geo_out) *geo_out = geo;
    return colorize_buffer(geo, W, H, cp, bg, M_bh, meta.r_isco,
                           meta.r_disk_in, meta.r_disk_out, debug_elliptic, bg_b);

#elif defined(USE_CUDA)
    static bool warned_cuda_fp64_default = false;
    static bool warned_cuda_fp64_strict = false;
    if (gpu_fp64) {
        if (!warned_cuda_fp64_strict) {
            std::cerr << "Info: CUDA FP64 strict mode enabled (--gpu-fp64): "
                         "native double kernel + capability check.\n";
            warned_cuda_fp64_strict = true;
        }
    } else if (!warned_cuda_fp64_default) {
        std::cerr << "Info: CUDA backend uses native double precision by default. "
                     "Use --gpu-fp64 for strict capability checking/reporting.\n";
        warned_cuda_fp64_default = true;
    }
    const bool ks_chart_supported = (std::abs(Lam) <= 1e-15);
    const bool gpu_chart_ok = (chart == CoordinateChart::BL) ||
                              (chart == CoordinateChart::KS && ks_chart_supported);
    if (!use_bundles && solver_mode == RaySolverMode::STANDARD && gpu_chart_ok && !fp.wormhole) {
        KNdSMetric g(M_bh, fp.a, Q_bh, Lam);
        const double r_isco = g.r_isco();
        Camera cam(fp.r_obs, fp.theta, fp.phi, fp.fov, W, H);
        KNdSParams_CUDA kpcuda{M_bh,fp.a,Q_bh,Lam,g.r_horizon(),r_isco,fp.disk_out};
        CameraParams_CUDA cpcuda{cam.r_obs,cam.theta_obs,cam.phi_obs,cam.fov_h,
                                 W,H,(chart==CoordinateChart::KS)?1:0};
        auto px32 = cuda_render(kpcuda, cpcuda, gpu_fp64);
        std::vector<RGB> image(W*H);
        for (int i=0;i<W*H;++i) {
            image[i].r=(px32[i])    &0xFF;
            image[i].g=(px32[i]>>8) &0xFF;
            image[i].b=(px32[i]>>16)&0xFF;
        }
        return image;
    }

    static bool warned_cuda_fallback = false;
    if (!warned_cuda_fallback) {
        if (use_bundles)
            std::cerr << "Info: CUDA backend does not support ray bundles yet; using CPU fallback for --bundles.\n";
        else if (solver_mode != RaySolverMode::STANDARD)
            std::cerr << "Info: solver mode '" << solver_mode_name(solver_mode)
                      << "' on CUDA currently falls back to CPU.\n";
        else if (chart == CoordinateChart::KS && !ks_chart_supported)
            std::cerr << "Info: KS chart on GPU currently supports Lambda=0 only; using CPU fallback.\n";
        else
            std::cerr << "Info: CUDA backend using CPU fallback.\n";
        warned_cuda_fallback = true;
    }
    KGeoMeta meta;
    auto geo = trace_geodesics(W, H, fp, use_bundles, solver_mode, chart, intg, ctl,
                               pixel_offset_x, pixel_offset_y,
                               M_bh, Q_bh, Lam, &meta, debug_elliptic);
    if (geo_out) *geo_out = geo;
    return colorize_buffer(geo, W, H, cp, bg, M_bh, meta.r_isco,
                           meta.r_disk_in, meta.r_disk_out, debug_elliptic, bg_b);

#else
    // CPU: two-phase
    KGeoMeta meta;
    auto geo = trace_geodesics(W, H, fp, use_bundles, solver_mode, chart, intg, ctl,
                               pixel_offset_x, pixel_offset_y,
                               M_bh, Q_bh, Lam, &meta, debug_elliptic);
    if (geo_out) *geo_out = geo;
    return colorize_buffer(geo, W, H, cp, bg, M_bh, meta.r_isco,
                           meta.r_disk_in, meta.r_disk_out, debug_elliptic, bg_b);
#endif
}

// ── File I/O ──────────────────────────────────────────────────
static void write_png(const char* path, const std::vector<RGB>& img, int W, int H) {
    stbi_write_png(path, W, H, 3,
                   reinterpret_cast<const unsigned char*>(img.data()), W*3);
}

// ── Interpolation helpers (animation) ────────────────────────
static double smooth_step(double t) { return t*t*(3.0-2.0*t); }
static double lerp_angle(double a, double b, double t) {
    double d=b-a;
    while(d> 180.0) d-=360.0;
    while(d<-180.0) d+=360.0;
    return a+d*t;
}

// ── main ──────────────────────────────────────────────────────
int main(int argc, char** argv) {

    // ── Resolution ───────────────────────────────────────────
    bool use_bundles=false, preview=false, hd_preview=false;
    bool res_720p=false, res_2k=false, res_4k=false;
    bool elliptic_fallback_black=false;
    bool anti_fireflies=false;
    bool debug_elliptic=false;
    int  custom_w=0, custom_h=0;
    Integrator intg=Integrator::RK4_DOUBLING;
    IntegratorControls int_ctl{};
    int camera_spp = 1;
    bool gpu_fp64 = false;
    CoordinateChart chart=CoordinateChart::KS;
    RaySolverMode solver_mode=RaySolverMode::STANDARD;
    IntersectionMode intersection_mode=IntersectionMode::HERMITE;
    int metal_kernel_mode=0; // 0=auto, 1=unified, 2=single, 3=bundle (Metal only)
    std::string bg_path = "assets/backgrounds/sfondo5.jpg";

    // ── Single-frame / base params ───────────────────────────
    double arg_a=0.998, arg_disk_out=12.0, arg_theta=80.0, arg_phi=0.0, arg_r_obs=-1.0;
    double arg_Q=0.0, arg_Lam=0.0, arg_fov=30.0;

    // ── Wormhole params ───────────────────────────────────────
    bool   arg_wormhole  = false;
    double arg_wh_rho    = 1.0;   // throat areal radius ρ
    double arg_wh_a      = 0.01;  // half tunnel length a
    double arg_wh_M      = 1.0;   // lensing parameter M
    std::string bg_b_path;        // Universe B background path (--bg-b)

    // ── Colorization params ───────────────────────────────────
    ColorParams cp;  // defaults: exposure=1, gamma=2.2, temp_scale=1

    // ── Two-phase modes ───────────────────────────────────────
    bool        geo_only    = false;
    std::string geo_file;         // path for .kgeo output (geo_only) or input (color_only)
    bool        color_only  = false;

    // ── Animation params ──────────────────────────────────────
    bool anim_mode=false, anim_ease=false, anim_keep_frames=false;
    bool anim_resume=false, anim_no_encode=false;
    int  anim_frames=60, anim_fps=30, anim_crf=18;
    double anim_orbits=0.0;
    std::string anim_output, anim_frames_dir;

    double NaN = std::numeric_limits<double>::quiet_NaN();
    double anim_theta_start=NaN, anim_theta_end=NaN;
    double anim_phi_start=NaN,   anim_phi_end=NaN;
    double anim_r_start=NaN,     anim_r_end=NaN;
    double anim_a_start=NaN,     anim_a_end=NaN;
    double anim_disk_out_start=NaN, anim_disk_out_end=NaN;

    for (int i=1;i<argc;++i) {
        std::string arg(argv[i]);

        // Resolution
        if (arg=="--bundles")  use_bundles=true;
        if (arg=="--dopri5")   intg=Integrator::DOPRI5;
        if (arg=="--max-steps" && i+1<argc) int_ctl.max_steps = std::max(1, std::stoi(argv[++i]));
        if ((arg=="--step-init" || arg=="--step-size") && i+1<argc) int_ctl.step_init = std::stod(argv[++i]);
        if ((arg=="--integrator-tol" || arg=="--tol") && i+1<argc) int_ctl.tol = std::stod(argv[++i]);
        if ((arg=="--camera-spp" || arg=="--spp") && i+1<argc) camera_spp = std::max(1, std::stoi(argv[++i]));
        if (arg=="--semi-analytic" || arg=="--elliptic") solver_mode=RaySolverMode::SEMI_ANALYTIC;
        if (arg=="--elliptic-closed") solver_mode=RaySolverMode::ELLIPTIC_CLOSED;
        if (arg=="--elliptic-fallback-black" || arg=="--elliptic-strict-black")
            elliptic_fallback_black = true;
        if (arg=="--debug-elliptic")
            debug_elliptic = true;
        if (arg=="--anti-fireflies" || arg=="--anti_fireflies")
            anti_fireflies = true;
        if (arg=="--no-anti-fireflies" || arg=="--no-anti_fireflies")
            anti_fireflies = false;
        if (arg=="--intersection-linear" || arg=="--linear-intersection")
            intersection_mode=IntersectionMode::LINEAR;
        if (arg=="--intersection-hermite" || arg=="--hermite-intersection")
            intersection_mode=IntersectionMode::HERMITE;
        if (arg=="--intersection" && i+1<argc) {
            const std::string m = argv[++i];
            if (m=="linear" || m=="lin") intersection_mode=IntersectionMode::LINEAR;
            else if (m=="hermite" || m=="cubic" || m=="event-hermite")
                intersection_mode=IntersectionMode::HERMITE;
            else
                std::cerr << "Warning: unknown --intersection value '" << m
                          << "', using hermite\n";
        }
        if (arg=="--solver-mode" && i+1<argc) {
            const std::string s = argv[++i];
            if (s=="standard") solver_mode=RaySolverMode::STANDARD;
            else if (s=="semi" || s=="semi-analytic" || s=="semi_analytic")
                solver_mode=RaySolverMode::SEMI_ANALYTIC;
            else if (s=="elliptic" || s=="elliptic-closed" || s=="elliptic_closed")
                solver_mode=RaySolverMode::ELLIPTIC_CLOSED;
        }
        if (arg=="--metal-kernel" && i+1<argc) {
            const std::string k = argv[++i];
            if (k=="auto") metal_kernel_mode = 0;
            else if (k=="unified" || k=="legacy") metal_kernel_mode = 1;
            else if (k=="single" || k=="single-ray" || k=="single_ray") metal_kernel_mode = 2;
            else if (k=="bundle" || k=="bundles" || k=="ray-bundle" || k=="ray_bundle")
                metal_kernel_mode = 3;
            else {
                std::cerr << "Warning: unknown --metal-kernel value '" << k
                          << "', using auto\n";
                metal_kernel_mode = 0;
            }
        }
        if (arg=="--gpu-fp64" || arg=="--gpu-fp64-on" || arg=="--metal-fp64" || arg=="--cuda-fp64")
            gpu_fp64 = true;
        if (arg=="--no-gpu-fp64" || arg=="--no-cuda-fp64")
            gpu_fp64 = false;
        if (arg=="--bl")       chart=CoordinateChart::BL;
        if (arg=="--ks")       chart=CoordinateChart::KS;
        if (arg=="--chart" && i+1<argc) {
            const std::string c = argv[++i];
            if (c=="bl" || c=="BL") chart=CoordinateChart::BL;
            else if (c=="ks" || c=="KS") chart=CoordinateChart::KS;
        }
        if (arg=="--preview")  preview=true;
        if (arg=="--hd")       hd_preview=true;
        if (arg=="--720p")     res_720p=true;
        if (arg=="--2k")       res_2k=true;
        if (arg=="--4k")       res_4k=true;
        if (arg=="--custom-res" && i+2<argc) {
            custom_w=std::stoi(argv[++i]); custom_h=std::stoi(argv[++i]);
        }

        // Single-frame physics
        if (arg=="--bg"       && i+1<argc) bg_path     = argv[++i];
        if (arg=="--a"        && i+1<argc) arg_a        = std::stod(argv[++i]);
        if (arg=="--disk-out" && i+1<argc) arg_disk_out = std::stod(argv[++i]);
        if (arg=="--theta"    && i+1<argc) arg_theta    = std::stod(argv[++i]);
        if (arg=="--phi"      && i+1<argc) arg_phi      = std::stod(argv[++i]);
        if (arg=="--r-obs"    && i+1<argc) arg_r_obs    = std::stod(argv[++i]);
        if (arg=="--charge"   && i+1<argc) arg_Q        = std::stod(argv[++i]);
        if (arg=="--lambda"   && i+1<argc) arg_Lam      = std::stod(argv[++i]);
        if (arg=="--fov"      && i+1<argc) arg_fov      = std::stod(argv[++i]);

        // Colorization
        if (arg=="--exposure"   && i+1<argc) cp.exposure   = std::stod(argv[++i]);
        if (arg=="--gamma"      && i+1<argc) cp.gamma      = std::stod(argv[++i]);
        if (arg=="--temp-scale" && i+1<argc) cp.temp_scale = std::stod(argv[++i]);
        if (arg=="--disk-interstellar") cp.palette = DiskPalette::INTERSTELLAR;
        if (arg=="--disk-blackbody")    cp.palette = DiskPalette::BLACKBODY;
        if (arg=="--disk-rings"      && i+1<argc) cp.disk_rings      = std::stoi(argv[++i]);
        if (arg=="--disk-sectors"    && i+1<argc) cp.disk_sectors    = std::stoi(argv[++i]);
        if (arg=="--disk-sigma"      && i+1<argc) cp.disk_sigma      = std::stod(argv[++i]);
        if (arg=="--disk-hue-offset" && i+1<argc) cp.disk_hue_offset = std::stod(argv[++i]);

        // Wormhole
        if (arg=="--wormhole")                   arg_wormhole = true;
        if (arg=="--wh-throat" && i+1<argc)      arg_wh_rho   = std::stod(argv[++i]);
        if (arg=="--wh-lensing"&& i+1<argc)      arg_wh_M     = std::stod(argv[++i]);
        if (arg=="--wh-tunnel" && i+1<argc)      arg_wh_a     = std::stod(argv[++i]);
        if (arg=="--bg-b"      && i+1<argc)      bg_b_path    = argv[++i];

        // Two-phase modes
        if (arg=="--geo-only")             geo_only  = true;
        if (arg=="--geo-file" && i+1<argc) geo_file  = argv[++i];
        if (arg=="--color-only"&& i+1<argc){ color_only=true; geo_file=argv[++i]; }

        // Animation
        if (arg=="--anim")                   anim_mode=true;
        if (arg=="--frames"   && i+1<argc)   anim_frames=std::stoi(argv[++i]);
        if (arg=="--fps"      && i+1<argc)   anim_fps=std::stoi(argv[++i]);
        if (arg=="--crf"      && i+1<argc)   anim_crf=std::stoi(argv[++i]);
        if (arg=="--ease")                   anim_ease=true;
        if (arg=="--keep-frames")            anim_keep_frames=true;
        if (arg=="--resume")                 anim_resume=true;
        if (arg=="--no-encode")              anim_no_encode=true;
        if (arg=="--orbits"   && i+1<argc)   anim_orbits=std::stod(argv[++i]);
        if (arg=="--output"   && i+1<argc)   anim_output=argv[++i];
        if (arg=="--frames-dir"&&i+1<argc)   anim_frames_dir=argv[++i];
        if (arg=="--theta-start"    &&i+1<argc) anim_theta_start    =std::stod(argv[++i]);
        if (arg=="--theta-end"      &&i+1<argc) anim_theta_end      =std::stod(argv[++i]);
        if (arg=="--phi-start"      &&i+1<argc) anim_phi_start      =std::stod(argv[++i]);
        if (arg=="--phi-end"        &&i+1<argc) anim_phi_end        =std::stod(argv[++i]);
        if (arg=="--r-start"        &&i+1<argc) anim_r_start        =std::stod(argv[++i]);
        if (arg=="--r-end"          &&i+1<argc) anim_r_end          =std::stod(argv[++i]);
        if (arg=="--a-start"        &&i+1<argc) anim_a_start        =std::stod(argv[++i]);
        if (arg=="--a-end"          &&i+1<argc) anim_a_end          =std::stod(argv[++i]);
        if (arg=="--disk-out-start" &&i+1<argc) anim_disk_out_start =std::stod(argv[++i]);
        if (arg=="--disk-out-end"   &&i+1<argc) anim_disk_out_end   =std::stod(argv[++i]);
    }

    // ── Background ───────────────────────────────────────────
    int_ctl.max_steps = std::max(1, int_ctl.max_steps);
    int_ctl.step_init = std::max(1e-10, int_ctl.step_init);
    int_ctl.tol = std::max(1e-10, int_ctl.tol);
    camera_spp = std::max(1, camera_spp);

    BackgroundImage bg;
    if (!bg_path.empty()) {
        if (bg.load(bg_path.c_str()))
            std::cout << "Background: " << bg_path
                      << " (" << bg.w << "x" << bg.h << ")\n";
        else
            std::cerr << "Warning: cannot load background '" << bg_path << "'\n";
    }

    BackgroundImage bg_b_img;
    if (!bg_b_path.empty()) {
        if (bg_b_img.load(bg_b_path.c_str()))
            std::cout << "Background B (Universe B): " << bg_b_path
                      << " (" << bg_b_img.w << "x" << bg_b_img.h << ")\n";
        else
            std::cerr << "Warning: cannot load --bg-b background '" << bg_b_path << "'\n";
    }
    const BackgroundImage* bg_b = bg_b_img.px.empty() ? nullptr : &bg_b_img;

    // ── Derived constants ────────────────────────────────────
    const double M_bh=1.0;
    const double Q_bh=arg_Q, Lam=arg_Lam;

    const int W = custom_w ? custom_w : res_4k ? 3840 : res_2k ? 2560
                : res_720p ? 1280 : hd_preview ? 854 : preview ? 480 : 1920;
    const int H = custom_h ? custom_h : res_4k ? 2160 : res_2k ? 1440
                : res_720p ? 720  : hd_preview ? 480 : preview ? 270 : 1080;

    const double default_r_obs = 40.0;
    const double base_r_obs    = (arg_r_obs>0) ? arg_r_obs : default_r_obs;

    const char* res_tag = res_4k ? "4k" : res_2k ? "2k" : custom_w ? "custom"
                        : res_720p ? "720p" : hd_preview ? "hd" : preview ? "preview"
                        : use_bundles ? "bundles" : "trace";

    // ── COLOR-ONLY mode ──────────────────────────────────────
    if (color_only) {
        if (geo_file.empty()) {
            std::cerr << "Error: --color-only requires a .kgeo file path\n";
            return 1;
        }
        std::vector<GeoPixel> geo;
        KGeoMeta meta;
        if (!load_kgeo(geo_file.c_str(), geo, meta)) {
            std::cerr << "Error: cannot load " << geo_file << "\n"; return 1;
        }
        int gW=(int)meta.W, gH=(int)meta.H;
        std::cout << "Loaded: " << geo_file << "  " << gW << "x" << gH
                  << "  a=" << meta.a_bh << "  r_isco=" << meta.r_isco << "\n"
                  << "ColorParams: exposure=" << cp.exposure
                  << " gamma=" << cp.gamma
                  << " temp_scale=" << cp.temp_scale << "\n";

        auto image = colorize_buffer(geo, gW, gH, cp, bg, meta.M_bh, meta.r_isco,
                                     meta.r_disk_in, meta.r_disk_out);

        std::time_t now=std::time(nullptr); char ts[32];
        std::strftime(ts,sizeof(ts),"%Y%m%d-%H%M%S",std::localtime(&now));
        std::string base = geo_file;
        // Remove .kgeo extension for the output name
        if (base.size()>5 && base.substr(base.size()-5)==".kgeo")
            base=base.substr(0,base.size()-5);
        std::string outfile = base + "_recolor_" + ts + ".png";
        write_png(outfile.c_str(), image, gW, gH);
        std::cout << "Saved: " << outfile << "\n";
        return 0;
    }

    // ── Helper: build timestamp ───────────────────────────────
    auto make_ts = []()->std::string {
        std::time_t now=std::time(nullptr); char ts[32];
        std::strftime(ts,sizeof(ts),"%Y%m%d-%H%M%S",std::localtime(&now));
        return ts;
    };

    auto effective_chart_for_naming = [&](CoordinateChart requested)->CoordinateChart {
        CoordinateChart eff = requested;
        if (eff == CoordinateChart::KS && std::abs(Lam) > 1e-15)
            eff = CoordinateChart::BL;
        return eff;
    };
    auto chart_tag = [](CoordinateChart c)->const char* {
        return (c == CoordinateChart::KS) ? "ks" : "bl";
    };
    auto integration_mode_tag = [&]()->std::string {
        const CoordinateChart eff_chart = effective_chart_for_naming(chart);
        const SolverSelection sel = select_solver_mode(
            solver_mode, use_bundles, eff_chart, Q_bh, Lam);

        if (use_bundles)
            return std::string(chart_tag(eff_chart)) + "-ray-bundle";

        if (sel.effective == RaySolverMode::SEMI_ANALYTIC)
            return std::string(chart_tag(eff_chart)) + "-semi-analytic";
        if (sel.effective == RaySolverMode::ELLIPTIC_CLOSED) {
            std::string tag = std::string(chart_tag(eff_chart)) + "-elliptic-closed";
            if (elliptic_fallback_black) tag += "-fallback-black";
            return tag;
        }

        // Standard solver: CPU KS is RK4-only; GPU KS can expose DOPRI5.
        bool ks_uses_rk4 = (eff_chart == CoordinateChart::KS);
#if defined(USE_METAL)
        const bool ks_chart_supported = (std::abs(Lam) <= 1e-15);
        const bool gpu_chart_ok = (eff_chart == CoordinateChart::BL) ||
                                  (eff_chart == CoordinateChart::KS && ks_chart_supported);
        const bool solver_gpu_supported = metal_solver_supported(sel.effective, eff_chart, Q_bh, Lam);
        const bool bundle_gpu_supported =
            use_bundles &&
            gpu_chart_ok &&
            sel.effective == RaySolverMode::STANDARD;
        const bool single_ray_gpu_supported =
            !use_bundles &&
            gpu_chart_ok &&
            solver_gpu_supported;
        const bool can_use_gpu = bundle_gpu_supported || single_ray_gpu_supported;
        if (can_use_gpu) ks_uses_rk4 = false;
#endif
        const char* intg_tag = (ks_uses_rk4 || intg == Integrator::RK4_DOUBLING)
            ? "rk4"
            : "dopri5";
        return std::string(chart_tag(eff_chart)) + "-standard-" + intg_tag;
    };
    auto backend_mode_tag = [&]()->const char* {
#if defined(USE_METAL)
        const bool ks_chart_supported = (std::abs(Lam) <= 1e-15);
        const bool gpu_chart_ok = (chart == CoordinateChart::BL) ||
                                  (chart == CoordinateChart::KS && ks_chart_supported);
        const SolverSelection gpu_solver_sel = select_solver_mode(
            solver_mode, use_bundles, chart, Q_bh, Lam);
        const RaySolverMode gpu_solver_mode = gpu_solver_sel.effective;
        const bool solver_gpu_supported = metal_solver_supported(gpu_solver_mode, chart, Q_bh, Lam);
        const bool bundle_gpu_supported =
            use_bundles &&
            gpu_chart_ok &&
            gpu_solver_mode == RaySolverMode::STANDARD;
        const bool single_ray_gpu_supported =
            !use_bundles &&
            gpu_chart_ok &&
            solver_gpu_supported;
        const bool base_can_use_gpu = bundle_gpu_supported || single_ray_gpu_supported;
        const bool metal_fp64_supported = false; // Current Metal kernel path is FP32 only.
        const bool can_use_gpu = base_can_use_gpu && (!gpu_fp64 || metal_fp64_supported);
        return can_use_gpu ? "gpu-metal" : "cpu";
#elif defined(USE_CUDA)
        const bool ks_chart_supported = (std::abs(Lam) <= 1e-15);
        const bool gpu_chart_ok = (chart == CoordinateChart::BL) ||
                                  (chart == CoordinateChart::KS && ks_chart_supported);
        const bool cuda_gpu_supported =
            !use_bundles &&
            solver_mode == RaySolverMode::STANDARD &&
            gpu_chart_ok;
        return cuda_gpu_supported ? "gpu-cuda" : "cpu";
#else
        return "cpu";
#endif
    };
    const std::string mode_tag = integration_mode_tag();
    const std::string backend_tag = backend_mode_tag();
    const std::string fp64_tag = gpu_fp64 ? "_gpufp64req" : "";
    const std::string spp_tag = (camera_spp > 1) ? ("_spp" + std::to_string(camera_spp)) : "";
    const std::string mode_backend_tag = mode_tag + "_" + backend_tag + fp64_tag + spp_tag;

    // ── SINGLE FRAME mode ────────────────────────────────────
    if (!anim_mode) {
        FrameParams fp;
        fp.a=arg_a; fp.theta=arg_theta; fp.phi=arg_phi;
        fp.r_obs=base_r_obs*M_bh; fp.disk_out=arg_disk_out*M_bh;
        fp.fov=arg_fov;
        fp.wormhole    = arg_wormhole;
        fp.wh_rho      = arg_wh_rho;
        fp.wh_a_tunnel = arg_wh_a;
        fp.wh_M_lens   = arg_wh_M;

        if (fp.wormhole) {
            std::cout << "DNEG Wormhole  ρ=" << fp.wh_rho
                      << " a=" << fp.wh_a_tunnel
                      << " M_lens=" << fp.wh_M_lens << "\n"
                      << "  throat_r=" << fp.wh_rho
                      << "  l_obs=" << fp.r_obs << "\n";
        } else {
            KNdSMetric g_info(M_bh,fp.a,Q_bh,Lam);
            std::cout << "KNdS  M=" << M_bh << " a=" << fp.a
                      << " Q=" << Q_bh << " Λ=" << Lam << "\n"
                      << "  r₊=" << g_info.r_horizon()
                      << "  r_ISCO=" << g_info.r_isco() << "\n";
        }
        std::cout << "Mode: " << (fp.wormhole?"wormhole":use_bundles?"ray bundles":"single ray")
                  << "  chart=" << (fp.wormhole?"wormhole":chart==CoordinateChart::KS?"KS":"BL")
                  << "  " << (intg==Integrator::DOPRI5?"DOPRI5":"RK4-doubling")
                  << "  " << (fp.wormhole?"rk4-adaptive":solver_mode_name(solver_mode))
                  << "  backend=" << (fp.wormhole?"cpu":backend_tag.c_str())
                  << "  intersection=" << intersection_mode_name(intersection_mode)
                  << "  elliptic-fallback-black=" << (elliptic_fallback_black ? "on" : "off")
                  << "  anti-fireflies=" << (anti_fireflies ? "on" : "off")
                  << "  max-steps=" << int_ctl.max_steps
                  << "  step-init=" << int_ctl.step_init
                  << "  tol=" << int_ctl.tol
                  << "  camera-spp=" << camera_spp
                  << "  gpu-fp64=" << (gpu_fp64 ? "on" : "off")
#if defined(USE_METAL)
                  << "  metal-kernel="
                  << metal_kernel_mode_name(static_cast<MetalKernelMode>(metal_kernel_mode))
#endif
                  << "  " << W << "x" << H << "\n"
                  << "ColorParams: exp=" << cp.exposure
                  << " γ=" << cp.gamma
                  << " T×=" << cp.temp_scale << "\n";

        double t0=get_time();

        if (geo_only) {
            // Phase 1 only: trace and save .kgeo
            KGeoMeta meta;
            auto geo = trace_geodesics(W, H, fp, use_bundles, solver_mode, chart, intg,
                                       int_ctl, 0.0, 0.0, M_bh, Q_bh, Lam, &meta);
            double elapsed=get_time()-t0;
            std::cout << "Trace: " << elapsed << "s  ("
                      << std::fixed << std::setprecision(1)
                      << W*H/elapsed/1e3 << " kpix/s)\n";

            std::string kgeo_path = geo_file.empty()
                ? std::string(OUT_DIR)+"/"+res_tag
                  +"_"+std::to_string(W)+"x"+std::to_string(H)
                  +"_"+mode_backend_tag
                  +"_"+make_ts()+".kgeo"
                : geo_file;
            save_kgeo(kgeo_path.c_str(), geo, meta);
            std::cout << "Geo saved: " << kgeo_path << "\n";
        } else {
            // Full render (Phase 1 + Phase 2, optionally save .kgeo)
            std::vector<GeoPixel> geo;
            auto image = render_image(W, H, fp, bg, use_bundles, solver_mode, chart, intg,
                                      int_ctl, camera_spp, M_bh, Q_bh, Lam,
                                      intersection_mode, metal_kernel_mode, gpu_fp64,
                                      elliptic_fallback_black, anti_fireflies, cp,
                                      geo_file.empty() ? nullptr : &geo, debug_elliptic,
                                      0.0, 0.0, bg_b);
            double elapsed=get_time()-t0;
            std::cout << "Time: " << elapsed << "s  ("
                      << std::fixed << std::setprecision(1)
                      << W*H/elapsed/1e3 << " kpix/s)\n";

            std::string ts_str = make_ts();
            std::string outfile = std::string(OUT_DIR)+"/"+res_tag
                                +"_"+std::to_string(W)+"x"+std::to_string(H)
                                +"_"+mode_backend_tag
                                +"_"+ts_str+".png";
            write_png(outfile.c_str(), image, W, H);
            std::cout << "Saved: " << outfile << "\n";

            // Optionally save geo file alongside PNG
            if (!geo_file.empty() && !geo.empty()) {
                KNdSMetric g_tmp(M_bh,fp.a,Q_bh,Lam);
                KGeoMeta meta{(uint32_t)W,(uint32_t)H,
                              M_bh,fp.a,Q_bh,Lam,
                              g_tmp.r_isco(),g_tmp.r_isco(),fp.disk_out,
                              fp.theta,fp.phi,fp.r_obs};
                save_kgeo(geo_file.c_str(), geo, meta);
                std::cout << "Geo saved: " << geo_file << "\n";
            }
        }
        return 0;
    }

    // ── ANIMATION mode ───────────────────────────────────────
    auto resolve=[](double v, double fb){ return std::isnan(v)?fb:v; };

    double theta_start    = resolve(anim_theta_start,    arg_theta);
    double theta_end      = resolve(anim_theta_end,      arg_theta);
    double phi_start      = resolve(anim_phi_start,      arg_phi);
    double phi_end        = resolve(anim_phi_end,        arg_phi);
    double r_start        = resolve(anim_r_start,        base_r_obs);
    double r_end          = resolve(anim_r_end,          base_r_obs);
    double a_start        = resolve(anim_a_start,        arg_a);
    double a_end          = resolve(anim_a_end,          arg_a);
    double disk_out_start = resolve(anim_disk_out_start, arg_disk_out);
    double disk_out_end   = resolve(anim_disk_out_end,   arg_disk_out);

    std::ostringstream tag_ss;
    tag_ss << res_tag;
    if (anim_orbits!=0.0)            tag_ss<<"_orbit"<<anim_orbits<<"x";
    else if (phi_start!=phi_end)     tag_ss<<"_phi"<<(int)phi_start<<"-"<<(int)phi_end;
    if (theta_start!=theta_end)      tag_ss<<"_th"<<(int)theta_start<<"-"<<(int)theta_end;
    if (r_start!=r_end)              tag_ss<<"_r"<<(int)r_start<<"-"<<(int)r_end;
    if (a_start!=a_end)              tag_ss<<"_a"<<a_start<<"-"<<a_end;
    tag_ss<<"_"<<mode_backend_tag;
    tag_ss<<"_"<<anim_frames<<"f"<<anim_fps<<"fps";
    std::string anim_tag=tag_ss.str();

    std::string frames_dir = anim_frames_dir.empty()
        ? std::string(OUT_DIR)+"/anim_"+anim_tag : anim_frames_dir;
    std::system(("mkdir -p \""+frames_dir+"\"").c_str());

    std::string output_file = anim_output.empty()
        ? std::string(OUT_DIR)+"/"+anim_tag+".mp4" : anim_output;

    std::cout<<"Animation: "<<anim_frames<<" frames @ "<<anim_fps<<" fps\n"
             <<"  theta:  "<<theta_start<<"° → "<<theta_end<<"°\n";
    if (anim_orbits!=0.0)
        std::cout<<"  phi:    "<<phi_start<<"° + "<<anim_orbits<<" orbit(s)\n";
    else
        std::cout<<"  phi:    "<<phi_start<<"° → "<<phi_end<<"°\n";
    std::cout<<"  r_obs:  "<<r_start<<" M → "<<r_end<<" M\n"
             <<"  a:      "<<a_start<<" → "<<a_end<<"\n"
             <<"  exp="<<cp.exposure<<" γ="<<cp.gamma<<" T×="<<cp.temp_scale<<"\n"
             <<"  output: "<<output_file<<"\n\n";

    double t_total=get_time();
    int rendered=0, skipped=0;

    for (int frame=0; frame<anim_frames; ++frame) {
        char fname[512];
        std::snprintf(fname,sizeof(fname),"%s/frame_%05d.png",frames_dir.c_str(),frame);
        if (anim_resume) {
            std::ifstream test(fname);
            if (test.good()) { ++skipped; continue; }
        }

        double phase=(anim_frames==1)?0.0:(double)frame/(anim_frames-1);
        double t=anim_ease?smooth_step(phase):phase;

        FrameParams fp;
        fp.a        = a_start       +(a_end       -a_start)       *t;
        fp.theta    = theta_start   +(theta_end   -theta_start)   *t;
        fp.r_obs    = (r_start      +(r_end       -r_start)       *t)*M_bh;
        fp.disk_out = (disk_out_start+(disk_out_end-disk_out_start)*t)*M_bh;
        fp.phi      = (anim_orbits!=0.0)
                    ? phi_start+360.0*anim_orbits*phase
                    : lerp_angle(phi_start,phi_end,t);
        fp.wormhole    = arg_wormhole;
        fp.wh_rho      = arg_wh_rho;
        fp.wh_a_tunnel = arg_wh_a;
        fp.wh_M_lens   = arg_wh_M;

        double t_frame=get_time();
        std::cout<<"Frame "<<(frame+1)<<"/"<<anim_frames
                 <<"  θ="<<std::fixed<<std::setprecision(1)<<fp.theta
                 <<"°  φ="<<fp.phi
                 <<"°  r="<<std::setprecision(2)<<fp.r_obs
                 <<"  a="<<std::setprecision(4)<<fp.a<<" ...\n"<<std::flush;

        auto image=render_image(W,H,fp,bg,use_bundles,solver_mode,chart,intg,
                                int_ctl, camera_spp, M_bh,Q_bh,Lam,
                                intersection_mode,metal_kernel_mode,gpu_fp64,
                                elliptic_fallback_black,anti_fireflies,cp,nullptr,debug_elliptic,
                                0.0, 0.0, bg_b);
        write_png(fname,image,W,H);

        double dt=get_time()-t_frame;
        ++rendered;
        double eta=(rendered>0)?(get_time()-t_total)/rendered*(anim_frames-frame-1):0.0;
        std::cout<<"  → "<<fname<<"  ("<<std::setprecision(1)<<dt<<"s"
                 <<"  ETA "<<(int)(eta/60)<<"m"<<(int)std::fmod(eta,60)<<"s)\n";
    }

    std::cout<<"\nFrames: "<<rendered<<" rendered";
    if (skipped) std::cout<<", "<<skipped<<" skipped";
    std::cout<<"  ("<<(get_time()-t_total)<<"s)\n";

    if (!anim_no_encode) {
        char ffcmd[2048];
        std::snprintf(ffcmd,sizeof(ffcmd),
            "ffmpeg -y -framerate %d -i \"%s/frame_%%05d.png\""
            " -c:v libx264 -pix_fmt yuv420p -crf %d -movflags +faststart"
            " \"%s\" 2>&1",
            anim_fps,frames_dir.c_str(),anim_crf,output_file.c_str());
        std::cout<<"Encoding...\n";
        int ret=std::system(ffcmd);
        if (ret==0) std::cout<<"Video saved: "<<output_file<<"\n";
        else        std::cerr<<"ffmpeg failed. Frames in: "<<frames_dir<<"\n";
    }
    if (!anim_keep_frames && !anim_no_encode)
        std::system(("rm -rf \""+frames_dir+"\"").c_str());

    return 0;
}
