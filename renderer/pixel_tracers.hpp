#pragma once
// Per-pixel tracers and the row loop that drives them.
//
// Part of the single-translation-unit renderer included by main.cpp;
// see docs/ARCHITETTURA.md for how the modules fit together.
//
// Each way of turning a pixel into a GeoPixel is its own small type with
//
//     void operator()(int px, int py, GeoPixel& pix) const;
//
// trace_geodesics builds exactly one of them per frame and stores it in a
// PixelTracer variant. trace_frame's std::visit then instantiates trace_rows for that
// concrete type, so the hot loop is monomorphic: the choice of tracer costs
// one dispatch per frame, not a chain of branches (or a virtual call) per pixel.
// Adding a tracer means adding a type to the variant; the loop, threading and
// progress output stay shared.

#include "tracer_bl.hpp"
#include "tracer_separable.hpp"
#include "tracer_elliptic.hpp"
#include "tracer_ks.hpp"
#include "tracer_wormhole.hpp"
#include "frame.hpp"
#include "parallel.hpp"

#include <variant>

// Everything a Kerr-Newman-de Sitter tracer needs that is fixed for the frame.
struct KndsFrameSetup {
    const Camera& cam;
    const KNdSMetric& g;
    double r_disk_in, r_disk_out, r_escape;
    const IntegratorControls& ctl;
    double pixel_offset_x, pixel_offset_y;
    const ColorParams* cp_ptr;
    bool debug_elliptic;
};

// The last two bytes of padding are bookkeeping, never geometry: _pad[0] keeps
// the elliptic debug tag only when it was asked for, _pad[2] is unused.
static inline void finish_knds_pixel(GeoPixel& pix, bool debug_elliptic) {
    if (!debug_elliptic) pix._pad[0] = 0;
    pix._pad[2] = 0;
}

static inline void store_single_ray(const TraceResult& res, GeoPixel& pix) {
    pix.outcome   = (res.out==Outcome::DISK_HIT)  ? 1
                  : (res.out==Outcome::HORIZON)    ? 2
                  : (res.out==Outcome::ESCAPED_B)  ? 3 : 0;
    pix.r         = (float)res.r;
    pix.redshift  = (float)res.redshift;
    pix.magnif    = 1.0f;
    pix.phi_disk  = (float)res.phi_disk;
    pix.fp_dr_a = pix.fp_dphi_a = pix.fp_dr_b = pix.fp_dphi_b = 0.0f;
    pix.sky_dth_a = pix.sky_dph_a = pix.sky_dth_b = pix.sky_dph_b = 0.0f;
    pix.coverage = 1.0f;
    pix.theta_esc = (float)res.theta_esc;
    pix.phi_esc   = (float)res.phi_esc;
    if (res.out == Outcome::DISK_HIT) {
        pix._pad[1] = (res.behind_out == Outcome::ESCAPED_B) ? 3
                    : (res.behind_out == Outcome::HORIZON)   ? 2 : 0;
    } else {
        pix._pad[1] = 0;
    }
}

// ── Ray bundle (Jacobi field + footprint) ────────────────────
struct BundlePixelTracer {
    const KndsFrameSetup& f;
    DiskTransparency disk_transp;

    void operator()(int px, int py, GeoPixel& pix) const {
        auto res = trace_bundle(px, py, f.cam, f.g, f.r_disk_in, f.r_disk_out, f.r_escape,
                                f.ctl.max_steps, f.ctl.step_init, f.ctl.tol,
                                f.pixel_offset_x, f.pixel_offset_y, disk_transp);
        pix.outcome   = res.disk_hit ? 1 : 0;
        pix.r         = res.disk_hit ? (float)res.r_hit : 0.0f;
        pix.redshift  = (float)res.redshift;
        pix.magnif    = (float)res.magnif;
        pix.phi_disk  = (float)res.phi_disk;
        pix.fp_dr_a   = (float)res.fp_dr_a;
        pix.fp_dphi_a = (float)res.fp_dphi_a;
        pix.fp_dr_b   = (float)res.fp_dr_b;
        pix.fp_dphi_b = (float)res.fp_dphi_b;
        pix.coverage  = (float)res.coverage;
        pix.sky_dth_a = (float)res.sky_dth_a;
        pix.sky_dph_a = (float)res.sky_dph_a;
        pix.sky_dth_b = (float)res.sky_dth_b;
        pix.sky_dph_b = (float)res.sky_dph_b;
        pix.theta_esc = (float)res.theta_esc;
        pix.phi_esc   = (float)res.phi_esc;
        // The uncovered fraction of a rim pixel sees the background, so
        // it must be told to sample it rather than composite on black.
        pix._pad[1] = res.disk_hit ? (res.behind_escaped ? 0 : 2) : 0;
        if (res.disk_hit && res.behind_escaped) {
            pix.theta_esc = (float)res.behind_theta;
            pix.phi_esc   = (float)res.behind_phi;
        }
        finish_knds_pixel(pix, f.debug_elliptic);
    }
};

// ── Elliptic closed form, with per-ray numerical fallback ─────
// Counts why rays fell back, for the end-of-frame summary.
struct EllipticFallbackStats {
    static constexpr size_t reason_count = static_cast<size_t>(EllipticFallbackReason::COUNT);
    std::array<std::atomic<uint64_t>, reason_count> by_reason{};
    std::atomic<uint64_t> total_rays{0};
    std::atomic<uint64_t> fallback_rays{0};

    void record(EllipticFallbackReason reason) {
        total_rays.fetch_add(1, std::memory_order_relaxed);
        if (reason == EllipticFallbackReason::NONE) return;
        fallback_rays.fetch_add(1, std::memory_order_relaxed);
        const size_t idx = static_cast<size_t>(reason);
        if (idx < reason_count)
            by_reason[idx].fetch_add(1, std::memory_order_relaxed);
    }

    void report() const {
        const uint64_t total = total_rays.load(std::memory_order_relaxed);
        const uint64_t fb = fallback_rays.load(std::memory_order_relaxed);
        if (total == 0) return;
        const double pct = 100.0 * double(fb) / double(total);
        std::cerr << "Info: elliptic-closed fallback summary: "
                  << fb << "/" << total << " rays ("
                  << std::fixed << std::setprecision(2) << pct << "%)\n";
        if (fb == 0) return;
        for (size_t i = 1; i < reason_count; ++i) {
            const uint64_t cnt = by_reason[i].load(std::memory_order_relaxed);
            if (cnt == 0) continue;
            const auto reason = static_cast<EllipticFallbackReason>(i);
            std::cerr << "  - " << elliptic_fallback_reason_name(reason)
                      << ": " << cnt << "\n";
        }
    }
};

struct EllipticPixelTracer {
    const KndsFrameSetup& f;
    CoordinateChart chart;
    Integrator intg;
    EllipticFallbackStats& stats;

    void operator()(int px, int py, GeoPixel& pix) const {
        auto s = f.cam.pixel_ray(px, py, f.g, f.pixel_offset_x, f.pixel_offset_y);
        EllipticFallbackReason fb_reason = EllipticFallbackReason::NONE;
        const TraceResult res = trace_single_elliptic_closed(
            s, f.g, f.r_disk_in, f.r_disk_out, f.r_escape,
            &fb_reason, chart, intg, f.ctl, f.cp_ptr);
        if (f.debug_elliptic)
            pix._pad[0] = static_cast<uint8_t>(fb_reason);
        stats.record(fb_reason);
        store_single_ray(res, pix);
        finish_knds_pixel(pix, f.debug_elliptic);
    }
};

// ── Kerr-Schild Cartesian, numerical ─────────────────────────
struct KsPixelTracer {
    const KndsFrameSetup& f;
    Integrator intg;

    void operator()(int px, int py, GeoPixel& pix) const {
        auto s = f.cam.pixel_ray(px, py, f.g, f.pixel_offset_x, f.pixel_offset_y);
        store_single_ray(trace_single_ks(s, f.g, f.r_disk_in, f.r_disk_out, f.r_escape,
                                         intg, f.ctl, f.cp_ptr), pix);
        finish_knds_pixel(pix, f.debug_elliptic);
    }
};

// ── Separable Kerr potentials (BL, Q=0, Lambda=0) ─────────────
struct SeparablePixelTracer {
    const KndsFrameSetup& f;

    void operator()(int px, int py, GeoPixel& pix) const {
        auto s = f.cam.pixel_ray(px, py, f.g, f.pixel_offset_x, f.pixel_offset_y);
        store_single_ray(trace_single_separable_kerr(s, f.g, f.r_disk_in, f.r_disk_out,
                                                     f.r_escape, f.ctl, f.cp_ptr), pix);
        finish_knds_pixel(pix, f.debug_elliptic);
    }
};

// ── Boyer-Lindquist, numerical: the reference path ───────────
struct BlPixelTracer {
    const KndsFrameSetup& f;
    Integrator intg;

    void operator()(int px, int py, GeoPixel& pix) const {
        auto s = f.cam.pixel_ray(px, py, f.g, f.pixel_offset_x, f.pixel_offset_y);
        store_single_ray(trace_single(s, f.g, f.r_disk_in, f.r_disk_out, f.r_escape,
                                      intg, f.ctl, f.cp_ptr), pix);
        finish_knds_pixel(pix, f.debug_elliptic);
    }
};

// ── DNEG wormhole: no disk, no horizon, two celestial spheres ─
struct WormholePixelTracer {
    WormholeCamera cam;
    DnegParams dp;
    double escape_radius;
    const IntegratorControls& ctl;
    double pixel_offset_x, pixel_offset_y;

    void operator()(int px, int py, GeoPixel& pix) const {
        WormholeState s = cam.pixel_ray(px, py, dp, pixel_offset_x, pixel_offset_y);
        TraceResult res = trace_single_wormhole(s, dp, escape_radius, ctl);

        pix.outcome   = (res.out == Outcome::ESCAPED_B) ? 3 : 0;
        pix.r         = (float)res.r;
        pix.redshift  = 1.0f;
        pix.magnif    = 1.0f;
        pix.coverage  = 1.0f;
        pix.sky_dth_a = pix.sky_dph_a = pix.sky_dth_b = pix.sky_dph_b = 0.0f;
        pix.phi_disk  = 0.0f;
        pix.theta_esc = (float)res.theta_esc;
        pix.phi_esc   = (float)res.phi_esc;
        pix._pad[0]   = pix._pad[1] = pix._pad[2] = 0;
    }
};

// Builds the wormhole tracer for a frame and fills the .kgeo metadata.
static WormholePixelTracer make_wormhole_tracer(
    int W, int H, const FrameParams& fp, const IntegratorControls& ctl,
    double pixel_offset_x, double pixel_offset_y, KGeoMeta* meta_out)
{
    // Celestial spheres live at l = ±infinity (James & Thorne 2015, Fig. 5), so the
    // sky must be far compared with the throat and with the camera distance; the old
    // 1.05*r_obs radius put it just past the camera and produced a fisheye artefact.
    // 1e4 rho reproduces the 1e5 rho exit angles to <0.03 deg (sub-pixel at HD/60 deg FOV).
    const double escape_radius = std::max(std::abs(fp.r_obs) * 20.0, fp.wh_sky_dist);

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
    return WormholePixelTracer{
        WormholeCamera(fp.r_obs, fp.theta, fp.phi, fp.fov, W, H, fp.wh_look_deg),
        DnegParams{fp.wh_rho, fp.wh_a_tunnel, fp.wh_M_lens},
        escape_radius, ctl, pixel_offset_x, pixel_offset_y};
}

using PixelTracer = std::variant<BundlePixelTracer, EllipticPixelTracer, KsPixelTracer,
                                 SeparablePixelTracer, BlPixelTracer, WormholePixelTracer>;

// The hot loop, instantiated once per tracer type.
template<class Tracer>
static std::vector<GeoPixel> trace_rows(int W, int H, const Tracer& tracer) {
    std::vector<GeoPixel> geo(W*H);
    std::atomic<int> rows_done{0};
    const double t0 = get_time();
    std::mutex progress_mu;

    parallel_rows(H, [&](int py) {
        for (int px = 0; px < W; ++px) tracer(px, py, geo[py*W+px]);
        int done = ++rows_done;
        if (done%4==0 || done==H) {
            std::lock_guard<std::mutex> lk(progress_mu);
            print_progress(done, H, get_time()-t0);
        }
    });
    std::cerr << "\n";
    return geo;
}

// The one runtime dispatch of phase 1: picks the instantiation for this frame.
static std::vector<GeoPixel> trace_frame(int W, int H, const PixelTracer& tracer) {
    return std::visit([&](const auto& t) { return trace_rows(W, H, t); }, tracer);
}
