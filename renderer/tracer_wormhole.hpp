#pragma once
// DNEG wormhole tracer.
//
// Part of the single-translation-unit renderer included by main.cpp;
// see docs/ARCHITETTURA.md for how the modules fit together.

#include "frame.hpp"
#include "trace_types.hpp"

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
    WormholeCamera cam(fp.r_obs, fp.theta, fp.phi, fp.fov, W, H, fp.wh_look_deg);
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
            pix.coverage  = 1.0f;
            pix.sky_dth_a = pix.sky_dph_a = pix.sky_dth_b = pix.sky_dph_b = 0.0f;
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
