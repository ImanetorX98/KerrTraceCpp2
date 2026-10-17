#pragma once
// Phase 1: trace every pixel into a GeoPixel buffer.
//
// Part of the single-translation-unit renderer included by main.cpp;
// see docs/ARCHITETTURA.md for how the modules fit together.

#include "tracer_bl.hpp"
#include "tracer_separable.hpp"
#include "tracer_elliptic.hpp"
#include "tracer_ks.hpp"
#include "tracer_wormhole.hpp"
#include "palettes.hpp"
#include "frame.hpp"

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
    const ColorParams* cp_ptr = nullptr,
    KGeoMeta* meta_out = nullptr,
    bool debug_elliptic = false)
{
    // Wormhole: delegate entirely to the DNEG integrator
    if (fp.wormhole)
        return trace_geodesics_wormhole(W, H, fp, ctl, pixel_offset_x, pixel_offset_y, meta_out);

    bool eff_use_bundles = use_bundles;
    if (chart == CoordinateChart::GKS && eff_use_bundles && std::abs(Lam) > 1e-15) {
        static bool warned_gks_disable_bundles = false;
        if (!warned_gks_disable_bundles) {
            std::cerr << "Info: GKS + Lambda != 0 currently disables ray bundles; using single-ray core.\n";
            warned_gks_disable_bundles = true;
        }
        eff_use_bundles = false;
    }

    KNdSMetric g(M_bh, fp.a, Q_bh, Lam);
    CoordinateChart eff_chart = resolve_runtime_chart_cpu(chart, Lam, eff_use_bundles);
    if (chart == CoordinateChart::GKS) {
        if (eff_use_bundles) {
            static bool warned_gks_bundle = false;
            if (!warned_gks_bundle) {
                std::cerr << "Info: GKS + ray bundles currently use BL core on CPU.\n";
                warned_gks_bundle = true;
            }
        } else if (std::abs(Lam) > 1e-15) {
            static bool warned_gks_lambda = false;
            if (!warned_gks_lambda) {
                std::cerr << "Info: GKS selected BL core for Lambda != 0.\n";
                warned_gks_lambda = true;
            }
        }
    }
    if (chart == CoordinateChart::KS && eff_chart == CoordinateChart::BL && std::abs(Lam) > 1e-15) {
        static bool warned_lam_ks = false;
        if (!warned_lam_ks) {
            std::cerr << "Info: KS chart currently supports Lambda=0 only; falling back to BL.\n";
            warned_lam_ks = true;
        }
    }
    if (chart == CoordinateChart::KS && eff_use_bundles && eff_chart == CoordinateChart::BL) {
        static bool warned_bundle_ks = false;
        if (!warned_bundle_ks) {
            std::cerr << "Info: Ray bundles currently run in BL chart; falling back to BL for --bundles.\n";
            warned_bundle_ks = true;
        }
    }

    const SolverSelection solver_sel = select_solver_mode(
        solver_mode, eff_use_bundles, eff_chart, Q_bh, Lam);
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
        (!eff_use_bundles && solver_sel.effective == RaySolverMode::ELLIPTIC_CLOSED);
    constexpr size_t elliptic_reason_count =
        static_cast<size_t>(EllipticFallbackReason::COUNT);
    std::array<std::atomic<uint64_t>, elliptic_reason_count> elliptic_fallback_by_reason{};
    for (auto& c : elliptic_fallback_by_reason) c.store(0, std::memory_order_relaxed);
    std::atomic<uint64_t> elliptic_total_rays{0};
    std::atomic<uint64_t> elliptic_fallback_rays{0};

    const double r_isco   = g.r_isco();
    if (!std::isfinite(r_isco) || !(r_isco > 0.0))
        throw std::runtime_error("No stable corotating ISCO for these metric parameters");
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

    // Same predicate the single-ray tracers apply, handed to the bundle tracer so
    // both agree on what counts as a hit. Without it the bundle stopped on the
    // faded rim of the Interstellar disk and rendered it black.
    struct BundleTranspCtx { const ColorParams* cp; double r_in, r_out; };
    const BundleTranspCtx transp_ctx{cp_ptr, r_disk_in, r_disk_out};
    DiskTransparency disk_transp{};
    if (cp_ptr) {
        disk_transp.ctx = &transp_ctx;
        disk_transp.fn = [](double rr, double pp, const void* vctx) -> bool {
            const BundleTranspCtx* c = static_cast<const BundleTranspCtx*>(vctx);
            if (!c || !c->cp) return false;
            if (c->cp->palette == DiskPalette::STRATIFIED)
                return stratified_disk_tile_transparent(rr, pp, c->r_in, c->r_out, *c->cp);
            if (c->cp->palette == DiskPalette::INTERSTELLAR ||
                c->cp->palette == DiskPalette::INTERSTELLAR_NASA)
                return interstellar_disk_soft_mask(rr, pp, c->r_in, c->r_out, *c->cp)
                       < std::max(0.0, c->cp->interstellar_edge_transparency);
            return false;
        };
    }

    std::vector<GeoPixel> geo(W*H);
    std::atomic<int> rows_done{0};
    const double t0_geo = get_time();
    std::mutex progress_mu;

    auto trace_row = [&](int py) {
        for (int px_=0; px_<W; ++px_) {
            GeoPixel& pix = geo[py*W+px_];
            if (eff_use_bundles) {
                auto res = trace_bundle(px_, py, cam, g, r_disk_in, r_disk_out, r_escape,
                                        ctl.max_steps, ctl.step_init, ctl.tol,
                                        pixel_offset_x, pixel_offset_y, disk_transp);
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
            } else {
                auto s   = cam.pixel_ray(px_, py, g, pixel_offset_x, pixel_offset_y);
                TraceResult res{};
                if (solver_sel.effective == RaySolverMode::ELLIPTIC_CLOSED) {
                    EllipticFallbackReason fb_reason = EllipticFallbackReason::NONE;
                    res = trace_single_elliptic_closed(
                        s, g, r_disk_in, r_disk_out, r_escape,
                        &fb_reason, eff_chart, intg, ctl, cp_ptr);
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
                    res = trace_single_ks(s, g, r_disk_in, r_disk_out, r_escape, intg, ctl, cp_ptr);
                } else if (solver_sel.effective == RaySolverMode::SEMI_ANALYTIC) {
                    res = trace_single_separable_kerr(s, g, r_disk_in, r_disk_out, r_escape, ctl, cp_ptr);
                } else {
                    res = trace_single(s, g, r_disk_in, r_disk_out, r_escape, intg, ctl, cp_ptr);
                }
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
            if (!debug_elliptic) pix._pad[0]=0;
            pix._pad[2]=0;
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

    // ── Edge resampling: A.3.1's own remedy ──────────────────────────────
    //
    // "In extreme cases these assumptions can break down, leading to a
    //  distortion in the shape of a star's image, flickering, and aliasing
    //  artefacts. In these cases we can trace multiple beams per pixel and
    //  resample."  -- DNGR, Appendix A.3.1
    //
    // The analytic coverage from the footprint is one-sided: a pixel whose
    // centre falls just inside the disk feathers, one whose centre falls just
    // outside carries no footprint at all and stays a hard cut, so the rim is
    // biased inward by half a footprint. Tracing a grid of beams across the
    // pixel measures the coverage from both sides instead of inferring it from
    // one, and it costs only what the boundary costs: a pixel is resampled only
    // when its four-neighbourhood disagrees about whether the disk is there.
    if (eff_use_bundles && cp_ptr && cp_ptr->bundle_filter && cp_ptr->bundle_edge_grid > 1) {
        const int n = cp_ptr->bundle_edge_grid;
        std::vector<uint32_t> edge;
        edge.reserve(size_t(W) * 2);
        for (int py = 0; py < H; ++py)
            for (int px_ = 0; px_ < W; ++px_) {
                const uint8_t here = geo[size_t(py)*W + px_].outcome;
                bool differs = false;
                for (int k = 0; k < 4 && !differs; ++k) {
                    static const int dxs[4] = {1,-1,0,0}, dys[4] = {0,0,1,-1};
                    const int qx = px_ + dxs[k], qy = py + dys[k];
                    if (qx < 0 || qy < 0 || qx >= W || qy >= H) continue;
                    const uint8_t there = geo[size_t(qy)*W + qx].outcome;
                    // Only the disk silhouette needs this; horizon against
                    // background is already a hard edge in the reference too.
                    if ((here == 1) != (there == 1)) differs = true;
                }
                if (differs) edge.push_back(uint32_t(size_t(py)*W + px_));
            }

        std::cerr << "Edge resampling: " << edge.size() << " pixels ("
                  << std::fixed << std::setprecision(2)
                  << (100.0 * double(edge.size()) / double(W*H)) << "% of frame), "
                  << n << "x" << n << " beams each\n";

        std::vector<GeoPixel> resolved(edge.size());
        auto resample_one = [&](size_t idx) {
            const uint32_t flat = edge[idx];
            const int py = int(flat / uint32_t(W));
            const int px_ = int(flat % uint32_t(W));
            GeoPixel out = geo[flat];

            int hits = 0, escapes = 0;
            double acc_r = 0.0, acc_red = 0.0, acc_cov = 0.0;
            double acc_phi_c = 0.0, acc_phi_s = 0.0;
            double acc_fp[4] = {0,0,0,0};
            double acc_th_e = 0.0, acc_ph_ec = 0.0, acc_ph_es = 0.0;
            double acc_sky[4] = {0,0,0,0};
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    const double ox = (double(i) + 0.5) / double(n) - 0.5;
                    const double oy = (double(j) + 0.5) / double(n) - 0.5;
                    const auto sub = trace_bundle(px_, py, cam, g,
                                                  r_disk_in, r_disk_out, r_escape,
                                                  ctl.max_steps, ctl.step_init, ctl.tol,
                                                  ox, oy, disk_transp);
                    if (sub.disk_hit) {
                        ++hits;
                        acc_r   += sub.r_hit;
                        acc_red += sub.redshift;
                        acc_cov += sub.coverage;
                        acc_phi_c += std::cos(sub.phi_disk);
                        acc_phi_s += std::sin(sub.phi_disk);
                        acc_fp[0] += sub.fp_dr_a;   acc_fp[1] += sub.fp_dphi_a;
                        acc_fp[2] += sub.fp_dr_b;   acc_fp[3] += sub.fp_dphi_b;
                        if (sub.behind_escaped) {
                            ++escapes;
                            acc_th_e   += sub.behind_theta;
                            acc_ph_ec  += std::cos(sub.behind_phi);
                            acc_ph_es  += std::sin(sub.behind_phi);
                            acc_sky[0] += sub.sky_dth_a; acc_sky[1] += sub.sky_dph_a;
                            acc_sky[2] += sub.sky_dth_b; acc_sky[3] += sub.sky_dph_b;
                        }
                    } else {
                        ++escapes;
                        acc_th_e  += sub.theta_esc;
                        acc_ph_ec += std::cos(sub.phi_esc);
                        acc_ph_es += std::sin(sub.phi_esc);
                        acc_sky[0] += sub.sky_dth_a; acc_sky[1] += sub.sky_dph_a;
                        acc_sky[2] += sub.sky_dth_b; acc_sky[3] += sub.sky_dph_b;
                    }
                }

            const int total = n*n;
            if (hits == 0) {
                out.outcome  = geo[flat].outcome == 2u ? 2u : 0u;
                out.coverage = 0.0f;
            } else {
                const double inv = 1.0 / double(hits);
                out.outcome   = 1u;
                out.r         = float(acc_r * inv);
                out.redshift  = float(acc_red * inv);
                // Azimuths are averaged as unit vectors: a plain mean would be
                // wrong for any pixel straddling the phi = pi branch cut.
                out.phi_disk  = float(std::atan2(acc_phi_s, acc_phi_c));
                out.fp_dr_a   = float(acc_fp[0] * inv);
                out.fp_dphi_a = float(acc_fp[1] * inv);
                out.fp_dr_b   = float(acc_fp[2] * inv);
                out.fp_dphi_b = float(acc_fp[3] * inv);
                // Coverage measured from both sides: the fraction of beams that
                // land on the disk, times each beam's own partial coverage.
                out.coverage  = float(clamp((acc_cov / double(total)), 0.0, 1.0));
                out._pad[1]   = escapes > 0 ? 0 : 2;
                if (escapes > 0) {
                    const double einv = 1.0 / double(escapes);
                    out.theta_esc = float(acc_th_e * einv);
                    out.phi_esc   = float(std::atan2(acc_ph_es, acc_ph_ec));
                    out.sky_dth_a = float(acc_sky[0] * einv);
                    out.sky_dph_a = float(acc_sky[1] * einv);
                    out.sky_dth_b = float(acc_sky[2] * einv);
                    out.sky_dph_b = float(acc_sky[3] * einv);
                }
            }
            resolved[idx] = out;
        };

#if defined(_OPENMP)
        #pragma omp parallel for schedule(dynamic, 32)
        for (long long idx = 0; idx < (long long)edge.size(); ++idx)
            resample_one(size_t(idx));
#else
        for (size_t idx = 0; idx < edge.size(); ++idx) resample_one(idx);
#endif
        for (size_t idx = 0; idx < edge.size(); ++idx) geo[edge[idx]] = resolved[idx];
    }

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
