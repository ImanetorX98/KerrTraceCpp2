#pragma once
// Phase 1: trace every pixel into a GeoPixel buffer.
//
// Part of the single-translation-unit renderer included by main.cpp;
// see docs/ARCHITETTURA.md for how the modules fit together.

#include "pixel_tracers.hpp"
#include "palettes.hpp"

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
    // Wormhole: a different spacetime, with its own camera and integrator.
    if (fp.wormhole)
        return trace_frame(W, H, make_wormhole_tracer(W, H, fp, ctl,
                                                      pixel_offset_x, pixel_offset_y, meta_out));

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
    EllipticFallbackStats elliptic_stats;

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

    // Pick the tracer once for the frame; trace_frame runs a loop specialised for it.
    const KndsFrameSetup setup{cam, g, r_disk_in, r_disk_out, r_escape, ctl,
                               pixel_offset_x, pixel_offset_y, cp_ptr, debug_elliptic};
    const PixelTracer tracer =
          eff_use_bundles
            ? PixelTracer{BundlePixelTracer{setup, disk_transp}}
        : solver_sel.effective == RaySolverMode::ELLIPTIC_CLOSED
            ? PixelTracer{EllipticPixelTracer{setup, eff_chart, intg, elliptic_stats}}
        : eff_chart == CoordinateChart::KS
            ? PixelTracer{KsPixelTracer{setup, intg}}
        : solver_sel.effective == RaySolverMode::SEMI_ANALYTIC
            ? PixelTracer{SeparablePixelTracer{setup}}
            : PixelTracer{BlPixelTracer{setup, intg}};
    std::vector<GeoPixel> geo = trace_frame(W, H, tracer);

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

    if (track_elliptic_fallbacks) elliptic_stats.report();
    return geo;
}
