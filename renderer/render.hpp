#pragma once
// Full render: phase 1 + phase 2 on the chosen backend.
//
// Part of the single-translation-unit renderer included by main.cpp;
// see docs/ARCHITETTURA.md for how the modules fit together.

#include "phase1.hpp"
#include "colorize.hpp"
#include "backend_select.hpp"

// ── Full render (trace + colorize, for animation / single frame) ──
static thread_local const char* last_render_backend = "cpu";

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
    last_render_backend = "cpu";
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
        const char* first_backend = nullptr;
        bool mixed_backends = false;
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
            if (!first_backend) first_backend = last_render_backend;
            else mixed_backends = mixed_backends || std::strcmp(first_backend, last_render_backend) != 0;
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
        last_render_backend = mixed_backends ? "mixed" : first_backend;
        return out;
    }

    (void)metal_kernel_mode;
    (void)intersection_mode;
    (void)gpu_fp64;
    (void)elliptic_fallback_black;
    (void)anti_fireflies;
    const BackendRequest backend_rq{fp.spacetime, chart, solver_mode, intg, use_bundles,
                                    Q_bh, Lam, gpu_fp64, geo_out != nullptr};
    (void)backend_rq;
#if defined(USE_METAL)
    const MetalPlan plan = plan_metal(backend_rq, cp);
    const CoordinateChart chart_gpu = plan.chart;
    const bool ks_chart_supported = plan.ks_chart_supported;
    const bool gpu_chart_ok = plan.chart_ok;
    const RaySolverMode gpu_solver_mode = plan.solver;
    const bool solver_gpu_supported = plan.solver_supported;
    const bool base_can_use_gpu = plan.base_can_use_gpu;
    const bool metal_fp64_supported = false; // Current Metal kernel path is FP32 only.
    const bool alpha_composite_cpu_only = cp.disk_opacity < (1.0 - 1e-9);
    const char* cpu_reason = plan.cpu_reason;
    const bool can_use_gpu = plan.can_use_gpu;
    if (cpu_reason) std::cerr << "Info: Metal using CPU: " << cpu_reason << ".\n";

    if (alpha_composite_cpu_only) {
        static bool warned_alpha_composite_cpu = false;
        if (!warned_alpha_composite_cpu) {
            std::cerr << "Info: disk-opacity < 1 enables true alpha compositing behind the disk; "
                         "using CPU path for parity.\n";
            warned_alpha_composite_cpu = true;
        }
    }

    if (can_use_gpu) {
        KNdSMetric g(M_bh, fp.a, Q_bh, Lam);
        const double r_isco = g.r_isco();
        const double flux_ref = disk_flux_reference(r_isco, fp.disk_out, fp.a, cp);
        Camera cam(fp.r_obs, fp.theta, fp.phi, fp.fov, W, H);
        KNdSParams_C kpc{(float)M_bh,(float)fp.a,(float)Q_bh,(float)Lam,
                         (float)g.r_horizon(),(float)r_isco,(float)fp.disk_out};
        CameraParams_C cpc{(float)cam.r_obs,(float)cam.theta_obs,(float)cam.phi_obs,(float)cam.fov_h,
                           W,H,(chart_gpu==CoordinateChart::KS)?1:0,metal_solver_mode_code(gpu_solver_mode),
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
                           (float)pixel_offset_y,
                           (float)cp.exposure,
                           (float)cp.gamma,
                           (float)cp.disk_brightness,
                           (float)cp.disk_opacity,
                           static_cast<int>(cp.palette),
                           static_cast<int>(cp.radial_profile),
                           (float)cp.interstellar_omega0,
                           (float)cp.interstellar_p,
                           (float)cp.interstellar_inner_falloff_scale,
                           (float)cp.interstellar_band_strength,
                           (float)cp.interstellar_band_frequency,
                           (float)cp.interstellar_band_warp,
                           (float)cp.interstellar_turbulence_strength,
                           (float)cp.interstellar_hdr_intensity,
                           (float)cp.interstellar_softness_in_scale,
                           (float)cp.interstellar_softness_out_scale,
                           (float)cp.interstellar_edge_transparency,
                           (float)cp.interstellar_outer_r,
                           (float)cp.interstellar_outer_g,
                           (float)cp.interstellar_outer_b,
                           (float)cp.interstellar_time,
                           cp.page_thorne_gaussian_taper ? 1 : 0,
                           (float)cp.page_thorne_taper_sigma_scale,
                           (float)flux_ref,
                           (float)cp.inner_emission_floor,
                           (float)cp.inner_emission_floor_width,
                           cp.doppler_enabled ? 1 : 0,
                           cp.radial_term_zero_torque ? 1 : 0,
                           cp.radial_term_r3_decay ? 1 : 0,
                           cp.radial_term_relativistic ? 1 : 0,
                           cp.radial_term_b_denom ? 1 : 0,
                           cp.interstellar_inner_glow ? 1 : 0,
                           cp.interstellar_physical_profile ? 1 : 0,
                           cp.bundle_magnif_luma ? 1 : 0,
                           (float)cp.bundle_magnif_max,
                           (float)cp.temp_scale, (float)cp.doppler_exp,
                           cp.temp_redshift_clamp ? 1 : 0, (float)cp.temp_redshift_floor,
                           (float)g.static_observer_ut(cam.r_obs, cam.theta_obs),
                           (float)cp.interstellar_inner_taper_scale};
        const uint8_t* bg_ptr = bg.px.empty() ? nullptr : bg.px.data();
        const int bg_w = bg.px.empty() ? 0 : bg.w;
        const int bg_h = bg.px.empty() ? 0 : bg.h;
        try {
            auto px32 = metal_render(kpc, cpc, bg_ptr, bg_w, bg_h);
            if (px32.size() != size_t(W)*size_t(H))
                throw std::runtime_error("Metal returned an incomplete pixel buffer");
            std::vector<RGB> image(px32.size());
            for (size_t i=0; i<image.size(); ++i)
                image[i] = {uint8_t(px32[i]), uint8_t(px32[i] >> 8), uint8_t(px32[i] >> 16)};
            last_render_backend = "gpu-metal";
            return image;
        } catch (const std::exception& error) {
            std::cerr << "Info: Metal failed (" << error.what() << "); using CPU.\n";
        }
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
        } else if (use_bundles && !gpu_chart_ok && chart_gpu == CoordinateChart::KS)
            std::cerr << "Info: Metal ray bundles on KS currently support Lambda=0 only; using CPU fallback for --bundles.\n";
        else if (use_bundles && gpu_solver_mode != RaySolverMode::STANDARD)
            std::cerr << "Info: Metal ray bundles currently support standard solver only; using CPU fallback for --bundles.\n";
        else if (use_bundles)
            std::cerr << "Info: Metal ray bundles fallback to CPU for this configuration.\n";
        else if (gpu_solver_mode == RaySolverMode::STANDARD &&
                 (std::abs(Q_bh) > 1e-15 || std::abs(Lam) > 1e-15))
            std::cerr << "Info: Metal standard kernel currently supports Kerr only (Q=0, Lambda=0); using CPU fallback.\n";
        else if (!solver_gpu_supported)
            std::cerr << "Info: solver mode '" << solver_mode_name(gpu_solver_mode)
                      << "' on Metal is not available for this chart/metric; using CPU fallback.\n";
        else if (chart_gpu == CoordinateChart::KS && !ks_chart_supported)
            std::cerr << "Info: KS chart on GPU currently supports Lambda=0 only; using CPU fallback.\n";
        else
            std::cerr << "Info: Metal backend using CPU fallback.\n";
        warned_metal_fallback = true;
    }
    KGeoMeta meta;
    auto geo = trace_geodesics(W, H, fp, use_bundles, solver_mode, chart, intg, ctl,
                               pixel_offset_x, pixel_offset_y,
                               M_bh, Q_bh, Lam, &cp, &meta, debug_elliptic);
    if (geo_out) *geo_out = geo;
    return colorize_buffer(geo, W, H, cp, bg, M_bh, fp.a, meta.r_isco,
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
    const CudaPlan plan = plan_cuda(backend_rq, cp);
    const CoordinateChart chart_gpu = plan.chart;
    const bool gpu_metric_ok = plan.metric_ok;
    const bool ks_chart_supported = (std::abs(Lam) <= 1e-15);
    if (plan.cpu_reason) std::cerr << "Info: CUDA using CPU: " << plan.cpu_reason << ".\n";
    if (plan.can_use_gpu) {
        KNdSMetric g(M_bh, fp.a, Q_bh, Lam);
        const double r_isco = g.r_isco();
        Camera cam(fp.r_obs, fp.theta, fp.phi, fp.fov, W, H);
        KNdSParams_CUDA metric{M_bh,fp.a,Q_bh,Lam,g.r_horizon(),r_isco,fp.disk_out};
        CameraParams_CUDA camera{cam.r_obs,cam.theta_obs,cam.phi_obs,cam.fov_h,
            W,H,chart_gpu==CoordinateChart::KS?1:0,ctl.max_steps,
            intersection_mode_code(intersection_mode),ctl.step_init,ctl.tol,
            pixel_offset_x,pixel_offset_y};
        try {
            auto geo = cuda_trace(metric, camera, gpu_fp64);
            auto image = colorize_buffer(geo,W,H,cp,bg,M_bh,fp.a,r_isco,r_isco,fp.disk_out,
                                         debug_elliptic,bg_b);
            if (geo_out) *geo_out = std::move(geo);
            last_render_backend = "gpu-cuda";
            return image;
        } catch (const std::exception& error) {
            std::cerr << "Info: CUDA failed (" << error.what() << "); using CPU.\n";
        }
    }

    static bool warned_cuda_fallback = false;
    if (!warned_cuda_fallback) {
        if (use_bundles)
            std::cerr << "Info: CUDA backend does not support ray bundles yet; using CPU fallback for --bundles.\n";
        else if (solver_mode != RaySolverMode::STANDARD)
            std::cerr << "Info: solver mode '" << solver_mode_name(solver_mode)
                      << "' on CUDA currently falls back to CPU.\n";
        else if (!gpu_metric_ok)
            std::cerr << "Info: CUDA standard kernel currently supports Kerr only (Q=0, Lambda=0); using CPU fallback.\n";
        else if (chart_gpu == CoordinateChart::KS && !ks_chart_supported)
            std::cerr << "Info: KS chart on GPU currently supports Lambda=0 only; using CPU fallback.\n";
        else
            std::cerr << "Info: CUDA backend using CPU fallback.\n";
        warned_cuda_fallback = true;
    }
    KGeoMeta meta;
    auto geo = trace_geodesics(W, H, fp, use_bundles, solver_mode, chart, intg, ctl,
                               pixel_offset_x, pixel_offset_y,
                               M_bh, Q_bh, Lam, &cp, &meta, debug_elliptic);
    if (geo_out) *geo_out = geo;
    return colorize_buffer(geo, W, H, cp, bg, M_bh, fp.a, meta.r_isco,
                           meta.r_disk_in, meta.r_disk_out, debug_elliptic, bg_b);

#else
    // CPU: two-phase
    KGeoMeta meta;
    auto geo = trace_geodesics(W, H, fp, use_bundles, solver_mode, chart, intg, ctl,
                               pixel_offset_x, pixel_offset_y,
                               M_bh, Q_bh, Lam, &cp, &meta, debug_elliptic);
    if (geo_out) *geo_out = geo;
    return colorize_buffer(geo, W, H, cp, bg, M_bh, fp.a, meta.r_isco,
                           meta.r_disk_in, meta.r_disk_out, debug_elliptic, bg_b);
#endif
}
