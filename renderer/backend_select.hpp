#pragma once
// Which backend (CPU, Metal, CUDA) can run a given configuration.
//
// Part of the single-translation-unit renderer included by main.cpp;
// see docs/ARCHITETTURA.md for how the modules fit together.

#include "trace_types.hpp"
#include "frame.hpp"

static const char* metal_cpu_reason(const ColorParams& cp, bool bundles, bool export_geo) {
    if (bundles) return "ray-bundle footprint filtering requires the CPU Jacobi pipeline";
    if (export_geo) return "KGEO export requires the CPU geometry pipeline";
    if (cp.palette == DiskPalette::INTERSTELLAR_NASA || cp.palette == DiskPalette::STRATIFIED)
        return "the selected disk palette requires CPU shading";
    if (cp.disk_opacity < 1.0 - 1e-9) return "disk transparency requires CPU compositing";
    if (cp.nt_gaussian_taper || cp.planck_emission || cp.radial_ring_taper || cp.zero_torque_taper)
        return "the selected emission/taper controls require CPU shading";
    return nullptr;
}

#if defined(USE_METAL)
static bool metal_solver_supported(RaySolverMode mode, CoordinateChart chart, double Q_bh, double Lam) {
    if (mode == RaySolverMode::STANDARD)
        return std::abs(Q_bh) <= 1e-15 && std::abs(Lam) <= 1e-15;
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

static const char* cuda_cpu_reason(const ColorParams& cp, bool bundles,
                                    RaySolverMode solver, Integrator integrator) {
    if (bundles) return "ray bundles require the CPU Jacobi pipeline";
    if (solver != RaySolverMode::STANDARD) return "the selected solver requires CPU integration";
    if (integrator != Integrator::RK4_DOUBLING) return "DOPRI5 requires CPU integration";
    if (cp.disk_opacity < 1.0 - 1e-9) return "disk transparency requires CPU compositing";
    if (cp.palette != DiskPalette::BLACKBODY)
        return "procedural disk visibility requires CPU integration";
    return nullptr;
}

// Actual completion backend, including runtime fallbacks and supersampling.

// ── Backend planning ──────────────────────────────────────────
// The one place that decides whether a frame can run on the GPU. render_image
// acts on the plan; the CLI asks the same question to label its output before
// rendering. They used to carry separate hand-synchronised copies of this
// logic (three, counting the integrator tag), one edit away from disagreeing.
struct BackendRequest {
    Spacetime spacetime;
    CoordinateChart chart;
    RaySolverMode solver_mode;
    Integrator intg;
    bool use_bundles;
    double Q_bh, Lam;
    bool gpu_fp64;
    bool export_geo;   ///< a .kgeo is written alongside the image
};

#if defined(USE_METAL)
struct MetalPlan {
    CoordinateChart chart;     ///< chart the kernels would integrate in
    RaySolverMode solver;      ///< solver after the availability fallback
    bool ks_chart_supported;
    bool chart_ok;
    bool solver_supported;
    bool base_can_use_gpu;     ///< the kernels support this geometry and solver
    const char* cpu_reason;    ///< shading or export feature that needs the CPU
    bool can_use_gpu;
};

static MetalPlan plan_metal(const BackendRequest& rq, const ColorParams& cp) {
    MetalPlan p{};
    p.chart = resolve_runtime_chart_gpu(rq.chart, rq.Lam);
    p.ks_chart_supported = (std::abs(rq.Lam) <= 1e-15);
    p.chart_ok = (p.chart == CoordinateChart::BL) ||
                 (p.chart == CoordinateChart::KS && p.ks_chart_supported);
    p.solver = select_solver_mode(rq.solver_mode, rq.use_bundles, p.chart, rq.Q_bh, rq.Lam).effective;
    p.solver_supported = metal_solver_supported(p.solver, p.chart, rq.Q_bh, rq.Lam);
    const bool bundle_ok = rq.use_bundles && p.chart_ok && p.solver == RaySolverMode::STANDARD;
    const bool single_ok = !rq.use_bundles && p.chart_ok && p.solver_supported;
    p.base_can_use_gpu = bundle_ok || single_ok;
    p.cpu_reason = metal_cpu_reason(cp, rq.use_bundles, rq.export_geo);
    p.can_use_gpu = p.base_can_use_gpu && !rq.gpu_fp64 &&
                    spacetime_traits(rq.spacetime).gpu_kernels && !p.cpu_reason;
    return p;
}
#endif

#if defined(USE_CUDA)
struct CudaPlan {
    CoordinateChart chart;
    bool chart_ok;
    bool metric_ok;            ///< the kernel is Kerr-only (Q = 0, Lambda = 0)
    const char* cpu_reason;
    bool can_use_gpu;
};

static CudaPlan plan_cuda(const BackendRequest& rq, const ColorParams& cp) {
    CudaPlan p{};
    p.chart = resolve_runtime_chart_gpu(rq.chart, rq.Lam);
    const bool ks_chart_supported = (std::abs(rq.Lam) <= 1e-15);
    p.chart_ok = (p.chart == CoordinateChart::BL) ||
                 (p.chart == CoordinateChart::KS && ks_chart_supported);
    p.metric_ok = (std::abs(rq.Q_bh) <= 1e-15 && std::abs(rq.Lam) <= 1e-15);
    p.cpu_reason = cuda_cpu_reason(cp, rq.use_bundles, rq.solver_mode, rq.intg);
    p.can_use_gpu = !p.cpu_reason && p.chart_ok && p.metric_ok &&
                    spacetime_traits(rq.spacetime).gpu_kernels;
    return p;
}
#endif

// Backend the frame is planned to run on, for labels and file names. A GPU
// failure at run time can still fall back to the CPU; render_image reports the
// backend actually used in last_render_backend.
static const char* planned_backend(const BackendRequest& rq, const ColorParams& cp, bool geo_only) {
    if (geo_only) return "cpu";   // phase 1 on its own always runs on the CPU
#if defined(USE_METAL)
    return plan_metal(rq, cp).can_use_gpu ? "gpu-metal" : "cpu";
#elif defined(USE_CUDA)
    return plan_cuda(rq, cp).can_use_gpu ? "gpu-cuda" : "cpu";
#else
    (void)rq; (void)cp;
    return "cpu";
#endif
}
