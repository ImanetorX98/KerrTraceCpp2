#pragma once
// Which backend (CPU, Metal, CUDA) can run a given configuration.
//
// Part of the single-translation-unit renderer included by main.cpp;
// see docs/ARCHITETTURA.md for how the modules fit together.

#include "trace_types.hpp"

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

// One sample of the pixel's footprint, in footprint coordinates (s, t) inside
// [-1,1]^2 with a truncated-Gaussian weight. Declared here because both the
// background sampler and the disk shading use the same pattern.
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
