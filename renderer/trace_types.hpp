#pragma once
// Ray outcomes, charts, solver modes, integrator controls.
//
// Part of the single-translation-unit renderer included by main.cpp;
// see docs/ARCHITETTURA.md for how the modules fit together.

#include "color_params.hpp"

// ── Geodesic trace helpers ────────────────────────────────────
enum class Outcome { ESCAPED, DISK_HIT, HORIZON, ESCAPED_B };
enum class CoordinateChart { KS, BL, GKS };
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
    Outcome behind_out=Outcome::HORIZON; // event reached after disk hit when disk_opacity < 1
};

struct IntegratorControls {
    int    max_steps = 500000;
    double step_init = 1.0;
    double tol       = 1e-7;
};

// Forward declarations for stratified segmented disk pass-through logic.
static double cell_hash(int cell_id, double salt_a, double salt_b);
static bool stratified_disk_tile_transparent(double r, double phi,
                                             double r_in, double r_out,
                                             const ColorParams& cp);
static double interstellar_disk_soft_mask(double r, double phi,
                                          double r_in, double r_out,
                                          const ColorParams& cp);

static const char* chart_label(CoordinateChart chart) {
    switch (chart) {
        case CoordinateChart::KS:  return "KS";
        case CoordinateChart::BL:  return "BL";
        case CoordinateChart::GKS: return "GKS";
    }
    return "BL";
}

static const char* chart_tag(CoordinateChart chart) {
    switch (chart) {
        case CoordinateChart::KS:  return "ks";
        case CoordinateChart::BL:  return "bl";
        case CoordinateChart::GKS: return "gks";
    }
    return "bl";
}

static const char* disk_radial_profile_name(DiskRadialProfile profile) {
    switch (profile) {
        case DiskRadialProfile::PHYSICAL_NT: return "physical_nt";
        case DiskRadialProfile::PAGE_THORNE:
        default: return "page_thorne";
    }
}

static CoordinateChart resolve_runtime_chart_cpu(CoordinateChart requested, double Lam, bool use_bundles) {
    if (requested == CoordinateChart::GKS) {
        if (use_bundles) return CoordinateChart::BL;
        return (std::abs(Lam) <= 1e-15) ? CoordinateChart::KS : CoordinateChart::BL;
    }
    return requested;
}

static CoordinateChart resolve_runtime_chart_gpu(CoordinateChart requested, double Lam) {
    if (requested == CoordinateChart::GKS)
        return (std::abs(Lam) <= 1e-15) ? CoordinateChart::KS : CoordinateChart::BL;
    return requested;
}

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
