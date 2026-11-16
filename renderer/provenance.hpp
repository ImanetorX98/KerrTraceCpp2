#pragma once
// What goes into a .kgeo provenance block, and the check --color-only makes
// against it.
//
// Part of the single-translation-unit renderer included by main.cpp;
// see docs/ARCHITETTURA.md for how the modules fit together.

#include "render.hpp"

#ifndef KERRTRACE_VERSION
#define KERRTRACE_VERSION "unknown"
#endif
#ifndef KERRTRACE_SOURCE_REV
#define KERRTRACE_SOURCE_REV "unknown"
#endif

static const char* disk_palette_name(DiskPalette p) {
    switch (p) {
        case DiskPalette::BLACKBODY:         return "blackbody";
        case DiskPalette::STRATIFIED:        return "stratified";
        case DiskPalette::INTERSTELLAR:      return "interstellar";
        case DiskPalette::INTERSTELLAR_NASA: return "interstellar-nasa";
    }
    return "unknown";
}

// The colour settings phase 1 reads: they decide which rays pass through the
// disk (transparent tiles, the soft Interstellar rim, global opacity), so a
// buffer traced with one value is not a valid geometry for another.
static void kgeo_visibility(KGeoProvenance& p, const ColorParams& cp) {
    p.set("visibility.palette", disk_palette_name(cp.palette));
    p.set("visibility.disk_opacity", cp.disk_opacity);
    if (cp.palette == DiskPalette::STRATIFIED) {
        p.set("visibility.disk_rings", cp.disk_rings);
        p.set("visibility.disk_sectors", cp.disk_sectors);
        p.set("visibility.disk_sigma", cp.disk_sigma);
        p.set("visibility.legacy_noise_hash", cp.legacy_noise_hash);
    }
    if (cp.palette == DiskPalette::INTERSTELLAR || cp.palette == DiskPalette::INTERSTELLAR_NASA) {
        p.set("visibility.interstellar_edge_transparency", cp.interstellar_edge_transparency);
        p.set("visibility.interstellar_softness_in_scale", cp.interstellar_softness_in_scale);
        p.set("visibility.interstellar_softness_out_scale", cp.interstellar_softness_out_scale);
        p.set("visibility.interstellar_time", cp.interstellar_time);
        p.set("visibility.interstellar_omega0", cp.interstellar_omega0);
        p.set("visibility.legacy_noise_hash", cp.legacy_noise_hash);
    }
}

static KGeoProvenance kgeo_provenance(int W, int H, const FrameParams& fp, bool use_bundles,
                                      RaySolverMode solver_mode, CoordinateChart chart,
                                      Integrator intg, const IntegratorControls& ctl,
                                      double M_bh, double Q_bh, double Lam,
                                      const ColorParams& cp, const std::string& backend) {
    KGeoProvenance p;
    p.complete = true;
    p.set("code.version", KERRTRACE_VERSION);
    p.set("code.source_rev_at_configure", KERRTRACE_SOURCE_REV);
    p.set("backend", backend);
    if (fp.spacetime == Spacetime::DNEG_WORMHOLE) {
        // KGeoMeta reuses the Kerr fields for these; the names say what they are.
        p.set("spacetime", "dneg-wormhole");
        p.set("wormhole.rho", fp.wh_rho);
        p.set("wormhole.a_tunnel", fp.wh_a_tunnel);
        p.set("wormhole.M_lens", fp.wh_M_lens);
        p.set("wormhole.look_deg", fp.wh_look_deg);
        p.set("wormhole.sky_dist", fp.wh_sky_dist);
    } else {
        p.set("spacetime", "knds");
        p.set("metric.M", M_bh);
        p.set("metric.a", fp.a);
        p.set("metric.Q", Q_bh);
        p.set("metric.Lambda", Lam);
    }
    p.set("camera.width", W);
    p.set("camera.height", H);
    p.set("camera.r_obs", fp.r_obs);
    p.set("camera.theta_deg", fp.theta);
    p.set("camera.phi_deg", fp.phi);
    p.set("camera.fov_deg", fp.fov);
    p.set("disk.r_out", fp.disk_out);
    p.set("solver.mode", solver_mode_name(solver_mode));
    p.set("solver.chart", chart == CoordinateChart::KS ? "ks" : chart == CoordinateChart::GKS ? "gks" : "bl");
    p.set("solver.integrator", intg == Integrator::DOPRI5 ? "dopri5" : "rk4-doubling");
    p.set("solver.tol", ctl.tol);
    p.set("solver.max_steps", ctl.max_steps);
    p.set("solver.step_init", ctl.step_init);
    p.set("solver.bundles", use_bundles);
    kgeo_visibility(p, cp);
    return p;
}

// --color-only on a buffer traced with other visibility settings shades the old
// geometry: say which settings differ instead of silently mixing the two.
static void warn_kgeo_visibility(const KGeoProvenance& saved, const ColorParams& cp) {
    if (!saved.complete) {
        std::cerr << "Info: this .kgeo has no provenance (version 4): cannot check that it was "
                     "traced with the current palette and opacity.\n";
        return;
    }
    KGeoProvenance now;
    kgeo_visibility(now, cp);
    const std::string* was_palette = saved.find("visibility.palette");
    const std::string& palette = now.entries.front().second;
    if (!was_palette || *was_palette != palette) {
        std::cerr << "Warning: the geometry was traced with palette "
                  << (was_palette ? *was_palette : std::string("(not recorded)")) << ", now " << palette
                  << "; which rays pass through the disk follows the old palette. Re-trace for an exact image.\n";
        return;
    }
    for (const auto& [key, value] : now.entries) {
        const std::string* was = saved.find(key);
        if (!was || *was != value)
            std::cerr << "Warning: " << key << " is " << value << " but the geometry was traced with "
                      << (was ? *was : std::string("(not recorded)"))
                      << "; rays through the disk follow the old setting. Re-trace for an exact image.\n";
    }
}
