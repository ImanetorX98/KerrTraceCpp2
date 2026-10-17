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
