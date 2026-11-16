#pragma once
// Separable Kerr fast path (Q=0, Lambda=0).
//
// Part of the single-translation-unit renderer included by main.cpp;
// see docs/ARCHITETTURA.md for how the modules fit together.

#include "tracer_bl.hpp"

// ── Separable Kerr fast path (Q=0, Lambda=0, BL chart) ──────
// Uses first integrals from Hamilton-Jacobi separation: R(r), Theta(theta) and
// the Carter constants. Integrated in Mino time tau (d lambda = Sigma d tau) in
// second-order form,
//   r'' = R'(r)/2,  theta'' = Theta'(theta)/2,  phi' = Lz/sin^2 + a(P/Delta - E),
// with r'^2 = R and theta'^2 = Theta as first integrals. The first-order form
// r' = +-sqrt(R) has r' = 0 as a solution at every turning point: a step into
// R < 0 left the ray frozen there until the step cap (audit A1).
struct SeparableKerrConsts {
    double M, a;
    double E;   // E = -p_t
    double Lz;  // p_phi
    double Qc;  // Carter constant
};

struct SeparableState {
    double r, theta, phi;
    double ur, uth;   // dr/dtau, dtheta/dtau
};

static bool init_separable_consts(const GeodesicState& s, const KNdSMetric& g,
                                  SeparableKerrConsts& c) {
    if (std::abs(g.Q) > 1e-15 || std::abs(g.Lambda) > 1e-15) return false;
    c.M  = g.M;
    c.a  = g.a;
    c.E  = -s.pt;
    c.Lz = s.pphi;
    if (!(std::isfinite(c.E) && std::isfinite(c.Lz) && c.E > 0.0)) return false;

    const double st  = std::sin(s.theta);
    const double ct  = std::cos(s.theta);
    const double st2 = std::max(st*st, 1e-14);
    c.Qc = s.ptheta*s.ptheta + ct*ct * (c.Lz*c.Lz/st2 - c.a*c.a*c.E*c.E);
    return std::isfinite(c.Qc);
}

static inline double kerr_sep_delta(const SeparableKerrConsts& c, double r) {
    return r*r - 2.0*c.M*r + c.a*c.a;
}

static inline double kerr_sep_R(const SeparableKerrConsts& c, double r) {
    const double r2 = r*r;
    const double P = c.E*(r2 + c.a*c.a) - c.a*c.Lz;
    const double K = c.Qc + (c.Lz - c.a*c.E)*(c.Lz - c.a*c.E);
    return P*P - kerr_sep_delta(c, r) * K;
}

static inline double kerr_sep_Theta(const SeparableKerrConsts& c, double theta) {
    const double st = std::sin(theta);
    const double ct = std::cos(theta);
    const double st2 = std::max(st*st, 1e-14);
    return c.Qc - ct*ct * (c.Lz*c.Lz/st2 - c.a*c.a*c.E*c.E);
}

struct SeparableRate { double r, theta, phi, ur, uth; };

static bool kerr_sep_rhs(const SeparableKerrConsts& c, const SeparableState& s,
                         SeparableRate& d) {
    const double st = std::max(std::sin(s.theta), 1e-7);
    const double ct = std::cos(s.theta);
    const double delta = kerr_sep_delta(c, s.r);
    const double P = c.E*(s.r*s.r + c.a*c.a) - c.a*c.Lz;
    const double K = c.Qc + (c.Lz - c.a*c.E)*(c.Lz - c.a*c.E);

    d.r     = s.ur;
    d.theta = s.uth;
    d.phi   = c.Lz/(st*st) + c.a*(P/std::max(delta, 1e-14) - c.E);
    d.ur    = 2.0*c.E*s.r*P - (s.r - c.M)*K;                               // R'/2
    d.uth   = c.Lz*c.Lz*ct/(st*st*st) - c.a*c.a*c.E*c.E*st*ct;             // Theta'/2
    return std::isfinite(d.phi) && std::isfinite(d.ur) && std::isfinite(d.uth);
}

static SeparableState sep_advance(const SeparableState& s, const SeparableRate& d, double h) {
    return {s.r + h*d.r, s.theta + h*d.theta, s.phi + h*d.phi, s.ur + h*d.ur, s.uth + h*d.uth};
}

static bool rk4_step_separable_kerr(const SeparableKerrConsts& c, SeparableState& s, double h) {
    SeparableRate k1, k2, k3, k4;
    if (!kerr_sep_rhs(c, s, k1)) return false;
    if (!kerr_sep_rhs(c, sep_advance(s, k1, 0.5*h), k2)) return false;
    if (!kerr_sep_rhs(c, sep_advance(s, k2, 0.5*h), k3)) return false;
    if (!kerr_sep_rhs(c, sep_advance(s, k3, h), k4)) return false;
    s.r     += h/6.0*(k1.r     + 2.0*k2.r     + 2.0*k3.r     + k4.r);
    s.theta += h/6.0*(k1.theta + 2.0*k2.theta + 2.0*k3.theta + k4.theta);
    s.phi   += h/6.0*(k1.phi   + 2.0*k2.phi   + 2.0*k3.phi   + k4.phi);
    s.ur    += h/6.0*(k1.ur    + 2.0*k2.ur    + 2.0*k3.ur    + k4.ur);
    s.uth   += h/6.0*(k1.uth   + 2.0*k2.uth   + 2.0*k3.uth   + k4.uth);
    return std::isfinite(s.r) && std::isfinite(s.theta) && std::isfinite(s.phi)
        && std::isfinite(s.ur) && std::isfinite(s.uth);
}

// Step doubling; h is a Mino-time step.
static bool rk4_adaptive_separable_kerr(const SeparableKerrConsts& c, SeparableState& s,
                                        double& h, double tol = 1e-8) {
    const SeparableState s0 = s;

    SeparableState sA = s0;
    if (!rk4_step_separable_kerr(c, sA, h)) {
        h *= 0.5;
        return false;
    }

    SeparableState sB = s0;
    if (!rk4_step_separable_kerr(c, sB, 0.5*h) ||
        !rk4_step_separable_kerr(c, sB, 0.5*h)) {
        h *= 0.5;
        return false;
    }

    // Same phase-space norm as the BL integrator: p_r = ur/Delta, p_theta = uth.
    const double delta = std::max(std::abs(kerr_sep_delta(c, sB.r)), 1e-14);
    const double dpr = (sA.ur - sB.ur) / delta;
    const double err = std::sqrt(
        (sA.r     - sB.r)     * (sA.r     - sB.r) +
        (sA.theta - sB.theta) * (sA.theta - sB.theta) +
        (sA.phi   - sB.phi)   * (sA.phi   - sB.phi) +
        dpr * dpr +
        (sA.uth   - sB.uth)   * (sA.uth   - sB.uth)
    ) / 15.0;

    if (!std::isfinite(err)) {
        h *= 0.5;
        return false;
    }

    if (err < tol) {
        s = sB;
        h *= (err > 1e-14) ? std::min(4.0, 0.9 * std::pow(tol/err, 0.2)) : 4.0;
        return true;
    }

    h *= std::max(0.1, 0.9 * std::pow(tol/err, 0.25));
    return false;
}

// The state at an event inside the accepted step s_prev -> s_prev + h: an RK4
// sub-step of alpha*h from s_prev, alpha corrected by Newton on f(state) = 0
// with df the rate of f. Cubic interpolation across a long Mino step is off by
// O(h^4); this keeps the integrator's own accuracy.
template <class F, class DF>
static SeparableState sep_locate_event(const SeparableKerrConsts& c, const SeparableState& s_prev,
                                       double h, double& alpha, F f, DF df) {
    SeparableState x = s_prev;
    for (int it = 0; it < 6; ++it) {
        x = s_prev;
        if (!rk4_step_separable_kerr(c, x, alpha*h)) return x;
        const double rate = df(x) * h;
        if (!(std::abs(rate) > 0.0)) return x;
        const double next = clamp(alpha - f(x)/rate, 0.0, 1.0);
        const bool done = std::abs(next - alpha) < 1e-13;
        alpha = next;
        if (done) break;
    }
    x = s_prev;
    rk4_step_separable_kerr(c, x, alpha*h);
    return x;
}

static TraceResult trace_single_separable_kerr(GeodesicState s_bl, const KNdSMetric& g,
                                               double r_disk_in, double r_disk_out,
                                               double r_escape,
                                               const IntegratorControls& ctl = IntegratorControls{},
                                               const ColorParams* cp_ptr = nullptr) {
    const double observer_frequency = -s_bl.pt * g.static_observer_ut(s_bl.r, s_bl.theta);
    if (cp_ptr && clamp(cp_ptr->disk_opacity, 0.0, 1.0) < (1.0 - 1e-9)) {
        // Semi-analytic path currently does not carry post-disk continuation metadata
        // required for alpha compositing; defer to standard BL tracer.
        return trace_single(s_bl, g, r_disk_in, r_disk_out, r_escape, Integrator::RK4_DOUBLING, ctl, cp_ptr);
    }

    SeparableKerrConsts c{};
    if (!init_separable_consts(s_bl, g, c))
        return trace_single(s_bl, g, r_disk_in, r_disk_out, r_escape, Integrator::RK4_DOUBLING, ctl, cp_ptr);

    // Mino-time velocities: dr/dtau = Delta p_r and dtheta/dtau = p_theta exactly.
    SeparableState s{ s_bl.r, s_bl.theta, s_bl.phi,
                      kerr_sep_delta(c, s_bl.r) * s_bl.pr, s_bl.ptheta };

    const double rh = g.r_horizon();
    const double rh_cut = rh * 1.03;
    // step_init is an affine step; d tau = d lambda / Sigma.
    const double sigma0 = s.r*s.r + c.a*c.a*std::cos(s.theta)*std::cos(s.theta);
    double h = std::max(ctl.step_init, 1e-10) / sigma0;
    const double h_min = 1e-12 * h;

    const int max_steps = std::max(1, ctl.max_steps);
    for (int it = 0; it < max_steps; ++it) {
        const SeparableState s_prev = s;
        double step_used = h;
        int rejects = 0;
        while (true) {
            step_used = h;
            if (rk4_adaptive_separable_kerr(c, s, h, ctl.tol)) break;
            if (!std::isfinite(h) || h < h_min || ++rejects > 64)
                return trace_single(s_bl, g, r_disk_in, r_disk_out, r_escape, Integrator::RK4_DOUBLING, ctl, cp_ptr);
        }

        // Back onto the first integrals: the second-order form keeps any error
        // in ur forever (ur' = R'/2 does not depend on ur), so an error made at
        // r_obs, where ur ~ E r^2, ruins the ray near the hole where ur ~ 1.
        // Where R <= 0 (a step past the turning point) the integrated value
        // stands: it is what carries the ray through the turn.
        const double Rnow = kerr_sep_R(c, s.r);
        const double Thnow = kerr_sep_Theta(c, s.theta);
        if (Rnow > 0.0)  s.ur  = std::copysign(std::sqrt(Rnow),  s.ur);
        if (Thnow > 0.0) s.uth = std::copysign(std::sqrt(Thnow), s.uth);

        // Through a pole (only reachable with Lz = 0): back into [0, pi].
        if (s.theta < 0.0 || s.theta > M_PI) {
            s.theta = (s.theta < 0.0) ? -s.theta : 2.0*M_PI - s.theta;
            s.uth = -s.uth;
            s.phi += M_PI;
        }

        double best_alpha = 2.0;
        enum class StepEvent { NONE, DISK, HORIZON, ESCAPE };
        StepEvent best_event = StepEvent::NONE;
        double disk_r_hit = 0.0;
        double disk_red_hit = 1.0;
        double disk_phi_hit = 0.0;

        const double q0 = s_prev.theta - M_PI/2.0;
        const double q1 = s.theta      - M_PI/2.0;
        const bool maybe_equator = sign_change(q0, q1) ||
                                   (std::min(std::abs(q0), std::abs(q1)) < 0.35);
        if (maybe_equator) {
            double alpha = 0.0;
            if (first_event_alpha_hermite(
                    s_prev.theta, s.theta, s_prev.uth, s.uth, step_used, M_PI/2.0,
                    alpha, 8, 8)) {
                const SeparableState x = sep_locate_event(c, s_prev, step_used, alpha,
                    [](const SeparableState& v) { return v.theta - M_PI/2.0; },
                    [](const SeparableState& v) { return v.uth; });
                const double r_hit = x.r;
                if (r_hit >= r_disk_in && r_hit <= r_disk_out) {
                    const double phi_hit = x.phi;
                    bool pass_through = false;
                    if (cp_ptr && cp_ptr->palette == DiskPalette::STRATIFIED) {
                        pass_through = stratified_disk_tile_transparent(
                            r_hit, phi_hit, r_disk_in, r_disk_out, *cp_ptr);
                    } else if (cp_ptr && (cp_ptr->palette == DiskPalette::INTERSTELLAR ||
                                          cp_ptr->palette == DiskPalette::INTERSTELLAR_NASA)) {
                        pass_through = interstellar_disk_soft_mask(
                            r_hit, phi_hit, r_disk_in, r_disk_out, *cp_ptr) < std::max(0.0, cp_ptr->interstellar_edge_transparency);
                    }
                    if (!pass_through) {
                        disk_r_hit = r_hit;
                        disk_red_hit = clamp(disk_redshift(r_hit, s_bl.pt, s_bl.pphi, g, observer_frequency), 0.0, 20.0);
                        disk_phi_hit = phi_hit;
                        best_alpha = alpha;
                        best_event = StepEvent::DISK;
                    }
                }
            }
        }

        const bool horizon_cross = ((s_prev.r > rh_cut) && (s.r <= rh_cut)) || (s.r <= rh_cut);
        if (horizon_cross) {
            const double denom_h = s_prev.r - s.r;
            double alpha_h = (std::abs(denom_h) > 1e-12) ? ((s_prev.r - rh_cut) / denom_h) : 0.0;
            alpha_h = clamp(alpha_h, 0.0, 1.0);
            if (alpha_h < best_alpha) {
                best_alpha = alpha_h;
                best_event = StepEvent::HORIZON;
            }
        }

        const bool escape_cross = ((s_prev.r < r_escape) && (s.r >= r_escape)) || (s.r >= r_escape);
        if (escape_cross) {
            const double denom_e = s.r - s_prev.r;
            double alpha_e = (std::abs(denom_e) > 1e-12) ? ((r_escape - s_prev.r) / denom_e) : 1.0;
            alpha_e = clamp(alpha_e, 0.0, 1.0);
            if (alpha_e < best_alpha) {
                best_alpha = alpha_e;
                best_event = StepEvent::ESCAPE;
            }
        }

        if (best_event == StepEvent::DISK)
            return {Outcome::DISK_HIT, disk_r_hit, disk_red_hit, disk_phi_hit};
        if (best_event == StepEvent::HORIZON) {
            const double r_h = s_prev.r + best_alpha * (s.r - s_prev.r);
            return {Outcome::HORIZON, r_h, 0.0};
        }
        if (best_event == StepEvent::ESCAPE) {
            const SeparableState x = sep_locate_event(c, s_prev, step_used, best_alpha,
                [r_escape](const SeparableState& v) { return v.r - r_escape; },
                [](const SeparableState& v) { return v.ur; });
            return {Outcome::ESCAPED, r_escape, 1.0, 0.0, x.theta, x.phi};
        }
    }

    // The step cap is not evidence of escape: let BL decide.
    return trace_single(s_bl, g, r_disk_in, r_disk_out, r_escape, Integrator::RK4_DOUBLING, ctl, cp_ptr);
}
