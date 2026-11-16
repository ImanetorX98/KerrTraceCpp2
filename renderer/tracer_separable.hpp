#pragma once
// Separable Kerr fast path (Q=0, Lambda=0).
//
// Part of the single-translation-unit renderer included by main.cpp;
// see docs/ARCHITETTURA.md for how the modules fit together.

#include "tracer_bl.hpp"

// ── Separable Kerr fast path (Q=0, Lambda=0, BL chart) ──────
// Uses first integrals from Hamilton-Jacobi separation:
//   R(r), Theta(theta), and dphi/dlambda from Carter constants.
// This is a semi-analytic step toward fully elliptic inversion.
struct SeparableKerrConsts {
    double M, a;
    double E;   // E = -p_t
    double Lz;  // p_phi
    double Qc;  // Carter constant
};

struct SeparableState {
    double r, theta, phi;
    int sgn_r;
    int sgn_th;
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

static bool kerr_sep_rhs(const SeparableKerrConsts& c, const SeparableState& s,
                         double& dr, double& dth, double& dphi) {
    const double st = std::sin(s.theta);
    const double ct = std::cos(s.theta);
    const double st2 = std::max(st*st, 1e-14);
    const double sigma = std::max(s.r*s.r + c.a*c.a*ct*ct, 1e-14);
    const double delta = kerr_sep_delta(c, s.r);
    const double P = c.E*(s.r*s.r + c.a*c.a) - c.a*c.Lz;
    const double R = kerr_sep_R(c, s.r);
    const double Th = kerr_sep_Theta(c, s.theta);

    dr  = double((s.sgn_r >= 0) ? 1 : -1) * std::sqrt(std::max(R,  0.0)) / sigma;
    dth = double((s.sgn_th>= 0) ? 1 : -1) * std::sqrt(std::max(Th, 0.0)) / sigma;
    dphi = (c.Lz/st2 + c.a*(P/std::max(delta, 1e-14) - c.E)) / sigma;

    return std::isfinite(dr) && std::isfinite(dth) && std::isfinite(dphi);
}

static bool rk4_step_separable_kerr(const SeparableKerrConsts& c, SeparableState& s, double h) {
    double dr1,dth1,dphi1;
    double dr2,dth2,dphi2;
    double dr3,dth3,dphi3;
    double dr4,dth4,dphi4;
    const SeparableState s0 = s;

    if (!kerr_sep_rhs(c, s0, dr1, dth1, dphi1)) return false;

    SeparableState s2 = s0;
    s2.r     = s0.r     + 0.5*h*dr1;
    s2.theta = s0.theta + 0.5*h*dth1;
    s2.phi   = s0.phi   + 0.5*h*dphi1;
    if (!kerr_sep_rhs(c, s2, dr2, dth2, dphi2)) return false;

    SeparableState s3 = s0;
    s3.r     = s0.r     + 0.5*h*dr2;
    s3.theta = s0.theta + 0.5*h*dth2;
    s3.phi   = s0.phi   + 0.5*h*dphi2;
    if (!kerr_sep_rhs(c, s3, dr3, dth3, dphi3)) return false;

    SeparableState s4 = s0;
    s4.r     = s0.r     + h*dr3;
    s4.theta = s0.theta + h*dth3;
    s4.phi   = s0.phi   + h*dphi3;
    if (!kerr_sep_rhs(c, s4, dr4, dth4, dphi4)) return false;

    s.r     += h/6.0*(dr1   + 2.0*dr2   + 2.0*dr3   + dr4);
    s.theta += h/6.0*(dth1  + 2.0*dth2  + 2.0*dth3  + dth4);
    s.phi   += h/6.0*(dphi1 + 2.0*dphi2 + 2.0*dphi3 + dphi4);
    return std::isfinite(s.r) && std::isfinite(s.theta) && std::isfinite(s.phi);
}

static bool rk4_adaptive_separable_kerr(const SeparableKerrConsts& c, SeparableState& s,
                                        double& h, double tol = 1e-8) {
    const SeparableState s0 = s;

    SeparableState sA = s0;
    if (!rk4_step_separable_kerr(c, sA, h)) {
        h = std::max(h*0.5, 1e-10);
        return false;
    }

    SeparableState sB = s0;
    if (!rk4_step_separable_kerr(c, sB, 0.5*h) ||
        !rk4_step_separable_kerr(c, sB, 0.5*h)) {
        h = std::max(h*0.5, 1e-10);
        return false;
    }

    const double err = std::sqrt(
        (sA.r     - sB.r)     * (sA.r     - sB.r) +
        (sA.theta - sB.theta) * (sA.theta - sB.theta) +
        (sA.phi   - sB.phi)   * (sA.phi   - sB.phi)
    ) / 15.0;

    if (!std::isfinite(err)) {
        h = std::max(h*0.5, 1e-10);
        return false;
    }

    if (err < tol || h < 1e-10) {
        s = sB;
        const double sc = (err > 1e-14) ? 0.9 * std::pow(tol/err, 0.2) : 4.0;
        h = clamp(h*sc, 1e-10, 100.0);
        return true;
    }

    h = std::max(h * 0.9 * std::pow(tol/err, 0.25), 1e-10);
    return false;
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

    // Initial signs from BL canonical velocity directions.
    double dr0=0.0, dth0=0.0, dpr0=0.0, dpth0=0.0;
    geodesic_rhs(g, s_bl.r, s_bl.theta, s_bl.pr, s_bl.ptheta, s_bl.pt, s_bl.pphi,
                 dr0, dth0, dpr0, dpth0);
    SeparableState s{ s_bl.r, s_bl.theta, s_bl.phi,
                      (dr0 >= 0.0 ? 1 : -1),
                      (dth0 >= 0.0 ? 1 : -1) };

    const double rh = g.r_horizon();
    const double rh_cut = rh * 1.03;
    double h = std::max(ctl.step_init, 1e-10);
    double prev_Rpot = kerr_sep_R(c, s.r);
    double prev_Thpot = kerr_sep_Theta(c, s.theta);
    const double R_turn_eps = 1e-9;
    const double Th_turn_eps = 1e-11;

    const int max_steps = std::max(1, ctl.max_steps);
    for (int it = 0; it < max_steps; ++it) {
        const SeparableState s_prev = s;
        const double prev_Rpot_step = prev_Rpot;
        const double prev_Thpot_step = prev_Thpot;
        double step_used = h;
        int rejects = 0;
        while (true) {
            step_used = h;
            if (rk4_adaptive_separable_kerr(c, s, h, ctl.tol)) break;
            if (!std::isfinite(h) || ++rejects > 64)
                return trace_single(s_bl, g, r_disk_in, r_disk_out, r_escape, Integrator::RK4_DOUBLING, ctl, cp_ptr);
        }

        // Keep theta in [0, pi].
        if (s.theta < 0.0) {
            s.theta = -s.theta;
            s.sgn_th *= -1;
        } else if (s.theta > M_PI) {
            s.theta = 2.0*M_PI - s.theta;
            s.sgn_th *= -1;
        }

        // Turning points in separated potentials: flip branch signs.
        const double Rnow = kerr_sep_R(c, s.r);
        const double Thnow = kerr_sep_Theta(c, s.theta);
        if (Rnow <= R_turn_eps && prev_Rpot_step > R_turn_eps) s.sgn_r *= -1;
        if (Thnow <= Th_turn_eps && prev_Thpot_step > Th_turn_eps) s.sgn_th *= -1;

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
            double dr0=0.0, dth0=0.0, dphi0=0.0;
            double dr1=0.0, dth1=0.0, dphi1=0.0;
            if (!kerr_sep_rhs(c, s_prev, dr0, dth0, dphi0) ||
                !kerr_sep_rhs(c, s,      dr1, dth1, dphi1)) {
                return trace_single(s_bl, g, r_disk_in, r_disk_out, r_escape, Integrator::RK4_DOUBLING, ctl, cp_ptr);
            }

            double alpha = 0.0;
            if (first_event_alpha_hermite(
                    s_prev.theta, s.theta, dth0, dth1, step_used, M_PI/2.0,
                    alpha, 8, 8)) {
                const double r_hit = hermite_interp_scalar(
                    s_prev.r, s.r, dr0, dr1, step_used, alpha);
                if (r_hit >= r_disk_in && r_hit <= r_disk_out) {
                    const double phi_hit = s_prev.phi + alpha * (s.phi - s_prev.phi);
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
            const double th_esc = s_prev.theta + best_alpha * (s.theta - s_prev.theta);
            const double ph_esc = s_prev.phi   + best_alpha * (s.phi   - s_prev.phi);
            return {Outcome::ESCAPED, r_escape, 1.0, 0.0, th_esc, ph_esc};
        }
        prev_Rpot = Rnow;
        prev_Thpot = Thnow;
    }

    return {Outcome::ESCAPED, s.r, 1.0, 0.0, s.theta, s.phi};
}
