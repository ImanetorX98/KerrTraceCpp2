#pragma once
// Boyer-Lindquist numerical tracer: the reference path.
//
// Part of the single-translation-unit renderer included by main.cpp;
// see docs/ARCHITETTURA.md for how the modules fit together.

#include "trace_types.hpp"
#include "disk_physics.hpp"

static TraceResult trace_terminal_no_disk(GeodesicState s, const KNdSMetric& g,
                                          double r_escape,
                                          Integrator intg,
                                          IntegratorControls ctl) {
    double dlam = std::max(ctl.step_init, 1e-10);
    const double rh_cut = g.r_horizon() * 1.03;
    Vec4d fsal = Vec4d::invalid();
    const int max_steps = std::max(1, ctl.max_steps);
    for (int it = 0; it < max_steps; ++it) {
        const GeodesicState s_prev = s;
        int rejects = 0;
        while (true) {
            if (adaptive_step(g, s, dlam, intg, fsal, ctl.tol)) break;
            if (!std::isfinite(dlam) || ++rejects > 64)
                return {Outcome::ESCAPED, s.r, 1.0, 0.0, s.theta, s.phi};
        }

        double best_alpha = 2.0;
        enum class TailEvent { NONE, HORIZON, ESCAPE };
        TailEvent best_event = TailEvent::NONE;

        const bool horizon_cross = ((s_prev.r > rh_cut) && (s.r <= rh_cut)) || (s.r <= rh_cut);
        if (horizon_cross) {
            const double denom_h = s_prev.r - s.r;
            double alpha_h = (std::abs(denom_h) > 1e-12) ? ((s_prev.r - rh_cut) / denom_h) : 0.0;
            alpha_h = clamp(alpha_h, 0.0, 1.0);
            if (alpha_h < best_alpha) {
                best_alpha = alpha_h;
                best_event = TailEvent::HORIZON;
            }
        }

        const bool escape_cross = ((s_prev.r < r_escape) && (s.r >= r_escape)) || (s.r >= r_escape);
        if (escape_cross) {
            const double denom_e = s.r - s_prev.r;
            double alpha_e = (std::abs(denom_e) > 1e-12) ? ((r_escape - s_prev.r) / denom_e) : 1.0;
            alpha_e = clamp(alpha_e, 0.0, 1.0);
            if (alpha_e < best_alpha) {
                best_alpha = alpha_e;
                best_event = TailEvent::ESCAPE;
            }
        }

        if (best_event == TailEvent::HORIZON) {
            const double r_h = s_prev.r + best_alpha * (s.r - s_prev.r);
            return {Outcome::HORIZON, r_h, 0.0};
        }
        if (best_event == TailEvent::ESCAPE) {
            const double th_esc = s_prev.theta + best_alpha * (s.theta - s_prev.theta);
            const double ph_esc = s_prev.phi   + best_alpha * (s.phi   - s_prev.phi);
            return {Outcome::ESCAPED, r_escape, 1.0, 0.0, th_esc, ph_esc};
        }
    }
    return {Outcome::ESCAPED, s.r, 1.0, 0.0, s.theta, s.phi};
}

static TraceResult trace_single(GeodesicState s, const KNdSMetric& g,
                                double r_disk_in, double r_disk_out,
                                double r_escape,
                                Integrator intg=Integrator::RK4_DOUBLING,
                                const IntegratorControls& ctl = IntegratorControls{},
                                const ColorParams* cp_ptr = nullptr) {
    const double observer_frequency = -s.pt * g.static_observer_ut(s.r, s.theta);
    double rh=g.r_horizon(), dlam=std::max(ctl.step_init, 1e-10);
    const double rh_cut = rh * 1.03;
    Vec4d fsal=Vec4d::invalid();
    const int max_steps = std::max(1, ctl.max_steps);
    for (int it=0; it<max_steps; ++it) {
        const GeodesicState s_prev = s;
        double step_used = dlam;
        int rejects = 0;
        while (true) {
            step_used = dlam;
            if (adaptive_step(g, s, dlam, intg, fsal, ctl.tol)) break;
            if (!std::isfinite(dlam) || ++rejects > 64)
                return {Outcome::ESCAPED, s.r, 1.0, 0.0, s.theta, s.phi};
        }

        double best_alpha = 2.0;
        enum class StepEvent { NONE, DISK, HORIZON, ESCAPE };
        StepEvent best_event = StepEvent::NONE;

        double disk_r_hit = 0.0;
        double disk_redshift_hit = 1.0;
        double disk_phi_hit = 0.0;

        const double q0 = s_prev.theta - M_PI/2.0;
        const double q1 = s.theta      - M_PI/2.0;
        const bool maybe_equator = sign_change(q0, q1) ||
                                   (std::min(std::abs(q0), std::abs(q1)) < 0.35);
        if (maybe_equator) {
            double dr0,dth0,dpr0,dpth0;
            double dr1,dth1,dpr1,dpth1;
            geodesic_rhs(g, s_prev.r, s_prev.theta, s_prev.pr, s_prev.ptheta,
                         s_prev.pt, s_prev.pphi, dr0,dth0,dpr0,dpth0);
            geodesic_rhs(g, s.r, s.theta, s.pr, s.ptheta,
                         s.pt, s.pphi, dr1,dth1,dpr1,dpth1);
            double alpha = 0.0;
            if (first_event_alpha_hermite(
                    s_prev.theta, s.theta, dth0, dth1, step_used, M_PI/2.0,
                    alpha, 8, 8)) {
                const double r_hit = hermite_interp_scalar(
                    s_prev.r, s.r, dr0, dr1, step_used, alpha);
                if (r_hit>=r_disk_in && r_hit<=r_disk_out) {
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
                        disk_redshift_hit = clamp(disk_redshift(r_hit, s.pt, s.pphi, g, observer_frequency), 0.0, 20.0);
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

        if (best_event == StepEvent::DISK) {
            TraceResult out{Outcome::DISK_HIT, disk_r_hit, disk_redshift_hit, disk_phi_hit};
            const double alpha = clamp(best_alpha, 0.0, 1.0);
            const double opacity = cp_ptr ? clamp(cp_ptr->disk_opacity, 0.0, 1.0) : 1.0;
            if (opacity < (1.0 - 1e-9)) {
                GeodesicState s_hit = s_prev;
                s_hit.r      = s_prev.r      + alpha * (s.r      - s_prev.r);
                s_hit.theta  = s_prev.theta  + alpha * (s.theta  - s_prev.theta);
                s_hit.phi    = s_prev.phi    + alpha * (s.phi    - s_prev.phi);
                s_hit.pr     = s_prev.pr     + alpha * (s.pr     - s_prev.pr);
                s_hit.ptheta = s_prev.ptheta + alpha * (s.ptheta - s_prev.ptheta);
                s_hit.pt     = s_prev.pt;
                s_hit.pphi   = s_prev.pphi;

                const double th_eps = 1e-6;
                const double q_dir = (s.theta - s_prev.theta);
                if (std::abs(s_hit.theta - M_PI/2.0) < 1e-5)
                    s_hit.theta += (q_dir >= 0.0 ? th_eps : -th_eps);
                s_hit.theta = clamp(s_hit.theta, th_eps, M_PI - th_eps);

                IntegratorControls tail_ctl = ctl;
                tail_ctl.step_init = std::max(1e-10, step_used * std::max(1e-3, 1.0 - alpha));
                TraceResult behind = trace_terminal_no_disk(s_hit, g, r_escape, intg, tail_ctl);
                out.behind_out = behind.out;
                out.theta_esc = behind.theta_esc;
                out.phi_esc = behind.phi_esc;
            }
            return out;
        }
        if (best_event == StepEvent::HORIZON) {
            const double r_h = s_prev.r + best_alpha * (s.r - s_prev.r);
            return {Outcome::HORIZON, r_h, 0.0};
        }
        if (best_event == StepEvent::ESCAPE) {
            const double th_esc = s_prev.theta + best_alpha * (s.theta - s_prev.theta);
            const double ph_esc = s_prev.phi   + best_alpha * (s.phi   - s_prev.phi);
            return {Outcome::ESCAPED, r_escape, 1.0, 0.0, th_esc, ph_esc};
        }
    }
    return {Outcome::ESCAPED, s.r, 1.0, 0.0, s.theta, s.phi};
}
