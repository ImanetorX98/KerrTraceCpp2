#pragma once
// Kerr-Schild Cartesian tracer, horizon-penetrating.
//
// Part of the single-translation-unit renderer included by main.cpp;
// see docs/ARCHITETTURA.md for how the modules fit together.

#include "tracer_bl.hpp"

// ── KS single-ray tracing (CPU) ───────────────────────────────
// Uses Kerr-Schild Cartesian coordinates (Lambda=0 chart) with
// numerical Hamiltonian gradients in (X,Y,Z).
struct KSState {
    double X, Y, Z;
    double pX, pY, pZ;
    double pT; // conserved (stationary metric)
};

static constexpr bool KS_INGOING = true;

static bool solve_3x3(const double A_in[3][3], const double b_in[3], double x[3]) {
    double A[3][4] = {
        {A_in[0][0], A_in[0][1], A_in[0][2], b_in[0]},
        {A_in[1][0], A_in[1][1], A_in[1][2], b_in[1]},
        {A_in[2][0], A_in[2][1], A_in[2][2], b_in[2]},
    };

    for (int col = 0; col < 3; ++col) {
        int piv = col;
        double best = std::abs(A[piv][col]);
        for (int r = col + 1; r < 3; ++r) {
            const double v = std::abs(A[r][col]);
            if (v > best) { best = v; piv = r; }
        }
        if (best < 1e-14) return false;
        if (piv != col) {
            for (int k = col; k < 4; ++k) std::swap(A[col][k], A[piv][k]);
        }

        const double inv = 1.0 / A[col][col];
        for (int k = col; k < 4; ++k) A[col][k] *= inv;
        for (int r = 0; r < 3; ++r) {
            if (r == col) continue;
            const double f = A[r][col];
            for (int k = col; k < 4; ++k) A[r][k] -= f * A[col][k];
        }
    }

    x[0] = A[0][3];
    x[1] = A[1][3];
    x[2] = A[2][3];
    return true;
}

static void jacobian_bl_to_ks(double r, double theta, double phi, double a_spin, double J[3][3]) {
    const double er  = 1e-6 * (std::abs(r) + 1.0);
    const double eth = 1e-6;
    const double eph = 1e-6;

    auto f = [&](double rr, double tt, double pp) {
        double X, Y, Z;
        KNdSMetric::BL_to_KS_spatial(rr, tt, pp, a_spin, X, Y, Z);
        return std::array<double, 3>{X, Y, Z};
    };

    const auto xr_p = f(r + er, theta, phi);
    const auto xr_m = f(r - er, theta, phi);
    const auto xt_p = f(r, theta + eth, phi);
    const auto xt_m = f(r, theta - eth, phi);
    const auto xp_p = f(r, theta, phi + eph);
    const auto xp_m = f(r, theta, phi - eph);

    for (int i = 0; i < 3; ++i) {
        J[i][0] = (xr_p[i] - xr_m[i]) / (2.0 * er);   // d(X,Y,Z)/dr
        J[i][1] = (xt_p[i] - xt_m[i]) / (2.0 * eth);  // d(X,Y,Z)/dtheta
        J[i][2] = (xp_p[i] - xp_m[i]) / (2.0 * eph);  // d(X,Y,Z)/dphi
    }
}

// Ingoing Kerr-Schild is NOT a purely spatial relabelling of Boyer-Lindquist:
//
//     dT       = dt   + (2Mr - Q^2)/Delta_r dr
//     dphi_KS  = dphi + a/Delta_r          dr
//
// so both the BL time and the BL azimuth depend on the KS *spatial* coordinates
// through r(X,Y,Z). The chain rule for the covector therefore carries two extra
// terms along dr/dX^i:
//
//     p_{X^i} = [p_r - F' p_t - G' p_phi] dr/dX^i
//             + p_theta dtheta/dX^i + p_phi dphi_KS/dX^i ,
//     F' = (2Mr - Q^2)/Delta_r ,  G' = a/Delta_r ,
//
// which is the same 3x3 Jacobian as before applied to a corrected radial
// component. Passing the raw p_r instead put a spurious radial momentum into the
// KS ray: at r = 40M, a = 0, the missing F' p_t term is 5.3% of E, and the KS
// chart rendered the Schwarzschild shadow 4.88% too large (7.6289 deg against the
// analytic 7.2740 deg) while BL got it to 0.05%. The error was independent of the
// integrator tolerance, because it is in the initial data, not the integration.
// The azimuth of the Kerr-Schild chart is not the Boyer-Lindquist one either:
// dphi_KS = dphi + (a/Delta_r) dr integrates to
//
//     phi_KS = phi_BL + G(r),   G(r) = a/(r+ - r-) ln|(r - r+)/(r - r-)| ,
//
// with the extremal limit G(r) = -a/(r - M) when r+ = r-. BL_to_KS_spatial takes
// the KS azimuth (its embedding X = sin(theta)(r cos phi - a sin phi) is the
// standard Kerr-Schild one), so a BL azimuth must be twisted on the way in and
// untwisted on the way out. Without this the two charts disagreed on phi_disk by
// 0.155 rad at a = 0.998 -- zero at a = 0, growing with spin -- which rotated the
// procedural disk textures and the star field between charts. G vanishes at
// infinity, so the effect on an escaping ray is the camera-radius value alone.
static double ks_azimuth_twist(double r, double a_spin, double M, double Q) {
    if (std::abs(a_spin) < 1e-15) return 0.0;
    const double disc = M*M - a_spin*a_spin - Q*Q;
    if (disc <= 1e-14) {                      // extremal (and near-extremal) limit
        const double d = r - M;
        return (std::abs(d) > 1e-12) ? (-a_spin / d) : 0.0;
    }
    const double root = std::sqrt(disc);
    const double rp = M + root, rm = M - root;
    const double num = r - rp, den = r - rm;
    if (!(std::abs(num) > 1e-12) || !(std::abs(den) > 1e-12)) return 0.0;
    return a_spin / (rp - rm) * std::log(std::abs(num / den));
}

static double ks_radial_covector_shift(double r, double a_spin, double M, double Q,
                                       double pt, double pphi) {
    const double Delta = r*r - 2.0*M*r + a_spin*a_spin + Q*Q;
    if (!(std::abs(Delta) > 1e-14)) return 0.0;
    const double Fp = (2.0*M*r - Q*Q) / Delta;   // dT/dr   at fixed t
    const double Gp = a_spin / Delta;            // dphi_KS/dr at fixed phi_BL
    return -Fp*pt - Gp*pphi;
}

static bool bl_covector_to_ks(double r, double theta, double phi, double a_spin,
                              double pr, double ptheta, double pphi,
                              double& pX, double& pY, double& pZ) {
    double J[3][3];
    jacobian_bl_to_ks(r, theta, phi, a_spin, J);

    // p_q = J^T p_xyz
    double A[3][3];
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i)
            A[j][i] = J[i][j];
    const double b[3] = {pr, ptheta, pphi};
    double x[3];
    if (!solve_3x3(A, b, x)) return false;
    pX = x[0]; pY = x[1]; pZ = x[2];
    return true;
}

static std::array<double, 3> ks_covector_to_bl(double r, double theta, double phi, double a_spin,
                                                double pX, double pY, double pZ) {
    double J[3][3];
    jacobian_bl_to_ks(r, theta, phi, a_spin, J);
    std::array<double, 3> pbl{};
    const double pxyz[3] = {pX, pY, pZ};
    // p_q = J^T p_xyz
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i)
            pbl[j] += J[i][j] * pxyz[i];
    return pbl;
}

static bool init_ks_state(const GeodesicState& s_bl, const KNdSMetric& g, KSState& s_ks) {
    if (std::abs(g.Lambda) > 1e-15) return false;

    const double phi_ks = s_bl.phi + ks_azimuth_twist(s_bl.r, g.a, g.M, g.Q);
    KNdSMetric::BL_to_KS_spatial(s_bl.r, s_bl.theta, phi_ks, g.a, s_ks.X, s_ks.Y, s_ks.Z);
    const double pr_ks = s_bl.pr + ks_radial_covector_shift(s_bl.r, g.a, g.M, g.Q,
                                                            s_bl.pt, s_bl.pphi);
    // The Jacobian is d(X,Y,Z)/d(r,theta,phi_KS), so it is evaluated at the KS
    // azimuth; the twist between the two azimuths is already carried by the
    // -G' p_phi term inside ks_radial_covector_shift.
    if (!bl_covector_to_ks(s_bl.r, s_bl.theta, phi_ks, g.a,
                           pr_ks, s_bl.ptheta, s_bl.pphi,
                           s_ks.pX, s_ks.pY, s_ks.pZ)) {
        return false;
    }

    double gUU[4][4];
    g.contravariant_KS(0.0, s_ks.X, s_ks.Y, s_ks.Z, KS_INGOING, gUU);
    const double A = gUU[0][0];
    const double B = 2.0 * (gUU[0][1] * s_ks.pX + gUU[0][2] * s_ks.pY + gUU[0][3] * s_ks.pZ);
    const double C =
        gUU[1][1] * s_ks.pX * s_ks.pX + gUU[2][2] * s_ks.pY * s_ks.pY + gUU[3][3] * s_ks.pZ * s_ks.pZ +
        2.0 * gUU[1][2] * s_ks.pX * s_ks.pY + 2.0 * gUU[1][3] * s_ks.pX * s_ks.pZ + 2.0 * gUU[2][3] * s_ks.pY * s_ks.pZ;
    const double disc = B * B - 4.0 * A * C;
    if (disc < 0.0 || std::abs(A) < 1e-15) return false;

    const double sq = std::sqrt(disc);
    const double pT1 = (-B - sq) / (2.0 * A);
    const double pT2 = (-B + sq) / (2.0 * A);
    s_ks.pT = (pT1 < 0.0) ? pT1 : pT2;
    if (s_ks.pT > 0.0) s_ks.pT = std::min(pT1, pT2);
    return std::isfinite(s_ks.pT);
}

static void geodesic_rhs_ks(const KNdSMetric& g, const KSState& s,
                            double& dX, double& dY, double& dZ,
                            double& dpX, double& dpY, double& dpZ) {
    double gUU[4][4];
    g.contravariant_KS(0.0, s.X, s.Y, s.Z, KS_INGOING, gUU);

    dX = gUU[1][0] * s.pT + gUU[1][1] * s.pX + gUU[1][2] * s.pY + gUU[1][3] * s.pZ;
    dY = gUU[2][0] * s.pT + gUU[2][1] * s.pX + gUU[2][2] * s.pY + gUU[2][3] * s.pZ;
    dZ = gUU[3][0] * s.pT + gUU[3][1] * s.pX + gUU[3][2] * s.pY + gUU[3][3] * s.pZ;

    auto Hxyz = [&](double X, double Y, double Z) {
        double p[4] = {s.pT, s.pX, s.pY, s.pZ};
        return g.hamiltonian_KS(0.0, X, Y, Z, p, KS_INGOING);
    };

    const double eX = 1e-5 * (std::abs(s.X) + 0.1);
    const double eY = 1e-5 * (std::abs(s.Y) + 0.1);
    const double eZ = 1e-5 * (std::abs(s.Z) + 0.1);

    dpX = -(Hxyz(s.X + eX, s.Y, s.Z) - Hxyz(s.X - eX, s.Y, s.Z)) / (2.0 * eX);
    dpY = -(Hxyz(s.X, s.Y + eY, s.Z) - Hxyz(s.X, s.Y - eY, s.Z)) / (2.0 * eY);
    dpZ = -(Hxyz(s.X, s.Y, s.Z + eZ) - Hxyz(s.X, s.Y, s.Z - eZ)) / (2.0 * eZ);
}

static void rk4_step_ks(const KNdSMetric& g, KSState& s, double dlam) {
    double dX1, dY1, dZ1, dpX1, dpY1, dpZ1;
    double dX2, dY2, dZ2, dpX2, dpY2, dpZ2;
    double dX3, dY3, dZ3, dpX3, dpY3, dpZ3;
    double dX4, dY4, dZ4, dpX4, dpY4, dpZ4;

    const KSState s0 = s;

    geodesic_rhs_ks(g, s0, dX1, dY1, dZ1, dpX1, dpY1, dpZ1);
    KSState s2{s0.X + 0.5 * dlam * dX1, s0.Y + 0.5 * dlam * dY1, s0.Z + 0.5 * dlam * dZ1,
               s0.pX + 0.5 * dlam * dpX1, s0.pY + 0.5 * dlam * dpY1, s0.pZ + 0.5 * dlam * dpZ1,
               s0.pT};
    geodesic_rhs_ks(g, s2, dX2, dY2, dZ2, dpX2, dpY2, dpZ2);

    KSState s3{s0.X + 0.5 * dlam * dX2, s0.Y + 0.5 * dlam * dY2, s0.Z + 0.5 * dlam * dZ2,
               s0.pX + 0.5 * dlam * dpX2, s0.pY + 0.5 * dlam * dpY2, s0.pZ + 0.5 * dlam * dpZ2,
               s0.pT};
    geodesic_rhs_ks(g, s3, dX3, dY3, dZ3, dpX3, dpY3, dpZ3);

    KSState s4{s0.X + dlam * dX3, s0.Y + dlam * dY3, s0.Z + dlam * dZ3,
               s0.pX + dlam * dpX3, s0.pY + dlam * dpY3, s0.pZ + dlam * dpZ3,
               s0.pT};
    geodesic_rhs_ks(g, s4, dX4, dY4, dZ4, dpX4, dpY4, dpZ4);

    s.X  += dlam / 6.0 * (dX1 + 2.0 * dX2 + 2.0 * dX3 + dX4);
    s.Y  += dlam / 6.0 * (dY1 + 2.0 * dY2 + 2.0 * dY3 + dY4);
    s.Z  += dlam / 6.0 * (dZ1 + 2.0 * dZ2 + 2.0 * dZ3 + dZ4);
    s.pX += dlam / 6.0 * (dpX1 + 2.0 * dpX2 + 2.0 * dpX3 + dpX4);
    s.pY += dlam / 6.0 * (dpY1 + 2.0 * dpY2 + 2.0 * dpY3 + dpY4);
    s.pZ += dlam / 6.0 * (dpZ1 + 2.0 * dpZ2 + 2.0 * dpZ3 + dpZ4);
}

static bool rk4_adaptive_ks(const KNdSMetric& g, KSState& s, double& dlam, double tol = 1e-7) {
    const KSState s0 = s;
    KSState sA = s0;
    rk4_step_ks(g, sA, dlam);
    KSState sB = s0;
    rk4_step_ks(g, sB, 0.5 * dlam);
    rk4_step_ks(g, sB, 0.5 * dlam);

    const double err = std::sqrt(
        (sA.X  - sB.X)  * (sA.X  - sB.X)  +
        (sA.Y  - sB.Y)  * (sA.Y  - sB.Y)  +
        (sA.Z  - sB.Z)  * (sA.Z  - sB.Z)  +
        (sA.pX - sB.pX) * (sA.pX - sB.pX) +
        (sA.pY - sB.pY) * (sA.pY - sB.pY) +
        (sA.pZ - sB.pZ) * (sA.pZ - sB.pZ)) / 15.0;

    if (!std::isfinite(err)) {
        s = s0;
        dlam = (std::isfinite(dlam) && dlam > 1e-10) ? dlam * 0.5 : 1e-6;
        if (dlam < 1e-10) dlam = 1e-10;
        return false;
    }

    const bool accepted = (err < tol || dlam < 1e-10);
    if (accepted) {
        s = sB;
        const double scale = (err > 1e-14) ? 0.9 * std::pow(tol / err, 0.2) : 4.0;
        double hnew = dlam * scale;
        if (!std::isfinite(hnew)) hnew = dlam;
        dlam = clamp(hnew, 1e-10, 100.0);
    } else {
        s = s0;
        const double half = dlam * 0.5;
        double hnew = dlam * 0.9 * std::pow(tol / err, 0.25);
        if (!std::isfinite(hnew)) hnew = half;
        dlam = hnew;
        if (dlam > half) dlam = half;
        if (dlam < 1e-10) dlam = 1e-10;
    }
    return accepted;
}

static TraceResult trace_single_ks(GeodesicState s_bl, const KNdSMetric& g,
                                   double r_disk_in, double r_disk_out,
                                   double r_escape,
                                   Integrator intg,
                                   const IntegratorControls& ctl,
                                   const ColorParams* cp_ptr) {
    const double observer_frequency = -s_bl.pt * g.static_observer_ut(s_bl.r, s_bl.theta);
    KSState s{};
    if (!init_ks_state(s_bl, g, s))
        return trace_single(s_bl, g, r_disk_in, r_disk_out, r_escape, intg, ctl, cp_ptr);

    if (intg == Integrator::DOPRI5) {
        static bool warned_dopri5_ks = false;
        if (!warned_dopri5_ks) {
            std::cerr << "Info: KS chart currently uses RK4-doubling; ignoring --dopri5.\n";
            warned_dopri5_ks = true;
        }
    }

    const double rh = g.r_horizon();
    const double rh_cut = rh * 1.03;
    double dlam = std::max(ctl.step_init, 1e-10);
    double prev_r_ks = KNdSMetric::r_KS(s.X, s.Y, s.Z, g.a);

    const int max_steps = std::max(1, ctl.max_steps);
    for (int it = 0; it < max_steps; ++it) {
        const KSState s_prev = s;
        double step_used = dlam;
        int rejects = 0;
        while (true) {
            step_used = dlam;
            if (rk4_adaptive_ks(g, s, dlam, ctl.tol)) break;
            if (!std::isfinite(dlam) || ++rejects > 64) {
                double r_tmp, th_tmp, ph_tmp;
                KNdSMetric::KS_to_BL_spatial(s.X, s.Y, s.Z, g.a, r_tmp, th_tmp, ph_tmp);
                ph_tmp -= ks_azimuth_twist(r_tmp, g.a, g.M, g.Q);
                return {Outcome::ESCAPED, r_tmp, 1.0, 0.0, th_tmp, ph_tmp};
            }
        }

        const double r_now = KNdSMetric::r_KS(s.X, s.Y, s.Z, g.a);
        double best_alpha = 2.0;
        enum class StepEvent { NONE, DISK, HORIZON, ESCAPE };
        StepEvent best_event = StepEvent::NONE;
        double disk_r_hit = 0.0;
        double disk_red_hit = 1.0;
        double disk_phi_hit = 0.0;

        const bool maybe_equator = sign_change(s_prev.Z, s.Z) ||
                                   (std::min(std::abs(s_prev.Z), std::abs(s.Z)) < 0.35);
        if (maybe_equator) {
            double dX0=0.0, dY0=0.0, dZ0=0.0, dpX0=0.0, dpY0=0.0, dpZ0=0.0;
            double dX1=0.0, dY1=0.0, dZ1=0.0, dpX1=0.0, dpY1=0.0, dpZ1=0.0;
            geodesic_rhs_ks(g, s_prev, dX0, dY0, dZ0, dpX0, dpY0, dpZ0);
            geodesic_rhs_ks(g, s,      dX1, dY1, dZ1, dpX1, dpY1, dpZ1);

            double alpha = 0.0;
            bool has_equator_hit = first_event_alpha_hermite(
                s_prev.Z, s.Z, dZ0, dZ1, step_used, 0.0, alpha, 8, 8);
            if (!has_equator_hit && sign_change(s_prev.Z, s.Z)) {
                // Fallback: if Hermite misses a genuine sign-crossing,
                // recover with linear interpolation instead of losing the event.
                const double denom = s_prev.Z - s.Z;
                alpha = (std::abs(denom) > 1e-12) ? (s_prev.Z / denom) : 0.5;
                alpha = clamp(alpha, 0.0, 1.0);
                has_equator_hit = true;
            }
            if (has_equator_hit) {
                const double Xh = hermite_interp_scalar(s_prev.X,  s.X,  dX0,  dX1,  step_used, alpha);
                const double Yh = hermite_interp_scalar(s_prev.Y,  s.Y,  dY0,  dY1,  step_used, alpha);
                const double Zh = hermite_interp_scalar(s_prev.Z,  s.Z,  dZ0,  dZ1,  step_used, alpha);
                const double pXh = hermite_interp_scalar(s_prev.pX, s.pX, dpX0, dpX1, step_used, alpha);
                const double pYh = hermite_interp_scalar(s_prev.pY, s.pY, dpY0, dpY1, step_used, alpha);
                const double pZh = hermite_interp_scalar(s_prev.pZ, s.pZ, dpZ0, dpZ1, step_used, alpha);
                const double pTh = s_prev.pT;

                double r_hit, th_hit, ph_hit;
                KNdSMetric::KS_to_BL_spatial(Xh, Yh, Zh, g.a, r_hit, th_hit, ph_hit);
                // ph_hit is the KS azimuth: the Jacobian inside ks_covector_to_bl
                // wants it as such, everything downstream wants the BL one.
                const double ph_hit_bl = ph_hit - ks_azimuth_twist(r_hit, g.a, g.M, g.Q);
                if (r_hit >= r_disk_in && r_hit <= r_disk_out) {
                    const auto pbl = ks_covector_to_bl(r_hit, th_hit, ph_hit, g.a, pXh, pYh, pZh);
                    bool pass_through = false;
                    if (cp_ptr && cp_ptr->palette == DiskPalette::STRATIFIED) {
                        pass_through = stratified_disk_tile_transparent(
                            r_hit, ph_hit_bl, r_disk_in, r_disk_out, *cp_ptr);
                    } else if (cp_ptr && (cp_ptr->palette == DiskPalette::INTERSTELLAR ||
                                          cp_ptr->palette == DiskPalette::INTERSTELLAR_NASA)) {
                        pass_through = interstellar_disk_soft_mask(
                            r_hit, ph_hit_bl, r_disk_in, r_disk_out, *cp_ptr) < std::max(0.0, cp_ptr->interstellar_edge_transparency);
                    }
                    if (!pass_through) {
                        disk_r_hit  = r_hit;
                        disk_red_hit = clamp(disk_redshift(r_hit, pTh, pbl[2], g, observer_frequency), 0.0, 20.0);
                        disk_phi_hit = ph_hit_bl;
                        best_alpha  = alpha;
                        best_event  = StepEvent::DISK;
                    }
                }
            }
        }

        const bool horizon_cross = ((prev_r_ks > rh_cut) && (r_now <= rh_cut)) || (r_now <= rh_cut);
        if (horizon_cross) {
            const double denom_h = prev_r_ks - r_now;
            double alpha_h = (std::abs(denom_h) > 1e-12) ? ((prev_r_ks - rh_cut) / denom_h) : 0.0;
            alpha_h = clamp(alpha_h, 0.0, 1.0);
            if (alpha_h < best_alpha) {
                best_alpha = alpha_h;
                best_event = StepEvent::HORIZON;
            }
        }

        const bool escape_cross = ((prev_r_ks < r_escape) && (r_now >= r_escape)) || (r_now >= r_escape);
        if (escape_cross) {
            const double denom_e = r_now - prev_r_ks;
            double alpha_e = (std::abs(denom_e) > 1e-12) ? ((r_escape - prev_r_ks) / denom_e) : 1.0;
            alpha_e = clamp(alpha_e, 0.0, 1.0);
            if (alpha_e < best_alpha) {
                best_alpha = alpha_e;
                best_event = StepEvent::ESCAPE;
            }
        }

        if (best_event == StepEvent::DISK) {
            TraceResult out{Outcome::DISK_HIT, disk_r_hit, disk_red_hit, disk_phi_hit};
            const double opacity = cp_ptr ? clamp(cp_ptr->disk_opacity, 0.0, 1.0) : 1.0;
            if (opacity < (1.0 - 1e-9)) {
                const double alpha = clamp(best_alpha, 0.0, 1.0);
                const double Xh = s_prev.X + alpha * (s.X - s_prev.X);
                const double Yh = s_prev.Y + alpha * (s.Y - s_prev.Y);
                const double Zh = s_prev.Z + alpha * (s.Z - s_prev.Z);
                const double pXh = s_prev.pX + alpha * (s.pX - s_prev.pX);
                const double pYh = s_prev.pY + alpha * (s.pY - s_prev.pY);
                const double pZh = s_prev.pZ + alpha * (s.pZ - s_prev.pZ);

                double r_hit, th_hit, ph_hit;
                KNdSMetric::KS_to_BL_spatial(Xh, Yh, Zh, g.a, r_hit, th_hit, ph_hit);
                const auto pbl = ks_covector_to_bl(r_hit, th_hit, ph_hit, g.a, pXh, pYh, pZh);
                // pbl[0] is the KS radial component p_r - F' p_t - G' p_phi; undo the
                // shift to hand trace_terminal_no_disk a genuine BL momentum, and
                // untwist the azimuth for the same reason.
                const double pr_bl = pbl[0] - ks_radial_covector_shift(r_hit, g.a, g.M, g.Q,
                                                                       s_prev.pT, pbl[2]);
                const double ph_bl = ph_hit - ks_azimuth_twist(r_hit, g.a, g.M, g.Q);
                GeodesicState s_hit{r_hit, th_hit, ph_bl, pr_bl, pbl[1], s_prev.pT, pbl[2]};
                const double th_eps = 1e-6;
                if (std::abs(s_hit.theta - M_PI/2.0) < 1e-5)
                    s_hit.theta += ((s.Z - s_prev.Z) >= 0.0 ? th_eps : -th_eps);
                s_hit.theta = clamp(s_hit.theta, th_eps, M_PI - th_eps);

                IntegratorControls tail_ctl = ctl;
                tail_ctl.step_init = std::max(1e-10, step_used * std::max(1e-3, 1.0 - alpha));
                TraceResult behind = trace_terminal_no_disk(s_hit, g, r_escape, Integrator::RK4_DOUBLING, tail_ctl);
                out.behind_out = behind.out;
                out.theta_esc = behind.theta_esc;
                out.phi_esc = behind.phi_esc;
            }
            return out;
        }
        if (best_event == StepEvent::HORIZON) {
            const double r_h = prev_r_ks + best_alpha * (r_now - prev_r_ks);
            return {Outcome::HORIZON, r_h, 0.0};
        }
        if (best_event == StepEvent::ESCAPE) {
            const double X_esc = s_prev.X + best_alpha * (s.X - s_prev.X);
            const double Y_esc = s_prev.Y + best_alpha * (s.Y - s_prev.Y);
            const double Z_esc = s_prev.Z + best_alpha * (s.Z - s_prev.Z);
            double r_esc, th_esc, ph_esc;
            KNdSMetric::KS_to_BL_spatial(X_esc, Y_esc, Z_esc, g.a, r_esc, th_esc, ph_esc);
            ph_esc -= ks_azimuth_twist(r_esc, g.a, g.M, g.Q);
            return {Outcome::ESCAPED, r_esc, 1.0, 0.0, th_esc, ph_esc};
        }

        prev_r_ks = r_now;
    }

    double r_end, th_end, ph_end;
    KNdSMetric::KS_to_BL_spatial(s.X, s.Y, s.Z, g.a, r_end, th_end, ph_end);
    ph_end -= ks_azimuth_twist(r_end, g.a, g.M, g.Q);
    return {Outcome::ESCAPED, r_end, 1.0, 0.0, th_end, ph_end};
}
