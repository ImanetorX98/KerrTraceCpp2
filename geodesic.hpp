#pragma once
// ============================================================
//  Geodesic integrator — Hamiltonian formulation
//
//  State vector for BL chart:
//    s = (r, θ, p_r, p_θ)
//    conserved: p_t = −E (energy), p_φ = L (ang. momentum)
//
//  Hamilton's equations:
//    dr/dλ   = ∂H/∂p_r   = g^rr p_r
//    dθ/dλ   = ∂H/∂p_θ   = g^θθ p_θ
//    dp_r/dλ = −∂H/∂r    (numerical diff)
//    dp_θ/dλ = −∂H/∂θ    (numerical diff)
//
//  Two adaptive integrators available (select via Integrator enum):
//
//  RK4_DOUBLING  — classic RK4 + step-doubling (Richardson error).
//                  12 RHS evals per accepted step.
//                  Simple, robust, good default.
//
//  DOPRI5        — Dormand-Prince RK45 (same as MATLAB ode45).
//                  Embedded 4th/5th order pair; error = y5 − y4.
//                  6 RHS evals per step (FSAL: last k reused as
//                  first k of next step → effectively 5 net evals).
//                  More efficient for smooth geodesics far from BH.
// ============================================================
#include "knds_metric.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

// ── Reduced BL state ─────────────────────────────────────────
struct GeodesicState {
    double r, theta, phi; ///< position  (BL coordinates; φ integrated for background)
    double pr, ptheta;    ///< covariant momenta
    double pt, pphi;      ///< conserved (set once at initialisation)
};

// ── φ velocity: dφ/dλ = g^{φt} p_t + g^{φφ} p_φ ─────────────
inline double dphi_vel(const KNdSMetric& g, double r, double th,
                       double pt, double pphi) {
    double gUU[4][4];
    g.contravariant_BL(r, th, gUU);
    return gUU[3][0]*pt + gUU[3][3]*pphi;
}

// ── RHS via Hamiltonian (numerical ∂H/∂r, ∂H/∂θ) ─────────────
inline void geodesic_rhs(const KNdSMetric& g,
                         double r,  double theta,
                         double pr, double ptheta,
                         double pt, double pphi,
                         double& dr, double& dth,
                         double& dpr, double& dpth) {
    double gUU[4][4];
    g.contravariant_BL(r, theta, gUU);

    // dr/dλ = g^rr p_r ,   dθ/dλ = g^θθ p_θ
    dr  = gUU[1][1] * pr;
    dth = gUU[2][2] * ptheta;

    // Finite differences for −∂H/∂r and −∂H/∂θ
    // Step sizes scale with local radius to keep relative accuracy
    const double eps_r  = 1e-5 * std::max(std::abs(r),  0.01);
    const double eps_th = 1e-6;

    dpr  = -(g.hamiltonian(r+eps_r, theta, pr, ptheta, pt, pphi)
           - g.hamiltonian(r-eps_r, theta, pr, ptheta, pt, pphi)) / (2.0*eps_r);

    dpth = -(g.hamiltonian(r, theta+eps_th, pr, ptheta, pt, pphi)
           - g.hamiltonian(r, theta-eps_th, pr, ptheta, pt, pphi)) / (2.0*eps_th);
}

// ── Classic RK4 single step ───────────────────────────────────
inline void rk4_step(const KNdSMetric& g, GeodesicState& s, double dlam) {
    double dr1,dth1,dpr1,dpth1;
    double dr2,dth2,dpr2,dpth2;
    double dr3,dth3,dpr3,dpth3;
    double dr4,dth4,dpr4,dpth4;

    const double r=s.r, th=s.theta, pr=s.pr, pth=s.ptheta;
    const double pt=s.pt, pp=s.pphi;

    geodesic_rhs(g, r,              th,              pr,              pth,              pt,pp, dr1,dth1,dpr1,dpth1);
    geodesic_rhs(g, r+.5*dlam*dr1, th+.5*dlam*dth1, pr+.5*dlam*dpr1,pth+.5*dlam*dpth1,pt,pp, dr2,dth2,dpr2,dpth2);
    geodesic_rhs(g, r+.5*dlam*dr2, th+.5*dlam*dth2, pr+.5*dlam*dpr2,pth+.5*dlam*dpth2,pt,pp, dr3,dth3,dpr3,dpth3);
    geodesic_rhs(g, r+   dlam*dr3, th+   dlam*dth3, pr+   dlam*dpr3, pth+  dlam*dpth3, pt,pp, dr4,dth4,dpr4,dpth4);

    // φ velocity at each stage (dφ/dλ = g^{φt} p_t + g^{φφ} p_φ)
    const double dp1 = dphi_vel(g, r,              th,              pt, pp);
    const double dp2 = dphi_vel(g, r+.5*dlam*dr1, th+.5*dlam*dth1, pt, pp);
    const double dp3 = dphi_vel(g, r+.5*dlam*dr2, th+.5*dlam*dth2, pt, pp);
    const double dp4 = dphi_vel(g, r+   dlam*dr3, th+   dlam*dth3, pt, pp);

    s.r      += dlam/6.0*(dr1   + 2*dr2   + 2*dr3   + dr4);
    s.theta  += dlam/6.0*(dth1  + 2*dth2  + 2*dth3  + dth4);
    s.phi    += dlam/6.0*(dp1   + 2*dp2   + 2*dp3   + dp4);
    s.pr     += dlam/6.0*(dpr1  + 2*dpr2  + 2*dpr3  + dpr4);
    s.ptheta += dlam/6.0*(dpth1 + 2*dpth2 + 2*dpth3 + dpth4);
}

// ── Adaptive RK4 — step doubling (Richardson extrapolation) ──
//
//  Error estimate = |full_step − two_half_steps| / 15
//  Accept:  use the more accurate two-half-step result,
//           grow step  ∝ (tol/err)^{1/5}
//  Reject:  shrink step ∝ (tol/err)^{1/4}, retry
//
//  Returns true when step accepted.
inline bool rk4_adaptive(const KNdSMetric& g, GeodesicState& s,
                         double& dlam, double tol = 1e-7) {
    // Save state
    const GeodesicState s0 = s;

    // Path A: one full step
    GeodesicState sA = s0;
    rk4_step(g, sA, dlam);

    // Path B: two half steps
    GeodesicState sB = s0;
    rk4_step(g, sB, 0.5*dlam);
    rk4_step(g, sB, 0.5*dlam);

    // Euclidean error in the 4D phase-space (r,θ,p_r,p_θ)
    const double err = std::sqrt(
        (sA.r      - sB.r)      * (sA.r      - sB.r)      +
        (sA.theta  - sB.theta)  * (sA.theta  - sB.theta)  +
        (sA.pr     - sB.pr)     * (sA.pr     - sB.pr)     +
        (sA.ptheta - sB.ptheta) * (sA.ptheta - sB.ptheta)
    ) / 15.0;   // Richardson factor 2^4 − 1

    if (!std::isfinite(err)) {
        // Degenerate step (typically near coordinate singularities): back off.
        s = s0;
        dlam = (std::isfinite(dlam) && dlam > 1e-10) ? dlam * 0.5 : 1e-6;
        if (dlam < 1e-10) dlam = 1e-10;
        return false;
    }

    const bool accepted = (err < tol || dlam < 1e-10);

    if (accepted) {
        s = sB;   // use higher-order result
        const double scale = (err > 1e-14)
                            ? 0.9 * std::pow(tol/err, 0.2)
                            : 4.0;
        double hnew = dlam * scale;
        if (!std::isfinite(hnew)) hnew = dlam;
        dlam = hnew;
        if (dlam > 100.0) dlam = 100.0;
        if (dlam < 1e-10) dlam = 1e-10;
    } else {
        // Reject: restore state and shrink step
        s = s0;
        const double half = dlam * 0.5;
        double hnew = dlam * 0.9 * std::pow(tol/err, 0.25);
        if (!std::isfinite(hnew)) hnew = half;
        dlam = hnew;
        if (dlam > half)  dlam = half;
        if (dlam < 1e-10) dlam = 1e-10;
    }
    return accepted;
}

// ── Dormand-Prince RK45 (DOPRI5) ─────────────────────────────
//
//  Butcher tableau (Dormand & Prince, 1980):
//
//   c  │ a
//  ────┼─────────────────────���──────────────────────────────
//   0  │
//  1/5 │ 1/5
//  3/10│ 3/40        9/40
//  4/5 │ 44/45      −56/15       32/9
//  8/9 │ 19372/6561 −25360/2187  64448/6561  −212/729
//   1  │ 9017/3168  −355/33      46732/5247   49/176  −5103/18656
//   1  │ 35/384      0           500/1113     125/192 −2187/6784   11/84
//  ────┼────────────────────────────────────────────────────
//  b5* │ 35/384      0           500/1113     125/192 −2187/6784   11/84    0
//  b4  │ 5179/57600  0           7571/16695   393/640 −92097/339200 187/2100 1/40
//
//  Error = y5* − y4  (embedded; no extra evaluations)
//  FSAL:  k7 = f(t+h, y5*)  →  reused as k1 of the next step.
//
//  Returns true when step accepted; dlam updated in place.
//  The caller must pass in k1 (= f at start of step) and receives
//  k7_out (= f at accepted end) to feed as k1 of the next call.
//  On first call set k1_fsal = {NaN,…} to trigger recomputation.

struct Vec4d {
    double v[4];
    static Vec4d nan_init() {
        Vec4d x;
        x.v[0] = x.v[1] = x.v[2] = x.v[3] = std::numeric_limits<double>::quiet_NaN();
        return x;
    }
};

inline void eval_rhs(const KNdSMetric& g, const GeodesicState& s,
                     Vec4d& k) {
    geodesic_rhs(g, s.r, s.theta, s.pr, s.ptheta, s.pt, s.pphi,
                 k.v[0], k.v[1], k.v[2], k.v[3]);
}

// Advance state by weighted sum of stages
inline GeodesicState advance(const GeodesicState& s0, double h,
                              const Vec4d* ks, const double* w, int nk) {
    GeodesicState s = s0;
    for (int i = 0; i < nk; ++i) {
        s.r      += h * w[i] * ks[i].v[0];
        s.theta  += h * w[i] * ks[i].v[1];
        s.pr     += h * w[i] * ks[i].v[2];
        s.ptheta += h * w[i] * ks[i].v[3];
    }
    return s;
}

inline bool dopri5_adaptive(const KNdSMetric& g, GeodesicState& s,
                             double& h, Vec4d& k1_fsal,
                             double tol = 1e-7) {
    // k1: reuse FSAL from previous step if valid
    Vec4d k1;
    if (!std::isnan(k1_fsal.v[0])) {
        k1 = k1_fsal;
    } else {
        eval_rhs(g, s, k1);
    }

    // ── Stage evaluations ─────────────────���────────────────
    // k2
    const double a21 = 1.0/5.0;
    const GeodesicState s2 = advance(s, h, &k1, (const double[]){a21}, 1);
    Vec4d k2; eval_rhs(g, s2, k2);

    // k3
    const double a3[] = {3.0/40.0, 9.0/40.0};
    const Vec4d ks3[] = {k1, k2};
    const GeodesicState s3 = advance(s, h, ks3, a3, 2);
    Vec4d k3; eval_rhs(g, s3, k3);

    // k4
    const double a4[] = {44.0/45.0, -56.0/15.0, 32.0/9.0};
    const Vec4d ks4[] = {k1, k2, k3};
    const GeodesicState s4 = advance(s, h, ks4, a4, 3);
    Vec4d k4; eval_rhs(g, s4, k4);

    // k5
    const double a5[] = {19372.0/6561.0, -25360.0/2187.0,
                          64448.0/6561.0,   -212.0/729.0};
    const Vec4d ks5[] = {k1, k2, k3, k4};
    const GeodesicState s5 = advance(s, h, ks5, a5, 4);
    Vec4d k5; eval_rhs(g, s5, k5);

    // k6
    const double a6[] = {9017.0/3168.0, -355.0/33.0,
                          46732.0/5247.0,    49.0/176.0, -5103.0/18656.0};
    const Vec4d ks6[] = {k1, k2, k3, k4, k5};
    const GeodesicState s6 = advance(s, h, ks6, a6, 5);
    Vec4d k6; eval_rhs(g, s6, k6);

    // ── 5th-order solution  y5  (= FSAL k7 base) ──────────
    const double b5[] = {35.0/384.0,   0.0,  500.0/1113.0,
                          125.0/192.0, -2187.0/6784.0, 11.0/84.0};
    const Vec4d  ks_b5[] = {k1, k2, k3, k4, k5, k6};
    const GeodesicState y5 = advance(s, h, ks_b5, b5, 6);

    // FSAL: k7 = f(y5)
    Vec4d k7; eval_rhs(g, y5, k7);

    // ── 4th-order solution  y4 ────────────────────────────
    const double b4[] = {5179.0/57600.0,   0.0,   7571.0/16695.0,
                          393.0/640.0, -92097.0/339200.0,
                          187.0/2100.0,   1.0/40.0};
    const Vec4d ks_b4[] = {k1, k2, k3, k4, k5, k6, k7};
    const GeodesicState y4 = advance(s, h, ks_b4, b4, 7);

    // ── Error: ||y5 − y4||  (mixed absolute/relative) ─────
    auto sc = [&](double y, double /*d*/) {
        return tol * (1.0 + std::abs(y));   // scale ~ atol + |y|*rtol
    };
    double err2 = 0.0;
    double dy[4] = { y5.r-y4.r, y5.theta-y4.theta,
                     y5.pr-y4.pr, y5.ptheta-y4.ptheta };
    double ref[4] = { s.r, s.theta, s.pr, s.ptheta };
    for (int i = 0; i < 4; ++i) {
        double ratio = dy[i] / sc(ref[i], dy[i]);
        err2 += ratio*ratio;
    }
    const double err = std::sqrt(err2 / 4.0);

    if (!std::isfinite(err)) {
        h = (std::isfinite(h) && h > 1e-10) ? h * 0.5 : 1e-6;
        if (h < 1e-10) h = 1e-10;
        k1_fsal.v[0] = std::numeric_limits<double>::quiet_NaN();
        return false;
    }

    const bool accepted = (err <= 1.0 || h < 1e-10);

    if (accepted) {
        s        = y5;          // accept 5th-order result
        k1_fsal  = k7;          // FSAL carry-over
        // PI controller step growth (Hairer §II.4)
        const double fac = (err > 1e-12)
                           ? 0.9 * std::pow(1.0/err, 0.2)
                           : 4.0;
        double hnew = h * fac;
        if (!std::isfinite(hnew)) hnew = h;
        if (hnew > 100.0) hnew = 100.0;
        if (hnew < 1e-10) hnew = 1e-10;
        h = hnew;
    } else {
        // Reject; shrink and do NOT advance state
        double hnew = h * 0.9 * std::pow(1.0/err, 0.25);
        if (!std::isfinite(hnew)) hnew = h * 0.5;
        h = std::max(hnew, 1e-10);
        k1_fsal.v[0] = std::numeric_limits<double>::quiet_NaN(); // force recompute
    }
    return accepted;
}

// ── Integrator selector ───────────────────────────────────────
enum class Integrator { RK4_DOUBLING, DOPRI5 };

// Unified adaptive step: hides the FSAL bookkeeping from callers
// that just want a drop-in replacement for rk4_adaptive().
// For DOPRI5 the caller must keep a persistent `fsal` Vec4d,
// initialised with NaN to signal "no previous step yet".
inline bool adaptive_step(const KNdSMetric& g, GeodesicState& s,
                           double& h, Integrator intg,
                           Vec4d& fsal,        // only used by DOPRI5
                           double tol = 1e-7) {
    if (intg == Integrator::DOPRI5)
        return dopri5_adaptive(g, s, h, fsal, tol);
    else
        return rk4_adaptive(g, s, h, tol);
}

// ── Null constraint monitor  (diagnostic only) ────────────────
//  H should remain ≈ 0; drift indicates accumulation error.
inline double null_residual(const KNdSMetric& g, const GeodesicState& s) {
    return std::abs(g.hamiltonian(s.r, s.theta, s.pr, s.ptheta, s.pt, s.pphi));
}
