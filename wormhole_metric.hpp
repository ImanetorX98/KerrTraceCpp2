#pragma once
// ============================================================
//  DNEG Wormhole metric — James & Thorne (2015), arXiv:1502.03809
//  "Visualizing Interstellar's Wormhole"
//
//  Metric (Φ = 0, gravitational potential neglected per the paper):
//    ds² = −dt² + dℓ² + r(ℓ)²(dθ² + sin²θ dφ²)
//
//  ℓ ∈ ℝ — signed proper radial distance:
//    ℓ > 0  → Universe A (observer side)
//    ℓ = 0  → throat
//    ℓ < 0  → Universe B (remote side)
//
//  Areal radius r(ℓ)  [eq. 8 in the paper]:
//    |ℓ| ≤ a :   r = ρ                     (cylindrical tunnel)
//    |ℓ| > a :   r = ρ + M·[x·arctan(x) − ½·ln(1+x²)]
//                x = 2(|ℓ|−a)/(πM)
//
//  Parameters:
//    ρ      — throat areal radius
//    a      — half-length of cylindrical throat  (a=0 → pointlike)
//    M_lens — lensing parameter; lensing width W ≈ 1.42953·M_lens
//
//  Interstellar defaults (paper Section IV):
//    ρ = M_lens = 1 M,  a = 0.01 M
//    Wormhole looks like a "crystal ball" — no long tunnel visible.
//
//  Conserved quantities (Killing symmetries):
//    E = −p_t  (energy)
//    L =  p_φ  (axial angular momentum)
//
//  Null Hamiltonian:
//    H = ½ g^μν p_μ p_ν = ½[−E² + p_ℓ² + p_θ²/r² + L²/(r²sin²θ)] = 0
//
//  Hamilton's equations  (state = (ℓ, θ, φ, p_ℓ, p_θ)):
//    dℓ/dλ     = p_ℓ
//    dθ/dλ     = p_θ / r²
//    dφ/dλ     = L / (r²·sin²θ)
//    dp_ℓ/dλ   = (r′/r³)·(p_θ² + L²/sin²θ)          ← lensing force
//    dp_θ/dλ   = L²·cosθ / (r²·sin²θ·sinθ)           ← centrifugal term
// ============================================================
#include <cmath>

// ── Physical parameters ───────────────────────────────────────
struct DnegParams {
    double rho    = 1.0;  ///< throat areal radius  [geometric units, G=c=1]
    double a      = 0.0;  ///< half-length of cylindrical tunnel  [same units]
    double M_lens = 1.0;  ///< lensing parameter; lensing width W ≈ 1.42953·M_lens
};

/// Areal radius r(ℓ) and its derivative dr/dℓ  (analytic, no finite diffs needed)
struct DnegProfile {
    double r;        ///< areal radius at ℓ
    double r_prime;  ///< dr/dℓ
};

inline DnegProfile dneg_profile(double l, const DnegParams& p) {
    const double abs_l = l >= 0.0 ? l : -l;
    if (abs_l <= p.a) {
        return {p.rho, 0.0};
    }
    const double pi_M   = M_PI * p.M_lens;
    const double x      = 2.0 * (abs_l - p.a) / pi_M;
    const double atan_x = std::atan(x);
    const double r      = p.rho + p.M_lens * (x * atan_x - 0.5 * std::log(1.0 + x * x));
    const double sgn    = l >= 0.0 ? 1.0 : -1.0;
    const double r_prime = sgn * atan_x * (2.0 / pi_M);   // sgn(ℓ)·arctan(x)·(2/πM)
    return {r, r_prime};
}

// ── Geodesic state ────────────────────────────────────────────
struct WormholeState {
    double l;        ///< proper radial distance (signed, ℓ ∈ ℝ)
    double theta;    ///< polar angle θ ∈ (0, π)
    double phi;      ///< azimuthal angle φ
    double p_l;      ///< momentum conjugate to ℓ
    double p_theta;  ///< momentum conjugate to θ
    double E;        ///< conserved energy  (−p_t > 0)
    double L;        ///< conserved angular momentum  (p_φ)
};

// ── Hamiltonian RHS  ──────────────────────────────────────────
inline void geodesic_rhs_wormhole(const WormholeState& s, const DnegParams& p,
                                   double& dl,      double& dtheta,  double& dphi,
                                   double& dp_l,    double& dp_theta)
{
    const DnegProfile prof = dneg_profile(s.l, p);
    const double r  = prof.r;
    const double rp = prof.r_prime;
    const double r2 = r * r;
    const double r3 = r2 * r;

    const double sin_t = std::sin(s.theta);
    const double cos_t = std::cos(s.theta);
    const double sin2  = sin_t * sin_t;
    const double sin2g = sin2 > 1e-20 ? sin2 : 1e-20;  // guard at poles
    const double L2    = s.L * s.L;

    dl       = s.p_l;
    dtheta   = s.p_theta / r2;
    dphi     = s.L / (r2 * sin2g);
    dp_l     = (rp / r3) * (s.p_theta * s.p_theta + L2 / sin2g);
    dp_theta = L2 * cos_t / (r2 * sin2g * (std::abs(sin_t) > 1e-10 ? sin_t : 1e-10));
}

// ── Null constraint: H residual (diagnostic / monitoring) ────
inline double wormhole_hamiltonian(const WormholeState& s, const DnegParams& p) {
    const double r2   = dneg_profile(s.l, p).r;
    const double r2sq = r2 * r2;
    const double _s2t = std::sin(s.theta) * std::sin(s.theta);
    const double sin2 = _s2t > 1e-20 ? _s2t : 1e-20;
    return 0.5 * (-s.E * s.E + s.p_l * s.p_l
                  + s.p_theta * s.p_theta / r2sq
                  + s.L * s.L / (r2sq * sin2));
}

// ── Classic RK4 single step ───────────────────────────────────
inline void rk4_step_wormhole(const DnegParams& p, WormholeState& s, double h) {
    double dl1,dth1,dph1,dpl1,dpth1;
    double dl2,dth2,dph2,dpl2,dpth2;
    double dl3,dth3,dph3,dpl3,dpth3;
    double dl4,dth4,dph4,dpl4,dpth4;

    const double E = s.E, L = s.L;

    // Stage 1 — at s
    geodesic_rhs_wormhole(s, p, dl1,dth1,dph1,dpl1,dpth1);

    // Stage 2 — at s + h/2 · k1
    WormholeState tmp = s;
    tmp.l = s.l + 0.5*h*dl1;  tmp.theta = s.theta + 0.5*h*dth1;
    tmp.p_l = s.p_l + 0.5*h*dpl1;  tmp.p_theta = s.p_theta + 0.5*h*dpth1;
    geodesic_rhs_wormhole(tmp, p, dl2,dth2,dph2,dpl2,dpth2);

    // Stage 3 — at s + h/2 · k2
    tmp.l = s.l + 0.5*h*dl2;  tmp.theta = s.theta + 0.5*h*dth2;
    tmp.p_l = s.p_l + 0.5*h*dpl2;  tmp.p_theta = s.p_theta + 0.5*h*dpth2;
    geodesic_rhs_wormhole(tmp, p, dl3,dth3,dph3,dpl3,dpth3);

    // Stage 4 — at s + h · k3
    tmp.l = s.l + h*dl3;  tmp.theta = s.theta + h*dth3;
    tmp.p_l = s.p_l + h*dpl3;  tmp.p_theta = s.p_theta + h*dpth3;
    geodesic_rhs_wormhole(tmp, p, dl4,dth4,dph4,dpl4,dpth4);

    const double inv6 = 1.0 / 6.0;
    s.l       += h * inv6 * (dl1   + 2.0*dl2   + 2.0*dl3   + dl4);
    s.theta   += h * inv6 * (dth1  + 2.0*dth2  + 2.0*dth3  + dth4);
    s.phi     += h * inv6 * (dph1  + 2.0*dph2  + 2.0*dph3  + dph4);
    s.p_l     += h * inv6 * (dpl1  + 2.0*dpl2  + 2.0*dpl3  + dpl4);
    s.p_theta += h * inv6 * (dpth1 + 2.0*dpth2 + 2.0*dpth3 + dpth4);
    s.E = E;  s.L = L;  // keep conserved quantities exact
}

// ── Adaptive RK4 via step-doubling (Richardson extrapolation) ─
// Same algorithm as geodesic.hpp::rk4_adaptive — Richardson factor = 2⁴−1 = 15.
// Returns true on accepted step; h updated in place.
inline bool rk4_adaptive_wormhole(const DnegParams& p, WormholeState& s,
                                   double& h, double tol = 1e-7) {
    const WormholeState s0 = s;

    WormholeState sA = s0;
    rk4_step_wormhole(p, sA, h);

    WormholeState sB = s0;
    rk4_step_wormhole(p, sB, 0.5 * h);
    rk4_step_wormhole(p, sB, 0.5 * h);

    const double err = std::sqrt(
        (sA.l       - sB.l)       * (sA.l       - sB.l)       +
        (sA.theta   - sB.theta)   * (sA.theta   - sB.theta)   +
        (sA.p_l     - sB.p_l)     * (sA.p_l     - sB.p_l)     +
        (sA.p_theta - sB.p_theta) * (sA.p_theta - sB.p_theta)
    ) / 15.0;   // Richardson factor

    if (!std::isfinite(err)) {
        s = s0;
        h = (std::isfinite(h) && h > 1e-10) ? h * 0.5 : 1e-6;
        if (h < 1e-10) h = 1e-10;
        return false;
    }

    const bool accepted = (err < tol || h < 1e-10);
    if (accepted) {
        s = sB;
        double hnew = h * (err > 1e-14 ? 0.9 * std::pow(tol / err, 0.2) : 4.0);
        if (!std::isfinite(hnew)) hnew = h;
        if (hnew > 50.0) hnew = 50.0;
        if (hnew < 1e-10) hnew = 1e-10;
        h = hnew;
    } else {
        s = s0;
        double hnew = h * 0.9 * std::pow(tol / err, 0.25);
        if (!std::isfinite(hnew)) hnew = h * 0.5;
        if (hnew > h * 0.5) hnew = h * 0.5;
        if (hnew < 1e-10) hnew = 1e-10;
        h = hnew;
    }
    return accepted;
}

// ── Wormhole camera ───────────────────────────────────────────
// Observer at (ℓ_obs, θ_obs, φ_obs) in Universe A (ℓ_obs > 0).
// Camera boresight points toward the wormhole (−ê_ℓ direction).
//
// At large ℓ_obs, metric → flat:  g_ℓℓ=1, g_θθ=r²,  g_φφ=r²sin²θ
// Orthonormal tetrad:  ê_ℓ, ê_θ = r⁻¹∂_θ, ê_φ = (r sinθ)⁻¹∂_φ
//
// For pixel angle offsets (α, β) from boresight:
//   tetrad direction n = (−cosα cosβ, −sinβ, −sinα cosβ)
//   → p_ℓ = n_ℓ,  p_θ = r_obs·n_θ,  L = r_obs·sinθ_obs·n_φ,  E=1
// Null condition: n_ℓ² + n_θ² + n_φ² = 1  ✓  (unit vector).
struct WormholeCamera {
    double l_obs;     ///< observer proper distance from throat (> 0, Universe A)
    double theta_obs; ///< polar inclination in radians
    double phi_obs;   ///< azimuthal angle in radians
    double fov_h;     ///< horizontal FOV in radians
    int    width, height;

    WormholeCamera(double l_obs_, double theta_deg, double phi_deg,
                   double fov_deg, int w, int h)
        : l_obs(l_obs_)
        , theta_obs(theta_deg * M_PI / 180.0)
        , phi_obs(phi_deg  * M_PI / 180.0)
        , fov_h(fov_deg    * M_PI / 180.0)
        , width(w), height(h) {}

    WormholeState pixel_ray(int px, int py, const DnegParams& p,
                            double offset_x = 0.0, double offset_y = 0.0) const
    {
        const int    span  = (width  > 1) ? (width - 1) : 1;
        const double xf    = double(px) + offset_x;
        const double yf    = double(py) + offset_y;
        const double alpha = fov_h * (xf - 0.5*(width-1))  / span;
        const double beta  = fov_h * (0.5*(height-1) - yf) / span;

        // Unit direction in observer tetrad frame
        // (same sign convention as the BL Camera in camera.hpp)
        const double ca = std::cos(alpha), sa = std::sin(alpha);
        const double cb = std::cos(beta),  sb = std::sin(beta);
        const double n_l   = -ca * cb;   // toward throat (−ê_ℓ)
        const double n_th  = -sb;        // upward is −ê_θ direction
        const double n_phi = -sa * cb;   // +α → left on screen

        // Effective r at observer
        const double r_obs_eff = dneg_profile(l_obs, p).r;
        const double sin_obs   = std::abs(std::sin(theta_obs));
        const double sin_obs_g = sin_obs > 1e-10 ? sin_obs : 1e-10;

        // Covariant momenta  (E=1 normalization, null condition satisfied)
        return {l_obs,
                theta_obs,
                phi_obs,
                n_l,                         // p_ℓ
                r_obs_eff * n_th,            // p_θ
                1.0,                         // E = 1
                r_obs_eff * sin_obs_g * n_phi};  // L = p_φ
    }
};
