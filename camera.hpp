#pragma once
// ============================================================
//  Camera and initial conditions (backwards ray-tracing)
//
//  Observer at (r_obs, θ_obs, φ=0) in BL coordinates,
//  looking toward the black hole (−r direction).
//
//  For each pixel (px, py):
//    1. Compute angular offsets (α, β) from FOV.
//    2. Build initial contravariant 4-momentum in the
//       approximately-flat frame at large r.
//    3. Convert to covariant p_μ via g_μν.
//    4. Enforce the null condition H=0 exactly by solving
//       a quadratic for p_t (keeping the future-directed root).
// ============================================================
#include "knds_metric.hpp"
#include "geodesic.hpp"
#include <cmath>

class Camera {
public:
    double r_obs;        ///< BL radial position  (should be ≫ M)
    double theta_obs;    ///< BL polar angle  (radians)
    double phi_obs;      ///< BL azimuthal angle (radians) — sets initial phi of each ray
    double fov_h;        ///< horizontal field-of-view  (radians)
    int    width, height;

    /// @param r_         observer radius (e.g. 500 M)
    /// @param theta_deg  inclination in degrees (90° = equatorial)
    /// @param phi_deg    azimuth in degrees (0° default)
    /// @param fov_deg    horizontal FOV in degrees
    Camera(double r_, double theta_deg, double phi_deg, double fov_deg, int w, int h)
        : r_obs(r_), theta_obs(theta_deg * M_PI / 180.0),
          phi_obs(phi_deg * M_PI / 180.0),
          fov_h(fov_deg * M_PI / 180.0), width(w), height(h) {}

    // ── pixel → initial geodesic state ───────────────────────
    GeodesicState pixel_ray(int px, int py, const KNdSMetric& g) const {
        const int span = (width > 1) ? (width - 1) : 1;
        const double alpha = fov_h * (px - 0.5*(width-1))  / span;
        const double beta  = fov_h * (0.5*(height-1) - py) / span;
        return angle_ray(alpha, beta, g);
    }

    // ── angle → initial state  (used by ray bundles) ─────────
    GeodesicState angle_ray(double alpha, double beta,
                            const KNdSMetric& g) const {
        const double r0  = r_obs;
        const double th0 = theta_obs;
        // ── At large r the metric is nearly flat. ─────────────
        // Local orthonormal frame (approx):
        //   ê_r̂ points outward  (away from BH)
        //   ê_θ̂ points toward south pole
        //   ê_φ̂ points in +φ direction
        //
        // Camera looks in  −ê_r̂.
        // Pixel direction: d = cos(β)[−cos(α)ê_r̂ + sin(α)ê_φ̂] − sin(β)ê_θ̂
        //
        // Coordinate-basis contravariant components:
        //   p^r   ~ −cos(α)cos(β) / sqrt(g_rr)
        //   p^θ   ~ −sin(β)       / sqrt(g_θθ)
        //   p^φ   ~  sin(α)cos(β) / sqrt(g_φφ)
        // Then set p^t = 1 and correct via null condition.
        double gLL[4][4];
        g.covariant_BL(r0, th0, gLL);

        const double ca = std::cos(alpha), sa = std::sin(alpha);
        const double cb = std::cos(beta),  sb = std::sin(beta);

        const double sqrt_grr  = std::sqrt(std::abs(gLL[1][1]));
        const double sqrt_gthth= std::sqrt(std::abs(gLL[2][2]));
        const double sqrt_gphph= std::sqrt(std::abs(gLL[3][3]));

        const double pUr   = -ca*cb / sqrt_grr;
        const double pUth  = -sb    / sqrt_gthth;
        const double pUphi =  sa*cb / sqrt_gphph;
        const double pUt   =  1.0;

        // Covariant momenta  p_μ = g_μν p^ν  (only non-zero off-diag: g_tφ)
        double pt   = gLL[0][0]*pUt + gLL[0][3]*pUphi;
        double pr   = gLL[1][1]*pUr;
        double pth  = gLL[2][2]*pUth;
        double pphi = gLL[3][0]*pUt + gLL[3][3]*pUphi;

        // ── Enforce null condition: solve  2H = 0  for p_t ───
        // 2H = g^tt p_t² + 2 g^tφ p_t p_φ + C = 0
        // where C = g^rr p_r² + g^θθ p_θ² + g^φφ p_φ²
        double gUU[4][4];
        g.contravariant_BL(r0, th0, gUU);

        const double A_coeff = gUU[0][0];
        const double B_coeff = 2.0 * gUU[0][3] * pphi;
        const double C_coeff = gUU[1][1]*pr*pr
                             + gUU[2][2]*pth*pth
                             + gUU[3][3]*pphi*pphi;

        const double disc = B_coeff*B_coeff - 4.0*A_coeff*C_coeff;
        if (disc >= 0.0 && std::abs(A_coeff) > 1e-15) {
            const double sq = std::sqrt(disc);
            // Two roots: pick the future-directed one  (p_t < 0 means E>0)
            const double pt1 = (-B_coeff - sq) / (2.0*A_coeff);
            const double pt2 = (-B_coeff + sq) / (2.0*A_coeff);
            pt = (pt1 < 0.0) ? pt1 : pt2;
            if (pt > 0.0) pt = std::min(pt1, pt2);  // force negative
        }

        return {r0, th0, phi_obs, pr, pth, pt, pphi};
    }
};
