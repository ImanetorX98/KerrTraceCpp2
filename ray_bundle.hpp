#pragma once
// ============================================================
//  Ray Bundle — geodesic-deviation approach
//
//  For every pixel we trace a small pencil of photons to model
//  the finite angular size of the pixel on the sky.  The bundle
//  deforms as it propagates through the curved spacetime; its
//  shape at the disk tells us:
//    • which region of the disk contributes to this pixel
//    • how much magnification / demagnification is applied
//    • a smooth, flicker-free luminosity estimate
//
//  ── Implementation: variational (Jacobi-field) equations ─────
//
//  Instead of tracing 5 separate rays we integrate the
//  linearised Hamiltonian flow alongside the main geodesic.
//
//  State:   z = (r, θ, p_r, p_θ)  (p_t, p_φ conserved)
//  Flow:    dz/dλ = f(z)   from Hamilton's equations
//  Variation: d(δz)/dλ = M(z) δz
//
//  M is the 4×4 symplectic-gradient matrix of H:
//         ┌  H_{p,q}   H_{p,p} ┐
//    M  = │                    │   (block notation)
//         └ −H_{q,q}  −H_{q,p} ┘
//
//  where  H_{ab} = ∂²H/∂z^a∂z^b  (Hessian, computed by
//  central finite differences — 9 metric evaluations total).
//
//  We evolve a 4×2 matrix of deviation vectors W = [δz_u | δz_v]
//  whose columns correspond to displacements along the two
//  orthogonal screen directions (α, β).
//
//  Initial conditions  (at the camera, large r, approx. flat):
//    δz_u(0) = ∂z/∂α  at pixel centre  (numerical)
//    δz_v(0) = ∂z/∂β  at pixel centre  (numerical)
//
//  At disk crossing the 4×2 matrix W projects onto the disk's
//  (r, φ) plane, giving a 2×2 Jacobi map J.
//  |det J| is the solid-angle / disk-area ratio → magnification.
//
//  References:
//    James et al. (2015) CQG 32 065001  [DNGR / Interstellar]
//    Luminet (2019) Universe 5(1) 9      [historical review]
//    Pihajoki et al. (2018) ApJ 863 8    [Arcmancer]
// ============================================================
#include "camera.hpp"
#include "geodesic.hpp"
#include "knds_metric.hpp"
#include <array>
#include <cmath>

// ── Deviation operator ───────────────────────────────────────
//  The bundle is a family of geodesics parameterised by the screen angles
//  (alpha, beta). Its deviation obeys the variational equation of the flow, and
//  the two things that were wrong before are both visible in the state list:
//
//    z = (r, theta, phi, p_r, p_theta),   parameters (p_t, p_phi)
//
//  1. p_t and p_phi are conserved ALONG a ray but DIFFER BETWEEN the rays of the
//     bundle, so they are parameters of the flow, not constants of the family.
//     The deviation is therefore inhomogeneous:
//
//       d(dz)/dl = (df/dz) dz + (df/dp_t) dp_t + (df/dp_phi) dp_phi
//
//     with dp_t and dp_phi constant per screen direction. The previous code
//     integrated only the homogeneous part, from the first step onward.
//
//  2. phi was not part of the deviation at all, yet the footprint on the disk
//     lives in the (dr, r dphi) plane. The old code substituted dtheta, which at
//     theta = pi/2 does not even lie in that plane.
//
//  Measured consequence: |det J| was ANTI-correlated with the finite-difference
//  Jacobian d(r,phi)/d(alpha,beta) -- log-log correlation -0.416 over 16 279
//  clean disk pixels, ratio spread 37x. See docs/PLAN-2026-09-05.md.
//
//  H = 1/2 g^{mu nu} p_mu p_nu is exactly quadratic in the momenta, so no
//  Hessian of H is needed: the momentum-momentum block IS the inverse metric,
//  and the mixed blocks are first derivatives of it contracted with p. Only
//  g^{mu nu} and its first and second derivatives in (r, theta) are required,
//  which is both cheaper and far better conditioned than differencing H itself.
//
//    r'      = g^{r nu} p_nu
//    theta'  = g^{th nu} p_nu
//    phi'    = g^{ph nu} p_nu
//    p_r'    = -1/2 d_r  g^{mu nu} p_mu p_nu
//    p_th'   = -1/2 d_th g^{mu nu} p_mu p_nu
//
//  Index order for the metric is BL: (t, r, theta, phi) = (0, 1, 2, 3).
struct BundleOps {
    double A[5][5];   ///< df/dz
    double B[5][2];   ///< df/d(p_t, p_phi)
    double f[5];      ///< the flow itself
};

static void bundle_ops(const KNdSMetric& g,
                       double r, double theta,
                       double pr, double pth,
                       double pt, double pphi,
                       BundleOps& op) {
    const double hr = 1e-5 * (std::abs(r) + 1.0);
    const double ht = 1e-5;

    double G[4][4], Gr_p[4][4], Gr_m[4][4], Gt_p[4][4], Gt_m[4][4];
    double Gpp[4][4], Gpm[4][4], Gmp[4][4], Gmm[4][4];
    g.contravariant_BL(r,      theta,      G);
    g.contravariant_BL(r + hr, theta,      Gr_p);
    g.contravariant_BL(r - hr, theta,      Gr_m);
    g.contravariant_BL(r,      theta + ht, Gt_p);
    g.contravariant_BL(r,      theta - ht, Gt_m);
    g.contravariant_BL(r + hr, theta + ht, Gpp);
    g.contravariant_BL(r + hr, theta - ht, Gpm);
    g.contravariant_BL(r - hr, theta + ht, Gmp);
    g.contravariant_BL(r - hr, theta - ht, Gmm);

    double dGr[4][4], dGt[4][4], d2Grr[4][4], d2Gtt[4][4], d2Grt[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            dGr[i][j]   = (Gr_p[i][j] - Gr_m[i][j]) / (2.0*hr);
            dGt[i][j]   = (Gt_p[i][j] - Gt_m[i][j]) / (2.0*ht);
            d2Grr[i][j] = (Gr_p[i][j] - 2.0*G[i][j] + Gr_m[i][j]) / (hr*hr);
            d2Gtt[i][j] = (Gt_p[i][j] - 2.0*G[i][j] + Gt_m[i][j]) / (ht*ht);
            d2Grt[i][j] = (Gpp[i][j] - Gpm[i][j] - Gmp[i][j] + Gmm[i][j])
                        / (4.0*hr*ht);
        }

    const double p[4] = {pt, pr, pth, pphi};
    // V_a(D) = D^{a nu} p_nu ,  Q(D) = D^{mu nu} p_mu p_nu
    auto V = [&](const double D[4][4], int a) {
        double s = 0.0;
        for (int n = 0; n < 4; ++n) s += D[a][n] * p[n];
        return s;
    };
    auto Q = [&](const double D[4][4]) {
        double s = 0.0;
        for (int m = 0; m < 4; ++m)
            for (int n = 0; n < 4; ++n) s += D[m][n] * p[m] * p[n];
        return s;
    };

    op.f[0] = V(G, 1);
    op.f[1] = V(G, 2);
    op.f[2] = V(G, 3);
    op.f[3] = -0.5 * Q(dGr);
    op.f[4] = -0.5 * Q(dGt);

    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) op.A[i][j] = 0.0;
        op.B[i][0] = op.B[i][1] = 0.0;
    }

    // Positions: d(x^a)'/dz. Column 2 (phi) stays zero -- H is axisymmetric.
    op.A[0][0] = V(dGr, 1);  op.A[0][1] = V(dGt, 1);
    op.A[0][3] = G[1][1];    op.A[0][4] = G[1][2];
    op.A[1][0] = V(dGr, 2);  op.A[1][1] = V(dGt, 2);
    op.A[1][3] = G[2][1];    op.A[1][4] = G[2][2];
    op.A[2][0] = V(dGr, 3);  op.A[2][1] = V(dGt, 3);
    op.A[2][3] = G[3][1];    op.A[2][4] = G[3][2];

    // Momenta: d(p_a)'/dz.
    op.A[3][0] = -0.5 * Q(d2Grr);  op.A[3][1] = -0.5 * Q(d2Grt);
    op.A[3][3] = -V(dGr, 1);       op.A[3][4] = -V(dGr, 2);
    op.A[4][0] = -0.5 * Q(d2Grt);  op.A[4][1] = -0.5 * Q(d2Gtt);
    op.A[4][3] = -V(dGt, 1);       op.A[4][4] = -V(dGt, 2);

    // Forcing by the conserved parameters, which differ across the bundle.
    op.B[0][0] = G[1][0];      op.B[0][1] = G[1][3];
    op.B[1][0] = G[2][0];      op.B[1][1] = G[2][3];
    op.B[2][0] = G[3][0];      op.B[2][1] = G[3][3];
    op.B[3][0] = -V(dGr, 0);   op.B[3][1] = -V(dGr, 3);
    op.B[4][0] = -V(dGt, 0);   op.B[4][1] = -V(dGt, 3);
}

// ── Bundle state: main geodesic + 5×2 deviation matrix W ─────
//  W rows are (dr, dtheta, dphi, dp_r, dp_theta); columns are the two screen
//  directions. dpt and dpphi are the constant parameter deviations of the same
//  two directions.
struct BundleState {
    GeodesicState geo;
    double W[5][2];
    double dpt[2];
    double dpphi[2];
};

static void bundle_rhs(const KNdSMetric& g,
                       const GeodesicState& s,
                       const double W[5][2],
                       const double dpt[2],
                       const double dpphi[2],
                       double dz[5],
                       double dW[5][2])
{
    BundleOps op;
    bundle_ops(g, s.r, s.theta, s.pr, s.ptheta, s.pt, s.pphi, op);
    for (int i = 0; i < 5; ++i) dz[i] = op.f[i];

    for (int i = 0; i < 5; ++i)
        for (int k = 0; k < 2; ++k) {
            double acc = op.B[i][0]*dpt[k] + op.B[i][1]*dpphi[k];
            for (int j = 0; j < 5; ++j) acc += op.A[i][j] * W[j][k];
            dW[i][k] = acc;
        }
}

// ── RK4 step for bundle ───────────────────────────────────────
//  phi is now integrated as part of the state (dz[2]) rather than by a separate
//  quadrature, so that the deviation and the ray see exactly the same phi.
static void bundle_rk4(const KNdSMetric& g, BundleState& bs, double dlam) {
    const GeodesicState s0 = bs.geo;
    double W0[5][2];
    for (int i = 0; i < 5; ++i) for (int k = 0; k < 2; ++k) W0[i][k] = bs.W[i][k];

    double dz[4][5], dW[4][2][2];
    double dWs[4][5][2];

    GeodesicState st = s0;
    double Wt[5][2];
    const double frac[4] = {0.0, 0.5, 0.5, 1.0};

    for (int stage = 0; stage < 4; ++stage) {
        if (stage == 0) {
            st = s0;
            for (int i = 0; i < 5; ++i) for (int k = 0; k < 2; ++k) Wt[i][k] = W0[i][k];
        } else {
            const double f = frac[stage] * dlam;
            const int prev = stage - 1;
            st = s0;
            st.r      = s0.r      + f*dz[prev][0];
            st.theta  = s0.theta  + f*dz[prev][1];
            st.phi    = s0.phi    + f*dz[prev][2];
            st.pr     = s0.pr     + f*dz[prev][3];
            st.ptheta = s0.ptheta + f*dz[prev][4];
            for (int i = 0; i < 5; ++i)
                for (int k = 0; k < 2; ++k)
                    Wt[i][k] = W0[i][k] + f*dWs[prev][i][k];
        }
        bundle_rhs(g, st, Wt, bs.dpt, bs.dpphi, dz[stage], dWs[stage]);
    }
    (void)dW;

    auto comb = [&](int i) {
        return (dz[0][i] + 2.0*dz[1][i] + 2.0*dz[2][i] + dz[3][i]) * dlam / 6.0;
    };
    bs.geo.r      += comb(0);
    bs.geo.theta  += comb(1);
    bs.geo.phi    += comb(2);
    bs.geo.pr     += comb(3);
    bs.geo.ptheta += comb(4);
    for (int i = 0; i < 5; ++i)
        for (int k = 0; k < 2; ++k)
            bs.W[i][k] += (dWs[0][i][k] + 2.0*dWs[1][i][k]
                         + 2.0*dWs[2][i][k] + dWs[3][i][k]) * dlam / 6.0;
}

// ── Adaptive bundle step (RK4 + step-doubling) ───────────────
static bool bundle_adaptive(const KNdSMetric& g, BundleState& bs,
                            double& dlam, double tol = 1e-7) {
    const BundleState s0 = bs;

    BundleState sA = s0;
    bundle_rk4(g, sA, dlam);

    BundleState sB = s0;
    bundle_rk4(g, sB, 0.5*dlam);
    bundle_rk4(g, sB, 0.5*dlam);

    const double err = std::sqrt(
        (sA.geo.r      - sB.geo.r)      * (sA.geo.r      - sB.geo.r)      +
        (sA.geo.theta  - sB.geo.theta)  * (sA.geo.theta  - sB.geo.theta)  +
        (sA.geo.pr     - sB.geo.pr)     * (sA.geo.pr     - sB.geo.pr)     +
        (sA.geo.ptheta - sB.geo.ptheta) * (sA.geo.ptheta - sB.geo.ptheta)
    ) / 15.0;

    if (!std::isfinite(err)) {
        bs = s0;
        dlam = (std::isfinite(dlam) && dlam > 1e-10) ? dlam * 0.5 : 1e-6;
        if (dlam < 1e-10) dlam = 1e-10;
        return false;
    }

    const bool accepted = (err < tol || dlam < 1e-10);
    if (accepted) {
        bs = sB;
        const double scale = (err > 1e-14)
                           ? 0.9 * std::pow(tol/err, 0.2)
                           : 4.0;
        double hnew = dlam * scale;
        if (!std::isfinite(hnew)) hnew = dlam;
        dlam = hnew;
        if (dlam > 100.0) dlam = 100.0;
        if (dlam < 1e-10) dlam = 1e-10;
    } else {
        bs = s0;
        const double half = dlam * 0.5;
        double hnew = dlam * 0.9 * std::pow(tol/err, 0.25);
        if (!std::isfinite(hnew)) hnew = half;
        dlam = hnew;
        if (dlam > half)  dlam = half;
        if (dlam < 1e-10) dlam = 1e-10;
    }
    return accepted;
}

// ── Bundle result ─────────────────────────────────────────────
struct BundleResult {
    bool   disk_hit    = false;
    double r_hit       = 0.0;
    double redshift    = 1.0;
    double magnif      = 1.0;  ///< |det J|  — solid-angle magnification
    double theta_esc   = 0.0;  ///< final θ on escape (for background lookup)
    double phi_esc     = 0.0;  ///< final φ on escape
    double phi_disk    = 0.0;  ///< BL azimuthal angle at disk crossing
    // Edge vectors of the pixel's footprint on the disk, in (r, φ), already
    // scaled by the pixel's angular size. DNGR §2.2(iii) integrates the emission
    // over this region rather than sampling the centre.
    double fp_dr_a = 0.0, fp_dphi_a = 0.0;
    double fp_dr_b = 0.0, fp_dphi_b = 0.0;
};

// ── Initial deviation of the bundle ──────────────────────────
//  Every ray of the bundle leaves the SAME event (the camera), so the position
//  deviations all start at zero and the whole bundle is born in the momenta:
//
//      dz(0) = (0, 0, 0, dp_r, dp_theta) ,  plus the constants dp_t, dp_phi.
//
//  dp_t and dp_phi are what the old code dropped. They are conserved along each
//  ray but differ between rays, so they enter the deviation as a constant
//  forcing rather than as part of the evolving vector.
static void init_bundle(const Camera& cam,
                        double alpha, double beta,
                        const KNdSMetric& g,
                        double W[5][2], double dpt[2], double dpphi[2]) {
    const double eps = cam.fov_h / cam.width * 0.5;  // half a pixel, in radians

    const GeodesicState s_ap = cam.angle_ray(alpha+eps, beta,   g);
    const GeodesicState s_am = cam.angle_ray(alpha-eps, beta,   g);
    const GeodesicState s_bp = cam.angle_ray(alpha,   beta+eps, g);
    const GeodesicState s_bm = cam.angle_ray(alpha,   beta-eps, g);
    const double inv = 1.0 / (2.0*eps);

    for (int i = 0; i < 3; ++i) { W[i][0] = 0.0; W[i][1] = 0.0; }
    W[3][0] = (s_ap.pr     - s_am.pr)     * inv;
    W[4][0] = (s_ap.ptheta - s_am.ptheta) * inv;
    W[3][1] = (s_bp.pr     - s_bm.pr)     * inv;
    W[4][1] = (s_bp.ptheta - s_bm.ptheta) * inv;

    dpt[0]   = (s_ap.pt   - s_am.pt)   * inv;
    dpphi[0] = (s_ap.pphi - s_am.pphi) * inv;
    dpt[1]   = (s_bp.pt   - s_bm.pt)   * inv;
    dpphi[1] = (s_bp.pphi - s_bm.pphi) * inv;
}

// ── Main bundle trace ─────────────────────────────────────────
static BundleResult trace_bundle(int px, int py,
                                  const Camera& cam,
                                  const KNdSMetric& g,
                                  double r_disk_in,
                                  double r_disk_out,
                                  double r_escape,
                                  int max_steps = 500000,
                                  double step_init = 1.0,
                                  double tol = 1e-7,
                                  double pixel_offset_x = 0.0,
                                  double pixel_offset_y = 0.0) {
    const int span = (cam.width > 1) ? (cam.width - 1) : 1;
    const double x = (double)px + pixel_offset_x;
    const double y = (double)py + pixel_offset_y;
    const double alpha = cam.fov_h*(x - 0.5*(cam.width-1))  / span;
    const double beta  = cam.fov_h*(0.5*(cam.height-1) - y) / span;
    const double dang  = cam.fov_h / span;   // angular size of one pixel

    BundleState bs;
    bs.geo = cam.angle_ray(alpha, beta, g);
    init_bundle(cam, alpha, beta, g, bs.W, bs.dpt, bs.dpphi);

    const double rh  = g.r_horizon();
    const double rh_cut = rh * 1.03;
    double dlam      = std::max(step_init, 1e-10);
    const int max_iter = std::max(1, max_steps);

    for (int iter = 0; iter < max_iter; ++iter) {
        const BundleState bs_prev = bs;
        double step_used = dlam;
        int rejects = 0;
        while (true) {
            step_used = dlam;
            if (bundle_adaptive(g, bs, dlam, tol)) break;
            if (!std::isfinite(dlam) || ++rejects > 64) return {};
        }

        const double r = bs.geo.r;

        double best_alpha = 2.0;
        enum class StepEvent { NONE, DISK, HORIZON, ESCAPE };
        StepEvent best_event = StepEvent::NONE;

        double disk_r_hit  = 0.0;
        double disk_red    = 1.0;
        double disk_det    = 1.0;
        double disk_phi    = 0.0;
        double disk_fp_dr_a = 0.0, disk_fp_dphi_a = 0.0;
        double disk_fp_dr_b = 0.0, disk_fp_dphi_b = 0.0;

        const double q0 = bs_prev.geo.theta - M_PI/2.0;
        const double q1 = bs.geo.theta      - M_PI/2.0;
        const bool maybe_equator = sign_change(q0, q1) ||
                                   (std::min(std::abs(q0), std::abs(q1)) < 0.35);
        if (maybe_equator) {
            double dr0,dth0,dpr0,dpth0;
            double dr1,dth1,dpr1,dpth1;
            geodesic_rhs(g, bs_prev.geo.r, bs_prev.geo.theta, bs_prev.geo.pr, bs_prev.geo.ptheta,
                         bs_prev.geo.pt, bs_prev.geo.pphi, dr0,dth0,dpr0,dpth0);
            geodesic_rhs(g, bs.geo.r, bs.geo.theta, bs.geo.pr, bs.geo.ptheta,
                         bs.geo.pt, bs.geo.pphi, dr1,dth1,dpr1,dpth1);
            double alpha = 0.0;
            // NOT a `continue`: this block only decides whether the step crossed
            // the disk. The horizon and escape tests live below, and skipping
            // them left the ray unable to terminate. maybe_equator is true
            // whenever the ray is within 0.35 rad of the equatorial plane, which
            // for a near-equatorial camera is almost every step, so almost every
            // ray ran to max_steps instead of stopping after a few hundred.
            const bool crossed_equator = first_event_alpha_hermite(
                    bs_prev.geo.theta, bs.geo.theta, dth0, dth1, step_used, M_PI/2.0,
                    alpha, 8, 8);
            const double r_hit = crossed_equator
                ? hermite_interp_scalar(bs_prev.geo.r, bs.geo.r, dr0, dr1, step_used, alpha)
                : 0.0;
            if (crossed_equator && r_hit >= r_disk_in && r_hit <= r_disk_out) {
                // ── Redshift ──────────────────────────────────────
                // Mirrors disk_redshift() in main.cpp. keplerian_omega() returns
                // −Ω_K by convention, so the physical prograde Ω is its negation;
                // this code used the raw value and so evaluated the disk-frame
                // normalisation d2 on the retrograde branch, where the 2·g_tφ·Ω
                // cross term flips sign. d2 then turns negative inside r ≈ 1.5M
                // and the whole inner disk fell back to red = 1, i.e. rendered as
                // if unshifted while the correct g there is ~0.1: with the g⁴
                // beaming that is a factor 10⁴ too bright, which was the saturated
                // white ring around the shadow. The two sign errors (Ω and b)
                // cancelled in the denominator, hiding the one in d2.
                const double Omega = -g.keplerian_omega(r_hit);
                const double b     = bs.geo.pphi / (-bs.geo.pt);
                double gLL[4][4];
                g.covariant_BL(r_hit, M_PI/2.0, gLL);
                const double d2 = -(gLL[0][0]+2.0*gLL[0][3]*Omega+gLL[3][3]*Omega*Omega);
                const double dv = 1.0 - Omega*b;
                // Same floors and ceiling as disk_redshift(): a photon reaching the
                // disk has E − ΩL > 0, so flooring dv rather than special-casing it
                // keeps g finite without inventing an unshifted value.
                const double d2_safe = d2 > 1.0e-8 ? d2 : 1.0e-8;
                const double dv_safe = dv > 1.0e-8 ? dv : 1.0e-8;
                double red = std::sqrt(d2_safe)/dv_safe;
                if (!std::isfinite(red)) red = 1.0;
                red = red < 0.0 ? 0.0 : red > 6.0 ? 6.0 : red;

                // ── Jacobi map  J: screen (α,β) → disk (r,φ) ────
                // Project the (r,θ) sub-block of W onto the disk tangent plane.
                // At equatorial crossing (θ≈π/2), φ-variation ≈ W[1]/sinθ·(dφ/dθ)
                // but more directly: use W[0] (δr) and approximate δφ from W[1]
                // via the disk metric: dφ ≈ (dθ/dr_disk) · W[1] ... complex.
                // Simpler: use only the 2×2 sub-block (δr, δθ) as proxy for
                // (δr_disk, δφ_disk)  — gives the right shape up to a constant.
                // Footprint in the plane of the disk.
                //
                // Two corrections over the old code, both needed. First, the
                // displacement of a bundle edge in the disk is (dr, r dphi), not
                // (dr, dtheta): at theta = pi/2 the dtheta direction is normal to
                // the disk and describes no area in it.
                //
                // Second, and this is the one that is easy to miss: W is the
                // deviation at equal affine parameter, but neighbouring rays do
                // not cross the equator at the same lambda. The footprint is the
                // deviation on the CROSSING SURFACE, so the flow direction has to
                // be projected out. Requiring the neighbour to be on theta = pi/2
                // as well fixes its lambda offset,
                //
                //     dtheta + theta' dl = 0   =>   dl = -dtheta/theta' ,
                //
                // and the in-plane displacement is then
                //
                //     dr_surf   = dr   + r'   dl ,
                //     dphi_surf = dphi + phi' dl .
                //
                // Without this the matrix mixes in the motion along the ray and
                // its determinant is unrelated to the area the pixel covers.
                double f_r, f_th, f_pr, f_pth;
                geodesic_rhs(g, r_hit, M_PI/2.0, bs.geo.pr, bs.geo.ptheta,
                             bs.geo.pt, bs.geo.pphi, f_r, f_th, f_pr, f_pth);
                const double f_phi = dphi_vel(g, r_hit, M_PI/2.0,
                                              bs.geo.pt, bs.geo.pphi);

                auto Wi = [&](int row, int col) {
                    return bs_prev.W[row][col]
                         + alpha * (bs.W[row][col] - bs_prev.W[row][col]);
                };
                double J00 = 0.0, J01 = 0.0, J10 = 0.0, J11 = 0.0;
                if (std::abs(f_th) > 1e-14) {
                    const double dl_a = -Wi(1,0) / f_th;
                    const double dl_b = -Wi(1,1) / f_th;
                    J00 = Wi(0,0) + f_r   * dl_a;
                    J01 = Wi(0,1) + f_r   * dl_b;
                    J10 = r_hit * (Wi(2,0) + f_phi * dl_a);
                    J11 = r_hit * (Wi(2,1) + f_phi * dl_b);
                } else {
                    J00 = Wi(0,0);            J01 = Wi(0,1);
                    J10 = r_hit * Wi(2,0);    J11 = r_hit * Wi(2,1);
                }

                double det = std::abs(J00*J11 - J01*J10);
                det = det < 1e-12 ? 1e-12 : det;
                // Per-pixel footprint: J is per unit (alpha, beta), so scale by
                // the angular size of one pixel. J10/J11 carry a factor r_hit
                // (they are r*dphi), which is removed to store a plain dphi.
                const double inv_r = 1.0 / std::max(r_hit, 1e-12);
                disk_fp_dr_a   = J00 * dang;
                disk_fp_dphi_a = J10 * dang * inv_r;
                disk_fp_dr_b   = J01 * dang;
                disk_fp_dphi_b = J11 * dang * inv_r;
                disk_r_hit  = r_hit;
                disk_red    = red;
                disk_det    = det;
                disk_phi    = bs_prev.geo.phi + alpha * (bs.geo.phi - bs_prev.geo.phi);
                best_alpha  = alpha;
                best_event  = StepEvent::DISK;
            }
        }
        const bool horizon_cross = ((bs_prev.geo.r > rh_cut) && (r <= rh_cut)) || (r <= rh_cut);
        if (horizon_cross) {
            const double denom_h = bs_prev.geo.r - r;
            double alpha_h = (std::abs(denom_h) > 1e-12) ? ((bs_prev.geo.r - rh_cut) / denom_h) : 0.0;
            alpha_h = alpha_h < 0.0 ? 0.0 : alpha_h > 1.0 ? 1.0 : alpha_h;
            if (alpha_h < best_alpha) {
                best_alpha = alpha_h;
                best_event = StepEvent::HORIZON;
            }
        }

        const bool escape_cross = ((bs_prev.geo.r < r_escape) && (r >= r_escape)) || (r >= r_escape);
        if (escape_cross) {
            const double denom_e = r - bs_prev.geo.r;
            double alpha_e = (std::abs(denom_e) > 1e-12) ? ((r_escape - bs_prev.geo.r) / denom_e) : 1.0;
            alpha_e = alpha_e < 0.0 ? 0.0 : alpha_e > 1.0 ? 1.0 : alpha_e;
            if (alpha_e < best_alpha) {
                best_alpha = alpha_e;
                best_event = StepEvent::ESCAPE;
            }
        }

        if (best_event == StepEvent::DISK) {
            return {true, disk_r_hit, disk_red, disk_det, 0.0, 0.0, disk_phi,
                    disk_fp_dr_a, disk_fp_dphi_a, disk_fp_dr_b, disk_fp_dphi_b};
        }
        if (best_event == StepEvent::HORIZON) {
            return {};
        }
        if (best_event == StepEvent::ESCAPE) {
            const double th_esc = bs_prev.geo.theta + best_alpha * (bs.geo.theta - bs_prev.geo.theta);
            const double ph_esc = bs_prev.geo.phi   + best_alpha * (bs.geo.phi   - bs_prev.geo.phi);
            return {false, r_escape, 1.0, 1.0, th_esc, ph_esc};
        }
    }
    return {false, bs.geo.r, 1.0, 1.0};
}
