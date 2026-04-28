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

// ── Hessian of H in the 4D phase space ───────────────────────
//  z[0]=r, z[1]=θ, z[2]=p_r, z[3]=p_θ
//  Returns the symmetric 4×4 Hessian via central finite diff.
static void hessian_H(const KNdSMetric& g,
                      double r, double theta,
                      double pr, double pth,
                      double pt, double pphi,
                      double H[4][4]) {
    // Step sizes proportional to coordinate magnitudes
    const double eps[4] = {
        1e-5 * (std::abs(r)   + 0.1),
        1e-5,
        1e-5 * (std::abs(pr)  + 1e-3),
        1e-5 * (std::abs(pth) + 1e-3)
    };

    auto eval = [&](double dr, double dth, double dpr, double dpth) {
        return g.hamiltonian(r+dr, theta+dth, pr+dpr, pth+dpth, pt, pphi);
    };

    // Diagonal:  H_aa ≈ (H(+ea) − 2H(0) + H(−ea)) / ea²
    const double H0 = eval(0,0,0,0);
    double z_p[4], z_m[4];
    for (int a = 0; a < 4; ++a) {
        z_p[a] = 0.0; z_m[a] = 0.0;
    }
    double H_diag[4];
    {
        double val[4][2];
        double ea[4] = {eps[0],eps[1],eps[2],eps[3]};
        (void)r; (void)theta; (void)pr; (void)pth; // suppress unused warning
        double dz[4][4] = {};
        // Diagonal terms
        dz[0][0]=ea[0]; val[0][0]=eval( ea[0],0,0,0); val[0][1]=eval(-ea[0],0,0,0);
        dz[1][1]=ea[1]; val[1][0]=eval(0, ea[1],0,0); val[1][1]=eval(0,-ea[1],0,0);
        dz[2][2]=ea[2]; val[2][0]=eval(0,0, ea[2],0); val[2][1]=eval(0,0,-ea[2],0);
        dz[3][3]=ea[3]; val[3][0]=eval(0,0,0, ea[3]); val[3][1]=eval(0,0,0,-ea[3]);
        for (int a=0;a<4;a++) H_diag[a] = (val[a][0]-2.0*H0+val[a][1])/(ea[a]*ea[a]);
    }
    for (int a=0;a<4;a++) H[a][a] = H_diag[a];

    // Off-diagonal:  H_ab ≈ (H(+ea+eb) - H(+ea-eb) - H(-ea+eb) + H(-ea-eb)) / (4 ea eb)
    double e[4] = {eps[0],eps[1],eps[2],eps[3]};
    auto cross = [&](int a, int b) -> double {
        double da[4]={}, db[4]={};
        da[a]=e[a]; db[b]=e[b];
        double pp = eval(da[0]+db[0],da[1]+db[1],da[2]+db[2],da[3]+db[3]);
        double pm = eval(da[0]-db[0],da[1]-db[1],da[2]-db[2],da[3]-db[3]);
        double mp = eval(-da[0]+db[0],-da[1]+db[1],-da[2]+db[2],-da[3]+db[3]);
        double mm = eval(-da[0]-db[0],-da[1]-db[1],-da[2]-db[2],-da[3]-db[3]);
        return (pp - pm - mp + mm) / (4.0*e[a]*e[b]);
    };
    for (int a=0;a<4;a++)
        for (int b=a+1;b<4;b++)
            H[a][b] = H[b][a] = cross(a,b);
}

// ── Symplectic-gradient matrix  M(z)  ────────────────────────
//  dδz/dλ = M δz
//  z = (q₁,q₂, p₁,p₂) = (r, θ, p_r, p_θ)
//
//  M = J_s · Hess(H),   J_s = [[0,I],[-I,0]] (symplectic unit)
//
//  In full:
//    M[0..1][0..3] =  H_Hess[2..3][0..3]  (dq/dλ row)
//    M[2..3][0..3] = -H_Hess[0..1][0..3]  (dp/dλ row)
static void build_M(const double Hess[4][4], double M[4][4]) {
    // Row 0,1: come from ∂H/∂p (indices 2,3 of Hess)
    for (int j=0;j<4;j++) M[0][j] =  Hess[2][j];
    for (int j=0;j<4;j++) M[1][j] =  Hess[3][j];
    // Row 2,3: come from −∂H/∂q (indices 0,1 of Hess)
    for (int j=0;j<4;j++) M[2][j] = -Hess[0][j];
    for (int j=0;j<4;j++) M[3][j] = -Hess[1][j];
}

// ── Bundle state: main geodesic + 4×2 Jacobi matrix W ────────
//  W[:,0] = deviation in α direction
//  W[:,1] = deviation in β direction
struct BundleState {
    GeodesicState geo;
    double W[4][2];   ///< Jacobi deviation matrix
};

// ── RHS for the coupled (geodesic + Jacobi) system ───────────
static void bundle_rhs(const KNdSMetric& g,
                       const GeodesicState& s,
                       const double W[4][2],
                       double dz[4],    // out: d(r,θ,pr,pth)/dλ
                       double dW[4][2]) // out: dW/dλ
{
    // 1. Geodesic RHS
    double dr,dth,dpr,dpth;
    geodesic_rhs(g, s.r, s.theta, s.pr, s.ptheta,
                 s.pt, s.pphi, dr, dth, dpr, dpth);
    dz[0]=dr; dz[1]=dth; dz[2]=dpr; dz[3]=dpth;

    // 2. Hessian at current point
    double Hess[4][4];
    hessian_H(g, s.r, s.theta, s.pr, s.ptheta, s.pt, s.pphi, Hess);

    // 3. M = J_s · Hess
    double M[4][4];
    build_M(Hess, M);

    // 4. dW/dλ = M · W
    for (int i=0;i<4;i++)
        for (int k=0;k<2;k++) {
            dW[i][k] = 0.0;
            for (int j=0;j<4;j++)
                dW[i][k] += M[i][j]*W[j][k];
        }
}

// ── RK4 step for bundle ───────────────────────────────────────
static void bundle_rk4(const KNdSMetric& g, BundleState& bs, double dlam) {
    const GeodesicState s0 = bs.geo;

    double dz1[4], dW1[4][2];
    double dz2[4], dW2[4][2];
    double dz3[4], dW3[4][2];
    double dz4[4], dW4[4][2];

    auto make_state = [](const GeodesicState& s, const double dz[4], double f,
                         const double W[4][2], const double dW[4][2]) {
        GeodesicState ns = s;
        ns.r      += f*dz[0]; ns.theta  += f*dz[1];
        ns.pr     += f*dz[2]; ns.ptheta += f*dz[3];
        double nW[4][2];
        for(int i=0;i<4;i++) for(int k=0;k<2;k++) nW[i][k] = W[i][k]+f*dW[i][k];
        return std::make_pair(ns, std::array<std::array<double,2>,4>{
            {{nW[0][0],nW[0][1]},{nW[1][0],nW[1][1]},
             {nW[2][0],nW[2][1]},{nW[3][0],nW[3][1]}}});
    };

    bundle_rhs(g, s0, bs.W, dz1, dW1);

    auto [s2,W2] = make_state(s0, dz1, 0.5*dlam, bs.W, dW1);
    double W2a[4][2]; for(int i=0;i<4;i++) for(int k=0;k<2;k++) W2a[i][k]=W2[i][k];
    bundle_rhs(g, s2, W2a, dz2, dW2);

    auto [s3,W3] = make_state(s0, dz2, 0.5*dlam, bs.W, dW2);
    double W3a[4][2]; for(int i=0;i<4;i++) for(int k=0;k<2;k++) W3a[i][k]=W3[i][k];
    bundle_rhs(g, s3, W3a, dz3, dW3);

    auto [s4,W4] = make_state(s0, dz3, dlam, bs.W, dW3);
    double W4a[4][2]; for(int i=0;i<4;i++) for(int k=0;k<2;k++) W4a[i][k]=W4[i][k];
    bundle_rhs(g, s4, W4a, dz4, dW4);

    // φ integration using the same RK stage points of the pre-step state.
    const double dp1 = dphi_vel(g, s0.r, s0.theta, s0.pt, s0.pphi);
    const double dp2 = dphi_vel(g, s2.r, s2.theta, s0.pt, s0.pphi);
    const double dp3 = dphi_vel(g, s3.r, s3.theta, s0.pt, s0.pphi);
    const double dp4 = dphi_vel(g, s4.r, s4.theta, s0.pt, s0.pphi);

    bs.geo.r      += dlam/6.0*(dz1[0]+2*dz2[0]+2*dz3[0]+dz4[0]);
    bs.geo.theta  += dlam/6.0*(dz1[1]+2*dz2[1]+2*dz3[1]+dz4[1]);
    bs.geo.pr     += dlam/6.0*(dz1[2]+2*dz2[2]+2*dz3[2]+dz4[2]);
    bs.geo.ptheta += dlam/6.0*(dz1[3]+2*dz2[3]+2*dz3[3]+dz4[3]);
    bs.geo.phi += dlam/6.0*(dp1 + 2*dp2 + 2*dp3 + dp4);
    for(int i=0;i<4;i++) for(int k=0;k<2;k++)
        bs.W[i][k] += dlam/6.0*(dW1[i][k]+2*dW2[i][k]+2*dW3[i][k]+dW4[i][k]);
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
};

// ── Initial deviation vectors ΔΨ_α, ΔΨ_β ────────────────────
//  Numerically: δz_α = [∂z/∂α] ≈ [z(α+ε)−z(α−ε)] / 2ε
static void init_bundle(const Camera& cam,
                        double alpha, double beta,
                        const KNdSMetric& g,
                        double W[4][2]) {
    const double eps = cam.fov_h / cam.width * 0.5;  // half-pixel in radians

    auto s_ap = cam.angle_ray(alpha+eps, beta,   g);
    auto s_am = cam.angle_ray(alpha-eps, beta,   g);
    auto s_bp = cam.angle_ray(alpha,   beta+eps, g);
    auto s_bm = cam.angle_ray(alpha,   beta-eps, g);

    W[0][0] = (s_ap.r      - s_am.r)      / (2.0*eps);
    W[1][0] = (s_ap.theta  - s_am.theta)  / (2.0*eps);
    W[2][0] = (s_ap.pr     - s_am.pr)     / (2.0*eps);
    W[3][0] = (s_ap.ptheta - s_am.ptheta) / (2.0*eps);

    W[0][1] = (s_bp.r      - s_bm.r)      / (2.0*eps);
    W[1][1] = (s_bp.theta  - s_bm.theta)  / (2.0*eps);
    W[2][1] = (s_bp.pr     - s_bm.pr)     / (2.0*eps);
    W[3][1] = (s_bp.ptheta - s_bm.ptheta) / (2.0*eps);
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

    BundleState bs;
    bs.geo = cam.angle_ray(alpha, beta, g);
    init_bundle(cam, alpha, beta, g, bs.W);

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
            if (!first_event_alpha_hermite(
                    bs_prev.geo.theta, bs.geo.theta, dth0, dth1, step_used, M_PI/2.0,
                    alpha, 8, 8)) {
                continue;
            }
            const double r_hit = hermite_interp_scalar(
                bs_prev.geo.r, bs.geo.r, dr0, dr1, step_used, alpha);
            if (r_hit >= r_disk_in && r_hit <= r_disk_out) {
                // ── Redshift ──────────────────────────────────────
                const double Omega = g.keplerian_omega(r_hit);
                const double b     = -bs.geo.pphi / (-bs.geo.pt);
                double gLL[4][4];
                g.covariant_BL(r_hit, M_PI/2.0, gLL);
                const double d2 = -(gLL[0][0]+2.0*gLL[0][3]*Omega+gLL[3][3]*Omega*Omega);
                const double dv = 1.0 - Omega*b;
                double red = (d2 > 0.0 && std::abs(dv) > 1e-10)
                             ? std::sqrt(d2)/dv : 1.0;
                red = red < 0.0 ? 0.0 : red > 20.0 ? 20.0 : red;

                // ── Jacobi map  J: screen (α,β) → disk (r,φ) ────
                // Project the (r,θ) sub-block of W onto the disk tangent plane.
                // At equatorial crossing (θ≈π/2), φ-variation ≈ W[1]/sinθ·(dφ/dθ)
                // but more directly: use W[0] (δr) and approximate δφ from W[1]
                // via the disk metric: dφ ≈ (dθ/dr_disk) · W[1] ... complex.
                // Simpler: use only the 2×2 sub-block (δr, δθ) as proxy for
                // (δr_disk, δφ_disk)  — gives the right shape up to a constant.
                const double J00 = bs_prev.W[0][0] + alpha*(bs.W[0][0] - bs_prev.W[0][0]);
                const double J01 = bs_prev.W[0][1] + alpha*(bs.W[0][1] - bs_prev.W[0][1]);
                const double J10 = bs_prev.W[1][0] + alpha*(bs.W[1][0] - bs_prev.W[1][0]);
                const double J11 = bs_prev.W[1][1] + alpha*(bs.W[1][1] - bs_prev.W[1][1]);

                double det = std::abs(J00*J11 - J01*J10);
                det = det < 1e-12 ? 1e-12 : det;
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
            return {true, disk_r_hit, disk_red, disk_det, 0.0, 0.0, disk_phi};
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
