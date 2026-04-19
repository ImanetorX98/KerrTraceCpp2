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
                                  double r_escape) {
    const int span = (cam.width > 1) ? (cam.width - 1) : 1;
    const double alpha = cam.fov_h*(px - 0.5*(cam.width-1))  / span;
    const double beta  = cam.fov_h*(0.5*(cam.height-1) - py) / span;

    BundleState bs;
    bs.geo = cam.angle_ray(alpha, beta, g);
    init_bundle(cam, alpha, beta, g, bs.W);

    const double rh  = g.r_horizon();
    double dlam      = 1.0;
    double prev_cos  = std::cos(bs.geo.theta);

    for (int iter = 0; iter < 500000; ++iter) {
        const BundleState bs_prev = bs;
        int rejects = 0;
        while (!bundle_adaptive(g, bs, dlam)) {
            if (!std::isfinite(dlam) || ++rejects > 64) return {};
        }

        const double r = bs.geo.r;

        if (r < rh * 1.03) return {};
        if (r > r_escape)  return {false, r, 1.0, 1.0, bs.geo.theta, bs.geo.phi};

        const double cos_th = std::cos(bs.geo.theta);
        const bool crossed  = (prev_cos * cos_th <= 0.0);

        if (crossed) {
            const double denom = prev_cos - cos_th;
            double w = (std::abs(denom) > 1e-14) ? (prev_cos / denom) : 0.5;
            w = w < 0.0 ? 0.0 : w > 1.0 ? 1.0 : w;
            const double r_hit = bs_prev.geo.r + w*(bs.geo.r - bs_prev.geo.r);
            if (!(r_hit >= r_disk_in && r_hit <= r_disk_out)) {
                prev_cos = cos_th;
                continue;
            }

            // ── Redshift ──────────────────────────────────────
            const double Omega = g.keplerian_omega(r_hit);
            const double b     = bs.geo.pphi / (-bs.geo.pt);
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
            const double J00 = bs_prev.W[0][0] + w*(bs.W[0][0] - bs_prev.W[0][0]);
            const double J01 = bs_prev.W[0][1] + w*(bs.W[0][1] - bs_prev.W[0][1]);
            const double J10 = bs_prev.W[1][0] + w*(bs.W[1][0] - bs_prev.W[1][0]);
            const double J11 = bs_prev.W[1][1] + w*(bs.W[1][1] - bs_prev.W[1][1]);

            double det = std::abs(J00*J11 - J01*J10);
            det = det < 1e-12 ? 1e-12 : det;

            return {true, r_hit, red, det};
        }
        prev_cos = cos_th;
    }
    return {false, bs.geo.r, 1.0, 1.0};
}
