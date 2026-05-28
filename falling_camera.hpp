#pragma once
// falling_camera.hpp — GPG coordinates for KNdS (Lin & Soo 2009, arXiv:0905.3244)
// Coordinate order: 0=T, 1=r, 2=θ, 3=φ_P

#include "knds_metric.hpp"
#include <cmath>

// ── GPG helper: f(r) ─────────────────────────────────────────────────────────
// Ensures f² > max(Δ, a²Ξ) for all (M,a,Q,Λ).
inline double gpg_f(const KNdSMetric& bh, double r) {
    const double R2  = r*r + bh.a*bh.a;
    const double Q2  = bh.Q*bh.Q;
    const double Lam = bh.Lambda;
    if (Lam >= 0.0)
        return std::sqrt(R2 + Q2 + Lam*bh.a*bh.a*bh.a*bh.a/3.0);
    else
        return std::sqrt(R2*(1.0 - Lam*r*r/3.0) + Q2);
}

// ── Covariant GPG metric g_μν ─────────────────────────────────────────────────
// Fills gLL[4][4]. Index order: 0=T,1=r,2=θ,3=φ_P.
inline void gpg_covariant(const KNdSMetric& bh, double r, double theta,
                           double gLL[4][4])
{
    const double a   = bh.a;
    const double M   = bh.M;
    const double Q2  = bh.Q*bh.Q;
    const double Lam = bh.Lambda;

    const double R2  = r*r + a*a;
    const double rho2= r*r + a*a*std::cos(theta)*std::cos(theta);
    const double Xi  = 1.0 + Lam*a*a/3.0;
    const double Xit = 1.0 + Lam*a*a*std::cos(theta)*std::cos(theta)/3.0;
    const double Del = R2*(1.0 - Lam*r*r/3.0) - 2.0*M*r + Q2;
    const double f   = gpg_f(bh, r);
    const double f2  = f*f;
    const double s   = std::sin(theta);
    const double s2  = s*s;

    const double D   = f2 - a*a*Xit*s2;
    const double sqD = std::sqrt(std::max(D, 0.0));
    const double sqFD= std::sqrt(std::max(f2 - Del, 0.0));
    const double sqXit= std::sqrt(std::max(Xit, 0.0));  // guard for AdS Λ<0

    const double rho = std::sqrt(rho2);

    const double e0T  =  sqD / (Xi*rho);
    const double e0ph = (sqD > 1e-15) ? a*s2*(Xit*R2 - f2) / (Xi*rho*sqD) : 0.0;

    const double e1T  =  sqFD / (Xi*rho);
    const double e1r  =  rho / f;
    const double e1ph = -a*s2*sqFD / (Xi*rho);

    const double e2th =  rho / std::max(std::sqrt(Xit), 1e-15);

    const double e3T  = (sqD > 1e-15) ? -a*rho*sqXit*s / (f*Xi*sqD) : 0.0;
    const double e3ph = (sqD > 1e-15) ?  rho*R2*sqXit*s / (f*Xi*sqD) : 0.0;

    for (int i=0;i<4;++i) for (int j=0;j<4;++j) gLL[i][j]=0.0;

    gLL[0][0] = -e0T*e0T + e1T*e1T + e3T*e3T;
    gLL[1][1] = e1r*e1r;
    gLL[2][2] = e2th*e2th;
    gLL[3][3] = -e0ph*e0ph + e1ph*e1ph + e3ph*e3ph;
    gLL[0][1] = gLL[1][0] = e1T*e1r;
    gLL[0][3] = gLL[3][0] = -e0T*e0ph + e1T*e1ph + e3T*e3ph;
    gLL[1][3] = gLL[3][1] = e1r*e1ph;
}

// ── Contravariant GPG metric g^μν ─────────────────────────────────────────────
inline void gpg_contravariant(const KNdSMetric& bh, double r, double theta,
                               double gUU[4][4])
{
    double gLL[4][4];
    gpg_covariant(bh, r, theta, gLL);

    const auto& m = gLL;
    auto cofactor = [&](int r0, int c0) -> double {
        double sub[3][3]; int ri=0;
        for (int i=0;i<4;++i) { if(i==r0) continue; int ci=0;
            for (int j=0;j<4;++j) { if(j==c0) continue; sub[ri][ci++]=m[i][j]; } ++ri; }
        return sub[0][0]*(sub[1][1]*sub[2][2]-sub[1][2]*sub[2][1])
              -sub[0][1]*(sub[1][0]*sub[2][2]-sub[1][2]*sub[2][0])
              +sub[0][2]*(sub[1][0]*sub[2][1]-sub[1][1]*sub[2][0]);
    };
    double det = 0.0;
    for (int j=0;j<4;++j) det += m[0][j]*cofactor(0,j)*((j%2==0)?1:-1);
    const double inv_det = (std::abs(det) > 1e-30) ? 1.0/det : 0.0;
    for (int i=0;i<4;++i)
        for (int j=0;j<4;++j)
            gUU[i][j] = cofactor(j,i)*((i+j)%2==0?1:-1)*inv_det;
}

// ── Falling camera parameter block ───────────────────────────────────────────
struct FallingParams {
    KNdSMetric bh;
    double r_start     = 20.0;
    double E           = 1.0;    // conserved energy (1 = fall from rest at ∞)
    double L           = 0.0;    // conserved angular momentum [M]
    double Qc          = 0.0;    // Carter constant [M²]
    double theta_start = M_PI/2;
    double phi_start   = 0.0;
    double fov_h       = 90.0 * M_PI/180.0;
    int    width       = 1280;
    int    height      = 720;
    int    frames      = 120;
    double dtau        = 0.1;
    double r_disk_in   = -1.0;   // <0 = use ISCO
    double r_disk_out  = 12.0;
    double r_escape    = 200.0;
    double r_singularity = 0.05;
    double disk_brightness = 1.0;
};

// ── Camera state in GPG coordinates ──────────────────────────────────────────
// x[0]=T, x[1]=r, x[2]=θ, x[3]=φ_P   (position)
// u[0..3]                              (contravariant four-velocity)
struct CameraState {
    double x[4];
    double u[4];
};

// ── Initial worldline from conserved quantities ───────────────────────────────
inline CameraState init_worldline(const FallingParams& fp) {
    const KNdSMetric& bh = fp.bh;
    const double r  = fp.r_start;
    const double th = fp.theta_start;

    double gLL[4][4], gUU[4][4];
    gpg_covariant(bh, r, th, gLL);
    gpg_contravariant(bh, r, th, gUU);

    // Covariant conserved components
    const double uT_low  = -fp.E;
    const double uph_low =  fp.L;

    // u_θ from Carter constant (Kerr-leading-order form)
    const double cos2 = std::cos(th)*std::cos(th);
    const double sin2 = std::sin(th)*std::sin(th);
    const double under = fp.Qc - cos2*(bh.a*bh.a*fp.E*fp.E
                                       - fp.L*fp.L/std::max(sin2, 1e-10));
    const double uth_low = std::sqrt(std::max(under, 0.0));

    // Solve for u_r: normalization g^μν u_μ u_ν = -1
    // g^rr ur² + 2(g^rT uT + g^rφ uφ) ur + C = -1
    const double C = gUU[0][0]*uT_low*uT_low
                   + 2.0*gUU[0][3]*uT_low*uph_low
                   + gUU[3][3]*uph_low*uph_low
                   + gUU[2][2]*uth_low*uth_low;
    const double A = gUU[1][1];
    const double B = 2.0*(gUU[1][0]*uT_low + gUU[1][3]*uph_low);
    const double disc = B*B - 4.0*A*(C+1.0);
    // Ingoing root (ur < 0)
    const double ur_low = (-B - std::sqrt(std::max(disc, 0.0))) / (2.0*A);

    // Raise indices: u^μ = g^μν u_ν
    CameraState cs;
    cs.x[0] = 0.0; cs.x[1] = r; cs.x[2] = th; cs.x[3] = fp.phi_start;
    cs.u[0] = gUU[0][0]*uT_low + gUU[0][1]*ur_low + gUU[0][3]*uph_low;
    cs.u[1] = gUU[1][0]*uT_low + gUU[1][1]*ur_low + gUU[1][3]*uph_low;
    cs.u[2] = gUU[2][2]*uth_low;
    cs.u[3] = gUU[3][0]*uT_low + gUU[3][1]*ur_low + gUU[3][3]*uph_low;
    return cs;
}
