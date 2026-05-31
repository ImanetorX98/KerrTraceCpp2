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
    double disk_brightness   = 1.0;
    double r_switch_factor   = 3.0;  // CPU refinement: r_min < factor*r_h → re-trace double
    double background_gray      = 30.0;  // escaped-photon background brightness [0-255]
    bool   look_outward         = false; // true → camera faces radially outward (+r), no horizon flip
    bool   interstellar_palette = false; // true → thermal gradient (gold→orange→dark red)
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

// ── Numerical Christoffel Γ^μ_{αβ} from GPG metric ───────────────────────────
// Central finite differences. Gamma[mu][alpha][beta].
inline void gpg_christoffel(const KNdSMetric& bh, double r, double th,
                             double Gamma[4][4][4])
{
    const double hr = r  * 1e-5 + 1e-9;
    const double ht = 1e-5;

    double gp[4][4], gm[4][4], gtp[4][4], gtm[4][4];
    gpg_covariant(bh, r+hr, th,    gp);
    gpg_covariant(bh, r-hr, th,    gm);
    gpg_covariant(bh, r,    th+ht, gtp);
    gpg_covariant(bh, r,    th-ht, gtm);

    // dg[coord][mu][nu]: coord=0→T(not computed,stationary), 1→r, 2→θ, 3→φ(not computed,axisymmetric)
    double dg[4][4][4] = {};
    for (int i=0;i<4;++i) for (int j=0;j<4;++j) {
        dg[1][i][j] = (gp[i][j] - gm[i][j]) / (2.0*hr);
        dg[2][i][j] = (gtp[i][j]- gtm[i][j])/ (2.0*ht);
        // dg[0] and dg[3] remain zero (stationary + axisymmetric metric)
    }

    double gUU[4][4];
    gpg_contravariant(bh, r, th, gUU);

    for (int mu=0;mu<4;++mu)
        for (int al=0;al<4;++al)
            for (int be=0;be<4;++be) {
                double s=0.0;
                for (int nu=0;nu<4;++nu)
                    s += gUU[mu][nu]*(dg[al][nu][be] + dg[be][nu][al] - dg[nu][al][be]);
                Gamma[mu][al][be] = 0.5*s;
            }
}

// ── RK4 step for worldline ────────────────────────────────────────────────────
inline CameraState step_worldline(const CameraState& cs,
                                   const KNdSMetric& bh, double dtau)
{
    auto deriv = [&](const CameraState& s, double dxdt[4], double dudt[4]) {
        double Gamma[4][4][4];
        gpg_christoffel(bh, s.x[1], s.x[2], Gamma);
        for (int mu=0;mu<4;++mu) {
            dxdt[mu] = s.u[mu];
            double acc=0.0;
            for (int al=0;al<4;++al)
                for (int be=0;be<4;++be)
                    acc -= Gamma[mu][al][be]*s.u[al]*s.u[be];
            dudt[mu] = acc;
        }
    };

    double dx1[4],du1[4], dx2[4],du2[4], dx3[4],du3[4], dx4[4],du4[4];
    CameraState tmp;

    deriv(cs, dx1, du1);
    for (int i=0;i<4;++i){tmp.x[i]=cs.x[i]+0.5*dtau*dx1[i]; tmp.u[i]=cs.u[i]+0.5*dtau*du1[i];}
    deriv(tmp, dx2, du2);
    for (int i=0;i<4;++i){tmp.x[i]=cs.x[i]+0.5*dtau*dx2[i]; tmp.u[i]=cs.u[i]+0.5*dtau*du2[i];}
    deriv(tmp, dx3, du3);
    for (int i=0;i<4;++i){tmp.x[i]=cs.x[i]+dtau*dx3[i]; tmp.u[i]=cs.u[i]+dtau*du3[i];}
    deriv(tmp, dx4, du4);

    CameraState next;
    for (int i=0;i<4;++i) {
        next.x[i] = cs.x[i] + (dtau/6.0)*(dx1[i]+2*dx2[i]+2*dx3[i]+dx4[i]);
        next.u[i] = cs.u[i] + (dtau/6.0)*(du1[i]+2*du2[i]+2*du3[i]+du4[i]);
    }

    // Normalization re-projection: adjust u^r so g_μν u^μ u^ν = -1
    {
        double gLL[4][4], gUU[4][4];
        gpg_covariant(bh, next.x[1], next.x[2], gLL);
        gpg_contravariant(bh, next.x[1], next.x[2], gUU);
        double C=0.0;
        for (int mu=0;mu<4;++mu) for (int nu=0;nu<4;++nu) {
            if (mu==1||nu==1) continue;
            C += gLL[mu][nu]*next.u[mu]*next.u[nu];
        }
        const double A2 = gLL[1][1];
        const double B2 = 2.0*(gLL[1][0]*next.u[0]+gLL[1][2]*next.u[2]+gLL[1][3]*next.u[3]);
        const double disc2 = B2*B2 - 4.0*A2*(C+1.0);
        if (disc2 >= 0.0) {
            const double ur_neg = (-B2 - std::sqrt(disc2))/(2.0*A2);
            const double ur_pos = (-B2 + std::sqrt(disc2))/(2.0*A2);
            next.u[1] = (next.u[1] < 0.0) ? ur_neg : ur_pos;
        }
    }
    return next;
}

// ── CameraTetrad: metric Gram-Schmidt ─────────────────────────────────────────
// e[a][mu]: a=tetrad index (0=time,1=radial,2=polar,3=azimuthal), mu=coord index
// Satisfies: g_μν e[a]^μ e[b]^ν = η_ab = diag(-1,+1,+1,+1)
inline void build_tetrad(const CameraState& cs, const KNdSMetric& bh,
                          double e[4][4])
{
    double gLL[4][4];
    gpg_covariant(bh, cs.x[1], cs.x[2], gLL);

    auto inner = [&](const double* v, const double* w) -> double {
        double s=0.0;
        for (int mu=0;mu<4;++mu) for (int nu=0;nu<4;++nu)
            s += gLL[mu][nu]*v[mu]*w[nu];
        return s;
    };
    auto normalize_v = [&](double* v, double sign) {
        double n2 = sign*inner(v,v);
        double n  = std::sqrt(std::max(n2, 1e-30));
        for (int mu=0;mu<4;++mu) v[mu] /= n;
    };
    auto subtract_proj = [&](double* v, const double* basis, double basis_norm2) {
        double coeff = inner(v, basis) / basis_norm2;
        for (int mu=0;mu<4;++mu) v[mu] -= coeff*basis[mu];
    };

    // e[0] = u^μ (timelike unit, already normalized by init_worldline)
    for (int mu=0;mu<4;++mu) e[0][mu] = cs.u[mu];
    normalize_v(e[0], -1.0);

    // Candidate coordinate seeds in order T,r,θ,φ
    static const double coord_seeds[4][4] = {
        {1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}
    };

    // Build spacelike basis vectors e[1], e[2], e[3] via Gram-Schmidt.
    // For each basis vector we try candidate seeds in priority order, picking
    // the first whose residual (after projecting out already-built vectors)
    // has n² = g(res,res) > 1e-10 to avoid degenerate directions near horizon.
    int used_seed[4] = {-1,-1,-1,-1}; // track which coord direction each slot used
    used_seed[0] = -1; // e[0] = u, no coord seed

    // Priority order for e[1]: r, θ, φ, T
    static const int pref1[4] = {1,2,3,0};
    // Priority order for e[2]: θ, φ, T, r
    static const int pref2[4] = {2,3,0,1};
    // Priority order for e[3]: φ, T, r, θ
    static const int pref3[4] = {3,0,1,2};

    auto try_build = [&](int slot, const int pref[4]) {
        for (int ki=0; ki<4; ++ki) {
            int k = pref[ki];
            // Don't reuse a seed already chosen for a previous slot
            bool already_used = false;
            for (int s=0; s<slot; ++s) if (used_seed[s] == k) { already_used=true; break; }
            if (already_used) continue;

            double seed[4];
            for (int mu=0;mu<4;++mu) seed[mu] = coord_seeds[k][mu];
            // Project out all previously built basis vectors
            subtract_proj(seed, e[0], inner(e[0],e[0]));
            for (int s=1; s<slot; ++s)
                subtract_proj(seed, e[s], inner(e[s],e[s]));
            double n2 = inner(seed, seed); // should be positive (spacelike)
            if (n2 > 1e-10) {
                for (int mu=0;mu<4;++mu) e[slot][mu] = seed[mu];
                normalize_v(e[slot], 1.0);
                used_seed[slot] = k;
                return;
            }
        }
        // Fallback: use whatever we got from the first preference (shouldn't happen
        // in well-behaved spacetimes, but avoids silent garbage)
        int k = pref[0];
        double seed[4];
        for (int mu=0;mu<4;++mu) seed[mu] = coord_seeds[k][mu];
        subtract_proj(seed, e[0], inner(e[0],e[0]));
        for (int s=1; s<slot; ++s)
            subtract_proj(seed, e[s], inner(e[s],e[s]));
        for (int mu=0;mu<4;++mu) e[slot][mu] = seed[mu];
        normalize_v(e[slot], 1.0);
        used_seed[slot] = k;
    };

    try_build(1, pref1);
    try_build(2, pref2);
    try_build(3, pref3);
}

// ── Apply roll ψ around ê_3 (azimuthal axis) ─────────────────────────────────
// Rotates ê_1 (radial) and ê_2 (polar) in the radial-polar plane.
inline void apply_roll(double e[4][4], double psi) {
    double e1[4], e2[4];
    for (int mu=0;mu<4;++mu) { e1[mu]=e[1][mu]; e2[mu]=e[2][mu]; }
    for (int mu=0;mu<4;++mu) {
        e[1][mu] =  std::cos(psi)*e1[mu] + std::sin(psi)*e2[mu];
        e[2][mu] = -std::sin(psi)*e1[mu] + std::cos(psi)*e2[mu];
    }
}

// ── HorizonFlip: ψ(r) ────────────────────────────────────────────────────────
// Returns ψ ∈ [0, π].
// r ≥ r_far  (far from horizon): ψ = π   → camera looks toward BH (inward)
// r ≤ r_near (inside horizon):   ψ = 0   → camera looks outward (back through horizon)
// Transition: cubic smoothstep between r_near and r_far.
inline double horizon_flip_psi(double r, double r_horizon,
                                double delta_out=2.0, double delta_in=0.8)
{
    const double r_far  = r_horizon * delta_out;
    const double r_near = r_horizon * delta_in;
    if (r >= r_far)  return M_PI;
    if (r <= r_near) return 0.0;
    double t = (r - r_near) / (r_far - r_near);
    double smooth = t*t*(3.0 - 2.0*t);
    return M_PI * smooth;
}
