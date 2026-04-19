#pragma once
// ============================================================
//  Kerr-Newman-de Sitter (KNdS) spacetime
//
//  Four-parameter family of exact solutions to Einstein-Maxwell:
//    M       — ADM mass
//    a       — spin  (|a| ≤ M)
//    Q       — electric charge
//    Lambda  — cosmological constant  (Λ>0: de Sitter, Λ<0: AdS)
//
//  Special cases by zeroing parameters:
//    Q=0, Λ=0          → Kerr
//    a=0, Λ=0          → Reissner-Nordström
//    a=0, Q=0          → Schwarzschild-de Sitter (Kottler)
//    a=0, Q=0, Λ=0     → Schwarzschild
//    Q=0                → Kerr-de Sitter
//
//  Two coordinate charts implemented:
//    (A) Boyer-Lindquist (BL): standard, singular at Δ_r = 0
//    (B) Kerr-Schild Cartesian (KS): regular through horizon, Λ=0
//
//  Signature: (−+++)    Units: G=c=1
// ============================================================
#include <cmath>
#include <algorithm>
#include <limits>

class KNdSMetric {
public:
    double M;       ///< mass
    double a;       ///< spin
    double Q;       ///< charge
    double Lambda;  ///< cosmological constant

    KNdSMetric(double M_=1.0, double a_=0.0,
               double Q_=0.0, double Lambda_=0.0)
        : M(M_), a(a_), Q(Q_), Lambda(Lambda_) {}

    // ──────────────────────────────────────────────────────────
    //  Chart (A) — Boyer-Lindquist
    //  x^μ = (t, r, θ, φ)
    // ──────────────────────────────────────────────────────────

    /// ρ² = r² + a² cos²θ
    double Sigma(double r, double theta) const {
        double ct = std::cos(theta);
        return r*r + a*a*ct*ct;
    }

    /// Δ_r = (r²+a²)(1 − Λr²/3) − 2Mr + Q²
    double Delta_r(double r) const {
        return (r*r + a*a) * (1.0 - Lambda*r*r/3.0) - 2.0*M*r + Q*Q;
    }

    /// Δ_θ = 1 + (Λa²/3) cos²θ   (= 1 for Λ=0)
    double Delta_theta(double theta) const {
        double ct = std::cos(theta);
        return 1.0 + Lambda*a*a*ct*ct/3.0;
    }

    /// Ξ = 1 + Λa²/3             (= 1 for Λ=0)
    double Xi() const { return 1.0 + Lambda*a*a/3.0; }

    // ── BL covariant  g_μν ────────────────────────────────────
    //
    //  ds² = [−Δ_r + Δ_θ a²s²]/(ρ²Ξ²) dt²
    //       + 2 a s² [Δ_r − Δ_θ(r²+a²)]/(ρ²Ξ²) dt dφ
    //       + ρ²/Δ_r dr² + ρ²/Δ_θ dθ²
    //       + s² [Δ_θ(r²+a²)² − Δ_r a²s²]/(ρ²Ξ²) dφ²
    //  (s² ≡ sin²θ)
    void covariant_BL(double r, double theta, double g[4][4]) const {
        const double sig  = Sigma(r, theta);
        const double dr   = Delta_r(r);
        const double dth  = Delta_theta(theta);
        const double xi2  = Xi()*Xi();
        const double st   = std::sin(theta);
        const double st2  = st*st;
        const double r2a2 = r*r + a*a;
        const double pre  = sig*xi2;   // common denominator

        for (int i=0;i<4;i++) for (int j=0;j<4;j++) g[i][j]=0.0;

        g[0][0] = (-dr + dth*a*a*st2) / pre;
        g[0][3] = g[3][0] = a*st2*(dr - dth*r2a2) / pre;
        g[1][1] = sig/dr;
        g[2][2] = sig/dth;
        g[3][3] = st2*(dth*r2a2*r2a2 - dr*a*a*st2) / pre;
    }

    // ── BL contravariant  g^μν ────────────────────────────────
    //
    //  Derived analytically; det of (t,φ) block = −Δ_r Δ_θ s²/Ξ⁴
    //
    //  g^tt   = −Ξ² [Δ_θ(r²+a²)² − Δ_r a²s²] / (ρ² Δ_r Δ_θ)
    //  g^tφ   =  Ξ² a [Δ_r − Δ_θ(r²+a²)]     / (ρ² Δ_r Δ_θ)
    //  g^rr   = Δ_r / ρ²
    //  g^θθ   = Δ_θ / ρ²
    //  g^φφ   =  Ξ² [Δ_r − Δ_θ a²s²]          / (ρ² Δ_r Δ_θ s²)
    void contravariant_BL(double r, double theta, double gUU[4][4]) const {
        const double sig  = Sigma(r, theta);
        const double dr   = Delta_r(r);
        const double dth  = Delta_theta(theta);
        const double xi2  = Xi()*Xi();
        const double st   = std::sin(theta);
        const double st2  = st*st;
        const double r2a2 = r*r + a*a;
        const double pre  = sig*dr*dth; // common denominator (before Ξ²)

        for (int i=0;i<4;i++) for (int j=0;j<4;j++) gUU[i][j]=0.0;

        gUU[0][0] = -xi2*(dth*r2a2*r2a2 - dr*a*a*st2) / pre;
        gUU[0][3] = gUU[3][0] = a*xi2*(dr - dth*r2a2) / pre;
        gUU[1][1] = dr/sig;
        gUU[2][2] = dth/sig;
        if (st2 > 1e-14)
            gUU[3][3] = xi2*(dr - dth*a*a*st2) / (pre*st2);
    }

    // ── Hamiltonian  H = ½ g^μν p_μ p_ν  (BL coords) ──────────
    //  p_t = −E  and  p_φ = L  are conserved (Killing symmetries).
    //  H = 0 on null geodesics, H = −½ on unit timelike geodesics.
    double hamiltonian(double r, double theta,
                       double pr, double ptheta,
                       double pt, double pphi) const {
        double gUU[4][4];
        contravariant_BL(r, theta, gUU);
        return 0.5*(  gUU[0][0]*pt*pt
                    + 2.0*gUU[0][3]*pt*pphi
                    + gUU[1][1]*pr*pr
                    + gUU[2][2]*ptheta*ptheta
                    + gUU[3][3]*pphi*pphi);
    }

    // ──────────────────────────────────────────────────────────
    //  Chart (B) — Kerr-Schild Cartesian  (Λ = 0 only)
    //  x^μ = (T, X, Y, Z)
    //
    //  g_ab = η_ab + H · l_a · l_b
    //
    //  H = (2Mr − Q²) / ρ²    ρ² = r² + a²Z²/r²  (= Σ in BL)
    //
    //  l_a^{in}  = ( 1,  (rX+aY)/(r²+a²),  (rY−aX)/(r²+a²),  Z/r )
    //  l_a^{out} = (−1,  (rX−aY)/(r²+a²),  (rY+aX)/(r²+a²),  Z/r )
    //
    //  r(X,Y,Z) is the largest positive root of
    //    r⁴ − (X²+Y²+Z²−a²) r² − a²Z² = 0
    // ──────────────────────────────────────────────────────────

    /// Implicit KS radial coordinate  r(X,Y,Z)
    static double r_KS(double X, double Y, double Z, double a_spin) {
        const double R2 = X*X + Y*Y + Z*Z;
        const double a2 = a_spin*a_spin;
        const double b  = R2 - a2;
        return std::sqrt(0.5*(b + std::sqrt(b*b + 4.0*a2*Z*Z)));
    }

    /// KS scalar  H(r, ρ²)
    double H_KS(double r, double rho2) const {
        return (2.0*M*r - Q*Q) / rho2;
    }

    /// KS null co-vector  l_a   (ingoing: sign=+1, outgoing: sign=−1)
    //
    //  Ingoing:  l_a = ( 1,  (rX+aY)/(r²+a²),  (rY−aX)/(r²+a²),  Z/r )
    //  Outgoing: l_a = (−1,  (rX−aY)/(r²+a²),  (rY+aX)/(r²+a²),  Z/r )
    //
    //  Note: the ±a flip is ON the cross-terms, not on the whole vector.
    static void null_covector(double X, double Y, double Z,
                              double r, double a_spin,
                              int sign,   // +1 ingoing, −1 outgoing
                              double l[4]) {
        const double r2a2 = r*r + a_spin*a_spin;
        const double sa   = double(sign) * a_spin;  // ±a
        l[0] =  double(sign);
        l[1] = (r*X + sa*Y) / r2a2;
        l[2] = (r*Y - sa*X) / r2a2;
        l[3] =  Z / r;
    }

    /// KS covariant metric  g_ab  (Λ=0)
    void covariant_KS(double /*T*/, double X, double Y, double Z,
                      bool ingoing, double g[4][4]) const {
        const double r    = r_KS(X, Y, Z, a);
        const double rho2 = r*r + a*a*Z*Z/(r*r);
        const double H    = H_KS(r, rho2);
        double l[4];
        null_covector(X, Y, Z, r, a, ingoing ? +1 : -1, l);

        // Minkowski background η = diag(−1,+1,+1,+1)
        for (int i=0;i<4;i++) for (int j=0;j<4;j++) g[i][j] = 0.0;
        g[0][0]=-1.0; g[1][1]=1.0; g[2][2]=1.0; g[3][3]=1.0;

        // Perturbation  H l_a l_b
        for (int mu=0;mu<4;mu++)
            for (int nu=0;nu<4;nu++)
                g[mu][nu] += H * l[mu] * l[nu];
    }

    // ── KS contravariant  g^ab = η^ab − H l^a l^b  (Λ=0) ─────
    //  Valid because l is null w.r.t. η: η^μν l_μ l_ν = 0
    //  ⟹ the Sherman-Morrison inverse collapses to this form.
    void contravariant_KS(double /*T*/, double X, double Y, double Z,
                          bool ingoing, double gUU[4][4]) const {
        const double r    = r_KS(X, Y, Z, a);
        const double rho2 = r*r + a*a*Z*Z/(r*r);
        const double H    = H_KS(r, rho2);
        double l[4];
        null_covector(X, Y, Z, r, a, ingoing ? +1 : -1, l);

        // l^a = η^ab l_b  →  l^T = -l[0], l^i = +l[i]
        const double lU[4] = {-l[0], l[1], l[2], l[3]};

        for (int i=0;i<4;i++) for (int j=0;j<4;j++) gUU[i][j]=0.0;
        gUU[0][0]=-1.0; gUU[1][1]=1.0; gUU[2][2]=1.0; gUU[3][3]=1.0;
        for (int mu=0;mu<4;mu++)
            for (int nu=0;nu<4;nu++)
                gUU[mu][nu] -= H * lU[mu] * lU[nu];
    }

    // ── Hamiltonian in KS coords  (Λ=0) ───────────────────────
    double hamiltonian_KS(double T, double X, double Y, double Z,
                          const double p[4], bool ingoing) const {
        double gUU[4][4];
        contravariant_KS(T, X, Y, Z, ingoing, gUU);
        double H = 0.0;
        for (int mu=0;mu<4;mu++)
            for (int nu=0;nu<4;nu++)
                H += gUU[mu][nu]*p[mu]*p[nu];
        return 0.5*H;
    }

    // ──────────────────────────────────────────────────────────
    //  Horizon / ISCO utilities
    // ──────────────────────────────────────────────────────────

    /// Outer event horizon  r₊  (largest root of Δ_r = 0)
    double r_horizon() const {
        // Flat case (Λ=0): exact Kerr-Newman outer horizon.
        if (std::abs(Lambda) < 1e-15) {
            const double disc = M*M - a*a - Q*Q;
            return M + std::sqrt(std::max(disc, 0.0));
        }

        // General Λ: detect sign changes of Δ_r on a logarithmic radial scan,
        // then refine each root by bisection.
        const double scale = std::max(std::abs(M), 1.0);
        const double rmin  = std::max(1e-6*scale, 1e-9);
        double rmax;
        if (Lambda > 0.0)
            rmax = std::max(20.0*scale, 1.25*std::sqrt(3.0/Lambda));
        else
            rmax = 200.0*scale + 20.0*std::abs(a) + 20.0*std::abs(Q);
        if (!(rmax > rmin)) rmax = rmin * 10.0;

        auto bisect = [&](double lo, double hi) -> double {
            double flo = Delta_r(lo), fhi = Delta_r(hi);
            if (!std::isfinite(flo) || !std::isfinite(fhi))
                return std::numeric_limits<double>::quiet_NaN();
            for (int i = 0; i < 160; ++i) {
                const double mid = 0.5*(lo + hi);
                const double fmid = Delta_r(mid);
                if (!std::isfinite(fmid)) break;
                if ((flo > 0.0 && fmid < 0.0) || (flo < 0.0 && fmid > 0.0)) {
                    hi  = mid;
                    fhi = fmid;
                } else {
                    lo  = mid;
                    flo = fmid;
                }
            }
            return 0.5*(lo + hi);
        };

        double roots[16];
        int nroots = 0;
        auto push_root = [&](double rr) {
            if (!std::isfinite(rr) || rr <= 0.0) return;
            for (int i = 0; i < nroots; ++i) {
                const double rel = std::max({1.0, std::abs(roots[i]), std::abs(rr)});
                if (std::abs(roots[i] - rr) <= 1e-8 * rel) return;
            }
            if (nroots < 16) roots[nroots++] = rr;
        };

        const int samples = 20000;
        const double log_span = std::log(rmax / rmin);
        double prev_r = rmin;
        double prev_f = Delta_r(prev_r);
        for (int i = 1; i <= samples; ++i) {
            const double t = double(i) / double(samples);
            const double r = rmin * std::exp(log_span * t);
            const double f = Delta_r(r);

            if (std::isfinite(prev_f) && std::isfinite(f)) {
                if (prev_f == 0.0) push_root(prev_r);
                if (f == 0.0)      push_root(r);
                if ((prev_f > 0.0 && f < 0.0) || (prev_f < 0.0 && f > 0.0))
                    push_root(bisect(prev_r, r));
            }
            prev_r = r;
            prev_f = f;
        }

        if (nroots == 0) {
            // Fallback for edge cases where the scan does not resolve a crossing.
            const double disc = M*M - a*a - Q*Q;
            return M + std::sqrt(std::max(disc, 0.0));
        }

        std::sort(roots, roots + nroots);
        // For Λ>0 the largest positive root is typically cosmological:
        // choose the next one (outer BH horizon) when available.
        if (Lambda > 0.0 && nroots >= 2) return roots[nroots - 2];
        return roots[nroots - 1];
    }

    /// ISCO radius (prograde equatorial)
    double r_isco() const {
        // Analytical for pure Kerr (Q=Λ=0)
        if (std::abs(Lambda) < 1e-15 && std::abs(Q) < 1e-15) {
            if (std::abs(a) < 1e-10) return 6.0*M;
            double ap = a/M;
            double z1 = 1.0 + std::cbrt(1.0-ap*ap)
                            * (std::cbrt(1.0+ap)+std::cbrt(1.0-ap));
            double z2 = std::sqrt(3.0*ap*ap + z1*z1);
            return M*(3.0 + z2 - std::sqrt((3.0-z1)*(3.0+z1+2.0*z2)));
        }
        // General: find innermost stable circular orbit numerically
        // dL_circ/dr = 0  (L = specific angular momentum of circular orbit)
        double rh  = r_horizon();
        double r   = rh * 1.05;
        double dr  = 1e-4*M;
        auto L_circ = [&](double rr) -> double {
            double Omega = keplerian_omega(rr);
            double gLL[4][4];
            covariant_BL(rr, M_PI/2.0, gLL);
            double N2 = -(gLL[0][0] + 2.0*gLL[0][3]*Omega + gLL[3][3]*Omega*Omega);
            if (N2 <= 0.0) return -1e30;
            double ut = 1.0/std::sqrt(N2);
            return -(gLL[3][0] + gLL[3][3]*Omega)*ut;
        };
        double L_prev = L_circ(r);
        for (; r < 30.0*M; r += dr) {
            double L = L_circ(r + dr);
            if (L < L_prev && L_prev > 0.0) return r; // passed the minimum
            L_prev = L;
        }
        return r;  // fallback
    }

    /// Prograde Keplerian Ω_K  (equatorial, circular orbit)
    //  For KNdS: Ω_K = [M − Q²/(2r) + Λar²/3] / [r^{3/2} + a·√(M − Q²/(2r))]
    //  (reduces to standard Kerr formula for Q=Λ=0)
    double keplerian_omega(double r) const {
        // Tie disk orbital direction to BH spin sign:
        // a > 0 -> Ω > 0, a < 0 -> Ω < 0.
        const double s = (a < 0.0) ? -1.0 : 1.0;
        const double Meff = M - Q*Q/(2.0*r) + Lambda*a*r*r/3.0;
        const double sq   = std::sqrt(std::max(Meff, 0.0));
        const double den  = r*std::sqrt(r) + s*a*sq;
        if (std::abs(den) < 1e-14) return 0.0;
        return s * sq / den;
    }

    // ── BL ↔ KS Cartesian coordinate transforms (Λ=0) ─────────

    /// BL (r, θ, φ)  →  KS Cartesian (X, Y, Z)
    static void BL_to_KS_spatial(double r, double theta, double phi,
                                  double a_spin,
                                  double& X, double& Y, double& Z) {
        // Standard Kerr-Schild embedding:
        //   X = √(r²+a²) sinθ cos(φ − arctan(a/r))
        //   Y = √(r²+a²) sinθ sin(φ − arctan(a/r))
        //   Z = r cosθ
        // Equivalently (avoiding arctan):
        //   X = sinθ (r cosφ + a sinφ)   [uses r and a combined]
        //   Y = sinθ (r sinφ − a cosφ)
        //   Z = r cosθ
        // Note: X²+Y² = (r²+a²)sin²θ  ✓
        const double st = std::sin(theta), ct = std::cos(theta);
        const double sf = std::sin(phi),   cf = std::cos(phi);
        X = st*(r*cf + a_spin*sf);
        Y = st*(r*sf - a_spin*cf);
        Z = r*ct;
    }

    /// KS (X, Y, Z)  →  BL (r, θ, φ)
    static void KS_to_BL_spatial(double X, double Y, double Z,
                                  double a_spin,
                                  double& r, double& theta, double& phi) {
        r     = r_KS(X, Y, Z, a_spin);
        theta = std::atan2(std::sqrt(X*X + Y*Y + Z*Z - r*r + a_spin*a_spin
                                     - a_spin*a_spin*(Z*Z)/(r*r)),
                           Z/r*r);   // simplified: just atan2(r·sinθ, Z)
        // Simpler: Z = r cosθ
        double Zr = Z/r;
        theta = std::acos(Zr < -1.0 ? -1.0 : Zr > 1.0 ? 1.0 : Zr);
        phi   = std::atan2(Y*r - X*a_spin, X*r + Y*a_spin);
        // Derived from:  X = sinθ(r cosφ + a sinφ),  Y = sinθ(r sinφ − a cosφ)
        // X*r + Y*a = sinθ(r²cosφ + ra sinφ − ra sinφ + a² cosφ) ... hmm
        // Let me use the direct formula:
        // r X/r2a2 = sinθ cosφ ... wait, no:
        // X = st(r cf + a sf), so X·r - Y·a = st(r²cf + rasf - rasf + a²cf) = st(r²+a²)cf
        // So cosφ = (Xr - Ya) / (st(r²+a²))  ... st = sinθ = √(1-(Z/r)²)
        double st_val = std::sqrt(std::max(1.0 - Z*Z/(r*r), 0.0));
        double r2a2   = r*r + a_spin*a_spin;
        if (st_val > 1e-10) {
            double cf_val = (X*r + Y*a_spin) / (st_val * r2a2);
            double sf_val = (Y*r - X*a_spin) / (st_val * r2a2);
            phi = std::atan2(sf_val, cf_val);
        } else {
            phi = 0.0;
        }
    }
};
