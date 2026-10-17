#pragma once
// Closed-form elliptic tracer (Carlson RF + Jacobi inversion).
//
// Part of the single-translation-unit renderer included by main.cpp;
// see docs/ARCHITETTURA.md for how the modules fit together.

#include "tracer_separable.hpp"

// ── Elliptic-closed helpers (Carlson RF + Jacobi inversion) ──
static double carlson_rf(double x, double y, double z) {
    if (!(x >= 0.0 && y >= 0.0 && z >= 0.0)) return std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
        return std::numeric_limits<double>::quiet_NaN();

    double xn = x, yn = y, zn = z;
    constexpr double ERR_TOL = 1e-10;
    for (int it = 0; it < 64; ++it) {
        const double mu = (xn + yn + zn) / 3.0;
        if (!(mu > 0.0) || !std::isfinite(mu))
            return std::numeric_limits<double>::quiet_NaN();

        const double X = 1.0 - xn / mu;
        const double Y = 1.0 - yn / mu;
        const double Z = 1.0 - zn / mu;
        const double eps = std::max({std::abs(X), std::abs(Y), std::abs(Z)});
        if (eps < ERR_TOL) {
            const double E2 = X*Y - Z*Z;
            const double E3 = X*Y*Z;
            const double corr =
                1.0
                - E2 / 10.0
                + E3 / 14.0
                + (E2*E2) / 24.0
                - (3.0 * E2 * E3) / 44.0;
            return corr / std::sqrt(mu);
        }

        const double sx = std::sqrt(xn);
        const double sy = std::sqrt(yn);
        const double sz = std::sqrt(zn);
        const double lam = sx*sy + sx*sz + sy*sz;
        xn = 0.25 * (xn + lam);
        yn = 0.25 * (yn + lam);
        zn = 0.25 * (zn + lam);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

static double ellint_F_incomplete(double phi, double m) {
    const double s = std::sin(phi);
    const double c = std::cos(phi);
    const double x = c*c;
    const double y = 1.0 - m*s*s;
    if (y <= 0.0) return std::numeric_limits<double>::quiet_NaN();
    const double rf = carlson_rf(x, y, 1.0);
    return s * rf;
}

static double ellint_K_complete(double m) {
    if (!(m >= 0.0 && m < 1.0)) return std::numeric_limits<double>::quiet_NaN();
    return carlson_rf(0.0, 1.0 - m, 1.0);
}

static double jacobi_sn2_from_u(double u, double m, double Kc) {
    if (!(m >= 0.0 && m < 1.0) || !(Kc > 0.0) || !std::isfinite(u))
        return std::numeric_limits<double>::quiet_NaN();

    double ur = std::fmod(u, 2.0 * Kc);
    if (ur < 0.0) ur += 2.0 * Kc;
    if (ur > Kc) ur = 2.0 * Kc - ur; // sn^2 symmetry on [0, 2K]

    double lo = 0.0, hi = 0.5 * M_PI;
    for (int it = 0; it < 64; ++it) {
        const double mid = 0.5 * (lo + hi);
        const double Fm = ellint_F_incomplete(mid, m);
        if (!std::isfinite(Fm)) return std::numeric_limits<double>::quiet_NaN();
        if (Fm < ur) lo = mid;
        else         hi = mid;
    }
    const double s = std::sin(0.5 * (lo + hi));
    return s*s;
}

static std::complex<double> eval_monic_quartic(const std::array<double, 4>& b,
                                               const std::complex<double>& z) {
    // z^4 + b[0] z^3 + b[1] z^2 + b[2] z + b[3]
    return (((z + b[0]) * z + b[1]) * z + b[2]) * z + b[3];
}

static void quartic_roots_monic_complex(const std::array<double, 4>& b,
                                        std::array<std::complex<double>, 4>& roots_out) {
    constexpr double PI = 3.141592653589793238462643383279502884;
    const double radius = 1.0 + std::max({std::abs(b[0]), std::abs(b[1]), std::abs(b[2]), std::abs(b[3])});
    std::array<std::complex<double>, 4> z = {
        std::polar(radius, 0.0 * PI),
        std::polar(radius, 0.5 * PI),
        std::polar(radius, 1.0 * PI),
        std::polar(radius, 1.5 * PI)
    };

    for (int it = 0; it < 160; ++it) {
        double max_delta = 0.0;
        std::array<std::complex<double>, 4> zn = z;
        for (int i = 0; i < 4; ++i) {
            std::complex<double> den = 1.0;
            for (int j = 0; j < 4; ++j) {
                if (i == j) continue;
                den *= (z[i] - z[j]);
            }
            if (std::abs(den) < 1e-20) den = std::complex<double>(1e-20, 0.0);
            const std::complex<double> delta = eval_monic_quartic(b, z[i]) / den;
            zn[i] = z[i] - delta;
            max_delta = std::max(max_delta, std::abs(delta));
        }
        z = zn;
        if (max_delta < 1e-13) break;
    }
    roots_out = z;
}

static bool quartic_real_roots_monic(const std::array<double, 4>& b,
                                     std::array<double, 4>& roots_out) {
    std::array<std::complex<double>, 4> z{};
    quartic_roots_monic_complex(b, z);

    std::vector<double> rr;
    rr.reserve(4);
    for (const auto& zi : z) {
        const double tol = 1e-7 * std::max(1.0, std::abs(zi.real()));
        if (std::abs(zi.imag()) <= tol && std::isfinite(zi.real()))
            rr.push_back(zi.real());
    }
    if (rr.size() != 4) return false;

    std::sort(rr.begin(), rr.end(), std::greater<double>());
    for (int i = 0; i < 4; ++i) roots_out[i] = rr[(size_t)i];
    return true;
}

static bool jacobi_sn_cn_from_u(double u, double m, double Kc, double& sn, double& cn) {
    sn = std::numeric_limits<double>::quiet_NaN();
    cn = std::numeric_limits<double>::quiet_NaN();
    if (!(m >= 0.0 && m < 1.0) || !(Kc > 0.0) || !std::isfinite(u)) return false;

    const double twoK = 2.0 * Kc;
    long long n = static_cast<long long>(std::floor(u / twoK));
    double ur = u - static_cast<double>(n) * twoK;
    if (ur < 0.0) {
        ur += twoK;
        --n;
    }

    const bool second_half = (ur > Kc);
    const double target = second_half ? (twoK - ur) : ur; // in [0, K]

    double lo = 0.0;
    double hi = 0.5 * M_PI;
    for (int it = 0; it < 72; ++it) {
        const double mid = 0.5 * (lo + hi);
        const double Fm = ellint_F_incomplete(mid, m);
        if (!std::isfinite(Fm)) return false;
        if (Fm < target) lo = mid;
        else             hi = mid;
    }

    const double amp = 0.5 * (lo + hi);
    const double s = std::sin(amp);
    const double c = std::cos(amp);
    double sn_local = s;
    double cn_local = second_half ? -c : c;

    if ((n & 1LL) != 0) {
        sn_local = -sn_local;
        cn_local = -cn_local;
    }

    sn = sn_local;
    cn = cn_local;
    return std::isfinite(sn) && std::isfinite(cn);
}

static bool jacobi_sc_from_u(double u, double m, double Kc, double& sc) {
    sc = std::numeric_limits<double>::quiet_NaN();
    double sn = 0.0, cn = 0.0;
    if (!jacobi_sn_cn_from_u(u, m, Kc, sn, cn)) return false;
    if (std::abs(cn) < 1e-14) return false;
    sc = sn / cn;
    return std::isfinite(sc);
}

enum class EllipticRadialCase : uint8_t {
    REGION_I_II_REAL4 = 0,
    REGION_III_REAL2_COMPLEX2,
    REGION_IV_COMPLEX4,
};

struct EllipticRadialMap {
    EllipticRadialCase radial_case = EllipticRadialCase::REGION_I_II_REAL4;

    // Real-root ordering (descending, code convention) when available.
    double r1=0.0, r2=0.0, r3=0.0, r4=0.0;

    double m=0.0;      // Jacobi modulus squared (k^2)
    double Kc=0.0;     // complete elliptic integral K(m)
    double omega=0.0;  // linear phase frequency in Mino time
    double X0=0.0;     // absolute source phase magnitude

    // X(tau) = phase_sign * X0 + tau_sign * omega * tau
    int phase_sign=1;
    int tau_sign=1;

    // Region III parameters (two real + one complex-conjugate pair)
    double A=0.0, B=0.0;
    double r_lo=0.0, r_hi=0.0; // ascending real roots: r_lo < r_hi

    // Region IV parameters (two complex-conjugate pairs)
    double a2=0.0, b1=0.0, g0=0.0;
};

static double elliptic_radial_r_from_phase(const EllipticRadialMap& mp, double X) {
    if (mp.radial_case == EllipticRadialCase::REGION_I_II_REAL4) {
        const double sn2 = jacobi_sn2_from_u(X, mp.m, mp.Kc);
        if (!std::isfinite(sn2)) return std::numeric_limits<double>::quiet_NaN();

        // GL Eq. (B46) alternative form using GL_r1 (code_r4) in denominator factors.
        // This form has a larger valid sn^2 range before the Möbius singularity.
        const double r24 = mp.r2 - mp.r4;
        const double r14 = mp.r1 - mp.r4;
        const double den = r24 - r14 * sn2;
        if (std::abs(den) < 1e-14) return std::numeric_limits<double>::quiet_NaN();
        return (mp.r1 * r24 - mp.r2 * r14 * sn2) / den;
    }

    if (mp.radial_case == EllipticRadialCase::REGION_III_REAL2_COMPLEX2) {
        double sn = 0.0, cn = 0.0;
        if (!jacobi_sn_cn_from_u(X, mp.m, mp.Kc, sn, cn))
            return std::numeric_limits<double>::quiet_NaN();

        // GL Eq. (B75)
        const double num = (mp.B * mp.r_hi - mp.A * mp.r_lo)
                         + (mp.B * mp.r_hi + mp.A * mp.r_lo) * cn;
        const double den = (mp.B - mp.A) + (mp.B + mp.A) * cn;
        if (std::abs(den) < 1e-14) return std::numeric_limits<double>::quiet_NaN();
        return num / den;
    }

    // REGION_IV_COMPLEX4
    double sc = 0.0;
    if (!jacobi_sc_from_u(X, mp.m, mp.Kc, sc))
        return std::numeric_limits<double>::quiet_NaN();

    // GL Eq. (B109)
    const double den = 1.0 + mp.g0 * sc;
    if (std::abs(den) < 1e-14) return std::numeric_limits<double>::quiet_NaN();
    return -mp.a2 * ((mp.g0 - sc) / den) - mp.b1;
}

static bool init_elliptic_radial_map(const SeparableKerrConsts& c,
                                     double r0, double dr0,
                                     EllipticRadialMap& out) {
    const double Kpot = c.Qc + (c.Lz - c.a*c.E)*(c.Lz - c.a*c.E);
    const double A0 = c.E*c.a*c.a - c.a*c.Lz;
    const double c4 = c.E*c.E;
    if (!(c4 > 1e-18)) return false;
    const double c2 = 2.0*c.E*A0 - Kpot;
    const double c1 = 2.0*c.M*Kpot;
    const double c0 = A0*A0 - c.a*c.a*Kpot;

    const std::array<double, 4> b = {0.0, c2/c4, c1/c4, c0/c4}; // monic quartic

    std::array<std::complex<double>, 4> z{};
    quartic_roots_monic_complex(b, z);

    std::vector<double> real_roots;
    std::vector<std::complex<double>> complex_roots;
    real_roots.reserve(4);
    complex_roots.reserve(4);

    for (const auto& zi : z) {
        if (!std::isfinite(zi.real()) || !std::isfinite(zi.imag())) return false;
        const double tol_im = 1e-7 * std::max(1.0, std::abs(zi.real()));
        if (std::abs(zi.imag()) <= tol_im) real_roots.push_back(zi.real());
        else                               complex_roots.push_back(zi);
    }

    auto align_initial_radial_sign = [&](EllipticRadialMap& mp_local, bool flip_phase, bool flip_tau) {
        if (std::abs(dr0) < 1e-14) return;
        const double probe_tau = 1e-6;
        const double Xp = double(mp_local.phase_sign) * mp_local.X0
                        + double(mp_local.tau_sign) * mp_local.omega * probe_tau;
        const double Xm = double(mp_local.phase_sign) * mp_local.X0
                        - double(mp_local.tau_sign) * mp_local.omega * probe_tau;
        const double rp = elliptic_radial_r_from_phase(mp_local, Xp);
        const double rm = elliptic_radial_r_from_phase(mp_local, Xm);
        if (!std::isfinite(rp) || !std::isfinite(rm)) return;

        const bool want_out = (dr0 >= 0.0);
        const bool is_out = (rp >= rm);
        if (want_out == is_out) return;

        if (flip_phase) mp_local.phase_sign *= -1;
        if (flip_tau)   mp_local.tau_sign *= -1;
    };

    // Region I/II: four real roots.
    if (real_roots.size() == 4) {
        std::sort(real_roots.begin(), real_roots.end(), std::greater<double>());

        out = EllipticRadialMap{};
        out.radial_case = EllipticRadialCase::REGION_I_II_REAL4;
        out.r1 = real_roots[0];
        out.r2 = real_roots[1];
        out.r3 = real_roots[2];
        out.r4 = real_roots[3];

        const double den_omega = (out.r1 - out.r3) * (out.r2 - out.r4);
        if (!(den_omega > 0.0)) return false;
        out.omega = 0.5 * std::sqrt(den_omega);

        // Alternative-form modulus consistent with the GL_r1 (code_r4) Möbius parametrization.
        // k'^2 = (r3_GL-r2_GL)*(r4_GL-r1_GL) / ((r3_GL-r1_GL)*(r4_GL-r2_GL))
        // In code-descending (r3_GL=r2, r2_GL=r3, r4_GL=r1, r1_GL=r4):
        // k'^2 = (r2-r3)*(r1-r4) / ((r2-r4)*(r1-r3))
        const double k_num = (out.r2 - out.r3) * (out.r1 - out.r4);
        const double k_den = (out.r2 - out.r4) * (out.r1 - out.r3);
        if (!(k_den > 0.0)) return false;
        out.m = k_num / k_den;
        if (!(out.m >= 0.0 && out.m < 1.0)) return false;
        out.Kc = ellint_K_complete(out.m);
        if (!std::isfinite(out.Kc) || !(out.Kc > 0.0)) return false;

        // Source phase consistent with the alternative (r4/code) Möbius form:
        // sn^2(F0,k') = (r0-r4_GL)*(r3_GL-r1_GL) / ((r0-r3_GL)*(r4_GL-r1_GL))
        // In code-descending: (r0-r1)*(r2-r4) / ((r0-r2)*(r1-r4))
        const double den_obs = (r0 - out.r2);
        const double den_root = (out.r1 - out.r4);
        if (std::abs(den_obs) < 1e-14 || std::abs(den_root) < 1e-14) return false;
        const double u0_raw = ((r0 - out.r1) / den_obs) * ((out.r2 - out.r4) / den_root);
        if (!std::isfinite(u0_raw)) return false;
        if (u0_raw < -1e-6 || u0_raw > 1.0 + 1e-6) return false;
        const double u0 = clamp(u0_raw, 0.0, 1.0);

        out.X0 = ellint_F_incomplete(std::asin(std::sqrt(u0)), out.m);
        if (!std::isfinite(out.X0)) return false;

        // Step 4 (GL B119): alpha = -1 in case (1), +1 otherwise.
        const bool case1 = (r0 >= out.r3 && r0 <= out.r2);
        const int alpha = case1 ? -1 : +1;
        const int nu_r = (dr0 >= 0.0) ? +1 : -1;
        out.phase_sign = alpha * nu_r;
        out.tau_sign = +1;

        align_initial_radial_sign(out, /*flip_phase=*/true, /*flip_tau=*/false);
        return true;
    }

    // Region III: two real roots + one complex-conjugate pair.
    if (real_roots.size() == 2 && complex_roots.size() == 2) {
        std::sort(real_roots.begin(), real_roots.end()); // ascending GL notation
        const double r1 = real_roots[0];
        const double r2 = real_roots[1];

        const double b1 = 0.5 * (complex_roots[0].real() + complex_roots[1].real());
        const double a1 = 0.5 * (std::abs(complex_roots[0].imag()) + std::abs(complex_roots[1].imag()));
        if (!(a1 > 0.0) || !std::isfinite(a1) || !std::isfinite(b1)) return false;

        const double A = std::sqrt(a1*a1 + (b1 - r2)*(b1 - r2));
        const double B = std::sqrt(a1*a1 + (b1 - r1)*(b1 - r1));
        if (!(A > 0.0 && B > 0.0 && B > A)) return false;

        const double r21 = r2 - r1;
        const double k3 = (((A + B) * (A + B)) - r21 * r21) / (4.0 * A * B);
        if (!(k3 > 0.0 && k3 < 1.0)) return false;
        const double K3 = ellint_K_complete(k3);
        if (!std::isfinite(K3) || !(K3 > 0.0)) return false;

        const double x3 = (A*(r0 - r1) - B*(r0 - r2))
                        / (A*(r0 - r1) + B*(r0 - r2));
        if (!std::isfinite(x3)) return false;
        const double phi0 = std::acos(clamp(x3, -1.0, 1.0));
        const double F0 = ellint_F_incomplete(phi0, k3);
        if (!std::isfinite(F0)) return false;

        out = EllipticRadialMap{};
        out.radial_case = EllipticRadialCase::REGION_III_REAL2_COMPLEX2;
        out.r1 = r2; // outer real root (descending-like convenience)
        out.r2 = r1;
        out.r_lo = r1;
        out.r_hi = r2;
        out.A = A;
        out.B = B;
        out.m = k3;
        out.Kc = K3;
        out.omega = std::sqrt(A * B);
        out.X0 = F0;
        out.phase_sign = (dr0 >= 0.0) ? +1 : -1; // X3 = nu_r*F0 + omega*tau
        out.tau_sign = +1;

        align_initial_radial_sign(out, /*flip_phase=*/true, /*flip_tau=*/false);
        return true;
    }

    // Region IV: two complex-conjugate pairs.
    if (real_roots.empty() && complex_roots.size() == 4) {
        std::vector<std::complex<double>> pos_im, neg_im;
        for (const auto& zi : complex_roots) {
            if (zi.imag() >= 0.0) pos_im.push_back(zi);
            else                  neg_im.push_back(zi);
        }
        if (pos_im.size() != 2 || neg_im.size() != 2) return false;

        auto conj_dist = [](const std::complex<double>& a, const std::complex<double>& b) {
            return std::abs(a - std::conj(b));
        };
        const int j0 = (conj_dist(pos_im[0], neg_im[0]) <= conj_dist(pos_im[0], neg_im[1])) ? 0 : 1;

        auto pair_params = [](const std::complex<double>& zp, const std::complex<double>& zn) {
            const double b = 0.5 * (zp.real() + zn.real());
            const double a = 0.5 * (std::abs(zp.imag()) + std::abs(zn.imag()));
            return std::pair<double,double>{b, a};
        };

        auto [bA, aA] = pair_params(pos_im[0], neg_im[j0]);
        auto [bB, aB] = pair_params(pos_im[1], neg_im[1 - j0]);
        if (!(aA > 0.0 && aB > 0.0)) return false;

        // GL notation: b1 > 0 > b2 and a1 > a2 > 0.
        double b1 = bA, a1 = aA, b2 = bB, a2 = aB;
        if (bB > bA) {
            b1 = bB; a1 = aB;
            b2 = bA; a2 = aA;
        }
        if (!(b1 > b2)) return false;

        const double C = std::sqrt((a1 - a2)*(a1 - a2) + (b1 - b2)*(b1 - b2));
        const double D = std::sqrt((a1 + a2)*(a1 + a2) + (b1 - b2)*(b1 - b2));
        if (!(C > 0.0 && D > 0.0)) return false;

        const double k4 = (4.0 * C * D) / ((C + D) * (C + D));
        if (!(k4 > 0.0 && k4 < 1.0)) return false;
        const double K4 = ellint_K_complete(k4);
        if (!std::isfinite(K4) || !(K4 > 0.0)) return false;

        const double g_num = 4.0 * a2 * a2 - (C - D) * (C - D);
        const double g_den = (C + D) * (C + D) - 4.0 * a2 * a2;
        if (!(g_num > 0.0 && g_den > 0.0)) return false;
        const double g0 = std::sqrt(g_num / g_den);
        if (!(g0 > 0.0) || !std::isfinite(g0)) return false;

        const double x4 = (r0 - b2) / std::max(a2, 1e-15);
        const double phi0 = std::atan(x4) + std::atan(g0);
        const double F0 = ellint_F_incomplete(phi0, k4);
        if (!std::isfinite(F0)) return false;

        out = EllipticRadialMap{};
        out.radial_case = EllipticRadialCase::REGION_IV_COMPLEX4;
        out.m = k4;
        out.Kc = K4;
        out.omega = 0.5 * (C + D);
        out.X0 = F0;
        out.phase_sign = +1;
        out.tau_sign = (dr0 >= 0.0) ? +1 : -1; // X4 = F0 + nu_r*omega*tau
        out.a2 = a2;
        out.b1 = b1;
        out.g0 = g0;

        align_initial_radial_sign(out, /*flip_phase=*/false, /*flip_tau=*/true);
        return true;
    }

    return false;
}

static double separable_radial_potential(const SeparableKerrConsts& c, double r) {
    const double Kpot = c.Qc + (c.Lz - c.a*c.E) * (c.Lz - c.a*c.E);
    const double A0 = c.E * (r*r + c.a*c.a) - c.a * c.Lz;
    const double Delta = r*r - 2.0*c.M*r + c.a*c.a;
    return A0*A0 - Delta*Kpot;
}

static bool first_equator_crossing_mino_time(const SeparableKerrConsts& c,
                                             double theta0, double dtheta0,
                                             double& tau_first, double& tau_period) {
    const double A = c.a*c.a*c.E*c.E;
    if (!(A > 1e-15)) return false;

    const double B = A - c.Lz*c.Lz - c.Qc;
    const double disc = B*B + 4.0*A*c.Qc;
    if (!(disc >= 0.0)) return false;
    const double sq = std::sqrt(disc);
    const double u_plus  = (B + sq) / (2.0*A);
    const double u_minus = (B - sq) / (2.0*A);
    if (!(u_plus > 0.0 && u_minus < 0.0)) return false;

    // For the physically relevant M_- < 0 branch (Dexter & Agol 2009, Eq. 42):
    //   mu(tau) = mu_+ * cn(u, k),   k^2 = u_+ / (u_+ - u_-)
    // with u evolving linearly in Mino time.
    const double m = u_plus / (u_plus - u_minus);
    if (!(m >= 0.0 && m < 1.0)) return false;
    const double Kc = ellint_K_complete(m);
    if (!std::isfinite(Kc) || !(Kc > 0.0)) return false;

    const double omega = std::sqrt(A * (u_plus - u_minus));
    if (!(omega > 0.0) || !std::isfinite(omega)) return false;

    const double mu_plus = std::sqrt(u_plus);
    const double mu0 = std::cos(theta0);
    const double c0 = clamp(std::abs(mu0) / std::max(mu_plus, 1e-15), 0.0, 1.0);
    const double phi0 = std::acos(c0);
    const double u_base = ellint_F_incomplete(phi0, m);
    if (!std::isfinite(u_base)) return false;

    // Full period of cn in Mino-time units.
    tau_period = 4.0 * Kc / omega;
    if (!(tau_period > 0.0) || !std::isfinite(tau_period)) return false;

    // If theta moves toward equator, use +u_base; otherwise start from -u_base.
    // First equator crossing is cn(u)=0 => u = K.
    const bool toward_equator = (std::cos(theta0) * dtheta0) > 0.0;
    const double u0 = toward_equator ? u_base : -u_base;
    tau_first = (Kc - u0) / omega;
    if (tau_first < 0.0) tau_first += tau_period;
    return std::isfinite(tau_first);
}

static TraceResult trace_single_ks(GeodesicState s_bl, const KNdSMetric& g,
                                   double r_disk_in, double r_disk_out,
                                   double r_escape,
                                   Integrator intg,
                                   const IntegratorControls& ctl = IntegratorControls{},
                                   const ColorParams* cp_ptr = nullptr);

static TraceResult trace_single_elliptic_closed(GeodesicState s_bl, const KNdSMetric& g,
                                                double r_disk_in, double r_disk_out,
                                                double r_escape,
                                                EllipticFallbackReason* fallback_reason = nullptr,
                                                CoordinateChart fallback_chart = CoordinateChart::BL,
                                                Integrator intg = Integrator::RK4_DOUBLING,
                                                const IntegratorControls& ctl = IntegratorControls{},
                                                const ColorParams* cp_ptr = nullptr) {
    const double observer_frequency = -s_bl.pt * g.static_observer_ut(s_bl.r, s_bl.theta);
    if (fallback_reason) *fallback_reason = EllipticFallbackReason::NONE;
    auto fallback_trace = [&](EllipticFallbackReason reason) -> TraceResult {
        if (fallback_reason) *fallback_reason = reason;
        if (fallback_chart == CoordinateChart::KS)
            return trace_single_ks(s_bl, g, r_disk_in, r_disk_out, r_escape, intg, ctl, cp_ptr);
        return trace_single(s_bl, g, r_disk_in, r_disk_out, r_escape, intg, ctl, cp_ptr);
    };

    SeparableKerrConsts c{};
    if (!init_separable_consts(s_bl, g, c))
        return fallback_trace(EllipticFallbackReason::INIT_SEPARABLE_CONSTS);

    double dr0=0.0, dth0=0.0, dpr0=0.0, dpth0=0.0;
    geodesic_rhs(g, s_bl.r, s_bl.theta, s_bl.pr, s_bl.ptheta, s_bl.pt, s_bl.pphi,
                 dr0, dth0, dpr0, dpth0);

    double tau_first = 0.0, tau_period = 0.0;
    if (!first_equator_crossing_mino_time(c, s_bl.theta, dth0, tau_first, tau_period))
        return fallback_trace(EllipticFallbackReason::THETA_CROSSING_SOLVE);
    (void)tau_period;

    EllipticRadialMap mp{};
    if (!init_elliptic_radial_map(c, s_bl.r, dr0, mp))
        return fallback_trace(EllipticFallbackReason::RADIAL_MAP_INIT);

    const double rh_cut = g.r_horizon() * 1.03;
    const double X = double(mp.phase_sign) * mp.X0
                   + double(mp.tau_sign) * mp.omega * tau_first;
    const double r_now = elliptic_radial_r_from_phase(mp, X);
    if (!std::isfinite(r_now))
        return fallback_trace(EllipticFallbackReason::DIRECT_RADIAL_INVALID);

    const double R = separable_radial_potential(c, r_now);
    const double R_scale = std::max(1.0, std::abs(c.E*c.E*r_now*r_now*r_now*r_now));
    if (R < -1e-8 * R_scale)
        return fallback_trace(EllipticFallbackReason::DIRECT_RADIAL_INVALID);

    // For Region I/II and III, enforce the observer-accessible outer branch.
    // Region IV has no real turning points.
    if (mp.radial_case != EllipticRadialCase::REGION_IV_COMPLEX4 && r_now < mp.r1)
        return fallback_trace(EllipticFallbackReason::DIRECT_RADIAL_INVALID);

    // For Region III, the GL B75 radial formula evaluates r on the hypothetical
    // post-bounce leg (after the inner turning point r_hi, which lies inside the
    // outer horizon for near-extremal Kerr).  In reality those photons cross
    // theta=pi/2 *before* the horizon and hit the disk — the formula gives the
    // wrong r.  Fall back to numerical for any sub-horizon r_now to be safe.
    if (r_now < rh_cut)
        return fallback_trace(EllipticFallbackReason::DIRECT_RADIAL_INVALID);
    if (r_now > r_escape)
        return fallback_trace(EllipticFallbackReason::DIRECT_OUTSIDE_DISK);
    if (r_now >= r_disk_in && r_now <= r_disk_out) {
        // Horizon-precedence safety check:
        // if chart-native standard tracing says this ray falls into horizon
        // before any disk intersection, prefer horizon to avoid leakage
        // inside the black-hole silhouette.
        const TraceResult verify = (fallback_chart == CoordinateChart::KS)
            ? trace_single_ks(s_bl, g, r_disk_in, r_disk_out, r_escape, intg, ctl, cp_ptr)
            : trace_single(s_bl, g, r_disk_in, r_disk_out, r_escape, intg, ctl, cp_ptr);
        if (verify.out == Outcome::HORIZON)
            return verify;

        // Stratified/Interstellar and global opacity < 1 are phi-dependent; this
        // direct branch does not compute phi_hit robustly, so defer to numerical trace.
        if (cp_ptr && (cp_ptr->palette != DiskPalette::BLACKBODY || cp_ptr->disk_opacity < (1.0 - 1e-9)))
            return fallback_trace(EllipticFallbackReason::DIRECT_OUTSIDE_DISK);

        return {Outcome::DISK_HIT, r_now,
                clamp(disk_redshift(r_now, s_bl.pt, s_bl.pphi, g, observer_frequency), 0.0, 20.0)};
    }
    return fallback_trace(EllipticFallbackReason::DIRECT_OUTSIDE_DISK);
}
