// The closed-form (elliptic) Kerr tracer must describe the same geodesic as
// the numerical one. Checked on the real helpers, in increasing order of scope:
//   1. the incomplete integral F(phi|m) beyond the first quadrant;
//   2. each radial map reproduces the camera: r(0) = r_obs, dr/dtau = +-sqrt(R);
//   3. a plunging photon reaches the horizon cut at the Mino time the solver uses;
//   4. a disk hit taken from the closed form lands where the BL integrator does,
//      and none of them is a ray the integrator sends into the horizon.
// The first two failed on 2026-11-15 (audit B2): F(pi - phi) returned F(phi),
// which put Region III photons ~59 M inside the camera at tau = 0, and the
// radial frequencies lacked the factor E that the monic quartic divides out.
#define KERRTRACE_NO_MAIN
#ifndef OUT_DIR
#define OUT_DIR "."
#endif
#include "../main.cpp"

namespace {
int failures = 0;
void check(bool ok, const std::string& description) {
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << description << '\n';
    if (!ok) ++failures;
}

void incomplete_integral_quadrants() {
    double worst = 0.0;
    for (double m : {0.0, 0.3, 0.5, 0.9, 0.999}) {
        const double K = ellint_K_complete(m);
        for (double phi : {0.1, 0.7, 1.2, M_PI / 2}) {
            const double F = ellint_F_incomplete(phi, m);
            worst = std::max({worst,
                std::abs(ellint_F_incomplete(M_PI - phi, m) - (2 * K - F)),
                std::abs(ellint_F_incomplete(phi + M_PI, m) - (2 * K + F)),
                std::abs(ellint_F_incomplete(phi + 3 * M_PI, m) - (6 * K + F)),
                std::abs(ellint_F_incomplete(-phi, m) + F)});
        }
    }
    std::cout << "  worst quadrant identity error " << worst << '\n';
    check(worst < 1e-9, "F(pi-phi)=2K-F, F(phi+n pi)=2nK+F, F odd");
}

struct MapCensus {
    int maps = 0, by_case[3] = {0, 0, 0}, plunging = 0;
    double r0_err = 0, rate_err = 0, cut_err = 0;
};

// Every pixel of a small frame whose radial map initialises: r(tau=0) must be
// the camera radius and the slope in Mino time must be the geodesic's.
void radial_maps(double a, double r_obs, double incl, MapCensus& out) {
    KNdSMetric g(1, a, 0, 0);
    Camera cam(r_obs, incl, 0, 50, 48, 27);
    for (int y = 0; y < 27; ++y) for (int x = 0; x < 48; ++x) {
        const auto s = cam.pixel_ray(x, y, g);
        SeparableKerrConsts c{};
        if (!init_separable_consts(s, g, c)) continue;
        double dr, dth, dpr, dpth;
        geodesic_rhs(g, s.r, s.theta, s.pr, s.ptheta, s.pt, s.pphi, dr, dth, dpr, dpth);
        EllipticRadialMap mp{};
        if (!init_elliptic_radial_map(c, s.r, dr, mp)) continue;
        ++out.maps;
        ++out.by_case[static_cast<int>(mp.radial_case)];

        const double X0 = mp.phase_sign * mp.X0;
        const double h = 1e-6 / mp.omega;
        const double r0 = elliptic_radial_r_from_phase(mp, X0);
        const double rate = (elliptic_radial_r_from_phase(mp, X0 + mp.tau_sign * mp.omega * h)
                           - elliptic_radial_r_from_phase(mp, X0 - mp.tau_sign * mp.omega * h)) / (2 * h);
        const double want = (dr >= 0 ? 1 : -1) * std::sqrt(std::max(0.0, kerr_sep_R(c, s.r)));
        const double r0_err = std::isfinite(r0) ? std::abs(r0 - s.r) / s.r : INFINITY;
        const double rate_err = std::isfinite(rate) ? std::abs(rate / want - 1) : INFINITY;
        out.r0_err = std::max(out.r0_err, r0_err);
        out.rate_err = std::max(out.rate_err, rate_err);

        const double r_cut = 1.03 * g.r_horizon();
        const double tau_cut = elliptic_mino_time_to_cut(mp, s.r, dr, r_cut);
        if (std::isinf(tau_cut)) continue;
        ++out.plunging;
        const double r_at = elliptic_radial_r_from_phase(mp, X0 + mp.tau_sign * mp.omega * tau_cut);
        out.cut_err = std::max(out.cut_err,
                               std::isfinite(r_at) ? std::abs(r_at - r_cut) / r_cut : INFINITY);
    }
}

void radial_map_initial_conditions() {
    for (double a : {0.5, 0.998}) for (double incl : {60.0, 84.0}) {
        MapCensus c;
        radial_maps(a, 60, incl, c);
        std::ostringstream label;
        label << "a=" << a << " incl=" << incl << ": " << c.maps << " maps (I/II "
              << c.by_case[0] << ", III " << c.by_case[1] << ", IV " << c.by_case[2] << ")";
        std::cout << "  " << label.str() << " max |r0-r_obs|/r_obs " << c.r0_err
                  << ", max |dr/dtau ratio - 1| " << c.rate_err << ", " << c.plunging
                  << " plunging, max |r(tau_cut)-r_cut|/r_cut " << c.cut_err << '\n';
        check(c.by_case[0] > 0 && c.by_case[1] > 0, label.str() + " covers Regions I/II and III");
        check(c.r0_err < 1e-9, label.str() + " starts at the camera radius");
        check(c.rate_err < 1e-5, label.str() + " has the geodesic radial velocity");
        check(c.plunging > 0 && c.cut_err < 1e-6, label.str() + " reaches the horizon cut at tau_cut");
    }
}

// Direct closed-form disk hits against the BL integrator on the same rays.
void direct_hits_match_numerical() {
    KNdSMetric g(1, 0.998, 0, 0);
    const double r_in = g.r_isco(), r_out = 20, r_esc = 70;
    Camera cam(60, 84, 0, 50, 160, 90);
    IntegratorControls tight;   // at the default 1e-7 the reference itself is off by ~1e-4
    tight.tol = 1e-10;
    int direct = 0, overridden = 0;
    double worst = 0;
    for (int y = 0; y < 90; ++y) for (int x = 0; x < 160; ++x) {
        const auto s = cam.pixel_ray(x, y, g);
        EllipticFallbackReason why = EllipticFallbackReason::NONE;
        const TraceResult e = trace_single_elliptic_closed(s, g, r_in, r_out, r_esc, &why);
        if (why != EllipticFallbackReason::NONE) continue;
        // A direct hit the solver's own numerical check turned into HORIZON.
        if (e.out == Outcome::HORIZON) ++overridden;
        if (e.out != Outcome::DISK_HIT) continue;
        const TraceResult n = trace_single(s, g, r_in, r_out, r_esc, Integrator::DOPRI5, tight);
        ++direct;
        worst = std::max(worst, n.out == Outcome::DISK_HIT ? std::abs(e.r - n.r) / n.r : INFINITY);
    }
    std::cout << "  " << direct << " direct closed-form disk hits, max relative r difference "
              << worst << '\n';
    std::cout << "  " << overridden << " direct hits overridden by the numerical horizon check\n";
    check(direct > 0, "closed form resolves some disk hits without falling back");
    check(overridden == 0, "closed form never claims a disk hit behind the horizon");
    check(worst < 5e-5, "closed-form hit radius matches the BL integrator");
}
} // namespace

int main() {
    incomplete_integral_quadrants();
    radial_map_initial_conditions();
    direct_hits_match_numerical();
    std::cout << (failures ? "FAILED" : "ALL PASSED") << '\n';
    return failures ? 1 : 0;
}
