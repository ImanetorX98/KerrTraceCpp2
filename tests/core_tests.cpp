#include "camera.hpp"
#include "geodesic.hpp"
#include "knds_metric.hpp"

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

bool approx(double a, double b, double tol) {
    return std::abs(a - b) <= tol;
}

bool test_metric_inverse_bl() {
    KNdSMetric g(1.0, 0.5, 0.1, 1e-4);
    double gLL[4][4], gUU[4][4];
    g.covariant_BL(10.0, 1.2, gLL);
    g.contravariant_BL(10.0, 1.2, gUU);

    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            double s = 0.0;
            for (int a = 0; a < 4; ++a)
                s += gUU[mu][a] * gLL[a][nu];
            const double expected = (mu == nu) ? 1.0 : 0.0;
            if (!approx(s, expected, 1e-8)) return false;
        }
    }
    return true;
}

bool test_camera_ray_is_null_and_future_directed() {
    KNdSMetric g(1.0, -0.5, 0.0, 0.0);
    Camera cam(40.0, 80.0, 15.0, 45.0, 640, 360);
    const GeodesicState s = cam.angle_ray(0.0, 0.0, g);
    const double H = g.hamiltonian(s.r, s.theta, s.pr, s.ptheta, s.pt, s.pphi);
    return std::isfinite(H) &&
           std::abs(H) < 1e-10 &&
           s.pt < 0.0 &&
           approx(s.phi, cam.phi_obs, 1e-14);
}

bool step_and_check(Integrator intg) {
    KNdSMetric g(1.0, -0.5, 0.0, 0.0);
    Camera cam(40.0, 80.0, 0.0, 45.0, 640, 360);
    GeodesicState s = cam.angle_ray(0.03, -0.02, g);
    double h = 1.0;
    Vec4d fsal = Vec4d::nan_init();

    bool accepted = false;
    for (int i = 0; i < 32; ++i) {
        if (adaptive_step(g, s, h, intg, fsal, 1e-7)) {
            accepted = true;
            break;
        }
    }
    if (!accepted) return false;
    if (!(std::isfinite(s.r) && std::isfinite(s.theta) && std::isfinite(s.phi) &&
          std::isfinite(s.pr) && std::isfinite(s.ptheta))) {
        return false;
    }
    return null_residual(g, s) < 5e-6;
}

bool test_rk4_adaptive_step_preserves_null() {
    return step_and_check(Integrator::RK4_DOUBLING);
}

bool test_dopri5_adaptive_step_preserves_null() {
    return step_and_check(Integrator::DOPRI5);
}

bool test_keplerian_omega_matches_screen_convention() {
    const double r = 8.0;
    KNdSMetric gp(1.0, 0.5, 0.0, 0.0);
    KNdSMetric gn(1.0, -0.5, 0.0, 0.0);
    const double op = gp.keplerian_omega(r);
    const double on = gn.keplerian_omega(r);
    return op < 0.0 && on > 0.0 && approx(std::abs(op), std::abs(on), 1e-14);
}

bool test_ks_radius_recovery() {
    const double a = 0.5;
    const double r0 = 8.0;
    const double th0 = 1.2;
    const double ph0 = 2.1;
    double X, Y, Z;
    KNdSMetric::BL_to_KS_spatial(r0, th0, ph0, a, X, Y, Z);
    const double r1 = KNdSMetric::r_KS(X, Y, Z, a);
    return approx(r0, r1, 1e-9) &&
           approx(Z, r0 * std::cos(th0), 1e-12);
}

} // namespace

int main() {
    using TestFn = std::function<bool()>;
    const std::vector<std::pair<std::string, TestFn>> tests = {
        {"metric inverse in BL", test_metric_inverse_bl},
        {"camera ray is null and future-directed", test_camera_ray_is_null_and_future_directed},
        {"RK4 adaptive step preserves null residual", test_rk4_adaptive_step_preserves_null},
        {"DOPRI5 adaptive step preserves null residual", test_dopri5_adaptive_step_preserves_null},
        {"keplerian omega matches screen convention", test_keplerian_omega_matches_screen_convention},
        {"KS radius recovery", test_ks_radius_recovery},
    };

    int failed = 0;
    for (const auto& tc : tests) {
        const bool ok = tc.second();
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << tc.first << "\n";
        if (!ok) ++failed;
    }

    if (failed > 0) {
        std::cerr << failed << " test(s) failed.\n";
        return 1;
    }
    std::cout << "All tests passed (" << tests.size() << ").\n";
    return 0;
}
