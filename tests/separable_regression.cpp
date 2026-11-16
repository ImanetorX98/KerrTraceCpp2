// The separable (semi-analytic) Kerr tracer must trace the same rays as BL.
// Per ray on a frame: same outcome, same disk radius, same escape direction,
// and no ray left at the step cap. On 2026-11-16 (audit A1) 82% of the rays
// froze at their first turning point: dr = +-sqrt(max(R, 0)) is zero there,
// a step into R < 0 left the velocity at zero, and the cap was returned as
// ESCAPED. 28% of the rays ended with a different outcome.
#define KERRTRACE_NO_MAIN
#ifndef OUT_DIR
#define OUT_DIR "."
#endif
#include "../main.cpp"
#include <chrono>

namespace {
int failures = 0;
void check(bool ok, const std::string& description) {
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << description << '\n';
    if (!ok) ++failures;
}

double angle_diff(double x, double y) {
    return std::abs(std::remainder(x - y, 2 * M_PI));
}

void frame(double a) {
    KNdSMetric g(1, a, 0, 0);
    const double r_in = g.r_isco(), r_out = 12, r_esc = 63;
    Camera cam(60, 80, 0, 30, 96, 54);
    IntegratorControls ctl;
    ctl.max_steps = 20000;   // a correct trace needs a few hundred
    int rays = 0, outcome = 0, disk_far = 0, sky_far = 0, capped = 0;
    double t_sep = 0, t_bl = 0;
    for (int y = 0; y < 54; ++y) for (int x = 0; x < 96; ++x) {
        const auto s = cam.pixel_ray(x, y, g);
        const auto t0 = std::chrono::steady_clock::now();
        const TraceResult e = trace_single_separable_kerr(s, g, r_in, r_out, r_esc, ctl);
        const auto t1 = std::chrono::steady_clock::now();
        const TraceResult b = trace_single(s, g, r_in, r_out, r_esc, Integrator::RK4_DOUBLING, ctl);
        const auto t2 = std::chrono::steady_clock::now();
        t_sep += std::chrono::duration<double>(t1 - t0).count();
        t_bl += std::chrono::duration<double>(t2 - t1).count();
        ++rays;
        if (e.out == Outcome::ESCAPED && e.r < 0.999 * r_esc) ++capped;
        if (e.out != b.out) { ++outcome; continue; }
        if (e.out == Outcome::DISK_HIT && std::abs(e.r - b.r) / b.r > 1e-3) ++disk_far;
        if (e.out == Outcome::ESCAPED &&
            (std::abs(e.theta_esc - b.theta_esc) > 1e-3 || angle_diff(e.phi_esc, b.phi_esc) > 1e-3))
            ++sky_far;
    }
    std::ostringstream label;
    label << "a=" << a << ", " << rays << " rays";
    std::cout << "  " << label.str() << ": " << capped << " at the step cap, " << outcome
              << " other outcome than BL, " << disk_far << " disk r off >0.1%, " << sky_far
              << " sky direction off >1e-3; time separable " << t_sep << " s, BL " << t_bl << " s\n";
    check(capped == 0, label.str() + ": no ray stalls at the step cap");
    // Rays grazing the shadow or the disk edge may land either side.
    check(outcome <= rays / 500, label.str() + ": outcome agrees with BL");
    check(disk_far <= rays / 500, label.str() + ": disk radius agrees with BL");
    check(sky_far <= rays / 500, label.str() + ": escape direction agrees with BL");
}
} // namespace

int main() {
    frame(0.998);
    frame(0.5);
    std::cout << (failures ? "FAILED" : "ALL PASSED") << '\n';
    return failures ? 1 : 0;
}
