// Versioned golden values for phase 1: per pixel outcome, r, redshift,
// magnification, phi on the disk, escape direction and coverage, for a set of
// frames covering every chart, solver and spacetime the CPU renderer has.
//
//   kerrtrace_golden_regression tests/golden/phase1_rays.txt            compare
//   kerrtrace_golden_regression tests/golden/phase1_rays.txt --update   rewrite
//
// Rewrite only for an intended change of the physics, in the same commit, and
// say in the message which cases moved and why. Unlike PNG hashes these values
// survive a different compiler: the tolerances are far above rounding and far
// below any physics change; a few grazing rays per case may still flip.
#define KERRTRACE_NO_MAIN
#ifndef OUT_DIR
#define OUT_DIR "."
#endif
#include "../main.cpp"
#include <map>
#include <set>

namespace {
constexpr int W = 24, H = 14;

struct Case {
    const char* name;
    FrameParams fp;
    bool bundles = false;
    RaySolverMode solver = RaySolverMode::STANDARD;
    CoordinateChart chart = CoordinateChart::BL;
    Integrator intg = Integrator::RK4_DOUBLING;
    double Q = 0, Lam = 0;
    DiskPalette palette = DiskPalette::BLACKBODY;
};

std::vector<Case> cases() {
    FrameParams base; base.a = 0.9; base.theta = 80; base.r_obs = 40; base.fov = 45; base.disk_out = 12;
    FrameParams extremal = base; extremal.a = 0.998;
    FrameParams retro = base; retro.a = -0.9;
    FrameParams worm; worm.spacetime = Spacetime::DNEG_WORMHOLE; worm.r_obs = 6; worm.fov = 60;
    std::vector<Case> c;
    c.push_back({"bl", base});
    c.push_back({"ks", base, false, RaySolverMode::STANDARD, CoordinateChart::KS});
    c.push_back({"gks", base, false, RaySolverMode::STANDARD, CoordinateChart::GKS});
    c.push_back({"bl_dopri5", base, false, RaySolverMode::STANDARD, CoordinateChart::BL, Integrator::DOPRI5});
    c.push_back({"semi_analytic", extremal, false, RaySolverMode::SEMI_ANALYTIC});
    c.push_back({"elliptic_closed", extremal, false, RaySolverMode::ELLIPTIC_CLOSED});
    c.push_back({"bundles", base, true});
    c.push_back({"extremal", extremal});
    c.push_back({"retrograde", retro});
    c.push_back({"charge", base, false, RaySolverMode::STANDARD, CoordinateChart::BL,
                 Integrator::RK4_DOUBLING, 0.3});
    c.push_back({"lambda", base, false, RaySolverMode::STANDARD, CoordinateChart::BL,
                 Integrator::RK4_DOUBLING, 0, 1e-4});
    c.push_back({"interstellar", base, false, RaySolverMode::STANDARD, CoordinateChart::BL,
                 Integrator::RK4_DOUBLING, 0, 0, DiskPalette::INTERSTELLAR});
    c.push_back({"stratified", base, false, RaySolverMode::STANDARD, CoordinateChart::BL,
                 Integrator::RK4_DOUBLING, 0, 0, DiskPalette::STRATIFIED});
    c.push_back({"wormhole", worm});
    return c;
}

// One value per field, in file order.
constexpr int NF = 8;
const char* const kFields[NF] = {"outcome", "r", "redshift", "magnif", "phi_disk",
                                 "theta_esc", "phi_esc", "coverage"};
std::array<double, NF> fields(const GeoPixel& p) {
    return {double(p.outcome), p.r, p.redshift, p.magnif, p.phi_disk, p.theta_esc, p.phi_esc, p.coverage};
}

std::vector<std::array<double, NF>> trace(const Case& c) {
    ColorParams cp;
    cp.palette = c.palette;
    IntegratorControls ctl;
    KGeoMeta meta;
    const auto geo = trace_geodesics(W, H, c.fp, c.bundles, c.solver, c.chart, c.intg, ctl,
                                     0.5, 0.5, 1.0, c.Q, c.Lam, &cp, &meta);
    std::vector<std::array<double, NF>> out;
    for (const auto& p : geo) out.push_back(fields(p));
    return out;
}

using Golden = std::map<std::string, std::vector<std::array<double, NF>>>;

Golden read_golden(const std::string& path) {
    Golden g;
    std::ifstream in(path);
    std::string line, current;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (line.rfind("case ", 0) == 0) { current = line.substr(5); g[current]; continue; }
        std::istringstream row(line);
        std::array<double, NF> v{};
        for (auto& x : v) row >> x;
        g[current].push_back(v);
    }
    return g;
}

// Angles compare modulo 2 pi; the rest relative with an absolute floor.
bool close(int field, double a, double b) {
    if (!std::isfinite(a) || !std::isfinite(b)) return std::isnan(a) == std::isnan(b) && (a == b || std::isnan(a));
    if (field == 4 || field == 6) return std::abs(std::remainder(a - b, 2 * M_PI)) <= 1e-5;
    return std::abs(a - b) <= 1e-5 * std::max(1.0, std::abs(b));
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: " << argv[0] << " golden.txt [--update]\n"; return 2; }
    const std::string path = argv[1];
    const bool update = argc > 2 && std::string(argv[2]) == "--update";

    if (update) {
        std::ofstream out(path);
        out << "# Phase-1 golden values, written by tests/golden_regression.cpp --update.\n"
            << "# " << W << "x" << H << " pixels per case, row-major. Columns:";
        for (const char* f : kFields) out << ' ' << f;
        out << "\n";
        out << std::setprecision(9);
        for (const Case& c : cases()) {
            out << "case " << c.name << '\n';
            for (const auto& v : trace(c)) {
                for (int k = 0; k < NF; ++k) out << (k ? " " : "") << v[k];
                out << '\n';
            }
        }
        std::cout << "wrote " << path << '\n';
        return 0;
    }

    const Golden golden = read_golden(path);
    if (golden.empty()) { std::cerr << "no golden values in " << path << '\n'; return 1; }
    int failures = 0;
    std::set<std::string> seen;
    for (const Case& c : cases()) {
        seen.insert(c.name);
        const auto it = golden.find(c.name);
        if (it == golden.end() || it->second.size() != size_t(W * H)) {
            std::cout << "[FAIL] " << c.name << ": missing or incomplete in the golden file\n";
            ++failures;
            continue;
        }
        const auto now = trace(c);
        int moved = 0;
        std::array<int, NF> by_field{};
        for (size_t i = 0; i < now.size(); ++i) {
            bool bad = false;
            for (int k = 0; k < NF; ++k)
                if (!close(k, now[i][k], it->second[i][k])) { ++by_field[k]; bad = true; }
            moved += bad;
        }
        // A grazing ray may land on the other side of a boundary.
        const bool ok = moved <= 1;
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << c.name << ": " << moved << " of " << now.size()
                  << " pixels differ";
        for (int k = 0; k < NF; ++k) if (by_field[k]) std::cout << ' ' << kFields[k] << '=' << by_field[k];
        std::cout << '\n';
        failures += !ok;
    }
    for (const auto& [name, rows] : golden)
        if (!seen.count(name)) { std::cout << "[FAIL] " << name << ": in the golden file, not in the test\n"; ++failures; }
    std::cout << (failures ? "FAILED (intended change? rerun with --update and commit the file)"
                           : "ALL PASSED") << '\n';
    return failures ? 1 : 0;
}
