// Chart-consistency regression test.
//
// Boyer-Lindquist and Kerr-Schild are two charts on the same spacetime, so they
// must produce the same image. They did not: KS rendered the Schwarzschild
// shadow 4.88% too large, and the error was independent of the integrator
// tolerance because it lived in the initial data, not the integration.
//
// The cause was in bl_covector_to_ks(). Ingoing Kerr-Schild is not a purely
// spatial relabelling of BL:
//
//     dT      = dt   + (2Mr - Q^2)/Delta_r dr
//     dphi_KS = dphi + a/Delta_r           dr
//
// so t and phi_BL both depend on the KS spatial coordinates through r(X,Y,Z),
// and the covector picks up two extra terms along dr/dX^i. Passing the raw p_r
// injected a spurious radial momentum: at r = 40M, a = 0 the missing term is
// (2Mr/Delta) p_t = 5.3% of E, which is the size of the shadow error.
//
// Two independent things are checked, because chart agreement alone would also
// be satisfied by both charts being wrong in the same way:
//
//   1. an ABSOLUTE check against theory -- for a = 0 the shadow seen by a static
//      observer at r has angular radius asin(3 sqrt(3) sqrt(1 - 2M/r) / r), which
//      is 7.2740 deg at r = 40M;
//   2. BL against KS, on r, on the disk-hit count, and on the redshift.
//
// Measured before the fix / after, at 641x361:
//   shadow radius   BL 7.2803 deg (+0.09%)   KS 7.6566 deg (+5.26%) -> 7.2803 deg
//   |dr| median (a=0.9)                      0.5806            -> 0.00004
//   disk-hit difference (a=0.9)              11.1%             -> 0.01%

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

// Must mirror GeoPixel in main.cpp exactly.
struct GeoPixelRaw {
    uint8_t outcome;
    uint8_t pad[3];
    float r;
    float redshift;
    float magnif;
    float phi_disk;
    float theta_esc;
    float phi_esc;
};
static_assert(sizeof(GeoPixelRaw) == 28, "GeoPixelRaw must match GeoPixel in main.cpp");

struct GeoFrame {
    uint32_t W = 0;
    uint32_t H = 0;
    std::vector<GeoPixelRaw> px;
};

int failures = 0;

void check(bool ok, const std::string& what, const std::string& detail) {
    std::cout << (ok ? "  ok   " : "  FAIL ") << what << "  [" << detail << "]\n";
    if (!ok) ++failures;
}

double median_of(std::vector<float> v) {
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    const size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    return double(v[mid]);
}

bool load_kgeo(const std::filesystem::path& path, GeoFrame& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    char magic[4] = {};
    f.read(magic, 4);
    if (!f || std::memcmp(magic, "KGEO", 4) != 0) return false;

    uint32_t version = 0;
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!f || version != 1) return false;

    f.read(reinterpret_cast<char*>(&out.W), sizeof(out.W));
    f.read(reinterpret_cast<char*>(&out.H), sizeof(out.H));
    if (!f || out.W == 0 || out.H == 0) return false;

    f.seekg(80, std::ios::cur);
    if (!f) return false;

    const std::uintmax_t total = std::filesystem::file_size(path);
    const std::uintmax_t header = 4 + 4 + 4 + 4 + 80;
    const std::uintmax_t payload = (total > header) ? (total - header) : 0;
    const std::uintmax_t expected =
        std::uintmax_t(out.W) * std::uintmax_t(out.H) * sizeof(GeoPixelRaw);
    if (payload != expected) {
        std::cerr << "kgeo record size mismatch: GeoPixelRaw is out of sync with "
                     "GeoPixel in main.cpp.\n";
        return false;
    }

    out.px.resize(size_t(out.W) * size_t(out.H));
    f.read(reinterpret_cast<char*>(out.px.data()),
           std::streamsize(out.px.size() * sizeof(GeoPixelRaw)));
    return bool(f);
}

bool render_geo(const std::string& tracer_bin,
                const std::filesystem::path& geo_path,
                const char* chart_flag,
                double a_spin,
                int W, int H,
                double disk_out,
                double fov_deg,
                double r_obs) {
    std::ostringstream cmd;
    cmd << "\"" << tracer_bin << "\""
        << " --geo-only"
        << " --geo-file \"" << geo_path.string() << "\""
        << " --custom-res " << W << " " << H
        << " --solver-mode standard"
        << " " << chart_flag
        << " --a " << a_spin
        << " --theta 80"
        << " --phi 0"
        << " --r-obs " << r_obs
        << " --fov " << fov_deg
        << " --disk-out " << disk_out
        << " --tol 1e-10"
        << " --intersection-hermite";
#if defined(_WIN32)
    cmd << " > NUL 2>&1";
#else
    cmd << " > /dev/null 2>&1";
#endif
    return std::system(cmd.str().c_str()) == 0;
}

// Equivalent angular radius of the horizon silhouette, from its area. Using the
// area rather than a scan line averages the staircase over the whole boundary
// and so resolves the radius to a fraction of a pixel.
double shadow_radius_deg(const GeoFrame& g, double fov_deg) {
    size_t n = 0;
    for (const GeoPixelRaw& p : g.px)
        if (p.outcome == 2u) ++n;
    if (n == 0) return std::numeric_limits<double>::quiet_NaN();
    const double r_px = std::sqrt(double(n) / M_PI);
    return r_px * fov_deg / double(g.W - 1);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <path-to-kerr_tracer>\n";
        return 2;
    }
    const std::string tracer = argv[1];

    std::error_code ec;
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path(ec) / "kerrtrace_chart_regression";
    std::filesystem::create_directories(dir, ec);

    // ── 1. Absolute check: Schwarzschild shadow against theory ────────────
    //
    // r_disk_out below r_isco (6M at a=0) leaves the frame free of disk hits, so
    // the silhouette is the horizon alone and nothing occludes it.
    const double fov = 45.0, r_obs = 40.0;
    const double b_crit = 3.0 * std::sqrt(3.0);                       // 3 sqrt(3) M
    const double theory_deg =
        std::asin(b_crit * std::sqrt(1.0 - 2.0 / r_obs) / r_obs) * 180.0 / M_PI;

    for (const char* chart : {"--bl", "--ks"}) {
        const std::string tag = (std::string(chart) == "--bl") ? "bl" : "ks";
        const auto p = dir / ("shadow_" + tag + ".kgeo");
        if (!render_geo(tracer, p, chart, 0.0, 641, 361, 1.5, fov, r_obs)) {
            std::cerr << "shadow render failed for " << chart << "\n";
            return 1;
        }
        GeoFrame g;
        if (!load_kgeo(p, g)) { std::cerr << "load failed\n"; return 1; }
        const double meas = shadow_radius_deg(g, fov);
        const double err = 100.0 * (meas - theory_deg) / theory_deg;
        std::ostringstream d;
        d << tag << ": " << meas << " deg vs theory " << theory_deg
          << " deg, error " << err << "%, limit 1%";
        // Broken KS measured +5.26% here; correct charts land near +0.09%.
        check(std::isfinite(meas) && std::abs(err) < 1.0,
              "Schwarzschild shadow matches the analytic radius", d.str());
    }

    // ── 2. Relative check: the two charts agree, at rest and spinning ─────
    for (double a : {0.0, 0.9}) {
        std::ostringstream stag; stag << a;
        const auto p_bl = dir / ("cmp_bl_" + stag.str() + ".kgeo");
        const auto p_ks = dir / ("cmp_ks_" + stag.str() + ".kgeo");
        if (!render_geo(tracer, p_bl, "--bl", a, 320, 180, 12.0, fov, r_obs) ||
            !render_geo(tracer, p_ks, "--ks", a, 320, 180, 12.0, fov, r_obs)) {
            std::cerr << "comparison render failed at a=" << a << "\n";
            return 1;
        }
        GeoFrame bl, ks;
        if (!load_kgeo(p_bl, bl) || !load_kgeo(p_ks, ks)) {
            std::cerr << "load failed at a=" << a << "\n";
            return 1;
        }

        size_t hits_bl = 0, hits_ks = 0;
        std::vector<float> dr, dg, dphi, dphi_esc;
        for (size_t i = 0; i < bl.px.size(); ++i) {
            if (bl.px[i].outcome == 1u) ++hits_bl;
            if (ks.px[i].outcome == 1u) ++hits_ks;
            if (bl.px[i].outcome == 1u && ks.px[i].outcome == 1u) {
                dr.push_back(std::abs(bl.px[i].r - ks.px[i].r));
                dg.push_back(std::abs(bl.px[i].redshift - ks.px[i].redshift));
                dphi.push_back(std::abs(bl.px[i].phi_disk - ks.px[i].phi_disk));
            }
            if (bl.px[i].outcome == 0u && ks.px[i].outcome == 0u)
                dphi_esc.push_back(std::abs(bl.px[i].phi_esc - ks.px[i].phi_esc));
        }
        const double hit_diff = (hits_bl > 0)
            ? 100.0 * std::abs(double(hits_ks) - double(hits_bl)) / double(hits_bl)
            : 100.0;
        const double dr_med = median_of(dr);
        const double dg_med = median_of(dg);

        {   // Broken: 0.6575 at a=0, 0.5806 at a=0.9. Fixed: under 1e-4.
            std::ostringstream d;
            d << "a=" << a << ": median |dr| = " << dr_med << " M over " << dr.size()
              << " common hits, limit 0.01";
            check(!dr.empty() && std::isfinite(dr_med) && dr_med < 0.01,
                  "BL and KS place the disk hit at the same radius", d.str());
        }
        {   // Broken: 10.6% at a=0, 11.1% at a=0.9. Fixed: 0.01%.
            std::ostringstream d;
            d << "a=" << a << ": " << hits_bl << " vs " << hits_ks
              << ", " << hit_diff << "%, limit 0.5%";
            check(hits_bl > 1000 && hit_diff < 0.5,
                  "BL and KS see the disk in the same pixels", d.str());
        }
        {
            std::ostringstream d;
            d << "a=" << a << ": median |dg| = " << dg_med << ", limit 0.01";
            check(std::isfinite(dg_med) && dg_med < 0.01,
                  "BL and KS agree on the redshift", d.str());
        }
        {   // The (a/Delta) dr twist between the two azimuths. Broken: 0.1267 rad
            // at a=0.9 and zero at a=0, since the twist is proportional to a.
            // Fixed: 4.2e-4 at every spin, i.e. the same residual as the
            // untwisted a=0 case, which is numerical noise rather than a twist.
            const double dphi_med = median_of(dphi);
            std::ostringstream d;
            d << "a=" << a << ": median |dphi_disk| = " << dphi_med
              << " rad, limit 0.01";
            check(std::isfinite(dphi_med) && dphi_med < 0.01,
                  "BL and KS agree on the disk azimuth", d.str());
        }
        {   // This one did NOT discriminate the twist bug and is not claimed to:
            // G(r) vanishes at infinity, so an escaping ray was already close to
            // right (0.00115 rad before the fix, 2.9e-5 after). It is here as a
            // guard on background sampling, not as a witness for that fix.
            const double dpe_med = median_of(dphi_esc);
            std::ostringstream d;
            d << "a=" << a << ": median |dphi_esc| = " << dpe_med
              << " rad over " << dphi_esc.size() << " escaping rays, limit 0.01";
            check(std::isfinite(dpe_med) && dpe_med < 0.01,
                  "BL and KS sample the background at the same azimuth", d.str());
        }
    }

    std::filesystem::remove_all(dir, ec);

    if (failures) {
        std::cout << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "all chart-consistency checks passed\n";
    return 0;
}
