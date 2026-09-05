// Ray-bundle regression test.
//
// Bundle mode had no coverage at all, and it broke twice in ways a test would
// have caught immediately:
//
//   1. Non-termination (fixed in 88e214c). The equator proximity test used a
//      `continue` that also skipped the horizon and escape tests, so almost
//      every ray ran to max_steps instead of stopping on an event. The render
//      time scaled with --max-steps and a 320x180 frame did not finish in ten
//      minutes; the same frame takes under three seconds once fixed.
//
//   2. Wrong sign of Omega in the redshift (fixed in dfde570). keplerian_omega()
//      returns -Omega_K by convention and the bundle path did not negate it, so
//      the disk-frame normalisation d2 was evaluated on the retrograde branch
//      and went negative inside r ~ 1.5M. The guard then fell back to g = 1,
//      i.e. the inner disk rendered as if unshifted, and with the g^4 beaming
//      that is a factor 1e4 too bright: a saturated white ring around the
//      shadow. Measured on the frame below, 100% of the disk hits inside 2M sat
//      at exactly g = 1.0 before the fix and 0% after.
//
// The thresholds below were calibrated against binaries built from both sides
// of that fix; each assertion is followed by the value it takes when broken and
// when correct, so the margins are not guesses.

#include <algorithm>
#include <chrono>
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

// Must mirror GeoPixel in main.cpp exactly: this is a raw binary record and the
// reader has no way to discover the writer's layout.
struct GeoPixelRaw {
    uint8_t outcome;
    uint8_t pad[3];
    float r;
    float redshift;
    float magnif;
    float phi_disk;
    float theta_esc;
    float phi_esc;
    float fp_dr_a, fp_dphi_a;   // pixel footprint on the disk (bundle mode)
    float fp_dr_b, fp_dphi_b;
};
static_assert(sizeof(GeoPixelRaw) == 44, "GeoPixelRaw must match GeoPixel in main.cpp");

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

double percentile_of(std::vector<float> v, double p01) {
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(v.begin(), v.end());
    const double t = p01 * double(v.size() - 1);
    const size_t i0 = size_t(std::floor(t));
    const size_t i1 = size_t(std::ceil(t));
    if (i0 == i1) return double(v[i0]);
    return double(v[i0]) + (t - double(i0)) * (double(v[i1]) - double(v[i0]));
}

bool load_kgeo(const std::filesystem::path& path, GeoFrame& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    char magic[4] = {};
    f.read(magic, 4);
    if (!f || std::memcmp(magic, "KGEO", 4) != 0) return false;

    uint32_t version = 0;
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!f || version != 2) return false;

    f.read(reinterpret_cast<char*>(&out.W), sizeof(out.W));
    f.read(reinterpret_cast<char*>(&out.H), sizeof(out.H));
    if (!f || out.W == 0 || out.H == 0) return false;

    f.seekg(80, std::ios::cur);  // rest of KGeoMeta after W/H
    if (!f) return false;

    // The record layout is not self-describing, so verify it against the file
    // size before trusting a single pixel.
    const std::uintmax_t total = std::filesystem::file_size(path);
    const std::uintmax_t header = 4 + 4 + 4 + 4 + 80;
    const std::uintmax_t payload = (total > header) ? (total - header) : 0;
    const std::uintmax_t expected =
        std::uintmax_t(out.W) * std::uintmax_t(out.H) * sizeof(GeoPixelRaw);
    if (payload != expected) {
        std::cerr << "kgeo record size mismatch: " << payload << " bytes for "
                  << out.W << "x" << out.H << " at " << sizeof(GeoPixelRaw)
                  << " bytes/pixel (expected " << expected << "). "
                  << "GeoPixelRaw is out of sync with GeoPixel in main.cpp.\n";
        return false;
    }

    out.px.resize(size_t(out.W) * size_t(out.H));
    f.read(reinterpret_cast<char*>(out.px.data()),
           std::streamsize(out.px.size() * sizeof(GeoPixelRaw)));
    return bool(f);
}

// Small enough to stay a unit test: 0.4 s single-ray, 2.7 s bundle on an M-series
// laptop. Near-extremal spin and a grazing inclination put a large fraction of
// the disk hits inside 2M, which is exactly where the redshift bug lived.
bool render_geo(const std::string& tracer_bin,
                const std::filesystem::path& geo_path,
                bool bundles,
                int max_steps,
                double& seconds) {
    std::ostringstream cmd;
    cmd << "\"" << tracer_bin << "\""
        << " --geo-only"
        << " --geo-file \"" << geo_path.string() << "\""
        << " --custom-res 320 180"
        << " --solver-mode standard"
        << " --ks"
        << " --a 0.998"
        << " --theta 80"
        << " --phi 0"
        << " --r-obs 40"
        << " --fov 45"
        << " --disk-out 12"
        << " --intersection-hermite";
    if (bundles) cmd << " --bundles";
    if (max_steps > 0) cmd << " --max-steps " << max_steps;
#if defined(_WIN32)
    cmd << " > NUL 2>&1";
#else
    cmd << " > /dev/null 2>&1";
#endif
    const auto t0 = std::chrono::steady_clock::now();
    const int rc = std::system(cmd.str().c_str());
    seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return rc == 0;
}

bool files_identical(const std::filesystem::path& a, const std::filesystem::path& b) {
    std::error_code ec;
    if (std::filesystem::file_size(a, ec) != std::filesystem::file_size(b, ec)) return false;
    std::ifstream fa(a, std::ios::binary), fb(b, std::ios::binary);
    if (!fa || !fb) return false;
    constexpr size_t kChunk = 1 << 16;
    std::vector<char> ba(kChunk), bb(kChunk);
    while (fa && fb) {
        fa.read(ba.data(), std::streamsize(kChunk));
        fb.read(bb.data(), std::streamsize(kChunk));
        const std::streamsize na = fa.gcount(), nb = fb.gcount();
        if (na != nb) return false;
        if (na == 0) break;
        if (std::memcmp(ba.data(), bb.data(), size_t(na)) != 0) return false;
    }
    return true;
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
        std::filesystem::temp_directory_path(ec) / "kerrtrace_bundle_regression";
    std::filesystem::create_directories(dir, ec);
    const auto p_single     = dir / "single.kgeo";
    const auto p_bundle     = dir / "bundle.kgeo";
    const auto p_bundle_big = dir / "bundle_maxsteps.kgeo";

    // A correct bundle render of this frame takes ~2.7 s. Before the termination
    // fix the same render did not finish in ten minutes, so a ceiling this loose
    // still separates the two by orders of magnitude while leaving room for a
    // slow shared CI runner.
    constexpr double kRenderCeilingSeconds = 300.0;

    double t_single = 0.0, t_bundle = 0.0, t_bundle_big = 0.0;
    if (!render_geo(tracer, p_single, /*bundles=*/false, 0, t_single)) {
        std::cerr << "single-ray render failed\n";
        return 1;
    }
    if (!render_geo(tracer, p_bundle, /*bundles=*/true, 0, t_bundle)) {
        std::cerr << "bundle render failed\n";
        return 1;
    }
    if (!render_geo(tracer, p_bundle_big, /*bundles=*/true, 200000, t_bundle_big)) {
        std::cerr << "bundle render with raised max-steps failed\n";
        return 1;
    }

    GeoFrame s, b;
    if (!load_kgeo(p_single, s) || !load_kgeo(p_bundle, b)) {
        std::cerr << "failed to load the rendered .kgeo buffers\n";
        return 1;
    }
    if (s.W != b.W || s.H != b.H) {
        std::cerr << "frame size mismatch between the two renders\n";
        return 1;
    }

    // ── Gather ────────────────────────────────────────────────────────────
    std::vector<float> inner_g;      // bundle disk hits inside 2M
    std::vector<float> dg_common;    // |g_bundle - g_single| where both hit the disk
    size_t hits_s = 0, hits_b = 0;
    size_t inner_exactly_one = 0;
    size_t bad_magnif = 0, bad_g = 0;

    for (size_t i = 0; i < b.px.size(); ++i) {
        const GeoPixelRaw& ps = s.px[i];
        const GeoPixelRaw& pb = b.px[i];
        if (ps.outcome == 1u) ++hits_s;
        if (pb.outcome != 1u) continue;
        ++hits_b;

        if (!std::isfinite(pb.magnif) || pb.magnif <= 0.0f) ++bad_magnif;
        if (!std::isfinite(pb.redshift) || pb.redshift <= 0.0f || pb.redshift > 6.0f) ++bad_g;

        if (pb.r < 2.0f) {
            inner_g.push_back(pb.redshift);
            // The signature of the Omega-sign bug: d2 < 0 made the code fall
            // back to a literal 1.0 rather than compute a shift.
            if (std::abs(pb.redshift - 1.0f) < 1e-6f) ++inner_exactly_one;
        }
        if (ps.outcome == 1u)
            dg_common.push_back(std::abs(pb.redshift - ps.redshift));
    }

    std::cout << "frame " << b.W << "x" << b.H
              << "  single-ray hits=" << hits_s << " (" << t_single << " s)"
              << "  bundle hits=" << hits_b << " (" << t_bundle << " s)\n";

    // ── Assertions ────────────────────────────────────────────────────────
    {
        std::ostringstream d;
        d << t_bundle << " s, ceiling " << kRenderCeilingSeconds << " s";
        check(t_bundle < kRenderCeilingSeconds && t_bundle_big < kRenderCeilingSeconds,
              "bundle render terminates on events, not on the step cap", d.str());
    }
    {
        // Every ray that stops on a real event is unaffected by a larger cap.
        // Rays that instead run out of steps return wherever they happened to
        // be, so raising the cap changes their record.
        const bool same = files_identical(p_bundle, p_bundle_big);
        std::ostringstream d;
        d << "max-steps 60000 vs 200000, byte-identical=" << (same ? "yes" : "no");
        check(same, "bundle output is independent of --max-steps", d.str());
    }
    {
        std::ostringstream d;
        d << hits_b << " bundle vs " << hits_s << " single-ray hits";
        check(hits_b > 1000 && double(hits_b) >= 0.75 * double(hits_s),
              "bundle finds the disk where single-ray does", d.str());
    }
    {
        std::ostringstream d;
        d << bad_magnif << " non-finite or non-positive |det J|";
        check(bad_magnif == 0, "every bundle hit carries a usable Jacobian", d.str());
    }
    {
        std::ostringstream d;
        d << bad_g << " redshifts outside (0, 6]";
        check(bad_g == 0, "bundle redshifts stay inside the physical clamp", d.str());
    }
    {
        // Broken: 1308 of 1308 (100%). Fixed: 0.
        const double frac = inner_g.empty() ? 1.0
                          : double(inner_exactly_one) / double(inner_g.size());
        std::ostringstream d;
        d << inner_exactly_one << " of " << inner_g.size() << " inner hits at exactly g=1 ("
          << (100.0 * frac) << "%), limit 1%";
        check(!inner_g.empty() && frac <= 0.01,
              "no inner-disk pixel falls back to an unshifted g=1", d.str());
    }
    {
        // Broken: 1.0000. Fixed: 0.173, against 0.168 for single-ray.
        const double med = median_of(inner_g);
        std::ostringstream d;
        d << "median g = " << med << " inside 2M, limit 0.50";
        check(std::isfinite(med) && med < 0.50,
              "the inner disk is strongly redshifted", d.str());
    }
    {
        // Broken: p90 = 0.604. Fixed: p90 = 0.066.
        const double p90 = percentile_of(dg_common, 0.90);
        std::ostringstream d;
        d << "p90 |dg| = " << p90 << " over " << dg_common.size()
          << " common hits, limit 0.25";
        check(std::isfinite(p90) && p90 < 0.25,
              "bundle and single-ray agree on the redshift", d.str());
    }

    std::filesystem::remove_all(dir, ec);

    if (failures) {
        std::cout << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "all ray-bundle checks passed\n";
    return 0;
}
