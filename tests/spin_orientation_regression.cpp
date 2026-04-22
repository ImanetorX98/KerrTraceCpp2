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

struct GeoPixelRaw {
    uint8_t outcome;
    uint8_t pad[3];
    float r;
    float redshift;
    float magnif;
    float theta_esc;
    float phi_esc;
};
static_assert(sizeof(GeoPixelRaw) == 24, "GeoPixelRaw size mismatch");

struct GeoFrame {
    uint32_t W = 0;
    uint32_t H = 0;
    std::vector<GeoPixelRaw> px;
};

struct OrientationMetrics {
    bool bright_right = false;
    bool gap_right = false;
    double cx = 0.0;
    double hot_x = 0.0;
    int best_gap = -1;
    int best_x = -1;
};

double percentile_linear(std::vector<float> v, double p01) {
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    if (p01 < 0.0) p01 = 0.0;
    if (p01 > 1.0) p01 = 1.0;
    std::sort(v.begin(), v.end());
    const double t = p01 * double(v.size() - 1);
    const size_t i0 = size_t(std::floor(t));
    const size_t i1 = size_t(std::ceil(t));
    if (i0 == i1) return double(v[i0]);
    const double a = double(v[i0]);
    const double b = double(v[i1]);
    return a + (t - double(i0)) * (b - a);
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

    // Skip remaining KGeoMeta payload after W/H (10 doubles = 80 bytes).
    f.seekg(80, std::ios::cur);
    if (!f) return false;

    out.px.resize(size_t(out.W) * size_t(out.H));
    f.read(reinterpret_cast<char*>(out.px.data()),
           std::streamsize(out.px.size() * sizeof(GeoPixelRaw)));
    return bool(f);
}

bool render_geo(const std::string& tracer_bin,
                const std::filesystem::path& geo_path,
                double a_spin) {
    std::ostringstream cmd;
    cmd << "\"" << tracer_bin << "\""
        << " --geo-only"
        << " --geo-file \"" << geo_path.string() << "\""
        << " --custom-res 320 180"
        << " --solver-mode standard"
        << " --ks"
        << " --a " << a_spin
        << " --theta 80"
        << " --phi 0"
        << " --r-obs 40"
        << " --fov 45"
        << " --disk-out 12"
        << " --intersection-hermite";
#if defined(_WIN32)
    cmd << " > NUL 2>&1";
#else
    cmd << " > /dev/null 2>&1";
#endif
    return std::system(cmd.str().c_str()) == 0;
}

bool analyze_orientation(const GeoFrame& g, OrientationMetrics& m) {
    if (g.px.empty()) return false;
    const int W = int(g.W);
    const int H = int(g.H);

    double sum_x_h = 0.0, sum_y_h = 0.0;
    int n_h = 0;
    std::vector<float> disk_red;
    disk_red.reserve(g.px.size() / 4);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const GeoPixelRaw& p = g.px[size_t(y) * size_t(W) + size_t(x)];
            if (p.outcome == 2u) {
                sum_x_h += double(x);
                sum_y_h += double(y);
                ++n_h;
            } else if (p.outcome == 1u && std::isfinite(p.redshift)) {
                disk_red.push_back(p.redshift);
            }
        }
    }
    if (n_h <= 0 || disk_red.empty()) return false;

    m.cx = sum_x_h / double(n_h);
    const double cy = sum_y_h / double(n_h);

    const double q97 = percentile_linear(disk_red, 0.97);
    double sum_hot_x = 0.0;
    int n_hot = 0;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const GeoPixelRaw& p = g.px[size_t(y) * size_t(W) + size_t(x)];
            if (p.outcome == 1u && std::isfinite(p.redshift) && double(p.redshift) >= q97) {
                sum_hot_x += double(x);
                ++n_hot;
            }
        }
    }
    if (n_hot <= 0) return false;
    m.hot_x = sum_hot_x / double(n_hot);
    m.bright_right = (m.hot_x > m.cx);

    const int y_top = std::max(1, int(std::floor(cy)));
    int best_gap = -1;
    int best_x = -1;

    for (int x = 0; x < W; ++x) {
        int y_h = -1;
        int y_d = -1;
        for (int y = 0; y < y_top; ++y) {
            const GeoPixelRaw& p = g.px[size_t(y) * size_t(W) + size_t(x)];
            if (y_h < 0 && p.outcome == 2u) y_h = y;
            if (y_d < 0 && p.outcome == 1u) y_d = y;
            if (y_h >= 0 && y_d >= 0) break;
        }
        if (y_h < 0 || y_d < 0) continue;
        const int gap = y_h - y_d;
        if (gap > best_gap) {
            best_gap = gap;
            best_x = x;
        }
    }
    if (best_x < 0) return false;

    m.best_gap = best_gap;
    m.best_x = best_x;
    m.gap_right = (double(best_x) > m.cx);
    return true;
}

bool check_case(const std::string& tracer_bin,
                double a_spin,
                bool expect_bright_right,
                bool expect_gap_right,
                const std::filesystem::path& geo_path) {
    if (!render_geo(tracer_bin, geo_path, a_spin)) {
        std::cerr << "Render failed for a=" << a_spin << "\n";
        return false;
    }

    GeoFrame gf;
    if (!load_kgeo(geo_path, gf)) {
        std::cerr << "Cannot load " << geo_path << "\n";
        return false;
    }

    OrientationMetrics m;
    if (!analyze_orientation(gf, m)) {
        std::cerr << "Cannot analyze " << geo_path << "\n";
        return false;
    }

    const bool ok = (m.bright_right == expect_bright_right) &&
                    (m.gap_right == expect_gap_right);
    std::cout << "a=" << a_spin
              << " bright=" << (m.bright_right ? "right" : "left")
              << " gap=" << (m.gap_right ? "right" : "left")
              << " (cx=" << m.cx << ", hot_x=" << m.hot_x
              << ", best_gap=" << m.best_gap << ", best_x=" << m.best_x << ")\n";
    if (!ok) {
        std::cerr << "Expected bright=" << (expect_bright_right ? "right" : "left")
                  << " and gap=" << (expect_gap_right ? "right" : "left") << "\n";
    }
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: spin_orientation_regression <path-to-kerr_tracer>\n";
        return 2;
    }

    const std::string tracer_bin = argv[1];
    const auto tmp = std::filesystem::temp_directory_path();
    const auto geo_pos = tmp / "kerrtrace_spin_pos.kgeo";
    const auto geo_neg = tmp / "kerrtrace_spin_neg.kgeo";

    const bool ok_pos = check_case(tracer_bin, +0.5, false, false, geo_pos);
    const bool ok_neg = check_case(tracer_bin, -0.5, true, true, geo_neg);

    std::error_code ec;
    std::filesystem::remove(geo_pos, ec);
    std::filesystem::remove(geo_neg, ec);

    if (!ok_pos || !ok_neg) {
        std::cerr << "Spin orientation regression failed.\n";
        return 1;
    }
    std::cout << "Spin orientation regression passed.\n";
    return 0;
}
