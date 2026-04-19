#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace {

struct ImageRGB {
    int w = 0;
    int h = 0;
    std::vector<uint8_t> px; // RGB
};

double clampd(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

bool load_rgb(const std::string& path, ImageRGB& out) {
    int c = 0;
    unsigned char* data = stbi_load(path.c_str(), &out.w, &out.h, &c, 3);
    if (!data) return false;
    out.px.assign(data, data + (size_t)out.w * (size_t)out.h * 3u);
    stbi_image_free(data);
    return true;
}

double luminance_at(const ImageRGB& img, int x, int y) {
    const size_t i = (size_t)(y * img.w + x) * 3u;
    const double r = img.px[i + 0];
    const double g = img.px[i + 1];
    const double b = img.px[i + 2];
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

struct BBox {
    int x0 = std::numeric_limits<int>::max();
    int x1 = std::numeric_limits<int>::min();
    int y0 = std::numeric_limits<int>::max();
    int y1 = std::numeric_limits<int>::min();
    int area = 0;
};

// Largest 4-connected dark component.
BBox largest_dark_component(const ImageRGB& img, double lum_thr = 4.0) {
    const int W = img.w, H = img.h;
    std::vector<uint8_t> dark((size_t)W * (size_t)H, 0u);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            dark[(size_t)y * (size_t)W + (size_t)x] =
                (luminance_at(img, x, y) < lum_thr) ? 1u : 0u;
        }
    }

    std::vector<uint8_t> vis((size_t)W * (size_t)H, 0u);
    BBox best{};
    for (int sy = 0; sy < H; ++sy) {
        for (int sx = 0; sx < W; ++sx) {
            const size_t sidx = (size_t)sy * (size_t)W + (size_t)sx;
            if (!dark[sidx] || vis[sidx]) continue;

            std::queue<std::pair<int, int>> q;
            q.push({sx, sy});
            vis[sidx] = 1u;

            BBox cur{};
            cur.area = 0;
            while (!q.empty()) {
                const auto [x, y] = q.front();
                q.pop();
                cur.area++;
                cur.x0 = std::min(cur.x0, x);
                cur.x1 = std::max(cur.x1, x);
                cur.y0 = std::min(cur.y0, y);
                cur.y1 = std::max(cur.y1, y);

                const int nx[4] = {x - 1, x + 1, x, x};
                const int ny[4] = {y, y, y - 1, y + 1};
                for (int k = 0; k < 4; ++k) {
                    if (nx[k] < 0 || nx[k] >= W || ny[k] < 0 || ny[k] >= H) continue;
                    const size_t j = (size_t)ny[k] * (size_t)W + (size_t)nx[k];
                    if (dark[j] && !vis[j]) {
                        vis[j] = 1u;
                        q.push({nx[k], ny[k]});
                    }
                }
            }
            if (cur.area > best.area) best = cur;
        }
    }
    return best;
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    p = clampd(p, 0.0, 100.0);
    const double t = (p / 100.0) * (double)(v.size() - 1);
    const size_t k0 = (size_t)std::floor(t);
    const size_t k1 = (size_t)std::ceil(t);
    std::nth_element(v.begin(), v.begin() + k0, v.end());
    const double a = v[k0];
    if (k1 == k0) return a;
    std::nth_element(v.begin(), v.begin() + k1, v.end());
    const double b = v[k1];
    return a + (t - (double)k0) * (b - a);
}

template <class T>
std::vector<T> moving_average(const std::vector<T>& x, int w) {
    if (x.empty()) return {};
    w = std::max(3, w | 1); // odd
    std::vector<T> out(x.size(), T(0));
    const int r = w / 2;
    for (int i = 0; i < (int)x.size(); ++i) {
        const int a = std::max(0, i - r);
        const int b = std::min((int)x.size() - 1, i + r);
        T s = T(0);
        for (int j = a; j <= b; ++j) s += x[j];
        out[i] = s / (T)(b - a + 1);
    }
    return out;
}

double bump_score(const ImageRGB& img) {
    if (img.w < 32 || img.h < 32) return std::numeric_limits<double>::quiet_NaN();

    const BBox bb = largest_dark_component(img, 4.0);
    if (bb.area <= 0) return std::numeric_limits<double>::quiet_NaN();

    const double cx = 0.5 * (bb.x0 + bb.x1);
    const double cy = 0.5 * (bb.y0 + bb.y1);
    const double rx = 0.5 * (bb.x1 - bb.x0);
    const double ry = 0.5 * (bb.y1 - bb.y0);
    if (rx < 8.0 || ry < 8.0) return std::numeric_limits<double>::quiet_NaN();

    const int x0 = std::max(0, (int)std::floor(cx - 1.5 * rx));
    const int x1 = std::min(img.w - 1, (int)std::ceil (cx + 1.5 * rx));
    const int y0 = std::max(0, (int)std::floor(cy - 0.55 * ry));
    const int y1 = std::min(img.h - 1, (int)std::ceil (cy + 0.15 * ry));
    if (x1 - x0 < 16 || y1 - y0 < 8) return std::numeric_limits<double>::quiet_NaN();

    std::vector<double> y_peak;
    std::vector<double> v_peak;
    y_peak.reserve((size_t)(x1 - x0 + 1));
    v_peak.reserve((size_t)(x1 - x0 + 1));
    for (int x = x0; x <= x1; ++x) {
        int best_y = y0;
        double best_v = -1.0;
        for (int y = y0; y <= y1; ++y) {
            const double v = luminance_at(img, x, y);
            if (v > best_v) {
                best_v = v;
                best_y = y;
            }
        }
        y_peak.push_back((double)best_y);
        v_peak.push_back(best_v);
    }

    // Longest contiguous bright run (approaching-side hotspot arc).
    // Fallback across percentiles for images where the top bright band is short.
    int best_a = -1, best_b = -1;
    auto try_percentile = [&](double p, int min_run_len) -> bool {
        const double v_thr = percentile(v_peak, p);
        std::vector<int> bright_idx;
        for (int i = 0; i < (int)v_peak.size(); ++i)
            if (v_peak[i] >= v_thr) bright_idx.push_back(i);
        if ((int)bright_idx.size() < min_run_len) return false;

        int a = bright_idx.front(), b = bright_idx.front();
        int la = a, lb = b;
        for (size_t k = 1; k < bright_idx.size(); ++k) {
            if (bright_idx[k] == b + 1) {
                b = bright_idx[k];
            } else {
                if ((b - a) > (lb - la)) { la = a; lb = b; }
                a = b = bright_idx[k];
            }
        }
        if ((b - a) > (lb - la)) { la = a; lb = b; }
        const int run_len = lb - la + 1;
        if (run_len < min_run_len) return false;
        const int w = std::max(5, ((run_len / 20) | 1));
        const int m = std::max(3, w);
        if (run_len < 2 * m + 5) return false;
        best_a = la;
        best_b = lb;
        return true;
    };
    if (!try_percentile(85.0, 12) &&
        !try_percentile(80.0, 10) &&
        !try_percentile(75.0, 8)  &&
        !try_percentile(70.0, 8)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    std::vector<double> y_seg(y_peak.begin() + best_a, y_peak.begin() + best_b + 1);
    const int w = std::max(5, (((int)y_seg.size() / 20) | 1));
    const std::vector<double> y_s = moving_average(y_seg, w);

    // Ignore boundaries where filter support is asymmetric.
    const int m = std::max(3, w);
    if ((int)y_s.size() < 2 * m + 5) return std::numeric_limits<double>::quiet_NaN();
    std::vector<double> core(y_s.begin() + m, y_s.end() - m);

    double max_abs_d2 = 0.0;
    for (int i = 1; i + 1 < (int)core.size(); ++i) {
        const double d2 = core[i + 1] - 2.0 * core[i] + core[i - 1];
        max_abs_d2 = std::max(max_abs_d2, std::abs(d2));
    }
    return max_abs_d2;
}

ImageRGB make_synthetic(bool with_bump) {
    const int W = 320, H = 180;
    ImageRGB img;
    img.w = W; img.h = H;
    img.px.assign((size_t)W * (size_t)H * 3u, 0u);

    // Dark "shadow" ellipse.
    const double cx = 160.0, cy = 90.0, rx = 78.0, ry = 48.0;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const double dx = (x - cx) / rx;
            const double dy = (y - cy) / ry;
            if (dx * dx + dy * dy <= 1.0) {
                const size_t i = (size_t)(y * W + x) * 3u;
                img.px[i + 0] = 0;
                img.px[i + 1] = 0;
                img.px[i + 2] = 0;
            }
        }
    }

    // Bright upper arc with optional local kink ("bump").
    for (int x = 30; x < W - 30; ++x) {
        const double t = (x - 30.0) / (W - 60.0);
        double y = 74.0 + 20.0 * (t - 0.5) * (t - 0.5);
        if (with_bump && x > 176 && x < 204) y += 18.0;
        for (int k = -2; k <= 2; ++k) {
            const int yy = (int)std::round(y) + k;
            if (yy < 0 || yy >= H) continue;
            const uint8_t v = (k == 0) ? 255u : 220u;
            const size_t i = (size_t)(yy * W + x) * 3u;
            img.px[i + 0] = v;
            img.px[i + 1] = v;
            img.px[i + 2] = v;
        }
    }
    return img;
}

int self_test() {
    const ImageRGB smooth = make_synthetic(false);
    const ImageRGB bump   = make_synthetic(true);
    const double s0 = bump_score(smooth);
    const double s1 = bump_score(bump);

    if (!(std::isfinite(s0) && std::isfinite(s1))) {
        std::cerr << "Self-test failed: non-finite scores.\n";
        return 1;
    }
    if (!(s0 < 3.0)) {
        std::cerr << "Self-test failed: smooth score too high: " << s0 << "\n";
        return 1;
    }
    if (!(s1 > s0 + 0.5)) {
        std::cerr << "Self-test failed: bumped score too low: smooth=" << s0
                  << " bump=" << s1 << "\n";
        return 1;
    }
    std::cout << "Self-test OK: smooth=" << s0 << " bump=" << s1 << "\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    std::string image_path;
    double threshold = 3.0;
    bool run_self = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--self-test") run_self = true;
        else if (a == "--image" && i + 1 < argc) image_path = argv[++i];
        else if (a == "--threshold" && i + 1 < argc) threshold = std::stod(argv[++i]);
    }

    if (run_self) return self_test();

    if (image_path.empty()) {
        std::cerr << "Usage:\n"
                  << "  bump_detector --self-test\n"
                  << "  bump_detector --image <path> [--threshold <value>]\n";
        return 2;
    }

    ImageRGB img;
    if (!load_rgb(image_path, img)) {
        std::cerr << "Cannot load image: " << image_path << "\n";
        return 2;
    }

    const double score = bump_score(img);
    if (!std::isfinite(score)) {
        std::cerr << "Cannot compute bump score for: " << image_path << "\n";
        return 2;
    }

    const bool has_bump = (score > threshold);
    std::cout << "image=" << image_path << " score=" << score
              << " threshold=" << threshold
              << " bump=" << (has_bump ? "YES" : "NO") << "\n";
    return has_bump ? 1 : 0;
}
