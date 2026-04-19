// ============================================================
//  main.cpp — KNdS relativistic ray-tracer
//
//  Backends selected at compile time via CMake options:
//    -DUSE_METAL=ON   → Metal GPU (macOS)
//    -DUSE_CUDA=ON    → CUDA GPU  (Linux/Windows)
//    default          → CPU with OpenMP
//
//  Run:  ./kerr_tracer [--bundles]
//    --bundles  : use Jacobi ray-bundle renderer (slower, antialiased)
// ============================================================
#include "camera.hpp"
#include "geodesic.hpp"
#include "knds_metric.hpp"
#include "ray_bundle.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#if defined(USE_METAL)
#  include "gpu/metal/metal_renderer.hpp"
#elif defined(USE_CUDA)
#  include "gpu/cuda/tracer.cuh"
#elif defined(_OPENMP)
#  include <omp.h>
#endif

#include <chrono>
static double get_time() {
#if defined(_OPENMP) && !defined(USE_METAL) && !defined(USE_CUDA)
    return omp_get_wtime();
#else
    using C = std::chrono::steady_clock;
    static auto t0 = C::now();
    return std::chrono::duration<double>(C::now()-t0).count();
#endif
}

// ── clamp helper ─────────────────────────────────────────────
template<class T> static T clamp(T v, T lo, T hi) {
    return v < lo ? lo : v > hi ? hi : v;
}

// ── Colour ────────────────────────────────────────────────────
struct RGB { uint8_t r, g, b; };

// ── Equirectangular background image ─────────────────────────
struct BackgroundImage {
    int w = 0, h = 0;
    std::vector<uint8_t> px;   // RGB, row-major

    bool load(const char* path) {
        int c;
        uint8_t* data = stbi_load(path, &w, &h, &c, 3);
        if (!data) return false;
        px.assign(data, data + (size_t)w*h*3);
        stbi_image_free(data);
        return true;
    }

    // θ ∈ [0,π], φ ∈ ℝ  →  bilinear sample
    RGB sample(double theta, double phi) const {
        if (px.empty()) return {0,0,0};
        // Wrap φ to [0, 2π)
        double phi_wrap = std::fmod(phi, 2.0*M_PI);
        if (phi_wrap < 0) phi_wrap += 2.0*M_PI;

        double uf = phi_wrap / (2.0*M_PI) * (w-1);
        double vf = theta   / M_PI        * (h-1);

        int u0 = (int)uf, u1 = std::min(u0+1, w-1);
        int v0 = (int)vf, v1 = std::min(v0+1, h-1);
        double fu = uf - u0, fv = vf - v0;

        auto at = [&](int u, int v) -> const uint8_t* {
            v = clamp(v, 0, h-1);
            return px.data() + 3*(v*w + clamp(u,0,w-1));
        };
        auto lerp = [](double a, double b, double t) { return a + t*(b-a); };

        double R = lerp(lerp(at(u0,v0)[0], at(u1,v0)[0], fu),
                        lerp(at(u0,v1)[0], at(u1,v1)[0], fu), fv);
        double G = lerp(lerp(at(u0,v0)[1], at(u1,v0)[1], fu),
                        lerp(at(u0,v1)[1], at(u1,v1)[1], fu), fv);
        double B = lerp(lerp(at(u0,v0)[2], at(u1,v0)[2], fu),
                        lerp(at(u0,v1)[2], at(u1,v1)[2], fu), fv);
        return {(uint8_t)R, (uint8_t)G, (uint8_t)B};
    }
};

// ── Trace outcome ─────────────────────────────────────────────
enum class Outcome { ESCAPED, DISK_HIT, HORIZON };
struct TraceResult {
    Outcome out;
    double  r;
    double  redshift;
    double  theta_esc = 0.0;   ///< final θ when ESCAPED (for background lookup)
    double  phi_esc   = 0.0;   ///< final φ when ESCAPED
};

static double disk_redshift(double r, double pt, double pphi,
                            const KNdSMetric& g) {
    double Omega = g.keplerian_omega(r);
    double b     = pphi / pt;    // pt<0; backward ray pphi sign is opposite to forward photon
    double gLL[4][4]; g.covariant_BL(r, M_PI/2.0, gLL);
    double d2 = -(gLL[0][0]+2.0*gLL[0][3]*Omega+gLL[3][3]*Omega*Omega);
    if (d2 <= 0.0) return 1.0;
    double dv = 1.0 - Omega*b;
    if (std::abs(dv) < 1e-12) return 10.0;
    return std::sqrt(d2)/dv;
}

static TraceResult trace_single(GeodesicState s, const KNdSMetric& g,
                                 double r_disk_in, double r_disk_out,
                                 double r_escape,
                                 Integrator intg = Integrator::RK4_DOUBLING) {
    double rh = g.r_horizon(), dlam = 1.0;
    double prev_cos = std::cos(s.theta);
    Vec4d  fsal = Vec4d::nan_init();   // DOPRI5 FSAL carry-over

    for (int it = 0; it < 500000; ++it) {
        while (!adaptive_step(g, s, dlam, intg, fsal)) {}
        if (s.r < rh*1.03) return {Outcome::HORIZON, s.r, 0.0};
        if (s.r > r_escape) return {Outcome::ESCAPED, s.r, 1.0, s.theta, s.phi};
        double cc = std::cos(s.theta);
        if (prev_cos*cc <= 0.0 && s.r >= r_disk_in && s.r <= r_disk_out)
            return {Outcome::DISK_HIT, s.r,
                    clamp(disk_redshift(s.r, s.pt, s.pphi, g), 0.0, 20.0)};
        prev_cos = cc;
    }
    return {Outcome::ESCAPED, s.r, 1.0, s.theta, s.phi};
}

static RGB blackbody(double T) {
    T = clamp(T, 800.0, 4e4);
    double t = std::log10(T/800.0)/std::log10(4e4/800.0);
    double R,G,B;
    if(t<0.25){R=1;G=t/0.25*0.4;B=0;}
    else if(t<0.5){double f=(t-0.25)/0.25;R=1;G=0.4+f*0.4;B=f*0.3;}
    else if(t<0.75){double f=(t-0.5)/0.25;R=1;G=0.8+f*0.2;B=0.3+f*0.5;}
    else{double f=(t-0.75)/0.25;R=1-f*0.2;G=1;B=0.8+f*0.2;}
    return {(uint8_t)(clamp(R,0.0,1.0)*255),
            (uint8_t)(clamp(G,0.0,1.0)*255),
            (uint8_t)(clamp(B,0.0,1.0)*255)};
}

// Reinhard tonemapping + gamma 2.2
static double tonemap(double x) {
    x = x / (1.0 + x);
    return std::pow(clamp(x, 0.0, 1.0), 1.0/2.2);
}

// Page-Thorne (Novikov-Thorne) emissivity profile.
// Zero at r_isco, peaks near 3*r_isco, falls as r^{-3} at large r.
// Simplified: f(r) = (1 - sqrt(r_isco/r)) / r^3
static double page_thorne(double r, double r_isco) {
    if (r <= r_isco) return 0.0;
    double x = std::sqrt(r_isco / r);          // = 1 at ISCO, → 0 at infinity
    return (1.0 - x) / (r * r * r);
}

static RGB disk_colour(double r, double red, double magnif,
                       double M, double r_isco) {
    // ── Colour temperature: warm at large r, hot near ISCO ───────
    // Reference: ~6500 K at r = 6M, scales as r^{-1/2} with redshift.
    // This keeps the inner disk blue-white and outer disk orange-red.
    double T = 6500.0 * std::sqrt(6.0*M / r) * clamp(red, 0.2, 5.0);

    // ── Intensity: Page-Thorne × g^4 × magnification correction ──
    double pt_norm = page_thorne(r, r_isco);
    // Normalise so peak (≈ 3·r_isco) maps to I ≈ 1
    double r_peak  = 3.0 * r_isco;
    double pt_peak = page_thorne(r_peak, r_isco);
    double I = (pt_peak > 0.0) ? pt_norm / pt_peak : 0.0;
    I *= std::pow(clamp(red, 0.1, 10.0), 4.0);   // g^4 surface brightness
    I *= clamp(1.0 / magnif, 0.05, 5.0);          // flux conservation

    auto c = blackbody(T);
    return {(uint8_t)(tonemap(c.r/255.0 * I) * 255),
            (uint8_t)(tonemap(c.g/255.0 * I) * 255),
            (uint8_t)(tonemap(c.b/255.0 * I) * 255)};
}

// ── Write PPM ─────────────────────────────────────────────────
static void write_ppm(const char* path, const std::vector<RGB>& img,
                      int W, int H) {
    std::ofstream o(path, std::ios::binary);
    o << "P6\n" << W << " " << H << "\n255\n";
    for (auto& p : img) { o.put(p.r); o.put(p.g); o.put(p.b); }
}

// ── Main ──────────────────────────────────────────────────────
// ── Progress bar ──────────────────────────────────────────────
static void print_progress(int done, int total, double elapsed) {
    const int BAR = 40;
    double frac = (double)done / total;
    int filled = (int)(frac * BAR);
    std::cerr << "\r[";
    for (int i=0;i<BAR;++i) std::cerr << (i<filled ? '#' : '-');
    std::cerr << "] " << (int)(frac*100) << "%";
    if (done > 0 && elapsed > 0.1) {
        double eta = elapsed / frac - elapsed;
        std::cerr << "  " << std::fixed;
        std::cerr.precision(1);
        std::cerr << elapsed << "s elapsed, " << eta << "s ETA";
    }
    std::cerr << "   " << std::flush;
}

int main(int argc, char** argv) {
    bool use_bundles  = false;
    bool preview      = false;
    bool hd_preview   = false;
    bool res_720p     = false;
    bool res_2k       = false;
    bool res_4k       = false;
    int  custom_w     = 0;
    int  custom_h     = 0;
    Integrator intg   = Integrator::RK4_DOUBLING;
    std::string bg_path;
    double arg_a        = 0.998;   // BH spin  (overridable)
    double arg_disk_out = 25.0;    // outer disk radius in M  (overridable)
    double arg_theta    = 80.0;    // camera inclination degrees (overridable)
    double arg_r_obs    = -1.0;    // <0 → use mode default
    for (int i=1;i<argc;++i) {
        std::string arg(argv[i]);
        if (arg=="--bundles")              use_bundles = true;
        if (arg=="--dopri5")               intg = Integrator::DOPRI5;
        if (arg=="--preview")              preview = true;
        if (arg=="--hd")                   hd_preview = true;
        if (arg=="--720p")                 res_720p = true;
        if (arg=="--2k")                   res_2k = true;
        if (arg=="--4k")                   res_4k = true;
        if (arg=="--custom-res" && i+2<argc) {
            custom_w = std::stoi(argv[++i]);
            custom_h = std::stoi(argv[++i]);
        }
        if (arg=="--bg"       && i+1<argc) bg_path     = argv[++i];
        if (arg=="--a"        && i+1<argc) arg_a        = std::stod(argv[++i]);
        if (arg=="--disk-out" && i+1<argc) arg_disk_out = std::stod(argv[++i]);
        if (arg=="--theta"    && i+1<argc) arg_theta    = std::stod(argv[++i]);
        if (arg=="--r-obs"    && i+1<argc) arg_r_obs    = std::stod(argv[++i]);
    }

    // ── Background ────────────────────────────────────────────
    BackgroundImage bg;
    if (!bg_path.empty()) {
        if (bg.load(bg_path.c_str()))
            std::cout << "Background: " << bg_path
                      << " (" << bg.w << "x" << bg.h << ")\n";
        else
            std::cerr << "Warning: could not load background '" << bg_path << "'\n";
    }

    // ── Spacetime ─────────────────────────────────────────────
    // Zero-out Q and/or Lambda to get simpler metrics:
    //   Q=0, Lambda=0  → Kerr
    //   a=0, Lambda=0  → Reissner-Nordström
    //   a=0, Q=0       → Schwarzschild-dS
    const double M_bh  = 1.0;
    const double a_bh  = arg_a;
    const double Q_bh  = 0.0;
    const double Lam   = 0.0;

    KNdSMetric g(M_bh, a_bh, Q_bh, Lam);
    const double rh     = g.r_horizon();
    const double r_isco = g.r_isco();
    std::cout << "KNdS  M=" << M_bh << " a=" << a_bh
              << " Q=" << Q_bh << " Λ=" << Lam << "\n"
              << "  r₊=" << rh << "  r_ISCO=" << r_isco << "\n";

    // ── Camera ────────────────────────────────────────────────
    const int    W      = custom_w   ? custom_w
                        : res_4k     ? 3840
                        : res_2k     ? 2560
                        : res_720p   ? 1280
                        : hd_preview ? 854  : preview ? 480  : 1920;
    const int    H      = custom_h   ? custom_h
                        : res_4k     ? 2160
                        : res_2k     ? 1440
                        : res_720p   ? 720
                        : hd_preview ? 480  : preview ? 270  : 1080;
    const double r_obs  = (arg_r_obs > 0) ? arg_r_obs*M_bh
                        : (res_720p || hd_preview) ? 30.0*M_bh : 500.0*M_bh;
    Camera cam(r_obs, arg_theta, 30.0, W, H);
    const double r_disk_in  = r_isco;
    const double r_disk_out = arg_disk_out * M_bh;
    const double r_escape   = cam.r_obs*1.05;

    std::vector<RGB> image(W*H, {0,0,0});
    const double t0 = get_time();

#if defined(USE_METAL)
    // ── Metal GPU path ─────────────────────────────────────────
    std::cout << "Backend: Metal GPU\n";
    KNdSParams_C kpc{(float)M_bh,(float)a_bh,(float)Q_bh,(float)Lam,
                     (float)rh,(float)r_isco,(float)r_disk_out};
    CameraParams_C cpc{(float)cam.r_obs,(float)cam.theta_obs,(float)cam.fov_h,W,H};
    auto px32 = metal_render(kpc, cpc);
    for (int i=0; i<W*H; ++i) {
        image[i].r = (px32[i])      & 0xFF;
        image[i].g = (px32[i]>>8)   & 0xFF;
        image[i].b = (px32[i]>>16)  & 0xFF;
    }

#elif defined(USE_CUDA)
    // ── CUDA GPU path ─────────────────────────────────────────
    std::cout << "Backend: CUDA GPU\n";
    KNdSParams_CUDA kpcuda{M_bh,a_bh,Q_bh,Lam,rh,r_isco,r_disk_out};
    CameraParams_CUDA cpcuda{cam.r_obs,cam.theta_obs,cam.fov_h,W,H};
    auto px32 = cuda_render(kpcuda, cpcuda);
    for (int i=0; i<W*H; ++i) {
        image[i].r = (px32[i])      & 0xFF;
        image[i].g = (px32[i]>>8)   & 0xFF;
        image[i].b = (px32[i]>>16)  & 0xFF;
    }

#else
    // ── CPU path (OpenMP) ──────────────────────────────────────
    std::cout << "Backend: CPU"
#ifdef _OPENMP
              << " (OpenMP " << omp_get_max_threads() << " threads)"
#endif
              << "\nMode: " << (use_bundles ? "ray bundles" : "single ray")
              << "  integrator: " << (intg==Integrator::DOPRI5 ? "DOPRI5" : "RK4-doubling") << "\n";

    std::atomic<int> rows_done{0};
    #pragma omp parallel for schedule(dynamic, 4)
    for (int py=0; py<H; ++py) {
        for (int px=0; px<W; ++px) {
            RGB col = {0,0,0};
            if (use_bundles) {
                auto res = trace_bundle(px, py, cam, g,
                                        r_disk_in, r_disk_out, r_escape);
                if (res.disk_hit)
                    col = disk_colour(res.r_hit, res.redshift, res.magnif, M_bh, r_isco);
                else if (!res.disk_hit && !bg.px.empty())
                    col = bg.sample(res.theta_esc, res.phi_esc);
            } else {
                auto s   = cam.pixel_ray(px, py, g);
                auto res = trace_single(s, g, r_disk_in, r_disk_out, r_escape, intg);
                if (res.out == Outcome::DISK_HIT)
                    col = disk_colour(res.r, res.redshift, 1.0, M_bh, r_isco);
                else if (res.out == Outcome::ESCAPED && !bg.px.empty())
                    col = bg.sample(res.theta_esc, res.phi_esc);
            }
            image[py*W+px] = col;
        }
        int done = ++rows_done;
        if (done % 4 == 0 || done == H) {
            #pragma omp critical
            print_progress(done, H, get_time()-t0);
        }
    }
    std::cerr << "\n";
#endif

    double elapsed = get_time()-t0;
    std::cout << "Time: " << elapsed << " s  ("
              << std::fixed << std::setprecision(1)
              << (W*H/elapsed/1e3) << " kpix/s)\n";

    // Build unique filename: <tag>_<WxH>_<YYYYMMDD-HHMMSS>.ppm
    const char* tag = res_4k      ? "4k"
                    : res_2k      ? "2k"
                    : custom_w    ? "custom"
                    : res_720p    ? "720p"
                    : hd_preview  ? "hd"
                    : preview     ? "preview"
                    : use_bundles ? "bundles"
                                  : "trace";
    {
        std::time_t now = std::time(nullptr);
        char ts[32];
        std::strftime(ts, sizeof(ts), "%Y%m%d-%H%M%S", std::localtime(&now));
        std::string outfile = std::string(OUT_DIR) + "/" + tag
                            + "_" + std::to_string(W) + "x" + std::to_string(H)
                            + "_" + ts + ".ppm";
        write_ppm(outfile.c_str(), image, W, H);
        std::cout << "Saved: " << outfile << "\n";
    }
    return 0;
}
