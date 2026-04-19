// ============================================================
//  main.cpp — KNdS relativistic ray-tracer
//
//  Backends selected at compile time via CMake options:
//    -DUSE_METAL=ON   → Metal GPU (macOS)
//    -DUSE_CUDA=ON    → CUDA GPU  (Linux/Windows)
//    default          → CPU with OpenMP
//
//  Single frame:
//    ./kerr_tracer [--720p] [--bundles] [--theta 80] ...
//
//  Animation:
//    ./kerr_tracer --anim --frames 120 --fps 30 \
//                  --theta-start 85 --theta-end 5   \
//                  --r-start 100   --r-end 15        \
//                  --orbits 1.0                      \
//                  --output out/flyby.mp4
// ============================================================
#include "camera.hpp"
#include "geodesic.hpp"
#include "knds_metric.hpp"
#include "ray_bundle.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
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
    double  theta_esc = 0.0;
    double  phi_esc   = 0.0;
};

static double disk_redshift(double r, double pt, double pphi,
                            const KNdSMetric& g) {
    double Omega = g.keplerian_omega(r);
    double b     = pphi / pt;
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
    Vec4d  fsal = Vec4d::nan_init();

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

static double tonemap(double x) {
    x = x / (1.0 + x);
    return std::pow(clamp(x, 0.0, 1.0), 1.0/2.2);
}

static double page_thorne(double r, double r_isco) {
    if (r <= r_isco) return 0.0;
    double x = std::sqrt(r_isco / r);
    return (1.0 - x) / (r * r * r);
}

static RGB disk_colour(double r, double red, double magnif,
                       double M, double r_isco) {
    double T = 6500.0 * std::sqrt(6.0*M / r) * clamp(red, 0.2, 5.0);
    double pt_norm = page_thorne(r, r_isco);
    double r_peak  = 3.0 * r_isco;
    double pt_peak = page_thorne(r_peak, r_isco);
    double I = (pt_peak > 0.0) ? pt_norm / pt_peak : 0.0;
    I *= std::pow(clamp(red, 0.1, 10.0), 4.0);
    I *= clamp(1.0 / magnif, 0.05, 5.0);
    auto c = blackbody(T);
    return {(uint8_t)(tonemap(c.r/255.0 * I) * 255),
            (uint8_t)(tonemap(c.g/255.0 * I) * 255),
            (uint8_t)(tonemap(c.b/255.0 * I) * 255)};
}

static void write_png(const char* path, const std::vector<RGB>& img,
                      int W, int H) {
    stbi_write_png(path, W, H, 3,
                   reinterpret_cast<const unsigned char*>(img.data()),
                   W * 3);
}

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

// ── Per-frame variable parameters ────────────────────────────
struct FrameParams {
    double a        = 0.998;  // BH spin
    double theta    = 80.0;   // camera inclination, degrees
    double phi      = 0.0;    // camera azimuth,     degrees
    double r_obs    = 500.0;  // observer radius,    units of M
    double disk_out = 25.0;   // outer disk radius,  units of M
};

// ── Core render function ──────────────────────────────────────
// Renders one frame given variable FrameParams + fixed context.
// Returns pixel buffer (W×H RGB).
static std::vector<RGB> render_image(
    int W, int H,
    const FrameParams& fp,
    const BackgroundImage& bg,
    bool use_bundles,
    Integrator intg,
    double M_bh, double Q_bh, double Lam)
{
    KNdSMetric g(M_bh, fp.a, Q_bh, Lam);
    const double r_isco   = g.r_isco();
    Camera cam(fp.r_obs, fp.theta, fp.phi, 30.0, W, H);
    const double r_disk_in  = r_isco;
    const double r_disk_out = fp.disk_out;
    const double r_escape   = cam.r_obs * 1.05;

    std::vector<RGB> image(W*H, {0,0,0});

#if defined(USE_METAL)
    KNdSParams_C kpc{(float)M_bh,(float)fp.a,(float)Q_bh,(float)Lam,
                     (float)g.r_horizon(),(float)r_isco,(float)r_disk_out};
    CameraParams_C cpc{(float)cam.r_obs,(float)cam.theta_obs,(float)cam.fov_h,W,H};
    auto px32 = metal_render(kpc, cpc);
    for (int i=0; i<W*H; ++i) {
        image[i].r = (px32[i])      & 0xFF;
        image[i].g = (px32[i]>>8)   & 0xFF;
        image[i].b = (px32[i]>>16)  & 0xFF;
    }

#elif defined(USE_CUDA)
    KNdSParams_CUDA kpcuda{M_bh,fp.a,Q_bh,Lam,g.r_horizon(),r_isco,r_disk_out};
    CameraParams_CUDA cpcuda{cam.r_obs,cam.theta_obs,cam.fov_h,W,H};
    auto px32 = cuda_render(kpcuda, cpcuda);
    for (int i=0; i<W*H; ++i) {
        image[i].r = (px32[i])      & 0xFF;
        image[i].g = (px32[i]>>8)   & 0xFF;
        image[i].b = (px32[i]>>16)  & 0xFF;
    }

#else
    std::atomic<int> rows_done{0};
    const double t0_render = get_time();
    #pragma omp parallel for schedule(dynamic, 4)
    for (int py=0; py<H; ++py) {
        for (int px=0; px<W; ++px) {
            RGB col = {0,0,0};
            if (use_bundles) {
                auto res = trace_bundle(px, py, cam, g,
                                        r_disk_in, r_disk_out, r_escape);
                if (res.disk_hit)
                    col = disk_colour(res.r_hit, res.redshift, res.magnif, M_bh, r_isco);
                else if (!bg.px.empty())
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
            print_progress(done, H, get_time()-t0_render);
        }
    }
    std::cerr << "\n";
#endif

    return image;
}

// ── Interpolation helpers ─────────────────────────────────────
static double smooth_step(double t) { return t*t*(3.0-2.0*t); }

static double lerp_angle(double a, double b, double t) {
    // Always take the short path
    double d = b - a;
    while (d >  180.0) d -= 360.0;
    while (d < -180.0) d += 360.0;
    return a + d*t;
}

// ── main ──────────────────────────────────────────────────────
int main(int argc, char** argv) {
    // ── Resolution flags ──────────────────────────────────────
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

    // ── Single-frame params ───────────────────────────────────
    double arg_a        = 0.998;
    double arg_disk_out = 25.0;
    double arg_theta    = 80.0;
    double arg_phi      = 0.0;
    double arg_r_obs    = -1.0;   // <0 → use mode default

    // ── Animation params ──────────────────────────────────────
    bool        anim_mode        = false;
    int         anim_frames      = 60;
    int         anim_fps         = 30;
    int         anim_crf         = 18;
    bool        anim_ease        = false;
    bool        anim_keep_frames = false;
    bool        anim_resume      = false;
    bool        anim_no_encode   = false;
    double      anim_orbits      = 0.0;  // phi = phi_start + 360*orbits*phase
    std::string anim_output;             // empty → auto
    std::string anim_frames_dir;         // empty → auto (out/anim_<tag>)

    // Sweep start/end (NaN = no sweep → use single-frame value)
    double anim_theta_start   = std::numeric_limits<double>::quiet_NaN();
    double anim_theta_end     = std::numeric_limits<double>::quiet_NaN();
    double anim_phi_start     = std::numeric_limits<double>::quiet_NaN();
    double anim_phi_end       = std::numeric_limits<double>::quiet_NaN();
    double anim_r_start       = std::numeric_limits<double>::quiet_NaN();
    double anim_r_end         = std::numeric_limits<double>::quiet_NaN();
    double anim_a_start       = std::numeric_limits<double>::quiet_NaN();
    double anim_a_end         = std::numeric_limits<double>::quiet_NaN();
    double anim_disk_out_start= std::numeric_limits<double>::quiet_NaN();
    double anim_disk_out_end  = std::numeric_limits<double>::quiet_NaN();

    for (int i=1;i<argc;++i) {
        std::string arg(argv[i]);

        // Resolution
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

        // Single-frame
        if (arg=="--bg"       && i+1<argc) bg_path     = argv[++i];
        if (arg=="--a"        && i+1<argc) arg_a        = std::stod(argv[++i]);
        if (arg=="--disk-out" && i+1<argc) arg_disk_out = std::stod(argv[++i]);
        if (arg=="--theta"    && i+1<argc) arg_theta    = std::stod(argv[++i]);
        if (arg=="--phi"      && i+1<argc) arg_phi      = std::stod(argv[++i]);
        if (arg=="--r-obs"    && i+1<argc) arg_r_obs    = std::stod(argv[++i]);

        // Animation
        if (arg=="--anim")                   anim_mode   = true;
        if (arg=="--frames"   && i+1<argc)   anim_frames = std::stoi(argv[++i]);
        if (arg=="--fps"      && i+1<argc)   anim_fps    = std::stoi(argv[++i]);
        if (arg=="--crf"      && i+1<argc)   anim_crf    = std::stoi(argv[++i]);
        if (arg=="--ease")                   anim_ease   = true;
        if (arg=="--keep-frames")            anim_keep_frames = true;
        if (arg=="--resume")                 anim_resume = true;
        if (arg=="--no-encode")              anim_no_encode = true;
        if (arg=="--orbits"   && i+1<argc)   anim_orbits = std::stod(argv[++i]);
        if (arg=="--output"   && i+1<argc)   anim_output = argv[++i];
        if (arg=="--frames-dir" && i+1<argc) anim_frames_dir = argv[++i];

        // Sweep params
        if (arg=="--theta-start"    && i+1<argc) anim_theta_start    = std::stod(argv[++i]);
        if (arg=="--theta-end"      && i+1<argc) anim_theta_end      = std::stod(argv[++i]);
        if (arg=="--phi-start"      && i+1<argc) anim_phi_start      = std::stod(argv[++i]);
        if (arg=="--phi-end"        && i+1<argc) anim_phi_end        = std::stod(argv[++i]);
        if (arg=="--r-start"        && i+1<argc) anim_r_start        = std::stod(argv[++i]);
        if (arg=="--r-end"          && i+1<argc) anim_r_end          = std::stod(argv[++i]);
        if (arg=="--a-start"        && i+1<argc) anim_a_start        = std::stod(argv[++i]);
        if (arg=="--a-end"          && i+1<argc) anim_a_end          = std::stod(argv[++i]);
        if (arg=="--disk-out-start" && i+1<argc) anim_disk_out_start = std::stod(argv[++i]);
        if (arg=="--disk-out-end"   && i+1<argc) anim_disk_out_end   = std::stod(argv[++i]);
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

    // ── Derived constants ─────────────────────────────────────
    const double M_bh  = 1.0;
    const double Q_bh  = 0.0;
    const double Lam   = 0.0;

    const int W = custom_w   ? custom_w
                : res_4k     ? 3840
                : res_2k     ? 2560
                : res_720p   ? 1280
                : hd_preview ? 854  : preview ? 480  : 1920;
    const int H = custom_h   ? custom_h
                : res_4k     ? 2160
                : res_2k     ? 1440
                : res_720p   ? 720
                : hd_preview ? 480  : preview ? 270  : 1080;

    // Default r_obs depends on resolution mode
    const double default_r_obs = (res_720p || hd_preview) ? 30.0 : 500.0;
    const double base_r_obs    = (arg_r_obs > 0) ? arg_r_obs : default_r_obs;

    // ── Resolution tag ────────────────────────────────────────
    const char* res_tag = res_4k      ? "4k"
                        : res_2k      ? "2k"
                        : custom_w    ? "custom"
                        : res_720p    ? "720p"
                        : hd_preview  ? "hd"
                        : preview     ? "preview"
                        : use_bundles ? "bundles"
                                      : "trace";

    // ── SINGLE FRAME mode ─────────────────────────────────────
    if (!anim_mode) {
        FrameParams fp;
        fp.a        = arg_a;
        fp.theta    = arg_theta;
        fp.phi      = arg_phi;
        fp.r_obs    = base_r_obs * M_bh;
        fp.disk_out = arg_disk_out * M_bh;

        // Print info
        KNdSMetric g_info(M_bh, fp.a, Q_bh, Lam);
        std::cout << "KNdS  M=" << M_bh << " a=" << fp.a
                  << " Q=" << Q_bh << " Λ=" << Lam << "\n"
                  << "  r₊=" << g_info.r_horizon()
                  << "  r_ISCO=" << g_info.r_isco() << "\n"
                  << "Backend: CPU"
#ifdef _OPENMP
                  << " (OpenMP " << omp_get_max_threads() << " threads)"
#endif
                  << "\nMode: " << (use_bundles ? "ray bundles" : "single ray")
                  << "  integrator: "
                  << (intg==Integrator::DOPRI5 ? "DOPRI5" : "RK4-doubling") << "\n"
                  << "Resolution: " << W << "x" << H << "\n";

        double t0 = get_time();
        auto image = render_image(W, H, fp, bg, use_bundles, intg, M_bh, Q_bh, Lam);
        double elapsed = get_time()-t0;

        std::cout << "Time: " << elapsed << " s  ("
                  << std::fixed << std::setprecision(1)
                  << (W*H/elapsed/1e3) << " kpix/s)\n";

        std::time_t now = std::time(nullptr);
        char ts[32];
        std::strftime(ts, sizeof(ts), "%Y%m%d-%H%M%S", std::localtime(&now));
        std::string outfile = std::string(OUT_DIR) + "/" + res_tag
                            + "_" + std::to_string(W) + "x" + std::to_string(H)
                            + "_" + ts + ".png";
        write_png(outfile.c_str(), image, W, H);
        std::cout << "Saved: " << outfile << "\n";
        return 0;
    }

    // ── ANIMATION mode ────────────────────────────────────────
    // Resolve sweep start/end values (NaN → use single-frame default)
    auto resolve = [](double val, double fallback) {
        return std::isnan(val) ? fallback : val;
    };

    double theta_start    = resolve(anim_theta_start,    arg_theta);
    double theta_end      = resolve(anim_theta_end,      arg_theta);
    double phi_start      = resolve(anim_phi_start,      arg_phi);
    double phi_end        = resolve(anim_phi_end,        arg_phi);
    double r_start        = resolve(anim_r_start,        base_r_obs);
    double r_end          = resolve(anim_r_end,          base_r_obs);
    double a_start        = resolve(anim_a_start,        arg_a);
    double a_end          = resolve(anim_a_end,          arg_a);
    double disk_out_start = resolve(anim_disk_out_start, arg_disk_out);
    double disk_out_end   = resolve(anim_disk_out_end,   arg_disk_out);

    // Build auto tag from active sweeps
    std::ostringstream tag_ss;
    tag_ss << res_tag;
    if (anim_orbits != 0.0)
        tag_ss << "_orbit" << anim_orbits << "x";
    else if (phi_start != phi_end)
        tag_ss << "_phi" << (int)phi_start << "-" << (int)phi_end;
    if (theta_start != theta_end)
        tag_ss << "_th" << (int)theta_start << "-" << (int)theta_end;
    if (r_start != r_end)
        tag_ss << "_r" << (int)r_start << "-" << (int)r_end;
    if (a_start != a_end)
        tag_ss << "_a" << a_start << "-" << a_end;
    tag_ss << "_" << anim_frames << "f" << anim_fps << "fps";
    std::string anim_tag = tag_ss.str();

    // Frames directory
    std::string frames_dir = anim_frames_dir.empty()
        ? std::string(OUT_DIR) + "/anim_" + anim_tag
        : anim_frames_dir;

    // Create frames dir
    {
        std::string cmd = "mkdir -p \"" + frames_dir + "\"";
        std::system(cmd.c_str());
    }

    // Output file
    std::string output_file = anim_output.empty()
        ? std::string(OUT_DIR) + "/" + anim_tag + ".mp4"
        : anim_output;

    std::cout << "Animation: " << anim_frames << " frames @ " << anim_fps << " fps\n"
              << "  theta:  " << theta_start << "° → " << theta_end << "°\n";
    if (anim_orbits != 0.0)
        std::cout << "  phi:    " << phi_start << "° + " << anim_orbits << " orbit(s)\n";
    else
        std::cout << "  phi:    " << phi_start << "° → " << phi_end << "°\n";
    std::cout << "  r_obs:  " << r_start << " M → " << r_end << " M\n"
              << "  a:      " << a_start << " → " << a_end << "\n"
              << "  output: " << output_file << "\n"
              << "  frames: " << frames_dir << "\n"
              << "  res:    " << W << "x" << H << "\n"
              << "  ease:   " << (anim_ease ? "yes (smooth-step)" : "no (linear)") << "\n\n";

    double t_total = get_time();
    int rendered = 0;
    int skipped  = 0;

    for (int frame = 0; frame < anim_frames; ++frame) {
        // Frame file path
        char fname[512];
        std::snprintf(fname, sizeof(fname),
                      "%s/frame_%05d.png", frames_dir.c_str(), frame);

        // Resume: skip existing frames
        if (anim_resume) {
            std::ifstream test(fname);
            if (test.good()) { ++skipped; continue; }
        }

        // Normalised phase [0,1]
        double phase = (anim_frames == 1) ? 0.0
                     : (double)frame / (anim_frames - 1);
        double t     = anim_ease ? smooth_step(phase) : phase;

        // Interpolate per-frame params
        FrameParams fp;
        fp.a        = a_start        + (a_end        - a_start)        * t;
        fp.theta    = theta_start    + (theta_end    - theta_start)    * t;
        fp.r_obs    = (r_start       + (r_end        - r_start)        * t) * M_bh;
        fp.disk_out = (disk_out_start+ (disk_out_end - disk_out_start) * t) * M_bh;

        // Phi: orbits override linear lerp
        if (anim_orbits != 0.0)
            fp.phi = phi_start + 360.0 * anim_orbits * phase;
        else
            fp.phi = lerp_angle(phi_start, phi_end, t);

        double t_frame = get_time();
        std::cout << "Frame " << (frame+1) << "/" << anim_frames
                  << "  θ=" << std::fixed << std::setprecision(1) << fp.theta
                  << "°  φ=" << fp.phi << "°"
                  << "  r=" << std::setprecision(2) << fp.r_obs
                  << "  a=" << std::setprecision(4) << fp.a
                  << " ...\n" << std::flush;

        auto image = render_image(W, H, fp, bg, use_bundles, intg, M_bh, Q_bh, Lam);
        write_png(fname, image, W, H);

        double dt = get_time() - t_frame;
        double eta_total = (rendered > 0)
            ? (get_time()-t_total) / rendered * (anim_frames - frame - 1)
            : 0.0;
        ++rendered;
        std::cout << "  → saved " << fname
                  << "  (" << std::setprecision(1) << dt << "s"
                  << "  ETA " << (int)(eta_total/60) << "m"
                  << (int)(std::fmod(eta_total,60)) << "s)\n";
    }

    double render_elapsed = get_time() - t_total;
    std::cout << "\nFrames done: " << rendered << " rendered";
    if (skipped) std::cout << ", " << skipped << " skipped";
    std::cout << "  (" << render_elapsed << "s total)\n";

    // ── ffmpeg encode ─────────────────────────────────────────
    if (!anim_no_encode) {
        char ffcmd[2048];
        std::snprintf(ffcmd, sizeof(ffcmd),
            "ffmpeg -y -framerate %d"
            " -i \"%s/frame_%%05d.png\""
            " -c:v libx264 -pix_fmt yuv420p"
            " -crf %d -movflags +faststart"
            " \"%s\" 2>&1",
            anim_fps, frames_dir.c_str(),
            anim_crf, output_file.c_str());

        std::cout << "Encoding: " << ffcmd << "\n";
        int ret = std::system(ffcmd);
        if (ret == 0)
            std::cout << "Video saved: " << output_file << "\n";
        else
            std::cerr << "ffmpeg failed (exit " << ret
                      << "). Frames preserved in: " << frames_dir << "\n";
    }

    // Cleanup frames dir unless --keep-frames or encode failed
    if (!anim_keep_frames && !anim_no_encode) {
        std::string rmcmd = "rm -rf \"" + frames_dir + "\"";
        std::system(rmcmd.c_str());
    }

    return 0;
}
