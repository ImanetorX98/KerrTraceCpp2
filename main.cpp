// ============================================================
//  main.cpp — KNdS relativistic ray-tracer
//
//  Pipeline (two-phase separation):
//    Phase 1 – Geodesic tracing  → GeoPixel buffer (+ optional .kgeo file)
//    Phase 2 – Colorization      → RGB image (loads GeoPixel buffer)
//
//  Modes:
//    Single frame    ./kerr_tracer [--720p] [--theta 80] ...
//    Geo only        ./kerr_tracer --geo-only [--geo-file out/foo.kgeo]
//    Color only      ./kerr_tracer --color-only out/foo.kgeo [--exposure 1.5]
//    Animation       ./kerr_tracer --anim --frames 120 --orbits 1 --720p
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
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

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

template<class T> static T clamp(T v, T lo, T hi) {
    return v < lo ? lo : v > hi ? hi : v;
}

// ── Colour ────────────────────────────────────────────────────
struct RGB { uint8_t r, g, b; };

// ── Colorization parameters (Phase 2 controls) ───────────────
struct ColorParams {
    double exposure    = 1.0;   // intensity multiplier before tonemapping
    double gamma       = 2.2;   // gamma correction exponent
    double temp_scale  = 1.0;   // disk blackbody temperature scale
};

// ── Per-pixel geodesic result (Phase 1 output) ───────────────
struct GeoPixel {
    uint8_t outcome;    // 0 = escaped, 1 = disk_hit, 2 = horizon
    uint8_t _pad[3];
    float   r;          // BL radius at disk crossing (or final r)
    float   redshift;   // g = ν_obs/ν_em
    float   magnif;     // flux magnification (bundle mode; 1 in single-ray)
    float   theta_esc;  // direction at escape (background lookup)
    float   phi_esc;
};
static_assert(sizeof(GeoPixel) == 24, "GeoPixel size mismatch");

// ── .kgeo file format ─────────────────────────────────────────
static const char   KGEO_MAGIC[4]  = {'K','G','E','O'};
static const uint32_t KGEO_VERSION = 1;

struct KGeoMeta {
    uint32_t W, H;
    double   M_bh, a_bh, Q_bh, Lam;
    double   r_isco, r_disk_in, r_disk_out;
    double   theta_obs, phi_obs, r_obs;
};

static void save_kgeo(const char* path,
                      const std::vector<GeoPixel>& geo,
                      const KGeoMeta& meta) {
    std::ofstream f(path, std::ios::binary);
    f.write(KGEO_MAGIC, 4);
    uint32_t ver = KGEO_VERSION;
    f.write(reinterpret_cast<const char*>(&ver), 4);
    f.write(reinterpret_cast<const char*>(&meta), sizeof(meta));
    f.write(reinterpret_cast<const char*>(geo.data()),
            (std::streamsize)(geo.size() * sizeof(GeoPixel)));
}

static bool load_kgeo(const char* path,
                      std::vector<GeoPixel>& geo,
                      KGeoMeta& meta) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[4]; f.read(magic, 4);
    if (std::memcmp(magic, KGEO_MAGIC, 4) != 0) {
        std::cerr << "load_kgeo: bad magic in " << path << "\n"; return false;
    }
    uint32_t ver; f.read(reinterpret_cast<char*>(&ver), 4);
    if (ver != KGEO_VERSION) {
        std::cerr << "load_kgeo: unsupported version " << ver << "\n"; return false;
    }
    f.read(reinterpret_cast<char*>(&meta), sizeof(meta));
    geo.resize(meta.W * meta.H);
    f.read(reinterpret_cast<char*>(geo.data()),
           (std::streamsize)(geo.size() * sizeof(GeoPixel)));
    return f.good();
}

// ── Background HDRI ───────────────────────────────────────────
struct BackgroundImage {
    int w = 0, h = 0;
    std::vector<uint8_t> px;

    bool load(const char* path) {
        int c;
        uint8_t* data = stbi_load(path, &w, &h, &c, 3);
        if (!data) return false;
        px.assign(data, data + (size_t)w*h*3);
        stbi_image_free(data);
        return true;
    }

    RGB sample(double theta, double phi) const {
        if (px.empty()) return {0,0,0};
        double pw = std::fmod(phi, 2.0*M_PI);
        if (pw < 0) pw += 2.0*M_PI;
        double uf = pw  / (2.0*M_PI) * (w-1);
        double vf = theta / M_PI     * (h-1);
        int u0=(int)uf, u1=std::min(u0+1,w-1);
        int v0=(int)vf, v1=std::min(v0+1,h-1);
        double fu=uf-u0, fv=vf-v0;
        auto at=[&](int u,int v)->const uint8_t*{
            return px.data()+3*(clamp(v,0,h-1)*w+clamp(u,0,w-1));
        };
        auto lp=[](double a,double b,double t){return a+t*(b-a);};
        double R=lp(lp(at(u0,v0)[0],at(u1,v0)[0],fu),lp(at(u0,v1)[0],at(u1,v1)[0],fu),fv);
        double G=lp(lp(at(u0,v0)[1],at(u1,v0)[1],fu),lp(at(u0,v1)[1],at(u1,v1)[1],fu),fv);
        double B=lp(lp(at(u0,v0)[2],at(u1,v0)[2],fu),lp(at(u0,v1)[2],at(u1,v1)[2],fu),fv);
        return {(uint8_t)R,(uint8_t)G,(uint8_t)B};
    }
};

// ── Geodesic trace helpers ────────────────────────────────────
enum class Outcome { ESCAPED, DISK_HIT, HORIZON };
struct TraceResult {
    Outcome out; double r, redshift;
    double theta_esc=0.0, phi_esc=0.0;
};

static double disk_redshift(double r, double pt, double pphi, const KNdSMetric& g) {
    double Omega = g.keplerian_omega(r);
    double b     = pphi / (-pt);
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
                                Integrator intg=Integrator::RK4_DOUBLING) {
    double rh=g.r_horizon(), dlam=1.0, prev_cos=std::cos(s.theta);
    Vec4d fsal=Vec4d::nan_init();
    for (int it=0; it<500000; ++it) {
        const GeodesicState s_prev = s;
        int rejects = 0;
        while (!adaptive_step(g, s, dlam, intg, fsal)) {
            if (!std::isfinite(dlam) || ++rejects > 64)
                return {Outcome::ESCAPED, s.r, 1.0, s.theta, s.phi};
        }
        if (s.r < rh*1.03)  return {Outcome::HORIZON,  s.r, 0.0};
        if (s.r > r_escape)  return {Outcome::ESCAPED,  s.r, 1.0, s.theta, s.phi};
        double cc=std::cos(s.theta);
        if (prev_cos*cc<=0.0) {
            const double denom = prev_cos - cc;
            double w = (std::abs(denom) > 1e-14) ? (prev_cos / denom) : 0.5;
            w = clamp(w, 0.0, 1.0);
            const double r_hit = s_prev.r + w*(s.r - s_prev.r);
            if (r_hit>=r_disk_in && r_hit<=r_disk_out)
                return {Outcome::DISK_HIT, r_hit,
                        clamp(disk_redshift(r_hit, s.pt, s.pphi, g), 0.0, 20.0)};
        }
        prev_cos=cc;
    }
    return {Outcome::ESCAPED, s.r, 1.0, s.theta, s.phi};
}

// ── Colorization (Phase 2) ────────────────────────────────────
static RGB blackbody(double T) {
    T=clamp(T,800.0,4e4);
    double t=std::log10(T/800.0)/std::log10(4e4/800.0);
    double R,G,B;
    if(t<0.25){R=1;G=t/0.25*0.4;B=0;}
    else if(t<0.5){double f=(t-0.25)/0.25;R=1;G=0.4+f*0.4;B=f*0.3;}
    else if(t<0.75){double f=(t-0.5)/0.25;R=1;G=0.8+f*0.2;B=0.3+f*0.5;}
    else{double f=(t-0.75)/0.25;R=1-f*0.2;G=1;B=0.8+f*0.2;}
    return {(uint8_t)(clamp(R,0.0,1.0)*255),
            (uint8_t)(clamp(G,0.0,1.0)*255),
            (uint8_t)(clamp(B,0.0,1.0)*255)};
}

static double page_thorne(double r, double r_isco) {
    if (r <= r_isco) return 0.0;
    double x=std::sqrt(r_isco/r);
    return (1.0-x)/(r*r*r);
}

// Reinhard tonemapping with variable gamma
static double tonemap(double x, double exposure, double gamma) {
    x = x * exposure;
    x = x / (1.0 + x);
    return std::pow(clamp(x, 0.0, 1.0), 1.0/gamma);
}

static RGB disk_colour(double r, double red, double magnif,
                       double M, double r_isco, const ColorParams& cp) {
    double T = 6500.0 * cp.temp_scale * std::sqrt(6.0*M/r) * clamp(red, 0.2, 5.0);
    double pt_norm = page_thorne(r, r_isco);
    double r_peak  = 3.0*r_isco;
    double pt_peak = page_thorne(r_peak, r_isco);
    double I = (pt_peak > 0.0) ? pt_norm/pt_peak : 0.0;
    I *= std::pow(clamp(red, 0.1, 10.0), 4.0);
    I *= clamp(1.0/magnif, 0.05, 5.0);
    auto c = blackbody(T);
    return {(uint8_t)(tonemap(c.r/255.0*I, cp.exposure, cp.gamma)*255),
            (uint8_t)(tonemap(c.g/255.0*I, cp.exposure, cp.gamma)*255),
            (uint8_t)(tonemap(c.b/255.0*I, cp.exposure, cp.gamma)*255)};
}

// Phase 2: GeoPixel buffer → RGB image
static std::vector<RGB> colorize_buffer(
    const std::vector<GeoPixel>& geo, int W, int H,
    const ColorParams& cp,
    const BackgroundImage& bg,
    double M_bh, double r_isco)
{
    std::vector<RGB> image(W*H, {0,0,0});
    for (int i = 0; i < W*H; ++i) {
        const GeoPixel& p = geo[i];
        if (p.outcome == 1)
            image[i] = disk_colour(p.r, p.redshift, p.magnif, M_bh, r_isco, cp);
        else if (p.outcome == 0 && !bg.px.empty())
            image[i] = bg.sample(p.theta_esc, p.phi_esc);
    }
    return image;
}

// ── Progress bar ──────────────────────────────────────────────
static void print_progress(int done, int total, double elapsed) {
    const int BAR=40;
    double frac=(double)done/total;
    int filled=(int)(frac*BAR);
    std::cerr<<"\r[";
    for(int i=0;i<BAR;++i) std::cerr<<(i<filled?'#':'-');
    std::cerr<<"] "<<(int)(frac*100)<<"%";
    if(done>0&&elapsed>0.1){
        double eta=elapsed/frac-elapsed;
        std::cerr<<"  "<<std::fixed;
        std::cerr.precision(1);
        std::cerr<<elapsed<<"s elapsed, "<<eta<<"s ETA";
    }
    std::cerr<<"   "<<std::flush;
}

// ── Per-frame camera/physics parameters ───────────────────────
struct FrameParams {
    double a=0.998, theta=80.0, phi=0.0, r_obs=500.0, disk_out=25.0, fov=30.0;
};

// ── Phase 1: trace all geodesics → GeoPixel buffer ───────────
static std::vector<GeoPixel> trace_geodesics(
    int W, int H,
    const FrameParams& fp,
    bool use_bundles,
    Integrator intg,
    double M_bh, double Q_bh, double Lam,
    KGeoMeta* meta_out = nullptr)
{
    KNdSMetric g(M_bh, fp.a, Q_bh, Lam);
    const double r_isco   = g.r_isco();
    Camera cam(fp.r_obs, fp.theta, fp.phi, fp.fov, W, H);
    const double r_disk_in  = r_isco;
    const double r_disk_out = fp.disk_out;
    const double r_escape   = cam.r_obs * 1.05;

    if (meta_out) {
        *meta_out = KGeoMeta{
            (uint32_t)W, (uint32_t)H,
            M_bh, fp.a, Q_bh, Lam,
            r_isco, r_disk_in, r_disk_out,
            fp.theta, fp.phi, fp.r_obs
        };
    }

    std::vector<GeoPixel> geo(W*H);
    std::atomic<int> rows_done{0};
    const double t0_geo = get_time();

    #pragma omp parallel for schedule(dynamic, 4)
    for (int py=0; py<H; ++py) {
        for (int px_=0; px_<W; ++px_) {
            GeoPixel& pix = geo[py*W+px_];
            if (use_bundles) {
                auto res = trace_bundle(px_, py, cam, g, r_disk_in, r_disk_out, r_escape);
                pix.outcome   = res.disk_hit ? 1 : 0;
                pix.r         = res.disk_hit ? (float)res.r_hit : 0.0f;
                pix.redshift  = (float)res.redshift;
                pix.magnif    = (float)res.magnif;
                pix.theta_esc = (float)res.theta_esc;
                pix.phi_esc   = (float)res.phi_esc;
            } else {
                auto s   = cam.pixel_ray(px_, py, g);
                auto res = trace_single(s, g, r_disk_in, r_disk_out, r_escape, intg);
                pix.outcome   = (res.out==Outcome::DISK_HIT) ? 1
                              : (res.out==Outcome::HORIZON)   ? 2 : 0;
                pix.r         = (float)res.r;
                pix.redshift  = (float)res.redshift;
                pix.magnif    = 1.0f;
                pix.theta_esc = (float)res.theta_esc;
                pix.phi_esc   = (float)res.phi_esc;
            }
            pix._pad[0]=pix._pad[1]=pix._pad[2]=0;
        }
        int done = ++rows_done;
        if (done%4==0 || done==H) {
            #pragma omp critical
            print_progress(done, H, get_time()-t0_geo);
        }
    }
    std::cerr << "\n";
    return geo;
}

// ── Full render (trace + colorize, for animation / single frame) ──
static std::vector<RGB> render_image(
    int W, int H,
    const FrameParams& fp,
    const BackgroundImage& bg,
    bool use_bundles,
    Integrator intg,
    double M_bh, double Q_bh, double Lam,
    const ColorParams& cp = ColorParams{},
    std::vector<GeoPixel>* geo_out = nullptr)
{
#if defined(USE_METAL)
    // Metal path: no geo separation yet, direct render
    KNdSMetric g(M_bh, fp.a, Q_bh, Lam);
    const double r_isco = g.r_isco();
    Camera cam(fp.r_obs, fp.theta, fp.phi, fp.fov, W, H);
    KNdSParams_C kpc{(float)M_bh,(float)fp.a,(float)Q_bh,(float)Lam,
                     (float)g.r_horizon(),(float)r_isco,(float)fp.disk_out};
    CameraParams_C cpc{(float)cam.r_obs,(float)cam.theta_obs,(float)cam.fov_h,W,H};
    auto px32 = metal_render(kpc, cpc);
    std::vector<RGB> image(W*H);
    for (int i=0;i<W*H;++i) {
        image[i].r=(px32[i])    &0xFF;
        image[i].g=(px32[i]>>8) &0xFF;
        image[i].b=(px32[i]>>16)&0xFF;
    }
    return image;

#elif defined(USE_CUDA)
    KNdSMetric g(M_bh, fp.a, Q_bh, Lam);
    const double r_isco = g.r_isco();
    Camera cam(fp.r_obs, fp.theta, fp.phi, fp.fov, W, H);
    KNdSParams_CUDA kpcuda{M_bh,fp.a,Q_bh,Lam,g.r_horizon(),r_isco,fp.disk_out};
    CameraParams_CUDA cpcuda{cam.r_obs,cam.theta_obs,cam.fov_h,W,H};
    auto px32 = cuda_render(kpcuda, cpcuda);
    std::vector<RGB> image(W*H);
    for (int i=0;i<W*H;++i) {
        image[i].r=(px32[i])    &0xFF;
        image[i].g=(px32[i]>>8) &0xFF;
        image[i].b=(px32[i]>>16)&0xFF;
    }
    return image;

#else
    // CPU: two-phase
    KGeoMeta meta;
    auto geo = trace_geodesics(W, H, fp, use_bundles, intg, M_bh, Q_bh, Lam, &meta);
    if (geo_out) *geo_out = geo;
    return colorize_buffer(geo, W, H, cp, bg, M_bh, meta.r_isco);
#endif
}

// ── File I/O ──────────────────────────────────────────────────
static void write_png(const char* path, const std::vector<RGB>& img, int W, int H) {
    stbi_write_png(path, W, H, 3,
                   reinterpret_cast<const unsigned char*>(img.data()), W*3);
}

// ── Interpolation helpers (animation) ────────────────────────
static double smooth_step(double t) { return t*t*(3.0-2.0*t); }
static double lerp_angle(double a, double b, double t) {
    double d=b-a;
    while(d> 180.0) d-=360.0;
    while(d<-180.0) d+=360.0;
    return a+d*t;
}

// ── main ──────────────────────────────────────────────────────
int main(int argc, char** argv) {

    // ── Resolution ───────────────────────────────────────────
    bool use_bundles=false, preview=false, hd_preview=false;
    bool res_720p=false, res_2k=false, res_4k=false;
    int  custom_w=0, custom_h=0;
    Integrator intg=Integrator::RK4_DOUBLING;
    std::string bg_path;

    // ── Single-frame / base params ───────────────────────────
    double arg_a=0.998, arg_disk_out=25.0, arg_theta=80.0, arg_phi=0.0, arg_r_obs=-1.0;
    double arg_Q=0.0, arg_Lam=0.0, arg_fov=30.0;

    // ── Colorization params ───────────────────────────────────
    ColorParams cp;  // defaults: exposure=1, gamma=2.2, temp_scale=1

    // ── Two-phase modes ───────────────────────────────────────
    bool        geo_only    = false;
    std::string geo_file;         // path for .kgeo output (geo_only) or input (color_only)
    bool        color_only  = false;

    // ── Animation params ──────────────────────────────────────
    bool anim_mode=false, anim_ease=false, anim_keep_frames=false;
    bool anim_resume=false, anim_no_encode=false;
    int  anim_frames=60, anim_fps=30, anim_crf=18;
    double anim_orbits=0.0;
    std::string anim_output, anim_frames_dir;

    double NaN = std::numeric_limits<double>::quiet_NaN();
    double anim_theta_start=NaN, anim_theta_end=NaN;
    double anim_phi_start=NaN,   anim_phi_end=NaN;
    double anim_r_start=NaN,     anim_r_end=NaN;
    double anim_a_start=NaN,     anim_a_end=NaN;
    double anim_disk_out_start=NaN, anim_disk_out_end=NaN;

    for (int i=1;i<argc;++i) {
        std::string arg(argv[i]);

        // Resolution
        if (arg=="--bundles")  use_bundles=true;
        if (arg=="--dopri5")   intg=Integrator::DOPRI5;
        if (arg=="--preview")  preview=true;
        if (arg=="--hd")       hd_preview=true;
        if (arg=="--720p")     res_720p=true;
        if (arg=="--2k")       res_2k=true;
        if (arg=="--4k")       res_4k=true;
        if (arg=="--custom-res" && i+2<argc) {
            custom_w=std::stoi(argv[++i]); custom_h=std::stoi(argv[++i]);
        }

        // Single-frame physics
        if (arg=="--bg"       && i+1<argc) bg_path     = argv[++i];
        if (arg=="--a"        && i+1<argc) arg_a        = std::stod(argv[++i]);
        if (arg=="--disk-out" && i+1<argc) arg_disk_out = std::stod(argv[++i]);
        if (arg=="--theta"    && i+1<argc) arg_theta    = std::stod(argv[++i]);
        if (arg=="--phi"      && i+1<argc) arg_phi      = std::stod(argv[++i]);
        if (arg=="--r-obs"    && i+1<argc) arg_r_obs    = std::stod(argv[++i]);
        if (arg=="--charge"   && i+1<argc) arg_Q        = std::stod(argv[++i]);
        if (arg=="--lambda"   && i+1<argc) arg_Lam      = std::stod(argv[++i]);
        if (arg=="--fov"      && i+1<argc) arg_fov      = std::stod(argv[++i]);

        // Colorization
        if (arg=="--exposure"   && i+1<argc) cp.exposure   = std::stod(argv[++i]);
        if (arg=="--gamma"      && i+1<argc) cp.gamma      = std::stod(argv[++i]);
        if (arg=="--temp-scale" && i+1<argc) cp.temp_scale = std::stod(argv[++i]);

        // Two-phase modes
        if (arg=="--geo-only")             geo_only  = true;
        if (arg=="--geo-file" && i+1<argc) geo_file  = argv[++i];
        if (arg=="--color-only"&& i+1<argc){ color_only=true; geo_file=argv[++i]; }

        // Animation
        if (arg=="--anim")                   anim_mode=true;
        if (arg=="--frames"   && i+1<argc)   anim_frames=std::stoi(argv[++i]);
        if (arg=="--fps"      && i+1<argc)   anim_fps=std::stoi(argv[++i]);
        if (arg=="--crf"      && i+1<argc)   anim_crf=std::stoi(argv[++i]);
        if (arg=="--ease")                   anim_ease=true;
        if (arg=="--keep-frames")            anim_keep_frames=true;
        if (arg=="--resume")                 anim_resume=true;
        if (arg=="--no-encode")              anim_no_encode=true;
        if (arg=="--orbits"   && i+1<argc)   anim_orbits=std::stod(argv[++i]);
        if (arg=="--output"   && i+1<argc)   anim_output=argv[++i];
        if (arg=="--frames-dir"&&i+1<argc)   anim_frames_dir=argv[++i];
        if (arg=="--theta-start"    &&i+1<argc) anim_theta_start    =std::stod(argv[++i]);
        if (arg=="--theta-end"      &&i+1<argc) anim_theta_end      =std::stod(argv[++i]);
        if (arg=="--phi-start"      &&i+1<argc) anim_phi_start      =std::stod(argv[++i]);
        if (arg=="--phi-end"        &&i+1<argc) anim_phi_end        =std::stod(argv[++i]);
        if (arg=="--r-start"        &&i+1<argc) anim_r_start        =std::stod(argv[++i]);
        if (arg=="--r-end"          &&i+1<argc) anim_r_end          =std::stod(argv[++i]);
        if (arg=="--a-start"        &&i+1<argc) anim_a_start        =std::stod(argv[++i]);
        if (arg=="--a-end"          &&i+1<argc) anim_a_end          =std::stod(argv[++i]);
        if (arg=="--disk-out-start" &&i+1<argc) anim_disk_out_start =std::stod(argv[++i]);
        if (arg=="--disk-out-end"   &&i+1<argc) anim_disk_out_end   =std::stod(argv[++i]);
    }

    // ── Background ───────────────────────────────────────────
    BackgroundImage bg;
    if (!bg_path.empty()) {
        if (bg.load(bg_path.c_str()))
            std::cout << "Background: " << bg_path
                      << " (" << bg.w << "x" << bg.h << ")\n";
        else
            std::cerr << "Warning: cannot load background '" << bg_path << "'\n";
    }

    // ── Derived constants ────────────────────────────────────
    const double M_bh=1.0;
    const double Q_bh=arg_Q, Lam=arg_Lam;

    const int W = custom_w ? custom_w : res_4k ? 3840 : res_2k ? 2560
                : res_720p ? 1280 : hd_preview ? 854 : preview ? 480 : 1920;
    const int H = custom_h ? custom_h : res_4k ? 2160 : res_2k ? 1440
                : res_720p ? 720  : hd_preview ? 480 : preview ? 270 : 1080;

    const double default_r_obs = (res_720p||hd_preview) ? 30.0 : 500.0;
    const double base_r_obs    = (arg_r_obs>0) ? arg_r_obs : default_r_obs;

    const char* res_tag = res_4k ? "4k" : res_2k ? "2k" : custom_w ? "custom"
                        : res_720p ? "720p" : hd_preview ? "hd" : preview ? "preview"
                        : use_bundles ? "bundles" : "trace";

    // ── COLOR-ONLY mode ──────────────────────────────────────
    if (color_only) {
        if (geo_file.empty()) {
            std::cerr << "Error: --color-only requires a .kgeo file path\n";
            return 1;
        }
        std::vector<GeoPixel> geo;
        KGeoMeta meta;
        if (!load_kgeo(geo_file.c_str(), geo, meta)) {
            std::cerr << "Error: cannot load " << geo_file << "\n"; return 1;
        }
        int gW=(int)meta.W, gH=(int)meta.H;
        std::cout << "Loaded: " << geo_file << "  " << gW << "x" << gH
                  << "  a=" << meta.a_bh << "  r_isco=" << meta.r_isco << "\n"
                  << "ColorParams: exposure=" << cp.exposure
                  << " gamma=" << cp.gamma
                  << " temp_scale=" << cp.temp_scale << "\n";

        auto image = colorize_buffer(geo, gW, gH, cp, bg, meta.M_bh, meta.r_isco);

        std::time_t now=std::time(nullptr); char ts[32];
        std::strftime(ts,sizeof(ts),"%Y%m%d-%H%M%S",std::localtime(&now));
        std::string base = geo_file;
        // Remove .kgeo extension for the output name
        if (base.size()>5 && base.substr(base.size()-5)==".kgeo")
            base=base.substr(0,base.size()-5);
        std::string outfile = base + "_recolor_" + ts + ".png";
        write_png(outfile.c_str(), image, gW, gH);
        std::cout << "Saved: " << outfile << "\n";
        return 0;
    }

    // ── Helper: build timestamp ───────────────────────────────
    auto make_ts = []()->std::string {
        std::time_t now=std::time(nullptr); char ts[32];
        std::strftime(ts,sizeof(ts),"%Y%m%d-%H%M%S",std::localtime(&now));
        return ts;
    };

    // ── SINGLE FRAME mode ────────────────────────────────────
    if (!anim_mode) {
        FrameParams fp;
        fp.a=arg_a; fp.theta=arg_theta; fp.phi=arg_phi;
        fp.r_obs=base_r_obs*M_bh; fp.disk_out=arg_disk_out*M_bh;
        fp.fov=arg_fov;

        KNdSMetric g_info(M_bh,fp.a,Q_bh,Lam);
        std::cout << "KNdS  M=" << M_bh << " a=" << fp.a
                  << " Q=" << Q_bh << " Λ=" << Lam << "\n"
                  << "  r₊=" << g_info.r_horizon()
                  << "  r_ISCO=" << g_info.r_isco() << "\n"
                  << "Mode: " << (use_bundles?"ray bundles":"single ray")
                  << "  " << (intg==Integrator::DOPRI5?"DOPRI5":"RK4-doubling")
                  << "  " << W << "x" << H << "\n"
                  << "ColorParams: exp=" << cp.exposure
                  << " γ=" << cp.gamma
                  << " T×=" << cp.temp_scale << "\n";

        double t0=get_time();

        if (geo_only) {
            // Phase 1 only: trace and save .kgeo
            KGeoMeta meta;
            auto geo = trace_geodesics(W, H, fp, use_bundles, intg,
                                       M_bh, Q_bh, Lam, &meta);
            double elapsed=get_time()-t0;
            std::cout << "Trace: " << elapsed << "s  ("
                      << std::fixed << std::setprecision(1)
                      << W*H/elapsed/1e3 << " kpix/s)\n";

            std::string kgeo_path = geo_file.empty()
                ? std::string(OUT_DIR)+"/"+res_tag
                  +"_"+std::to_string(W)+"x"+std::to_string(H)
                  +"_"+make_ts()+".kgeo"
                : geo_file;
            save_kgeo(kgeo_path.c_str(), geo, meta);
            std::cout << "Geo saved: " << kgeo_path << "\n";
        } else {
            // Full render (Phase 1 + Phase 2, optionally save .kgeo)
            std::vector<GeoPixel> geo;
            auto image = render_image(W, H, fp, bg, use_bundles, intg,
                                      M_bh, Q_bh, Lam, cp,
                                      geo_file.empty() ? nullptr : &geo);
            double elapsed=get_time()-t0;
            std::cout << "Time: " << elapsed << "s  ("
                      << std::fixed << std::setprecision(1)
                      << W*H/elapsed/1e3 << " kpix/s)\n";

            std::string ts_str = make_ts();
            std::string outfile = std::string(OUT_DIR)+"/"+res_tag
                                +"_"+std::to_string(W)+"x"+std::to_string(H)
                                +"_"+ts_str+".png";
            write_png(outfile.c_str(), image, W, H);
            std::cout << "Saved: " << outfile << "\n";

            // Optionally save geo file alongside PNG
            if (!geo_file.empty() && !geo.empty()) {
                KNdSMetric g_tmp(M_bh,fp.a,Q_bh,Lam);
                KGeoMeta meta{(uint32_t)W,(uint32_t)H,
                              M_bh,fp.a,Q_bh,Lam,
                              g_tmp.r_isco(),g_tmp.r_isco(),fp.disk_out,
                              fp.theta,fp.phi,fp.r_obs};
                save_kgeo(geo_file.c_str(), geo, meta);
                std::cout << "Geo saved: " << geo_file << "\n";
            }
        }
        return 0;
    }

    // ── ANIMATION mode ───────────────────────────────────────
    auto resolve=[](double v, double fb){ return std::isnan(v)?fb:v; };

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

    std::ostringstream tag_ss;
    tag_ss << res_tag;
    if (anim_orbits!=0.0)            tag_ss<<"_orbit"<<anim_orbits<<"x";
    else if (phi_start!=phi_end)     tag_ss<<"_phi"<<(int)phi_start<<"-"<<(int)phi_end;
    if (theta_start!=theta_end)      tag_ss<<"_th"<<(int)theta_start<<"-"<<(int)theta_end;
    if (r_start!=r_end)              tag_ss<<"_r"<<(int)r_start<<"-"<<(int)r_end;
    if (a_start!=a_end)              tag_ss<<"_a"<<a_start<<"-"<<a_end;
    tag_ss<<"_"<<anim_frames<<"f"<<anim_fps<<"fps";
    std::string anim_tag=tag_ss.str();

    std::string frames_dir = anim_frames_dir.empty()
        ? std::string(OUT_DIR)+"/anim_"+anim_tag : anim_frames_dir;
    std::system(("mkdir -p \""+frames_dir+"\"").c_str());

    std::string output_file = anim_output.empty()
        ? std::string(OUT_DIR)+"/"+anim_tag+".mp4" : anim_output;

    std::cout<<"Animation: "<<anim_frames<<" frames @ "<<anim_fps<<" fps\n"
             <<"  theta:  "<<theta_start<<"° → "<<theta_end<<"°\n";
    if (anim_orbits!=0.0)
        std::cout<<"  phi:    "<<phi_start<<"° + "<<anim_orbits<<" orbit(s)\n";
    else
        std::cout<<"  phi:    "<<phi_start<<"° → "<<phi_end<<"°\n";
    std::cout<<"  r_obs:  "<<r_start<<" M → "<<r_end<<" M\n"
             <<"  a:      "<<a_start<<" → "<<a_end<<"\n"
             <<"  exp="<<cp.exposure<<" γ="<<cp.gamma<<" T×="<<cp.temp_scale<<"\n"
             <<"  output: "<<output_file<<"\n\n";

    double t_total=get_time();
    int rendered=0, skipped=0;

    for (int frame=0; frame<anim_frames; ++frame) {
        char fname[512];
        std::snprintf(fname,sizeof(fname),"%s/frame_%05d.png",frames_dir.c_str(),frame);
        if (anim_resume) {
            std::ifstream test(fname);
            if (test.good()) { ++skipped; continue; }
        }

        double phase=(anim_frames==1)?0.0:(double)frame/(anim_frames-1);
        double t=anim_ease?smooth_step(phase):phase;

        FrameParams fp;
        fp.a        = a_start       +(a_end       -a_start)       *t;
        fp.theta    = theta_start   +(theta_end   -theta_start)   *t;
        fp.r_obs    = (r_start      +(r_end       -r_start)       *t)*M_bh;
        fp.disk_out = (disk_out_start+(disk_out_end-disk_out_start)*t)*M_bh;
        fp.phi      = (anim_orbits!=0.0)
                    ? phi_start+360.0*anim_orbits*phase
                    : lerp_angle(phi_start,phi_end,t);

        double t_frame=get_time();
        std::cout<<"Frame "<<(frame+1)<<"/"<<anim_frames
                 <<"  θ="<<std::fixed<<std::setprecision(1)<<fp.theta
                 <<"°  φ="<<fp.phi
                 <<"°  r="<<std::setprecision(2)<<fp.r_obs
                 <<"  a="<<std::setprecision(4)<<fp.a<<" ...\n"<<std::flush;

        auto image=render_image(W,H,fp,bg,use_bundles,intg,M_bh,Q_bh,Lam,cp);
        write_png(fname,image,W,H);

        double dt=get_time()-t_frame;
        ++rendered;
        double eta=(rendered>0)?(get_time()-t_total)/rendered*(anim_frames-frame-1):0.0;
        std::cout<<"  → "<<fname<<"  ("<<std::setprecision(1)<<dt<<"s"
                 <<"  ETA "<<(int)(eta/60)<<"m"<<(int)std::fmod(eta,60)<<"s)\n";
    }

    std::cout<<"\nFrames: "<<rendered<<" rendered";
    if (skipped) std::cout<<", "<<skipped<<" skipped";
    std::cout<<"  ("<<(get_time()-t_total)<<"s)\n";

    if (!anim_no_encode) {
        char ffcmd[2048];
        std::snprintf(ffcmd,sizeof(ffcmd),
            "ffmpeg -y -framerate %d -i \"%s/frame_%%05d.png\""
            " -c:v libx264 -pix_fmt yuv420p -crf %d -movflags +faststart"
            " \"%s\" 2>&1",
            anim_fps,frames_dir.c_str(),anim_crf,output_file.c_str());
        std::cout<<"Encoding...\n";
        int ret=std::system(ffcmd);
        if (ret==0) std::cout<<"Video saved: "<<output_file<<"\n";
        else        std::cerr<<"ffmpeg failed. Frames in: "<<frames_dir<<"\n";
    }
    if (!anim_keep_frames && !anim_no_encode)
        std::system(("rm -rf \""+frames_dir+"\"").c_str());

    return 0;
}
