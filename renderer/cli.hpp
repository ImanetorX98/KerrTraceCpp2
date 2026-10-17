#pragma once
// Command-line interface, single frames and animation.
//
// Part of the single-translation-unit renderer included by main.cpp;
// see docs/ARCHITETTURA.md for how the modules fit together.

#include "render.hpp"

// ── File I/O ──────────────────────────────────────────────────
static void write_png(const char* path, const std::vector<RGB>& img, int W, int H) {
    if (W <= 0 || H <= 0 || checked_pixel_count(uint32_t(W), uint32_t(H)) != img.size())
        throw std::runtime_error("PNG dimensions do not match the pixel buffer");
    if (!stbi_write_png(path, W, H, 3,
                        reinterpret_cast<const unsigned char*>(img.data()), W*3))
        throw std::runtime_error(std::string("Cannot write PNG file: ") + path);
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
static int run_cli(int argc, char** argv) {

    // ── Resolution ───────────────────────────────────────────
    bool use_bundles=false, preview=false, hd_preview=false;
    bool res_720p=false, res_2k=false, res_4k=false;
    bool elliptic_fallback_black=false;
    bool anti_fireflies=false;
    bool debug_elliptic=false;
    int  custom_w=0, custom_h=0;
    Integrator intg=Integrator::RK4_DOUBLING;
    IntegratorControls int_ctl{};
    int camera_spp = 1;
    bool gpu_fp64 = false;
    CoordinateChart chart=CoordinateChart::KS;
    RaySolverMode solver_mode=RaySolverMode::STANDARD;
    IntersectionMode intersection_mode=IntersectionMode::HERMITE;
    int metal_kernel_mode=0; // 0=auto, 1=unified, 2=single, 3=bundle (Metal only)
    std::string bg_path;

    // ── Single-frame / base params ───────────────────────────
    double arg_a=0.5, arg_disk_out=12.0, arg_theta=80.0, arg_phi=0.0, arg_r_obs=-1e300;
    bool   arg_r_obs_set = false;
    double arg_Q=0.0, arg_Lam=0.0, arg_fov=30.0;

    // ── Wormhole params ───────────────────────────────────────
    bool   arg_wormhole  = false;
    double arg_wh_rho    = 1.0;   // throat areal radius ρ
    double arg_wh_a      = 0.01;  // half tunnel length a
    double arg_wh_M      = 1.0;   // lensing parameter M
    double arg_wh_look   = 0.0;   // boresight rotation (deg): 0=toward wormhole, 90=perpendicular
    double arg_wh_sky    = 1.0e4; // |l| of the two celestial spheres
    std::string bg_b_path;        // Universe B background path (--bg-b)

    // ── Colorization params ───────────────────────────────────
    ColorParams cp;  // defaults: exposure=1, gamma=2.2, temp_scale=1
    double cli_disk_brightness = cp.disk_brightness;
    double cli_disk_opacity    = cp.disk_opacity;
    int    cli_disk_rings      = cp.disk_rings;
    int    cli_disk_sectors    = cp.disk_sectors;
    double cli_disk_sigma      = cp.disk_sigma;
    double cli_disk_hue_offset = cp.disk_hue_offset;
    DiskRadialProfile cli_disk_radial_profile = cp.radial_profile;
    double cli_interstellar_omega0              = cp.interstellar_omega0;
    double cli_interstellar_p                   = cp.interstellar_p;
    double cli_interstellar_inner_falloff_scale = cp.interstellar_inner_falloff_scale;
    bool   cli_interstellar_inner_glow = cp.interstellar_inner_glow;
    bool   cli_interstellar_physical_profile = cp.interstellar_physical_profile;
    double cli_interstellar_band_strength       = cp.interstellar_band_strength;
    double cli_interstellar_band_frequency      = cp.interstellar_band_frequency;
    double cli_interstellar_band_warp           = cp.interstellar_band_warp;
    double cli_interstellar_turbulence_strength = cp.interstellar_turbulence_strength;
    double cli_interstellar_hdr_intensity       = cp.interstellar_hdr_intensity;
    double cli_interstellar_softness_in_scale   = cp.interstellar_softness_in_scale;
    double cli_interstellar_softness_out_scale  = cp.interstellar_softness_out_scale;
    double cli_interstellar_edge_transparency = cp.interstellar_edge_transparency;
    double cli_interstellar_outer_r             = cp.interstellar_outer_r;
    double cli_interstellar_outer_g             = cp.interstellar_outer_g;
    double cli_interstellar_outer_b             = cp.interstellar_outer_b;
    double cli_interstellar_time                = cp.interstellar_time;
    bool   cli_pt_gaussian_taper                = cp.page_thorne_gaussian_taper;
    double cli_pt_taper_sigma_scale             = cp.page_thorne_taper_sigma_scale;
    bool   cli_nt_gaussian_taper                = cp.nt_gaussian_taper;
    double cli_nt_taper_sigma_scale             = cp.nt_taper_sigma_scale;
    bool   cli_planck_emission                  = cp.planck_emission;
    double cli_inner_emission_floor             = cp.inner_emission_floor;
    double cli_inner_emission_floor_width       = cp.inner_emission_floor_width;
    bool   cli_doppler_enabled                  = cp.doppler_enabled;
    bool   cli_radial_term_zero_torque          = cp.radial_term_zero_torque;
    bool   cli_radial_term_r3_decay             = cp.radial_term_r3_decay;
    bool   cli_radial_term_relativistic         = cp.radial_term_relativistic;
    bool   cli_radial_term_b_denom              = cp.radial_term_b_denom;

    // ── Two-phase modes ───────────────────────────────────────
    bool        geo_only    = false;
    std::string geo_file;         // path for .kgeo output (geo_only) or input (color_only)
    bool        color_only  = false;

    // ── Bake mode ─────────────────────────────────────────────
    bool        bake_mode   = false;

    // ── Animation params ──────────────────────────────────────
    bool anim_mode=false, anim_ease=false, anim_keep_frames=false;
    bool anim_resume=false, anim_no_encode=false;
    int  anim_frames=60, anim_fps=30, anim_crf=18;
    double anim_orbits=0.0;
    std::string anim_output, anim_frames_dir;

    std::optional<double> anim_theta_start, anim_theta_end;
    std::optional<double> anim_phi_start, anim_phi_end;
    std::optional<double> anim_r_start, anim_r_end;
    std::optional<double> anim_a_start, anim_a_end;
    std::optional<double> anim_disk_out_start, anim_disk_out_end;

    for (int i=1;i<argc;++i) {
        std::string arg(argv[i]);

        // Resolution
        if (arg=="--bundles")  use_bundles=true;
        if (arg=="--dopri5")   intg=Integrator::DOPRI5;
        if (arg=="--max-steps" && i+1<argc) int_ctl.max_steps = std::max(1, std::stoi(argv[++i]));
        if ((arg=="--step-init" || arg=="--step-size") && i+1<argc) int_ctl.step_init = std::stod(argv[++i]);
        if ((arg=="--integrator-tol" || arg=="--tol") && i+1<argc) int_ctl.tol = std::stod(argv[++i]);
        if ((arg=="--camera-spp" || arg=="--spp") && i+1<argc) camera_spp = std::max(1, std::stoi(argv[++i]));
        if (arg=="--semi-analytic" || arg=="--elliptic") solver_mode=RaySolverMode::SEMI_ANALYTIC;
        if (arg=="--elliptic-closed") solver_mode=RaySolverMode::ELLIPTIC_CLOSED;
        if (arg=="--elliptic-fallback-black" || arg=="--elliptic-strict-black")
            elliptic_fallback_black = true;
        if (arg=="--debug-elliptic")
            debug_elliptic = true;
        if (arg=="--anti-fireflies" || arg=="--anti_fireflies")
            anti_fireflies = true;
        if (arg=="--no-anti-fireflies" || arg=="--no-anti_fireflies")
            anti_fireflies = false;
        if (arg=="--intersection-linear" || arg=="--linear-intersection")
            intersection_mode=IntersectionMode::LINEAR;
        if (arg=="--intersection-hermite" || arg=="--hermite-intersection")
            intersection_mode=IntersectionMode::HERMITE;
        if (arg=="--intersection" && i+1<argc) {
            const std::string m = argv[++i];
            if (m=="linear" || m=="lin") intersection_mode=IntersectionMode::LINEAR;
            else if (m=="hermite" || m=="cubic" || m=="event-hermite")
                intersection_mode=IntersectionMode::HERMITE;
            else
                std::cerr << "Warning: unknown --intersection value '" << m
                          << "', using hermite\n";
        }
        if (arg=="--solver-mode" && i+1<argc) {
            const std::string s = argv[++i];
            if (s=="standard") solver_mode=RaySolverMode::STANDARD;
            else if (s=="semi" || s=="semi-analytic" || s=="semi_analytic")
                solver_mode=RaySolverMode::SEMI_ANALYTIC;
            else if (s=="elliptic" || s=="elliptic-closed" || s=="elliptic_closed")
                solver_mode=RaySolverMode::ELLIPTIC_CLOSED;
        }
        if (arg=="--metal-kernel" && i+1<argc) {
            const std::string k = argv[++i];
            if (k=="auto") metal_kernel_mode = 0;
            else if (k=="unified" || k=="legacy") metal_kernel_mode = 1;
            else if (k=="single" || k=="single-ray" || k=="single_ray") metal_kernel_mode = 2;
            else if (k=="bundle" || k=="bundles" || k=="ray-bundle" || k=="ray_bundle")
                metal_kernel_mode = 3;
            else {
                std::cerr << "Warning: unknown --metal-kernel value '" << k
                          << "', using auto\n";
                metal_kernel_mode = 0;
            }
        }
        if (arg=="--gpu-fp64" || arg=="--gpu-fp64-on" || arg=="--metal-fp64" || arg=="--cuda-fp64")
            gpu_fp64 = true;
        if (arg=="--no-gpu-fp64" || arg=="--no-cuda-fp64")
            gpu_fp64 = false;
        if (arg=="--bl")       chart=CoordinateChart::BL;
        if (arg=="--ks")       chart=CoordinateChart::KS;
        if (arg=="--gks")      chart=CoordinateChart::GKS;
        if (arg=="--chart" && i+1<argc) {
            const std::string c = argv[++i];
            if (c=="bl" || c=="BL") chart=CoordinateChart::BL;
            else if (c=="ks" || c=="KS") chart=CoordinateChart::KS;
            else if (c=="gks" || c=="GKS") chart=CoordinateChart::GKS;
        }
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
        if (arg=="--r-obs"    && i+1<argc) { arg_r_obs = std::stod(argv[++i]); arg_r_obs_set = true; }
        if (arg=="--charge"   && i+1<argc) arg_Q        = std::stod(argv[++i]);
        if (arg=="--lambda"   && i+1<argc) arg_Lam      = std::stod(argv[++i]);
        if (arg=="--fov"      && i+1<argc) arg_fov      = std::stod(argv[++i]);

        // Colorization
        if (arg=="--exposure"   && i+1<argc) cp.exposure   = std::stod(argv[++i]);
        if (arg=="--gamma"      && i+1<argc) cp.gamma      = std::stod(argv[++i]);
        if (arg=="--temp-scale" && i+1<argc) cp.temp_scale = std::stod(argv[++i]);
        if (arg=="--disk-brightness" && i+1<argc) cli_disk_brightness = std::stod(argv[++i]);
        if (arg=="--disk-opacity" && i+1<argc) cli_disk_opacity = std::stod(argv[++i]);
        if (arg=="--disk-blackbody")    cp.palette = DiskPalette::BLACKBODY;
        if (arg=="--disk-stratified")   cp.palette = DiskPalette::STRATIFIED;
        if (arg=="--disk-interstellar") cp.palette = DiskPalette::INTERSTELLAR;
        if (arg=="--disk-nasa") cp.palette = DiskPalette::INTERSTELLAR_NASA;
        if (arg=="--disk-nasa-symmetric") {
            cp.palette = DiskPalette::INTERSTELLAR_NASA;
            cli_doppler_enabled = false;  // artistic: no approaching side
        }
        if (arg=="--disk-radial-profile" && i+1<argc) {
            const std::string mode = argv[++i];
            if (mode=="physical_nt" || mode=="physical-nt" || mode=="nt")
                cli_disk_radial_profile = DiskRadialProfile::PHYSICAL_NT;
            else
                cli_disk_radial_profile = DiskRadialProfile::PAGE_THORNE;
        }
        if (arg=="--disk-profile-physical-nt" || arg=="--disk-physical-nt")
            cli_disk_radial_profile = DiskRadialProfile::PHYSICAL_NT;
        if (arg=="--disk-profile-page-thorne" || arg=="--disk-page-thorne")
            cli_disk_radial_profile = DiskRadialProfile::PAGE_THORNE;
        if (arg=="--disk-rings"      && i+1<argc) cli_disk_rings      = std::stoi(argv[++i]);
        if (arg=="--disk-sectors"    && i+1<argc) cli_disk_sectors    = std::stoi(argv[++i]);
        if (arg=="--disk-sigma"      && i+1<argc) cli_disk_sigma      = std::stod(argv[++i]);
        if (arg=="--disk-hue-offset" && i+1<argc) cli_disk_hue_offset = std::stod(argv[++i]);
        if (arg=="--disk-interstellar-omega0" && i+1<argc) cli_interstellar_omega0 = std::stod(argv[++i]);
        if (arg=="--disk-interstellar-p" && i+1<argc) cli_interstellar_p = std::stod(argv[++i]);
        if (arg=="--disk-interstellar-artistic-profile") cli_interstellar_physical_profile = false;
        if (arg=="--disk-interstellar-inner-glow") cli_interstellar_inner_glow = true;
        // Setting the falloff only makes sense with the glow on, so imply it.
        if (arg=="--disk-interstellar-inner-falloff" && i+1<argc) {
            cli_interstellar_inner_falloff_scale = std::stod(argv[++i]);
            cli_interstellar_inner_glow = true;
        }
        if (arg=="--disk-interstellar-band-strength" && i+1<argc) cli_interstellar_band_strength = std::stod(argv[++i]);
        if (arg=="--disk-interstellar-band-frequency" && i+1<argc) cli_interstellar_band_frequency = std::stod(argv[++i]);
        if (arg=="--disk-interstellar-band-warp" && i+1<argc) cli_interstellar_band_warp = std::stod(argv[++i]);
        if (arg=="--disk-interstellar-turbulence" && i+1<argc) cli_interstellar_turbulence_strength = std::stod(argv[++i]);
        if (arg=="--disk-interstellar-hdr" && i+1<argc) cli_interstellar_hdr_intensity = std::stod(argv[++i]);
        if (arg=="--disk-interstellar-softness-in" && i+1<argc) cli_interstellar_softness_in_scale = std::stod(argv[++i]);
        if (arg=="--disk-interstellar-softness-out" && i+1<argc) cli_interstellar_softness_out_scale = std::stod(argv[++i]);
        if (arg=="--disk-interstellar-edge-transparency" && i+1<argc) cli_interstellar_edge_transparency = std::stod(argv[++i]);
        if (arg=="--disk-interstellar-outer-r" && i+1<argc) cli_interstellar_outer_r = std::stod(argv[++i]);
        if (arg=="--disk-interstellar-outer-g" && i+1<argc) cli_interstellar_outer_g = std::stod(argv[++i]);
        if (arg=="--disk-interstellar-outer-b" && i+1<argc) cli_interstellar_outer_b = std::stod(argv[++i]);
        if (arg=="--disk-interstellar-time" && i+1<argc) cli_interstellar_time = std::stod(argv[++i]);
        if (arg=="--disk-interstellar-inner-taper" && i+1<argc) cp.interstellar_inner_taper_scale = std::max(1e-6, std::stod(argv[++i]));
        if (arg=="--disk-pt-gaussian-taper" || arg=="--pt-gaussian-taper")
            cli_pt_gaussian_taper = true;
        if (arg=="--disk-no-pt-gaussian-taper" || arg=="--no-pt-gaussian-taper")
            cli_pt_gaussian_taper = false;
        if ((arg=="--disk-pt-taper-sigma-scale" || arg=="--pt-taper-sigma-scale") && i+1<argc)
            cli_pt_taper_sigma_scale = std::stod(argv[++i]);
        if (arg=="--disk-inner-emission-floor" && i+1<argc)
            cli_inner_emission_floor = std::stod(argv[++i]);
        if (arg=="--disk-inner-emission-floor-width" && i+1<argc)
            cli_inner_emission_floor_width = std::stod(argv[++i]);
        if (arg=="--bundle-edge-grid" && i+1<argc)
            cp.bundle_edge_grid = std::max(1, std::stoi(argv[++i]));
        if (arg=="--disk-nasa-gain" && i+1<argc)
            cp.nasa_gain = std::max(0.0, std::stod(argv[++i]));
        if (arg=="--disk-temp-clamp") cp.temp_redshift_clamp = true;
        if (arg=="--disk-temp-clamp-floor" && i+1<argc) {
            cp.temp_redshift_clamp = true;
            cp.temp_redshift_floor = std::max(0.0, std::stod(argv[++i]));
        }
        if (arg=="--bundle-filter")    cp.bundle_filter = true;
        if (arg=="--no-bundle-filter") cp.bundle_filter = false;
        if (arg=="--bundle-filter-rings" && i+1<argc)
            cp.bundle_filter_rings = std::max(0, std::stoi(argv[++i]));
        if (arg=="--bundle-filter-sigma" && i+1<argc)
            cp.bundle_filter_sigma = std::max(1e-3, std::stod(argv[++i]));
        if (arg=="--bundle-magnification")    cp.bundle_magnif_luma = true;
        if (arg=="--no-bundle-magnification") cp.bundle_magnif_luma = false;
        if (arg=="--bundle-magnification-max" && i+1<argc)
            cp.bundle_magnif_max = std::max(1.0, std::stod(argv[++i]));
        if (arg=="--no-doppler") cli_doppler_enabled = false;
        if (arg=="--doppler") cli_doppler_enabled = true;
        if (arg=="--doppler-exp" && i+1<argc) cp.doppler_exp = std::max(0.0, std::stod(argv[++i]));
        if (arg=="--zero-torque-taper")    cp.zero_torque_taper = true;
        if (arg=="--no-zero-torque-taper") cp.zero_torque_taper = false;
        if (arg=="--no-radial-term-zero-torque" || arg=="--no-radial-zero-torque")
            cli_radial_term_zero_torque = false;
        if (arg=="--radial-term-zero-torque" || arg=="--radial-zero-torque")
            cli_radial_term_zero_torque = true;
        if (arg=="--no-radial-term-r3" || arg=="--no-radial-r3")
            cli_radial_term_r3_decay = false;
        if (arg=="--radial-term-r3" || arg=="--radial-r3")
            cli_radial_term_r3_decay = true;
        if (arg=="--no-radial-term-relativistic" || arg=="--no-radial-relativistic")
            cli_radial_term_relativistic = false;
        if (arg=="--radial-term-relativistic" || arg=="--radial-relativistic")
            cli_radial_term_relativistic = true;
        if (arg=="--no-radial-term-b-denom" || arg=="--no-radial-b-denom")
            cli_radial_term_b_denom = false;
        if (arg=="--radial-term-b-denom" || arg=="--radial-b-denom")
            cli_radial_term_b_denom = true;
        if (arg=="--nt-taper")
            cli_nt_gaussian_taper = true;
        if (arg=="--no-nt-taper")
            cli_nt_gaussian_taper = false;
        if (arg=="--nt-taper-sigma" && i+1<argc)
            cli_nt_taper_sigma_scale = std::stod(argv[++i]);
        if (arg=="--planck-emission")
            cli_planck_emission = true;
        if (arg=="--no-planck-emission")
            cli_planck_emission = false;
        if (arg=="--ring-taper")      cp.radial_ring_taper = true;
        if (arg=="--no-ring-taper")   cp.radial_ring_taper = false;
        if (arg=="--ring-taper-isco") { cp.radial_ring_taper = true; cp.radial_ring_taper_use_isco = true; }
        if (arg=="--ring-taper-r0"    && i+1<argc) { cp.radial_ring_taper_r0 = std::stod(argv[++i]); cp.radial_ring_taper_use_isco = false; }
        if (arg=="--ring-taper-sigma" && i+1<argc) cp.radial_ring_taper_sigma = std::max(1e-6, std::stod(argv[++i]));

        // Wormhole
        if (arg=="--wormhole")                   arg_wormhole = true;
        if (arg=="--wh-throat" && i+1<argc)      arg_wh_rho   = std::stod(argv[++i]);
        if (arg=="--wh-lensing"&& i+1<argc)      arg_wh_M     = std::stod(argv[++i]);
        if (arg=="--wh-tunnel" && i+1<argc)      arg_wh_a     = std::stod(argv[++i]);
        if (arg=="--wh-look"   && i+1<argc)      arg_wh_look  = std::stod(argv[++i]);
        if (arg=="--wh-sky-dist"&& i+1<argc)     arg_wh_sky   = std::stod(argv[++i]);
        if (arg=="--bg-b"      && i+1<argc)      bg_b_path    = argv[++i];

        // Two-phase modes
        if (arg=="--geo-only")             geo_only  = true;
        if (arg=="--geo-file" && i+1<argc) geo_file  = argv[++i];
        if (arg=="--color-only"&& i+1<argc){ color_only=true; geo_file=argv[++i]; }

        // Bake mode (batch stdin loop: one process, many frames)
        if (arg=="--bake-mode") bake_mode = true;

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

    cp.disk_brightness = std::max(0.0, cli_disk_brightness);
    cp.disk_opacity = clamp(cli_disk_opacity, 0.0, 1.0);
    cp.radial_profile = cli_disk_radial_profile;
    cp.page_thorne_gaussian_taper = cli_pt_gaussian_taper;
    cp.page_thorne_taper_sigma_scale = std::max(1.0e-6, cli_pt_taper_sigma_scale);
    cp.nt_gaussian_taper = cli_nt_gaussian_taper;
    cp.nt_taper_sigma_scale = std::max(1.0e-6, cli_nt_taper_sigma_scale);
    cp.planck_emission = cli_planck_emission;
    cp.inner_emission_floor = std::max(0.0, cli_inner_emission_floor);
    cp.inner_emission_floor_width = clamp(cli_inner_emission_floor_width, 1.0e-4, 2.0);
    cp.doppler_enabled = cli_doppler_enabled;
    cp.radial_term_zero_torque = cli_radial_term_zero_torque;
    cp.radial_term_r3_decay = cli_radial_term_r3_decay;
    cp.radial_term_relativistic = cli_radial_term_relativistic;
    cp.radial_term_b_denom = cli_radial_term_b_denom;

    if (cp.palette == DiskPalette::STRATIFIED) {
        cp.disk_rings = std::max(1, cli_disk_rings);
        cp.disk_sectors = std::max(1, cli_disk_sectors);
        cp.disk_sigma = std::max(0.01, cli_disk_sigma);
        cp.disk_hue_offset = std::fmod(cli_disk_hue_offset, 1.0);
        if (cp.disk_hue_offset < 0.0) cp.disk_hue_offset += 1.0;
    }
    if (cp.palette == DiskPalette::INTERSTELLAR) {
        cp.interstellar_omega0 = std::max(1.0e-4, cli_interstellar_omega0);
        cp.interstellar_p = std::max(0.1, cli_interstellar_p);
        cp.interstellar_inner_falloff_scale = std::max(1.0e-4, cli_interstellar_inner_falloff_scale);
        cp.interstellar_inner_glow = cli_interstellar_inner_glow;
        cp.interstellar_physical_profile = cli_interstellar_physical_profile;
        cp.interstellar_band_strength = std::max(0.0, cli_interstellar_band_strength);
        cp.interstellar_band_frequency = std::max(0.0, cli_interstellar_band_frequency);
        cp.interstellar_band_warp = std::max(0.0, cli_interstellar_band_warp);
        cp.interstellar_turbulence_strength = std::max(0.0, cli_interstellar_turbulence_strength);
        cp.interstellar_hdr_intensity = std::max(0.0, cli_interstellar_hdr_intensity);
        cp.interstellar_softness_in_scale = std::max(0.0, cli_interstellar_softness_in_scale);
        cp.interstellar_softness_out_scale = std::max(0.0, cli_interstellar_softness_out_scale);
        cp.interstellar_edge_transparency = clamp(cli_interstellar_edge_transparency, 0.0, 1.0);
        cp.interstellar_outer_r = clamp(cli_interstellar_outer_r, 0.0, 1.0);
        cp.interstellar_outer_g = clamp(cli_interstellar_outer_g, 0.0, 1.0);
        cp.interstellar_outer_b = clamp(cli_interstellar_outer_b, 0.0, 1.0);
        cp.interstellar_time = cli_interstellar_time;
    }

    // ── Background ───────────────────────────────────────────
    int_ctl.max_steps = std::max(1, int_ctl.max_steps);
    int_ctl.step_init = std::max(1e-10, int_ctl.step_init);
    int_ctl.tol = std::max(1e-10, int_ctl.tol);
    camera_spp = std::max(1, camera_spp);

    BackgroundImage bg;
    if (!bg_path.empty()) {
        if (bg.load(bg_path.c_str()))
            std::cout << "Background: " << bg_path
                      << " (" << bg.w << "x" << bg.h << ")\n";
        else
            std::cerr << "Warning: cannot load background '" << bg_path << "'\n";
    }

    BackgroundImage bg_b_img;
    if (!bg_b_path.empty()) {
        if (bg_b_img.load(bg_b_path.c_str()))
            std::cout << "Background B (Universe B): " << bg_b_path
                      << " (" << bg_b_img.w << "x" << bg_b_img.h << ")\n";
        else
            std::cerr << "Warning: cannot load --bg-b background '" << bg_b_path << "'\n";
    }
    const BackgroundImage* bg_b = bg_b_img.px.empty() ? nullptr : &bg_b_img;

    // ── Derived constants ────────────────────────────────────
    const double M_bh=1.0;
    const double Q_bh=arg_Q, Lam=arg_Lam;
    if (chart == CoordinateChart::GKS && use_bundles && std::abs(Lam) > 1e-15) {
        std::cerr << "Info: GKS + Lambda != 0 disables ray bundles in this build; running single-ray mode.\n";
        use_bundles = false;
        anti_fireflies = false;
    }

    // In bake mode, default to 320×180 if no explicit resolution was given
    if (bake_mode && !custom_w && !custom_h &&
        !preview && !hd_preview && !res_720p && !res_2k && !res_4k) {
        custom_w = 320; custom_h = 180;
    }

    const int W = custom_w ? custom_w : res_4k ? 3840 : res_2k ? 2560
                : res_720p ? 1280 : hd_preview ? 854 : preview ? 480 : 1920;
    const int H = custom_h ? custom_h : res_4k ? 2160 : res_2k ? 1440
                : res_720p ? 720  : hd_preview ? 480 : preview ? 270 : 1080;
    checked_pixel_count(uint32_t(W), uint32_t(H));

    const double default_r_obs = 60.0;
    // Wormhole mode allows negative l_obs (observer in Universe B); non-wormhole needs positive r_obs.
    const double base_r_obs = arg_r_obs_set
        ? (arg_wormhole ? arg_r_obs : std::max(arg_r_obs, 1.0))
        : default_r_obs;

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
                  << " temp_scale=" << cp.temp_scale
                  << " disk_brightness=" << cp.disk_brightness
                  << " radial_profile=" << disk_radial_profile_name(cp.radial_profile)
                  << " radial_terms={zt:" << (cp.radial_term_zero_torque ? "on" : "off")
                  << ",r3:" << (cp.radial_term_r3_decay ? "on" : "off")
                  << ",rel:" << (cp.radial_term_relativistic ? "on" : "off")
                  << ",b:" << (cp.radial_term_b_denom ? "on" : "off")
                  << "}"
                  << " doppler=" << (cp.doppler_enabled ? "on" : "off")
                  << "\n";

        auto image = colorize_buffer(geo, gW, gH, cp, bg, meta.M_bh, meta.a_bh, meta.r_isco,
                                     meta.r_disk_in, meta.r_disk_out);

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

    // A fresh checkout has no out/ directory (render outputs are gitignored).
    std::filesystem::create_directories(OUT_DIR);

    // ── Helper: build timestamp ───────────────────────────────
    auto make_ts = []()->std::string {
        std::time_t now=std::time(nullptr); char ts[32];
        std::strftime(ts,sizeof(ts),"%Y%m%d-%H%M%S",std::localtime(&now));
        return ts;
    };

    auto effective_chart_for_naming = [&](CoordinateChart requested)->CoordinateChart {
        CoordinateChart eff = resolve_runtime_chart_gpu(requested, Lam);
        if (eff == CoordinateChart::KS && std::abs(Lam) > 1e-15)
            eff = CoordinateChart::BL;
        return eff;
    };
    auto integration_mode_tag = [&]()->std::string {
        const CoordinateChart eff_chart = effective_chart_for_naming(chart);
        const char* out_chart_tag = (chart == CoordinateChart::GKS)
            ? chart_tag(CoordinateChart::GKS)
            : chart_tag(eff_chart);
        const SolverSelection sel = select_solver_mode(
            solver_mode, use_bundles, eff_chart, Q_bh, Lam);

        if (use_bundles)
            return std::string(out_chart_tag) + "-ray-bundle";

        if (sel.effective == RaySolverMode::SEMI_ANALYTIC)
            return std::string(out_chart_tag) + "-semi-analytic";
        if (sel.effective == RaySolverMode::ELLIPTIC_CLOSED) {
            std::string tag = std::string(out_chart_tag) + "-elliptic-closed";
            if (elliptic_fallback_black) tag += "-fallback-black";
            return tag;
        }

        // Standard solver: CPU KS is RK4-only; GPU KS can expose DOPRI5.
        bool ks_uses_rk4 = (eff_chart == CoordinateChart::KS);
#if defined(USE_METAL)
        const bool ks_chart_supported = (std::abs(Lam) <= 1e-15);
        const bool gpu_chart_ok = (eff_chart == CoordinateChart::BL) ||
                                  (eff_chart == CoordinateChart::KS && ks_chart_supported);
        const bool solver_gpu_supported = metal_solver_supported(sel.effective, eff_chart, Q_bh, Lam);
        const bool bundle_gpu_supported =
            use_bundles &&
            gpu_chart_ok &&
            sel.effective == RaySolverMode::STANDARD;
        const bool single_ray_gpu_supported =
            !use_bundles &&
            gpu_chart_ok &&
            solver_gpu_supported;
        const bool alpha_composite_cpu_only = cp.disk_opacity < (1.0 - 1e-9);
        const bool can_use_gpu = (bundle_gpu_supported || single_ray_gpu_supported) && !gpu_fp64 && !arg_wormhole
            && !metal_cpu_reason(cp, use_bundles, geo_only || !geo_file.empty());
        if (can_use_gpu) ks_uses_rk4 = false;
#endif
        const char* intg_tag = (ks_uses_rk4 || intg == Integrator::RK4_DOUBLING)
            ? "rk4"
            : "dopri5";
        return std::string(out_chart_tag) + "-standard-" + intg_tag;
    };
    auto backend_mode_tag = [&]()->const char* {
#if defined(USE_METAL)
        const CoordinateChart eff_chart = effective_chart_for_naming(chart);
        const bool ks_chart_supported = (std::abs(Lam) <= 1e-15);
        const bool gpu_chart_ok = (eff_chart == CoordinateChart::BL) ||
                                  (eff_chart == CoordinateChart::KS && ks_chart_supported);
        const SolverSelection gpu_solver_sel = select_solver_mode(
            solver_mode, use_bundles, eff_chart, Q_bh, Lam);
        const RaySolverMode gpu_solver_mode = gpu_solver_sel.effective;
        const bool solver_gpu_supported = metal_solver_supported(gpu_solver_mode, eff_chart, Q_bh, Lam);
        const bool bundle_gpu_supported =
            use_bundles &&
            gpu_chart_ok &&
            gpu_solver_mode == RaySolverMode::STANDARD;
        const bool single_ray_gpu_supported =
            !use_bundles &&
            gpu_chart_ok &&
            solver_gpu_supported;
        const bool base_can_use_gpu = bundle_gpu_supported || single_ray_gpu_supported;
        const bool metal_fp64_supported = false; // Current Metal kernel path is FP32 only.
        const bool alpha_composite_cpu_only = cp.disk_opacity < (1.0 - 1e-9);
        const bool can_use_gpu = base_can_use_gpu && !gpu_fp64 && !arg_wormhole
            && !metal_cpu_reason(cp, use_bundles, geo_only || !geo_file.empty());
        return can_use_gpu ? "gpu-metal" : "cpu";
#elif defined(USE_CUDA)
        const CoordinateChart eff_chart = effective_chart_for_naming(chart);
        const bool ks_chart_supported = (std::abs(Lam) <= 1e-15);
        const bool gpu_chart_ok = (eff_chart == CoordinateChart::BL) ||
                                  (eff_chart == CoordinateChart::KS && ks_chart_supported);
        const bool cuda_gpu_supported = !arg_wormhole && !geo_only &&
            !cuda_cpu_reason(cp, use_bundles, solver_mode, intg) && gpu_chart_ok &&
            std::abs(Q_bh) <= 1e-15 && std::abs(Lam) <= 1e-15;
        return cuda_gpu_supported ? "gpu-cuda" : "cpu";
#else
        return "cpu";
#endif
    };
    const std::string mode_tag = integration_mode_tag();
    const std::string backend_tag = backend_mode_tag();
    const std::string fp64_tag = gpu_fp64 ? "_gpufp64req" : "";
    const std::string spp_tag = (camera_spp > 1) ? ("_spp" + std::to_string(camera_spp)) : "";
    const std::string mode_backend_tag = mode_tag + "_" + backend_tag + fp64_tag + spp_tag;

    // ── SINGLE FRAME mode / BAKE mode ───────────────────────────
    if (!anim_mode) {
        FrameParams fp;
        fp.a=arg_a; fp.theta=arg_theta; fp.phi=arg_phi;
        fp.r_obs=base_r_obs*M_bh; fp.disk_out=arg_disk_out*M_bh;
        fp.fov=arg_fov;
        fp.wormhole    = arg_wormhole;
        fp.wh_rho      = arg_wh_rho;
        fp.wh_a_tunnel = arg_wh_a;
        fp.wh_M_lens   = arg_wh_M;
        fp.wh_look_deg = arg_wh_look;
        fp.wh_sky_dist = arg_wh_sky;

        // ── BAKE mode: read (theta phi) pairs from stdin, render each frame ──
        if (bake_mode) {
            // Suppress verbose output; only "Saved: <path>" goes to stdout.
            // Render each frame as it arrives, then flush stdout.
            std::string line;
            while (std::getline(std::cin, line)) {
                if (line.empty()) continue;
                double bake_theta = fp.theta, bake_phi = fp.phi;
                std::istringstream iss(line);
                if (!(iss >> bake_theta >> bake_phi)) continue;
                fp.theta = bake_theta;
                fp.phi   = bake_phi;

                auto image = render_image(W, H, fp, bg, use_bundles, solver_mode, chart, intg,
                                          int_ctl, camera_spp, M_bh, Q_bh, Lam,
                                          intersection_mode, metal_kernel_mode, gpu_fp64,
                                          elliptic_fallback_black, anti_fireflies, cp,
                                          nullptr, debug_elliptic,
                                          0.0, 0.0, bg_b);

                std::cout << "Backend used: " << last_render_backend << "\n";
                const std::string actual_mode_backend = mode_tag + "_" + last_render_backend + fp64_tag + spp_tag;
                std::string ts_str = make_ts();
                std::string outfile = std::string(OUT_DIR)+"/"+res_tag
                                    +"_"+std::to_string(W)+"x"+std::to_string(H)
                                    +"_"+actual_mode_backend
                                    +"_"+ts_str+".png";
                write_png(outfile.c_str(), image, W, H);
                std::cout << "Saved: " << outfile << "\n";
                std::cout.flush();
            }
            return 0;
        }

        if (fp.wormhole) {
            std::cout << "DNEG Wormhole  ρ=" << fp.wh_rho
                      << " a=" << fp.wh_a_tunnel
                      << " M_lens=" << fp.wh_M_lens << "\n"
                      << "  throat_r=" << fp.wh_rho
                      << "  l_obs=" << fp.r_obs << "\n";
        } else {
            KNdSMetric g_info(M_bh,fp.a,Q_bh,Lam);
            std::cout << "KNdS  M=" << M_bh << " a=" << fp.a
                      << " Q=" << Q_bh << " Λ=" << Lam << "\n"
                      << "  r₊=" << g_info.r_horizon()
                      << "  r_ISCO=" << g_info.r_isco() << "\n";
        }
        std::cout << "Mode: " << (fp.wormhole?"wormhole":use_bundles?"ray bundles":"single ray")
                  << "  chart=" << (fp.wormhole ? "wormhole" : chart_label(chart))
                  << "  " << (intg==Integrator::DOPRI5?"DOPRI5":"RK4-doubling")
                  << "  " << (fp.wormhole?"rk4-adaptive":solver_mode_name(solver_mode))
                  << "  backend=" << (fp.wormhole?"cpu":backend_tag.c_str())
                  << "  intersection=" << intersection_mode_name(intersection_mode)
                  << "  elliptic-fallback-black=" << (elliptic_fallback_black ? "on" : "off")
                  << "  anti-fireflies=" << (anti_fireflies ? "on" : "off")
                  << "  max-steps=" << int_ctl.max_steps
                  << "  step-init=" << int_ctl.step_init
                  << "  tol=" << int_ctl.tol
                  << "  camera-spp=" << camera_spp
                  << "  gpu-fp64=" << (gpu_fp64 ? "on" : "off")
#if defined(USE_METAL)
                  << "  metal-kernel="
                  << metal_kernel_mode_name(static_cast<MetalKernelMode>(metal_kernel_mode))
#endif
                  << "  " << W << "x" << H << "\n"
                  << "ColorParams: exp=" << cp.exposure
                  << " γ=" << cp.gamma
                  << " T×=" << cp.temp_scale
                  << " diskB×=" << cp.disk_brightness
                  << " diskOpacity=" << cp.disk_opacity
                  << " radialProfile=" << disk_radial_profile_name(cp.radial_profile)
                  << " radialTerms={zt:" << (cp.radial_term_zero_torque ? "on" : "off")
                  << ",r3:" << (cp.radial_term_r3_decay ? "on" : "off")
                  << ",rel:" << (cp.radial_term_relativistic ? "on" : "off")
                  << ",b:" << (cp.radial_term_b_denom ? "on" : "off")
                  << "}"
                  << " innerFloor=" << cp.inner_emission_floor
                  << " innerFloorW=" << cp.inner_emission_floor_width
                  << " doppler=" << (cp.doppler_enabled ? "on" : "off")
                  << "\n";

        double t0=get_time();

        if (geo_only) {
            // Phase 1 only: trace and save .kgeo
            KGeoMeta meta;
            auto geo = trace_geodesics(W, H, fp, use_bundles, solver_mode, chart, intg,
                                       int_ctl, 0.0, 0.0, M_bh, Q_bh, Lam, &cp, &meta);
            double elapsed=get_time()-t0;
            std::cout << "Trace: " << elapsed << "s  ("
                      << std::fixed << std::setprecision(1)
                      << W*H/elapsed/1e3 << " kpix/s)\n";

            std::string kgeo_path = geo_file.empty()
                ? std::string(OUT_DIR)+"/"+res_tag
                  +"_"+std::to_string(W)+"x"+std::to_string(H)
                  +"_"+mode_backend_tag
                  +"_"+make_ts()+".kgeo"
                : geo_file;
            save_kgeo(kgeo_path.c_str(), geo, meta);
            std::cout << "Geo saved: " << kgeo_path << "\n";
        } else {
            // Full render (Phase 1 + Phase 2, optionally save .kgeo)
            std::vector<GeoPixel> geo;
            auto image = render_image(W, H, fp, bg, use_bundles, solver_mode, chart, intg,
                                      int_ctl, camera_spp, M_bh, Q_bh, Lam,
                                      intersection_mode, metal_kernel_mode, gpu_fp64,
                                      elliptic_fallback_black, anti_fireflies, cp,
                                      geo_file.empty() ? nullptr : &geo, debug_elliptic,
                                      0.0, 0.0, bg_b);
            double elapsed=get_time()-t0;
            std::cout << "Time: " << elapsed << "s  ("
                      << std::fixed << std::setprecision(1)
                      << W*H/elapsed/1e3 << " kpix/s)\n";

            std::cout << "Backend used: " << last_render_backend << "\n";
            const std::string actual_mode_backend = mode_tag + "_" + last_render_backend + fp64_tag + spp_tag;
            std::string ts_str = make_ts();
            std::string outfile = std::string(OUT_DIR)+"/"+res_tag
                                +"_"+std::to_string(W)+"x"+std::to_string(H)
                                +"_"+actual_mode_backend
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
    auto resolve=[](const std::optional<double>& v, double fb){ return v.value_or(fb); };

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
    tag_ss<<"_"<<mode_backend_tag;
    tag_ss<<"_"<<anim_frames<<"f"<<anim_fps<<"fps";
    std::string anim_tag=tag_ss.str();

    std::string frames_dir = anim_frames_dir.empty()
        ? std::string(OUT_DIR)+"/anim_"+anim_tag : anim_frames_dir;
    std::filesystem::create_directories(frames_dir);

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
             <<"  exp="<<cp.exposure<<" γ="<<cp.gamma<<" T×="<<cp.temp_scale<<" diskB×="<<cp.disk_brightness
             <<" doppler="<<(cp.doppler_enabled?"on":"off")<<"\n"
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
        fp.fov      = arg_fov;
        fp.phi      = (anim_orbits!=0.0)
                    ? phi_start+360.0*anim_orbits*phase
                    : lerp_angle(phi_start,phi_end,t);
        fp.wormhole    = arg_wormhole;
        fp.wh_rho      = arg_wh_rho;
        fp.wh_a_tunnel = arg_wh_a;
        fp.wh_M_lens   = arg_wh_M;
        fp.wh_look_deg = arg_wh_look;
        fp.wh_sky_dist = arg_wh_sky;

        double t_frame=get_time();
        std::cout<<"Frame "<<(frame+1)<<"/"<<anim_frames
                 <<"  θ="<<std::fixed<<std::setprecision(1)<<fp.theta
                 <<"°  φ="<<fp.phi
                 <<"°  r="<<std::setprecision(2)<<fp.r_obs
                 <<"  a="<<std::setprecision(4)<<fp.a<<" ...\n"<<std::flush;

        auto image=render_image(W,H,fp,bg,use_bundles,solver_mode,chart,intg,
                                int_ctl, camera_spp, M_bh,Q_bh,Lam,
                                intersection_mode,metal_kernel_mode,gpu_fp64,
                                elliptic_fallback_black,anti_fireflies,cp,nullptr,debug_elliptic,
                                0.0, 0.0, bg_b);
        std::cout << "Backend used: " << last_render_backend << "\n";
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
        std::cout<<"Encoding...\n";
        const int ret=run_process({"ffmpeg", "-y", "-framerate", std::to_string(anim_fps),
            "-i", frames_dir+"/frame_%05d.png", "-frames:v", std::to_string(anim_frames),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", std::to_string(anim_crf),
            "-movflags", "+faststart", output_file});
        if (ret==0) std::cout<<"Video saved: "<<output_file<<"\n";
        else {
            std::cerr << "ffmpeg failed. Frames preserved in: " << frames_dir << "\n";
            return 1;
        }
    }
    if (!anim_keep_frames && !anim_no_encode) {
        for (int frame=0; frame<anim_frames; ++frame) {
            char name[32];
            std::snprintf(name,sizeof(name),"frame_%05d.png",frame);
            std::filesystem::remove(std::filesystem::path(frames_dir)/name);
        }
        if (std::filesystem::is_empty(frames_dir)) std::filesystem::remove(frames_dir);
    }

    return 0;
}
