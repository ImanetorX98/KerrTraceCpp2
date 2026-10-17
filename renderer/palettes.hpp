#pragma once
// Disk palettes and the radial emission profiles (phase 2).
//
// Part of the single-translation-unit renderer included by main.cpp;
// see docs/ARCHITETTURA.md for how the modules fit together.

#include "disk_physics.hpp"
#include "trace_types.hpp"

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

static double page_thorne_flux_raw(double r, double r_isco,
                                   const ColorParams& cp) {
    if (r <= r_isco) return 0.0;
    const double x = std::sqrt(r_isco / r);
    const double zero_torque = cp.radial_term_zero_torque
        ? clamp(1.0 - x, 0.0, 1.0)
        : 1.0;
    const double r3_decay = cp.radial_term_r3_decay
        ? (1.0 / std::max(r * r * r, 1.0e-12))
        : 1.0;
    double flux = zero_torque * r3_decay;
    if (cp.page_thorne_gaussian_taper) {
        const double dr = std::max(0.0, r - r_isco);
        const double sigma = std::max(1.0e-6, cp.page_thorne_taper_sigma_scale * r_isco);
        const double taper = 1.0 - std::exp(-0.5 * (dr / sigma) * (dr / sigma));
        flux *= taper;
    }
    return std::max(0.0, flux);
}

static double physical_nt_flux_proxy_raw(double r, double r_isco, double a,
                                         const ColorParams& cp) {
    if (r <= r_isco) return 0.0;
    const double r_safe = std::max(r, r_isco * 1.0005);
    const double r2 = r_safe * r_safe;
    const double r3 = r2 * r_safe;
    const double r32 = std::max(r_safe * std::sqrt(std::max(r_safe, 1.0e-12)), 1.0e-12);
    const double a2 = a * a;
    const double orbit_spin = std::abs(a); // disk corotates for either spin sign

    const double A = 1.0 - 2.0 / std::max(r_safe, 1.0e-12) + a2 / std::max(r2, 1.0e-12);
    const double B = 1.0 - 3.0 / std::max(r_safe, 1.0e-12) + 2.0 * orbit_spin / r32;
    const double C = 1.0 - 4.0 * orbit_spin / r32 + 3.0 * a2 / std::max(r2, 1.0e-12);

    const double zero_torque = cp.radial_term_zero_torque
        ? clamp(1.0 - std::sqrt(clamp(r_isco / std::max(r_safe, 1.0e-12), 0.0, 1.0)), 0.0, 1.0)
        : 1.0;
    const double relativistic = cp.radial_term_relativistic
        ? std::sqrt(clamp(C / std::max(A, 1.0e-6), 0.02, 50.0))
        : 1.0;
    const double inv_r3 = cp.radial_term_r3_decay
        ? (1.0 / std::max(r3, 1.0e-12))
        : 1.0;
    const double inv_B = cp.radial_term_b_denom
        ? (1.0 / std::max(B, 0.02))
        : 1.0;
    double flux = zero_torque * relativistic * inv_r3 * inv_B;
    if (cp.nt_gaussian_taper) {
        const double dr = std::max(0.0, r_safe - r_isco);
        const double sigma = std::max(1.0e-6, cp.nt_taper_sigma_scale * r_isco);
        const double taper = 1.0 - std::exp(-0.5 * (dr / sigma) * (dr / sigma));
        flux *= taper;
    }
    return std::max(0.0, flux);
}

static double disk_flux_raw(double r, double r_isco, double a, const ColorParams& cp) {
    if (cp.radial_profile == DiskRadialProfile::PHYSICAL_NT)
        return physical_nt_flux_proxy_raw(r, r_isco, a, cp);
    return page_thorne_flux_raw(r, r_isco, cp);
}

static double inner_emission_floor_value(double r,
                                         double r_in,
                                         double r_out,
                                         const ColorParams& cp) {
    const double floor0 = std::max(0.0, cp.inner_emission_floor);
    if (floor0 <= 0.0) return 0.0;
    const double span = std::max(r_out - r_in, 1.0e-12);
    const double width = std::max(1.0e-6, cp.inner_emission_floor_width * span);
    const double t = std::max(0.0, (r - r_in) / width);
    // Smoothly fades the floor away from ISCO to avoid a hard luminous ring.
    return floor0 * std::exp(-t * t);
}

static double disk_flux_reference(double r_isco, double r_disk_out, double a,
                                  const ColorParams& cp) {
    if (cp.radial_profile == DiskRadialProfile::PHYSICAL_NT) {
        const double rin = std::max(1.0e-6, r_isco * 1.0005);
        const double rout = std::max(r_disk_out, rin + 1.0e-3);
        const int n = 1024;
        double peak = 0.0;
        for (int i = 0; i < n; ++i) {
            const double t = (double(i) + 0.5) / double(n);
            const double r = rin + (rout - rin) * t;
            const double f = disk_flux_raw(r, r_isco, a, cp);
            if (f > 0.0 && std::isfinite(f) && f > peak) peak = f;
        }
        if (std::isfinite(peak) && peak > 0.0) return peak;
        const double fallback = disk_flux_raw(std::max(rin, 1.5 * r_isco), r_isco, a, cp);
        return (std::isfinite(fallback) && fallback > 0.0) ? fallback : 1.0;
    }

    const double r_peak = 3.0 * r_isco;
    const double pt_peak = page_thorne_flux_raw(r_peak, r_isco, cp);
    return (std::isfinite(pt_peak) && pt_peak > 0.0) ? pt_peak : 1.0;
}

// Reinhard tonemapping with variable gamma
static double tonemap(double x, double exposure, double gamma) {
    x = x * exposure;
    x = x / (1.0 + x);
    return std::pow(clamp(x, 0.0, 1.0), 1.0/gamma);
}

// HSV → RGB (all values in [0,1])
static RGB hsv_to_rgb(double h, double s, double v) {
    h = h - std::floor(h);  // wrap to [0,1)
    const double hh = h * 6.0;
    const int    hi = (int)hh;
    const double f  = hh - hi;
    const double p  = v * (1.0 - s);
    const double q  = v * (1.0 - s * f);
    const double t  = v * (1.0 - s * (1.0 - f));
    double rr, gg, bb;
    switch (hi % 6) {
        case 0: rr=v; gg=t; bb=p; break;
        case 1: rr=q; gg=v; bb=p; break;
        case 2: rr=p; gg=v; bb=t; break;
        case 3: rr=p; gg=q; bb=v; break;
        case 4: rr=t; gg=p; bb=v; break;
        default:rr=v; gg=p; bb=q; break;
    }
    return {(uint8_t)(rr*255+0.5), (uint8_t)(gg*255+0.5), (uint8_t)(bb*255+0.5)};
}

// Deterministic per-cell hash in [0,1) using a sin-based scramble
static double cell_hash(int cell_id, double salt_a, double salt_b) {
    double v = std::sin((double)cell_id * salt_a + salt_b) * 43758.5453;
    return v - std::floor(v);
}

// Returns true when the local segmented Interstellar disk cell blend is effectively
// transparent (no opaque contribution). Used in Phase 1 to let rays pass through
// those tiles and continue to background/horizon events.
static bool stratified_disk_tile_transparent(double r, double phi,
                                             double r_in, double r_out,
                                             const ColorParams& cp) {
    const int    NR    = cp.disk_rings   > 0 ? cp.disk_rings   : 1;
    const int    NS    = cp.disk_sectors > 0 ? cp.disk_sectors : 1;
    const double sigma = cp.disk_sigma   > 0 ? cp.disk_sigma   : 0.5;
    const double span_r = std::max(r_out - r_in, 1e-12);

    const double r_norm   = clamp((r - r_in) / span_r, 0.0, 1.0);
    const double phi_wrap = std::fmod(phi, 2.0 * M_PI);
    const double phi_norm = (phi_wrap < 0.0 ? phi_wrap + 2.0 * M_PI : phi_wrap) / (2.0 * M_PI);
    const double ring_pos = r_norm * NR;
    const double sec_pos  = phi_norm * NS;

    const double inv2sig2 = 1.0 / (2.0 * sigma * sigma);
    const int    SEARCH   = (int)std::ceil(3.0 * sigma) + 1;

    double w_total = 0.0;
    double w_opaque = 0.0;

    for (int ri = -SEARCH; ri <= NR + SEARCH; ++ri) {
        const double dr = ring_pos - (double)ri;
        if (dr * dr * inv2sig2 > 16.0) continue;
        const int ri_clamped = ri < 0 ? 0 : ri >= NR ? NR - 1 : ri;

        for (int si_raw = -SEARCH; si_raw <= NS + SEARCH; ++si_raw) {
            double ds = sec_pos - (double)si_raw;
            ds = ds - std::floor((ds + NS * 0.5) / NS) * NS;
            const double dist2 = dr * dr + ds * ds;
            const double w = std::exp(-dist2 * inv2sig2);
            if (w < 1e-6) continue;

            const int si_clamped = ((si_raw % NS) + NS) % NS;
            const int cell_id = ri_clamped * NS + si_clamped;
            const double h3 = cell_hash(cell_id, 419.2, 371.9);

            w_total += w;
            if (h3 >= 0.30) w_opaque += w;
        }
    }

    if (w_total < 1e-9) return true;
    const double opacity = w_opaque / w_total;
    return opacity < 1e-3;
}

// Stratified disk coloring (legacy "interstellar"):
// n_rings concentric bands × n_sectors azimuthal slices, each assigned
// a deterministic warm color (red/orange/yellow, some transparent).
// Colors are Gaussian-blended across cell boundaries so there are no hard edges.
// Physical modulation (redshift, selected radial emissivity profile, magnification) applied on top.
static RGB disk_colour_stratified(double r, double phi,
                                  double red, double magnif,
                                  double r_in, double r_out,
                                  double M, double a, double r_isco,
                                  const ColorParams& cp,
                                  double flux_ref) {
    (void)M;
    const int    NR    = cp.disk_rings   > 0 ? cp.disk_rings   : 1;
    const int    NS    = cp.disk_sectors > 0 ? cp.disk_sectors : 1;
    const double sigma = cp.disk_sigma   > 0 ? cp.disk_sigma   : 0.5;

    // Normalize position to cell-index space
    const double r_norm   = clamp((r - r_in) / (r_out - r_in), 0.0, 1.0);
    const double phi_norm = (phi - std::floor(phi / (2.0*M_PI)) * (2.0*M_PI)) / (2.0*M_PI); // [0,1)
    const double ring_pos = r_norm * NR;
    const double sec_pos  = phi_norm * NS;

    // Gaussian accumulation over nearby cells
    double acc_r = 0.0, acc_g = 0.0, acc_b = 0.0, w_total = 0.0;
    const double inv2sig2 = 1.0 / (2.0 * sigma * sigma);
    const int    SEARCH   = (int)std::ceil(3.0 * sigma) + 1;

    for (int ri = -SEARCH; ri <= NR + SEARCH; ++ri) {
        const double dr = ring_pos - (double)ri;
        if (dr * dr * inv2sig2 > 16.0) continue;  // early cull
        const int ri_clamped = ri < 0 ? 0 : ri >= NR ? NR-1 : ri;

        for (int si_raw = -SEARCH; si_raw <= NS + SEARCH; ++si_raw) {
            // Sector distance with wrap-around
            double ds = sec_pos - (double)si_raw;
            // Wrap ds to [-NS/2, NS/2]
            ds = ds - std::floor((ds + NS * 0.5) / NS) * NS;
            const double dist2 = dr * dr + ds * ds;
            const double w = std::exp(-dist2 * inv2sig2);
            if (w < 1e-4) continue;

            const int si_clamped = ((si_raw % NS) + NS) % NS;
            const int cell_id = ri_clamped * NS + si_clamped;

            // Three independent hashes per cell
            const double h1 = cell_hash(cell_id, 127.1, 311.7);
            const double h2 = cell_hash(cell_id, 269.5, 183.3);
            const double h3 = cell_hash(cell_id, 419.2, 371.9);

            // ~30% of cells are transparent (gaps in the disk)
            if (h3 < 0.30) continue;

            // Warm accretion palette: hue in [0, 0.14] → deep red through yellow
            const double hue = std::fmod(h1 * 0.14 + cp.disk_hue_offset, 1.0);
            const double sat = 0.70 + h2 * 0.30;
            const double val = 0.50 + h1 * 0.50;

            const RGB c = hsv_to_rgb(hue, sat, val);
            acc_r += w * c.r;
            acc_g += w * c.g;
            acc_b += w * c.b;
            w_total += w;
        }
    }

    if (w_total < 1e-6) return {0, 0, 0};  // transparent region

    // Physical intensity modulation (same as blackbody path)
    const double flux = disk_flux_raw(r, r_isco, a, cp);
    const double safe_ref = std::max(1.0e-12, flux_ref);
    double I = flux / safe_ref;
    I = std::max(I, inner_emission_floor_value(r, r_isco, r_out, cp));
    I *= doppler_luma_scale(red, cp);
    I *= magnif_luma_scale(magnif, cp);
    I *= std::max(0.0, cp.disk_brightness);

    const double norm = w_total * 255.0;
    return {
        (uint8_t)(tonemap(acc_r / norm * I, cp.exposure, cp.gamma) * 255),
        (uint8_t)(tonemap(acc_g / norm * I, cp.exposure, cp.gamma) * 255),
        (uint8_t)(tonemap(acc_b / norm * I, cp.exposure, cp.gamma) * 255)
    };
}

static double smoothstep_range(double edge0, double edge1, double x) {
    if (edge1 <= edge0) return (x >= edge1) ? 1.0 : 0.0;
    const double t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

static double fract01(double x) {
    return x - std::floor(x);
}

static double hash2d(double x, double y, double seed) {
    return fract01(std::sin(x * 127.1 + y * 311.7 + seed) * 43758.5453123);
}

static double value_noise2d(double x, double y, double seed) {
    const double ix = std::floor(x);
    const double iy = std::floor(y);
    const double fx = x - ix;
    const double fy = y - iy;

    const double ux = fx * fx * (3.0 - 2.0 * fx);
    const double uy = fy * fy * (3.0 - 2.0 * fy);

    const double n00 = hash2d(ix + 0.0, iy + 0.0, seed);
    const double n10 = hash2d(ix + 1.0, iy + 0.0, seed);
    const double n01 = hash2d(ix + 0.0, iy + 1.0, seed);
    const double n11 = hash2d(ix + 1.0, iy + 1.0, seed);

    const double nx0 = n00 + (n10 - n00) * ux;
    const double nx1 = n01 + (n11 - n01) * ux;
    return nx0 + (nx1 - nx0) * uy;
}

// Tileable variants: the lattice index is wrapped modulo an integer period, so
// the noise is exactly periodic in that axis. Needed because the disk textures
// are functions of the azimuth, which has a branch cut: phi_disk jumps by 2*pi
// down one column of the image (x = 637 in a 1280-wide frame at phi_obs = 0),
// and untiled lattice noise reads a completely unrelated value across it. That
// was the vertical seam through the disk.
static double value_noise2d_tiled(double x, double y, double seed,
                                  int per_x, int per_y) {
    const double ix = std::floor(x), iy = std::floor(y);
    const double fx = x - ix,        fy = y - iy;
    const double ux = fx * fx * (3.0 - 2.0 * fx);
    const double uy = fy * fy * (3.0 - 2.0 * fy);

    auto wrap = [](double v, int per) {
        if (per <= 0) return v;
        const double p = double(per);
        double m = std::fmod(v, p);
        if (m < 0.0) m += p;
        return m;
    };
    auto H = [&](double gx, double gy) {
        return hash2d(wrap(gx, per_x), wrap(gy, per_y), seed);
    };

    const double n00 = H(ix, iy),       n10 = H(ix + 1.0, iy);
    const double n01 = H(ix, iy + 1.0), n11 = H(ix + 1.0, iy + 1.0);
    const double nx0 = n00 + (n10 - n00) * ux;
    const double nx1 = n01 + (n11 - n01) * ux;
    return nx0 + (nx1 - nx0) * uy;
}

// Each octave doubles the frequency, so its period in lattice cells doubles too.
static double fbm2d_tiled(double x, double y, double seed, int octaves,
                          int per_x, int per_y) {
    double sum = 0.0, amp = 0.5, freq = 1.0, norm = 0.0;
    for (int i = 0; i < octaves; ++i) {
        const int px = (per_x > 0) ? int(std::lround(per_x * freq)) : 0;
        const int py = (per_y > 0) ? int(std::lround(per_y * freq)) : 0;
        sum += amp * value_noise2d_tiled(x * freq, y * freq, seed + 17.0 * i, px, py);
        norm += amp;
        freq *= 2.0;
        amp *= 0.5;
    }
    return (norm > 0.0) ? (sum / norm) : 0.0;
}

static double fbm2d(double x, double y, double seed, int octaves = 5) {
    double sum = 0.0;
    double amp = 0.5;
    double freq = 1.0;
    double norm = 0.0;
    for (int i = 0; i < octaves; ++i) {
        sum += amp * value_noise2d(x * freq, y * freq, seed + 17.0 * i);
        norm += amp;
        freq *= 2.0;
        amp *= 0.5;
    }
    return (norm > 0.0) ? (sum / norm) : 0.0;
}

static double interstellar_disk_soft_mask(double r, double phi,
                                          double r_in, double r_out,
                                          const ColorParams& cp) {
    const double phi_norm = fract01(phi / (2.0 * M_PI));
    const double t = cp.interstellar_time;

    // phi_norm already wraps into [0,1); the noise must wrap with it, or the
    // disk's inner and outer edges step at the branch cut.
    const double edgeInNoise = 2.0 * fbm2d_tiled(phi_norm * 3.0, t * 0.10, 11.0, 4, 3, 0) - 1.0;
    const double edgeOutNoise = 2.0 * fbm2d_tiled(phi_norm * 2.0, t * 0.07, 29.0, 4, 2, 0) - 1.0;

    const double rin_local = r_in * (1.0 + 0.03 * edgeInNoise);
    const double rout_local = r_out * (1.0 + 0.06 * edgeOutNoise);
    const double rout_safe = std::max(rout_local, rin_local + 1e-4);

    const double softness_in = std::max(1e-6, cp.interstellar_softness_in_scale * r_in);
    const double softness_out = std::max(1e-6, cp.interstellar_softness_out_scale * r_out);

    const double inner_mask = smoothstep_range(rin_local, rin_local + softness_in, r);
    const double outer_mask = 1.0 - smoothstep_range(rout_safe - softness_out, rout_safe, r);
    return clamp(inner_mask * outer_mask, 0.0, 1.0);
}

// Interstellar cinematic disk coloring (artist-driven):
// annulus mask + radial emissivity + differential rotation advection +
// stretched turbulence + logarithmic radial bands + warm HDR palette.
static RGB disk_colour_interstellar(double r, double phi,
                                    double red, double magnif,
                                    double r_in, double r_out,
                                    double M, double a, double r_isco,
                                    const ColorParams& cp,
                                    double flux_ref) {
    (void)M;

    const double x = clamp((r - r_in) / std::max(r_out - r_in, 1e-12), 0.0, 1.0);
    const double t = cp.interstellar_time;

    const double omega0 = cp.interstellar_omega0;
    const double omega = omega0 * std::pow(std::max(r / std::max(r_in, 1e-12), 1e-6), -1.5);
    const double phit = phi - omega * t;

    const double mask = interstellar_disk_soft_mask(r, phi, r_in, r_out, cp);
    if (mask <= 1e-6) return {0, 0, 0};

    const double p = std::max(0.1, cp.interstellar_p);
    const double inner_falloff = std::max(1e-6, cp.interstellar_inner_falloff_scale * r_in);
    // Physical: Novikov-Thorne flux normalised to its peak, the same quantity the
    // blackbody and stratified paths use. Artistic: bare power law, kept as an option.
    const double radial = cp.interstellar_physical_profile
        ? disk_flux_raw(r, r_isco, a, cp) / std::max(1.0e-12, flux_ref)
        : std::pow(std::max(r / std::max(r_in, 1e-12), 1e-6), -p);
    const double inner_glow = cp.interstellar_inner_glow
        ? std::exp(-(r - r_in) / inner_falloff)
        : 1.0;
    const double profile = radial * inner_glow;

    // The angular coordinate is wrapped and scaled to a whole number of noise
    // cells per revolution: 2*pi*8 = 50.3 becomes 50 and 2*pi*25 = 157.1 becomes
    // 157, so the texture tiles with essentially the same visual scale as before.
    const double phit_w = fract01(phit / (2.0 * M_PI));
    const double n1 = 2.0 * fbm2d_tiled(r * 1.2, phit_w * 50.0, 41.0, 5, 0, 50) - 1.0;
    const double n2 = 2.0 * fbm2d_tiled(r * 5.0, phit_w * 157.0 + 2.0 * n1, 73.0, 4, 0, 157) - 1.0;
    const double turb_str = cp.interstellar_turbulence_strength;
    const double turbulence = 0.75 + turb_str * (0.35 * n1 + 0.12 * n2);

    const double band_strength = cp.interstellar_band_strength;
    const double band_frequency = cp.interstellar_band_frequency;
    const double band_warp = cp.interstellar_band_warp;
    const double bands = 1.0 + band_strength *
        std::sin(band_frequency * std::log(std::max(r / std::max(r_in, 1e-12), 1e-6)) + band_warp * n1);

    // Thermal palette: inner warm-white -> orange -> deep red/brown.
    const double inner_r = 1.0, inner_g = 0.82, inner_b = 0.45;
    const double mid_r   = 1.0, mid_g   = 0.32, mid_b   = 0.06;
    const double out_r   = clamp(cp.interstellar_outer_r, 0.0, 1.0),
                 out_g   = clamp(cp.interstellar_outer_g, 0.0, 1.0),
                 out_b   = clamp(cp.interstellar_outer_b, 0.0, 1.0);

    const double t1 = smoothstep_range(0.08, 0.38, x);
    double cr = inner_r + (mid_r - inner_r) * t1;
    double cg = inner_g + (mid_g - inner_g) * t1;
    double cb = inner_b + (mid_b - inner_b) * t1;
    const double t2 = smoothstep_range(0.35, 1.0, x);
    cr += (out_r - cr) * t2;
    cg += (out_g - cg) * t2;
    cb += (out_b - cb) * t2;

    const double lens = magnif_luma_scale(magnif, cp);

    // Gaussian taper from inner edge to suppress bright ring at r_in.
    const double inner_taper_sigma = std::max(1e-6, cp.interstellar_inner_taper_scale * r_in);
    const double dr_in = std::max(0.0, r - r_in);
    const double inner_taper = 1.0 - std::exp(-0.5 * (dr_in / inner_taper_sigma) * (dr_in / inner_taper_sigma));

    double intensity = mask * profile * turbulence * bands * inner_taper;
    // doppler_luma_scale_inverted already mirrors red → pass raw red, not 2-red.
    intensity *= doppler_luma_scale(red, cp);
    intensity *= lens;
    intensity *= cp.interstellar_hdr_intensity;
    intensity *= std::max(0.0, cp.disk_brightness);
    if (cp.radial_ring_taper) {
        const double r0  = cp.radial_ring_taper_use_isco ? r_isco : cp.radial_ring_taper_r0;
        const double sig = std::max(1e-6, cp.radial_ring_taper_sigma);
        intensity *= std::exp(-std::abs(r - r0) / sig);
    }
    if (cp.zero_torque_taper && r_isco > 0.0 && r > r_isco) {
        intensity *= 1.0 - std::sqrt(r_isco / r);
    }
    intensity = std::max(intensity, 0.0);

    return {
        (uint8_t)(tonemap(cr * intensity, cp.exposure, cp.gamma) * 255),
        (uint8_t)(tonemap(cg * intensity, cp.exposure, cp.gamma) * 255),
        (uint8_t)(tonemap(cb * intensity, cp.exposure, cp.gamma) * 255)
    };
}

// NASA Goddard-style accretion disk (Schnittman 2019 visualization).
// Orange-amber palette, no Doppler (artistic symmetric choice), strong bands.
static RGB disk_colour_interstellar_nasa(double r, double phi,
                                         double red, double magnif,
                                         double r_in, double r_out,
                                         double r_isco,
                                         const ColorParams& cp) {
    const double x = clamp((r - r_in) / std::max(r_out - r_in, 1e-12), 0.0, 1.0);
    const double t = cp.interstellar_time;

    const double mask = interstellar_disk_soft_mask(r, phi, r_in, r_out, cp);
    if (mask <= 1e-6) return {0, 0, 0};

    // Keplerian differential rotation advection (same as INTERSTELLAR).
    const double omega = cp.interstellar_omega0 *
        std::pow(std::max(r / std::max(r_in, 1e-12), 1e-6), -1.5);
    const double phit = phi - omega * t;

    // Power-law only: no exponential inner_glow so the outer disk stays visible.
    // The NASA render shows significant emission out to large r.
    const double p = std::max(0.1, cp.interstellar_p);
    const double profile = std::pow(std::max(r / std::max(r_in, 1e-12), 1e-6), -p);

    // Low turbulence: NASA viz has smooth, regular banding without swirling.
    // Same azimuthal wrap as the Interstellar palette: 2*pi*5 = 31.4 cells per
    // revolution becomes 31, so the texture closes on itself at the phi branch
    // cut. v0.2.29 tiled the shared soft mask and the Interstellar turbulence but
    // missed this one, which is why the seam survived in NASA frames.
    const double phit_w = fract01(phit / (2.0 * M_PI));
    const double n1 = 2.0 * fbm2d_tiled(r * 0.7, phit_w * 31.0, 41.0, 3, 0, 31) - 1.0;
    const double turb_str = cp.interstellar_turbulence_strength;
    const double turbulence = 0.88 + turb_str * 0.18 * n1;

    // Strong, regular logarithmic bands – dominant visual of NASA render.
    const double band_strength  = cp.interstellar_band_strength;
    const double band_frequency = cp.interstellar_band_frequency;
    const double band_warp      = cp.interstellar_band_warp;
    const double bands = 1.0 + band_strength *
        std::sin(band_frequency * std::log(std::max(r / std::max(r_in, 1e-12), 1e-6))
                 + band_warp * n1);

    // NASA orange-amber palette: warm-orange core → vivid orange → deep orange-red.
    // Match Schnittman (2019) NASA Goddard visualization: saturated, no white inner.
    const double inner_r = 1.00, inner_g = 0.30, inner_b = 0.02;  // vivid orange
    const double mid_r   = 0.90, mid_g   = 0.17, mid_b   = 0.008; // deep orange
    const double out_r   = 0.55, out_g   = 0.09, out_b   = 0.003; // burnt orange-red

    const double t1 = smoothstep_range(0.04, 0.25, x);
    double cr = inner_r + (mid_r - inner_r) * t1;
    double cg = inner_g + (mid_g - inner_g) * t1;
    double cb = inner_b + (mid_b - inner_b) * t1;
    const double t2 = smoothstep_range(0.22, 0.80, x);
    cr += (out_r - cr) * t2;
    cg += (out_g - cg) * t2;
    cb += (out_b - cb) * t2;

    const double lens = magnif_luma_scale(magnif, cp);

    const double inner_taper_sigma = std::max(1e-6, cp.interstellar_inner_taper_scale * r_in);
    const double dr_in = std::max(0.0, r - r_in);
    const double inner_taper = 1.0 - std::exp(-0.5 * (dr_in / inner_taper_sigma) * (dr_in / inner_taper_sigma));

    // Relativistic beaming, on unless explicitly switched off. This palette used
    // to drop it unconditionally -- the factor was not even a parameter of the
    // function, so --doppler silently did nothing here and the disk came out
    // uniformly bright with no approaching side. A symmetric disk is an artistic
    // choice, so it is now --disk-nasa-symmetric rather than the default.
    double intensity = mask * profile * turbulence * bands * inner_taper;
    intensity *= doppler_luma_scale(red, cp);
    intensity *= std::max(0.0, cp.nasa_gain);
    intensity *= lens;
    intensity *= cp.interstellar_hdr_intensity;
    intensity *= std::max(0.0, cp.disk_brightness);
    if (cp.radial_ring_taper) {
        const double r0  = cp.radial_ring_taper_use_isco ? r_isco : cp.radial_ring_taper_r0;
        const double sig = std::max(1e-6, cp.radial_ring_taper_sigma);
        intensity *= std::exp(-std::abs(r - r0) / sig);
    }
    if (cp.zero_torque_taper && r_isco > 0.0 && r > r_isco) {
        intensity *= 1.0 - std::sqrt(r_isco / r);
    }
    intensity = std::max(intensity, 0.0);

    return {
        (uint8_t)(tonemap(cr * intensity, cp.exposure, cp.gamma) * 255),
        (uint8_t)(tonemap(cg * intensity, cp.exposure, cp.gamma) * 255),
        (uint8_t)(tonemap(cb * intensity, cp.exposure, cp.gamma) * 255)
    };
}

static RGB disk_colour(double r, double red, double magnif,
                       double M, double a, double r_isco, double r_disk_out,
                       const ColorParams& cp,
                       double flux_ref) {
    // Physical g: >1 on approaching side → hotter temperature there, and no floor
    // on the cold side. blackbody() already clamps T to [800, 40000] K, so a very
    // small g simply lands on the deepest red the ramp has rather than producing
    // a degenerate colour. Measured at a=0.998, theta=82, r_obs=60: g reaches
    // 0.0295 and 4.72% of disk pixels sat below the old 0.2 floor (7.29% at
    // theta=88, r_obs=30; only 0.34% at a=0.9). Where it bit, at r ~ 1.6M, it
    // turned (255,0,0) into (255,119,13).
    const double red_phys = cp.temp_redshift_clamp
        ? clamp(red, cp.temp_redshift_floor, 5.0)
        : std::max(red, 0.0);
    double T = 6500.0 * cp.temp_scale * std::sqrt(6.0*M/r) * red_phys;
    const double flux = disk_flux_raw(r, r_isco, a, cp);
    const double safe_ref = std::max(1.0e-12, flux_ref);
    double I = flux / safe_ref;
    I = std::max(I, inner_emission_floor_value(r, r_isco, r_disk_out, cp));
    I *= doppler_luma_scale(red, cp);
    I *= magnif_luma_scale(magnif, cp);
    I *= std::max(0.0, cp.disk_brightness);
    if (cp.planck_emission && T > 0.0) {
        // Weight intensity by Planck B(T,λ=550nm) / B(T_ref,λ=550nm).
        // hc/kλ for λ=550nm ≈ 26145 K.  Hotter regions brightened, cooler suppressed.
        const double hc_kl = 26145.0;
        auto b_planck = [&](double t) -> double {
            const double x = hc_kl / std::max(t, 200.0);
            return 1.0 / std::expm1(std::min(x, 700.0));
        };
        const double T_ref_planck = std::max(200.0, 6500.0 * cp.temp_scale);
        const double b_ref = b_planck(T_ref_planck);
        if (b_ref > 1.0e-30)
            I *= std::max(0.0, b_planck(T) / b_ref);
    }
    if (cp.radial_ring_taper) {
        const double r0  = cp.radial_ring_taper_use_isco ? r_isco : cp.radial_ring_taper_r0;
        const double sig = std::max(1e-6, cp.radial_ring_taper_sigma);
        I *= std::exp(-std::abs(r - r0) / sig);
    }
    if (cp.zero_torque_taper && r_isco > 0.0 && r > r_isco) {
        I *= 1.0 - std::sqrt(r_isco / r);
    }
    auto c = blackbody(T);
    return {(uint8_t)(tonemap(c.r/255.0*I, cp.exposure, cp.gamma)*255),
            (uint8_t)(tonemap(c.g/255.0*I, cp.exposure, cp.gamma)*255),
            (uint8_t)(tonemap(c.b/255.0*I, cp.exposure, cp.gamma)*255)};
}

// Truncated-Gaussian sample pattern over the pixel's footprint on the disk.
//
// DNGR §2.2(iii) adds up the emission from within the whole ellipse the bundle
// covers, and A.3.1 modulates it with a truncated Gaussian, both to suppress
// moire on extended sources and to stop a feature flickering as it crosses the
// pixel grid. Sampling only the centre, which is what a single ray does, is the
// thing that produces the aliasing.
//
// The pattern is a centre sample plus concentric rings in footprint coordinates
// (s, t) in [-1, 1]^2, mapped to the disk by the two edge vectors. Weights are
// exp(-d^2 / 2 sigma^2) truncated at the footprint edge, so a sample never
// reaches beyond the region the pixel actually sees.
