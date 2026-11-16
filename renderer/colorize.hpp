#pragma once
// Phase 2: GeoPixel buffer to RGB image.
//
// Part of the single-translation-unit renderer included by main.cpp;
// see docs/ARCHITETTURA.md for how the modules fit together.

#include "palettes.hpp"
#include "background.hpp"

static const std::vector<FootprintSample>& footprint_pattern(int rings, double sigma) {
    static std::vector<FootprintSample> cache;
    static int cached_rings = -1;
    static double cached_sigma = -1.0;
    if (rings == cached_rings && sigma == cached_sigma) return cache;

    cache.clear();
    const int n_rings = std::max(0, rings);
    const double sig = std::max(1e-3, sigma);
    cache.push_back({0.0, 0.0, 1.0});
    for (int ring = 1; ring <= n_rings; ++ring) {
        const double rad = double(ring) / double(n_rings);
        const int n_pts = 6 * ring;                  // denser rings further out
        const double w = std::exp(-0.5 * (rad*rad) / (sig*sig));
        for (int k = 0; k < n_pts; ++k) {
            const double ang = 2.0 * M_PI * (double(k) + 0.5*double(ring % 2))
                             / double(n_pts);
            cache.push_back({rad * std::cos(ang), rad * std::sin(ang), w});
        }
    }
    double tot = 0.0;
    for (const FootprintSample& fs : cache) tot += fs.w;
    if (tot > 0.0) for (FootprintSample& fs : cache) fs.w /= tot;
    cached_rings = rings;
    cached_sigma = sigma;
    return cache;
}

// Phase 2: GeoPixel buffer → RGB image
static std::vector<RGB> colorize_buffer(
    const std::vector<GeoPixel>& geo, int W, int H,
    const ColorParams& cp,
    const BackgroundImage& bg,
    double M_bh, double a_bh, double r_isco,
    double r_disk_in, double r_disk_out,
    bool debug_elliptic = false,
    const BackgroundImage* bg_b = nullptr)  // Universe B background (wormhole)
{
    std::vector<RGB> image(W*H, {0,0,0});
    const double flux_ref = disk_flux_reference(r_isco, r_disk_out, a_bh, cp);

    if (debug_elliptic) {
        // Color by elliptic solver tag stored in _pad[0]:
        //   NONE(0)=green   INIT_SEPARABLE_CONSTS(1)=white   THETA_CROSSING_SOLVE(2)=cyan
        //   RADIAL_MAP_INIT(3)=red   DIRECT_RADIAL_INVALID(4)=yellow   DIRECT_OUTSIDE_DISK(5)=blue
        static const RGB debug_colors[] = {
            {  0, 220,   0},  // 0 NONE           = elliptic success  → green
            {255, 255, 255},  // 1 INIT_SEP_CONSTS                    → white
            {  0, 220, 220},  // 2 THETA_CROSSING                     → cyan
            {220,   0,   0},  // 3 RADIAL_MAP_INIT                    → red  (dominant)
            {220, 220,   0},  // 4 DIRECT_RADIAL_INVALID              → yellow
            { 60,  60, 220},  // 5 DIRECT_OUTSIDE_DISK                → blue
        };
        for (int i = 0; i < W*H; ++i) {
            const uint8_t tag = geo[i]._pad[0];
            image[i] = (tag < 6) ? debug_colors[tag] : RGB{128, 0, 128};
        }
        return image;
    }

    // For an escaping ray the footprint fields hold (dtheta, dphi) on the
    // celestial sphere, so the background can be averaged over the area the
    // pixel really sees instead of point-sampled at its centre.
    auto sample_sky = [&](const BackgroundImage& img, const GeoPixel& p) -> RGB {
        const bool has_fp =
            cp.bundle_filter &&
            (std::abs(p.sky_dth_a) + std::abs(p.sky_dph_a) +
             std::abs(p.sky_dth_b) + std::abs(p.sky_dph_b)) > 0.0f &&
            std::isfinite(p.sky_dth_a) && std::isfinite(p.sky_dph_a) &&
            std::isfinite(p.sky_dth_b) && std::isfinite(p.sky_dph_b);
        if (!has_fp) return img.sample(p.theta_esc, p.phi_esc);
        return img.sample_footprint(p.theta_esc, p.phi_esc,
                                    p.sky_dth_a, p.sky_dph_a,
                                    p.sky_dth_b, p.sky_dph_b,
                                    footprint_pattern(cp.bundle_filter_rings,
                                                      cp.bundle_filter_sigma));
    };

    for (int i = 0; i < W*H; ++i) {
        const GeoPixel& p = geo[i];
        if (p.outcome == 1) {
            RGB disk_col{};
            const double magnif_n = double(p.magnif);
            // Evaluates the chosen palette at one point of the disk. Sampling the
            // footprint means calling this several times; the palettes are pure
            // functions of (r, phi) plus per-pixel scalars, so the filter lives
            // out here and none of them had to change.
            auto shade_at = [&](double rr, double pp) -> RGB {
                if (cp.palette == DiskPalette::STRATIFIED)
                    return disk_colour_stratified(rr, pp, p.redshift, magnif_n,
                                                  r_disk_in, r_disk_out,
                                                  M_bh, a_bh, r_isco, cp, flux_ref);
                if (cp.palette == DiskPalette::INTERSTELLAR)
                    return disk_colour_interstellar(rr, pp, p.redshift, magnif_n,
                                                    r_disk_in, r_disk_out,
                                                    M_bh, a_bh, r_isco, cp, flux_ref);
                if (cp.palette == DiskPalette::INTERSTELLAR_NASA)
                    return disk_colour_interstellar_nasa(rr, pp, p.redshift, magnif_n,
                                                         r_disk_in, r_disk_out,
                                                         r_isco, cp);
                return disk_colour(rr, p.redshift, magnif_n,
                                   M_bh, a_bh, r_isco, r_disk_out, cp, flux_ref);
            };

            const bool has_footprint =
                cp.bundle_filter &&
                (std::abs(p.fp_dr_a) + std::abs(p.fp_dr_b) +
                 std::abs(p.fp_dphi_a) + std::abs(p.fp_dphi_b)) > 0.0f &&
                std::isfinite(p.fp_dr_a) && std::isfinite(p.fp_dphi_a) &&
                std::isfinite(p.fp_dr_b) && std::isfinite(p.fp_dphi_b);

            if (!has_footprint) {
                disk_col = shade_at(p.r, p.phi_disk);
            } else {
                // Half the edge vectors: the footprint spans one pixel, so the
                // offsets from its centre reach half a pixel each way.
                const double ha_r = 0.5*double(p.fp_dr_a),  ha_p = 0.5*double(p.fp_dphi_a);
                const double hb_r = 0.5*double(p.fp_dr_b),  hb_p = 0.5*double(p.fp_dphi_b);
                const std::vector<FootprintSample>& pat =
                    footprint_pattern(cp.bundle_filter_rings, cp.bundle_filter_sigma);
                double acc_r = 0.0, acc_g = 0.0, acc_b = 0.0;
                for (const FootprintSample& fs : pat) {
                    const double rr = p.r        + fs.s*ha_r + fs.t*hb_r;
                    const double pp = p.phi_disk + fs.s*ha_p + fs.t*hb_p;
                    const RGB c = shade_at(rr, pp);
                    acc_r += fs.w * c.r; acc_g += fs.w * c.g; acc_b += fs.w * c.b;
                }
                disk_col = { (uint8_t)clamp(acc_r + 0.5, 0.0, 255.0),
                             (uint8_t)clamp(acc_g + 0.5, 0.0, 255.0),
                             (uint8_t)clamp(acc_b + 0.5, 0.0, 255.0) };
            }

            const double cov = (cp.bundle_filter && std::isfinite(p.coverage))
                             ? clamp(double(p.coverage), 0.0, 1.0) : 1.0;
            const double alpha = clamp(cp.disk_opacity, 0.0, 1.0) * cov;
            if (alpha >= 1.0 - 1e-9) {
                image[i] = disk_col;
            } else {
                RGB behind{0,0,0};
                const uint8_t behind_code = p._pad[1];
                if (behind_code == 0) {
                    if (!bg.px.empty()) behind = sample_sky(bg, p);
                } else if (behind_code == 3) {
                    const BackgroundImage& bgB = (bg_b && !bg_b->px.empty()) ? *bg_b : bg;
                    if (!bgB.px.empty()) behind = sample_sky(bgB, p);
                }
                const double ia = 1.0 - alpha;
                image[i] = {
                    (uint8_t)clamp(alpha * disk_col.r + ia * behind.r, 0.0, 255.0),
                    (uint8_t)clamp(alpha * disk_col.g + ia * behind.g, 0.0, 255.0),
                    (uint8_t)clamp(alpha * disk_col.b + ia * behind.b, 0.0, 255.0),
                };
            }
        } else if (p.outcome == 3) {
            // Universe B — use secondary background if provided, else bg
            const BackgroundImage& bgB = (bg_b && !bg_b->px.empty()) ? *bg_b : bg;
            if (!bgB.px.empty())
                image[i] = sample_sky(bgB, p);
        } else if (p.outcome == 0 && !bg.px.empty())
            image[i] = sample_sky(bg, p);
    }
    return image;
}
