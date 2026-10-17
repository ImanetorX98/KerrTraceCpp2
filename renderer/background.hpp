#pragma once
// Celestial-sphere background image and footprint resampling.
//
// Part of the single-translation-unit renderer included by main.cpp;
// see docs/ARCHITETTURA.md for how the modules fit together.

#include "color_params.hpp"
#include "util.hpp"

// One sample of the pixel's footprint, in footprint coordinates (s, t) inside
// [-1,1]^2 with a truncated-Gaussian weight. Declared here because both the
// background sampler and the disk shading use the same pattern.
struct FootprintSample { double s, t, w; };

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

    // Averages the lookup over the beam's footprint on the celestial sphere.
    // The background is a resolved texture, so what the bundle gives it is a
    // resampling kernel: A.3.1 minimises moire on extended structures "by
    // adapting the resampling filter according to the shape of the beam". With
    // a single bilinear tap the star field aliases badly wherever the lensing
    // stretches or compresses the sky -- measured at twice the reference's
    // high-frequency energy, and 3181 of the 3546 pixels still off the 16-spp
    // reference after the disk work of P1.
    RGB sample_footprint(double theta, double phi,
                         double dth_a, double dph_a,
                         double dth_b, double dph_b,
                         const std::vector<FootprintSample>& pat) const {
        if (px.empty()) return {0,0,0};
        const double ha_t = 0.5*dth_a, ha_p = 0.5*dph_a;
        const double hb_t = 0.5*dth_b, hb_p = 0.5*dph_b;
        double ar = 0.0, ag = 0.0, ab = 0.0;
        for (const FootprintSample& fs : pat) {
            const RGB c = sample(theta + fs.s*ha_t + fs.t*hb_t,
                                 phi   + fs.s*ha_p + fs.t*hb_p);
            ar += fs.w * c.r; ag += fs.w * c.g; ab += fs.w * c.b;
        }
        return { (uint8_t)clamp(ar + 0.5, 0.0, 255.0),
                 (uint8_t)clamp(ag + 0.5, 0.0, 255.0),
                 (uint8_t)clamp(ab + 0.5, 0.0, 255.0) };
    }

    RGB sample(double theta, double phi) const {
        if (px.empty()) return {0,0,0};
        double pw = std::fmod(phi, 2.0*M_PI);
        if (pw < 0) pw += 2.0*M_PI;
        // Texel i covers [i, i+1)/w with its centre at (i + 0.5)/w: the convention
        // of the Metal sampler (repeat in u, clamp in v). Scaling by (w-1) instead
        // put CPU and GPU up to half a texel apart across the sky and left a
        // one-texel jump at phi = 2pi.
        const double uf = pw / (2.0*M_PI) * w - 0.5;
        const double vf = clamp(theta / M_PI, 0.0, 1.0) * h - 0.5;
        const double uf0 = std::floor(uf), vf0 = std::floor(vf);
        int u0 = int(uf0) % w; if (u0 < 0) u0 += w;
        const int u1 = (w > 1) ? ((u0 + 1) % w) : 0;
        const int v0 = int(vf0), v1 = v0 + 1;   // at() clamps v to the edge rows
        const double fu = uf - uf0, fv = vf - vf0;
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
