#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// ── Per-pixel geodesic result (Phase 1 output) ───────────────
struct GeoPixel {
    uint8_t outcome;    // 0 = escaped, 1 = disk_hit, 2 = horizon
    uint8_t _pad[3];    // _pad[0]: debug solver tag (EllipticFallbackReason) when --debug-elliptic
    float   r;          // BL radius at disk crossing (or final r)
    float   redshift;   // g = ν_obs/ν_em
    float   magnif;     // flux magnification (bundle mode; 1 in single-ray)
    float   phi_disk;   // BL azimuthal angle at disk crossing (0 if not disk hit)
    float   theta_esc;  // direction at escape (background lookup)
    float   phi_esc;
    // Footprint of the pixel on the disk, from the ray bundle: the two edge
    // vectors of the parallelogram the pixel maps onto, in (r, phi). Already
    // scaled by the pixel's angular size, so they are per-pixel, not per-radian.
    // Zero in single-ray mode, where the pixel has no measured extent.
    float   fp_dr_a, fp_dphi_a;
    float   fp_dr_b, fp_dphi_b;
    // Fraction of the pixel covered by the disk. 1 everywhere except where the
    // footprint straddles the disk's radial bounds; this is what turns the
    // binary hit/miss decision at the rim into a gradient.
    float   coverage;
    // Footprint on the celestial sphere, for filtering the background. A rim
    // pixel needs this as well as the disk footprint above: it shades the disk
    // over one and composites the sky over the other.
    float   sky_dth_a, sky_dph_a;
    float   sky_dth_b, sky_dph_b;
};
static_assert(sizeof(GeoPixel) == 64, "GeoPixel size mismatch");

// ── .kgeo file format ─────────────────────────────────────────
static const char   KGEO_MAGIC[4]  = {'K','G','E','O'};
// 4: split the sky footprint out of the disk footprint, so a partly covered rim
//    pixel can carry both.
// 3: added the coverage fraction.
// 2: added the disk footprint vectors to GeoPixel. Version 1 files are 28 bytes
// per record against 44 and cannot be read; the layout is not self-describing, so
// the version is the only guard. It was left at 1 through the v0.2.3 layout change,
// which is what made that drift silent.
static const uint32_t KGEO_VERSION = 4;

struct KGeoMeta {
    uint32_t W, H;
    double   M_bh, a_bh, Q_bh, Lam;
    double   r_isco, r_disk_in, r_disk_out;
    double   theta_obs, phi_obs, r_obs;
};


inline size_t checked_pixel_count(uint32_t width, uint32_t height) {
    // Renderer indexing and PNG row strides are signed ints. Check before any
    // multiplication or allocation, including files supplied to --color-only.
    const uint64_t count = uint64_t(width) * uint64_t(height);
    if (width == 0 || height == 0 || width > uint32_t(std::numeric_limits<int>::max()/3)
        || count > uint64_t(std::numeric_limits<int>::max())
        || count > std::numeric_limits<size_t>::max()/sizeof(GeoPixel))
        throw std::runtime_error("Invalid or oversized image dimensions");
    return size_t(count);
}

inline void save_kgeo(const char* path, const std::vector<GeoPixel>& geo,
                      const KGeoMeta& meta) {
    if (checked_pixel_count(meta.W, meta.H) != geo.size())
        throw std::runtime_error("KGEO dimensions do not match the pixel buffer");
    std::ofstream f(path, std::ios::binary);
    f.write(KGEO_MAGIC, 4);
    const uint32_t ver = KGEO_VERSION;
    f.write(reinterpret_cast<const char*>(&ver), sizeof(ver));
    f.write(reinterpret_cast<const char*>(&meta), sizeof(meta));
    f.write(reinterpret_cast<const char*>(geo.data()),
            std::streamsize(geo.size()*sizeof(GeoPixel)));
    f.close();
    if (!f) throw std::runtime_error(std::string("Cannot write KGEO file: ") + path);
}

inline bool load_kgeo(const char* path, std::vector<GeoPixel>& geo, KGeoMeta& meta) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const auto size = f.tellg();
    constexpr size_t header_size = 8 + sizeof(KGeoMeta);
    if (size < std::streamoff(header_size)) return false;
    f.seekg(0);
    char magic[4] = {};
    uint32_t version = 0;
    KGeoMeta candidate{};
    f.read(magic, sizeof(magic));
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    f.read(reinterpret_cast<char*>(&candidate), sizeof(candidate));
    if (!f || std::memcmp(magic, KGEO_MAGIC, 4) != 0 || version != KGEO_VERSION)
        return false;
    const double values[] = {candidate.M_bh, candidate.a_bh, candidate.Q_bh, candidate.Lam,
        candidate.r_isco, candidate.r_disk_in, candidate.r_disk_out,
        candidate.theta_obs, candidate.phi_obs, candidate.r_obs};
    for (double value : values) if (!std::isfinite(value)) return false;
    try {
        const size_t count = checked_pixel_count(candidate.W, candidate.H);
        const uint64_t expected = uint64_t(header_size) + uint64_t(count)*sizeof(GeoPixel);
        if (uint64_t(std::streamoff(size)) != expected) return false;
        std::vector<GeoPixel> pixels(count);
        if (!f.read(reinterpret_cast<char*>(pixels.data()), std::streamsize(count*sizeof(GeoPixel))))
            return false;
        geo.swap(pixels);
        meta = candidate;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}
