#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <string>
#include <utility>
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
// Version 5 layout (offsets in bytes, integers in the writer's byte order):
//    0  "KGEO"                 4  u32 version = 5
//    8  u32 byte-order tag     12 u32 header bytes (where the records start)
//   16  u32 record size        20 u32 KGeoMeta size
//   24  KGeoMeta              112 u32 provenance length, then provenance JSON
//   records (GeoPixel), W*H of them, from the header-bytes offset.
// The provenance is a flat JSON object, one "key": value per line: the record
// schema, and how the buffer was made (metric, camera, solver, the colour
// settings that decide which rays pass through the disk, code revision).
// Bump the version, or rename the field in the schema, whenever a field's
// meaning changes, even if the size does not.
static const char   KGEO_MAGIC[4]  = {'K','G','E','O'};
static const uint32_t KGEO_BYTE_ORDER_TAG = 0x01020304u;
// 5: self-describing header (byte order, record and meta sizes, schema) and the
//    provenance block. Version 4 files still load, with provenance incomplete.
// 4: split the sky footprint out of the disk footprint, so a partly covered rim
//    pixel can carry both.
// 3: added the coverage fraction.
// 2: added the disk footprint vectors to GeoPixel. Version 1 files are 28 bytes
// per record against 44 and cannot be read; the layout is not self-describing, so
// the version is the only guard. It was left at 1 through the v0.2.3 layout change,
// which is what made that drift silent.
static const uint32_t KGEO_VERSION = 5;

struct KGeoMeta {
    uint32_t W, H;
    double   M_bh, a_bh, Q_bh, Lam;
    double   r_isco, r_disk_in, r_disk_out;
    double   theta_obs, phi_obs, r_obs;
};


// GeoPixel as written to disk: name:type:offset:bytes per field. A reader
// compares it with its own; a moved, resized or renamed field refuses to load.
inline std::string kgeo_geopixel_schema() {
    struct F { const char* name; const char* type; size_t offset, bytes; };
#define KGEO_F(field, type) F{#field, type, offsetof(GeoPixel, field), sizeof(GeoPixel::field)}
    const F fields[] = {
        KGEO_F(outcome, "u1"), KGEO_F(_pad, "u1"), KGEO_F(r, "f4"), KGEO_F(redshift, "f4"),
        KGEO_F(magnif, "f4"), KGEO_F(phi_disk, "f4"), KGEO_F(theta_esc, "f4"), KGEO_F(phi_esc, "f4"),
        KGEO_F(fp_dr_a, "f4"), KGEO_F(fp_dphi_a, "f4"), KGEO_F(fp_dr_b, "f4"), KGEO_F(fp_dphi_b, "f4"),
        KGEO_F(coverage, "f4"), KGEO_F(sky_dth_a, "f4"), KGEO_F(sky_dph_a, "f4"),
        KGEO_F(sky_dth_b, "f4"), KGEO_F(sky_dph_b, "f4")};
#undef KGEO_F
    std::string out;
    size_t covered = 0;
    for (const F& f : fields) {
        if (!out.empty()) out += ',';
        out += std::string(f.name) + ':' + f.type + ':' + std::to_string(f.offset) + ':' + std::to_string(f.bytes);
        covered += f.bytes;
    }
    // A field added to GeoPixel but not listed here leaves bytes uncovered.
    if (covered != sizeof(GeoPixel)) throw std::logic_error("kgeo_geopixel_schema misses a GeoPixel field");
    return out;
}

// How a .kgeo buffer was made: flat "key": value pairs, values as JSON literals.
struct KGeoProvenance {
    std::vector<std::pair<std::string, std::string>> entries;
    bool complete = false;   // false: some of what made the buffer is unknown (v4 files)

    void set_raw(const std::string& key, const std::string& literal) {
        for (auto& e : entries) if (e.first == key) { e.second = literal; return; }
        entries.emplace_back(key, literal);
    }
    void set(const std::string& key, double v) {
        if (!std::isfinite(v)) { set_raw(key, "null"); return; }
        std::string text;   // shortest form that reads back exactly
        for (int digits = 15; digits <= 17; ++digits) {
            std::ostringstream o; o.precision(digits); o << v;
            text = o.str();
            if (std::stod(text) == v) break;
        }
        set_raw(key, text);
    }
    void set(const std::string& key, int v) { set_raw(key, std::to_string(v)); }
    void set(const std::string& key, bool v) { set_raw(key, v ? "true" : "false"); }
    void set(const std::string& key, const char* v) { set(key, std::string(v)); }
    void set(const std::string& key, const std::string& v) {
        std::string q = "\"";
        for (char c : v) { if (c == '"' || c == '\\') q += '\\'; q += c; }
        set_raw(key, q + "\"");
    }
    const std::string* find(const std::string& key) const {
        for (const auto& e : entries) if (e.first == key) return &e.second;
        return nullptr;
    }
    std::string to_json() const {
        std::string out = "{\n";
        for (size_t i = 0; i < entries.size(); ++i)
            out += "  \"" + entries[i].first + "\": " + entries[i].second
                 + (i + 1 < entries.size() ? ",\n" : "\n");
        return out + "}\n";
    }
    // Reads what to_json writes (one pair per line); anything else is ignored.
    static KGeoProvenance from_json(const std::string& json) {
        KGeoProvenance p;
        std::istringstream in(json);
        std::string line;
        while (std::getline(in, line)) {
            const size_t k0 = line.find('"');
            const size_t k1 = (k0 == std::string::npos) ? k0 : line.find('"', k0 + 1);
            const size_t colon = (k1 == std::string::npos) ? k1 : line.find(':', k1);
            if (colon == std::string::npos) continue;
            std::string value = line.substr(colon + 1);
            while (!value.empty() && (value.back() == ',' || value.back() == ' ' || value.back() == '\r')) value.pop_back();
            while (!value.empty() && value.front() == ' ') value.erase(0, 1);
            p.entries.emplace_back(line.substr(k0 + 1, k1 - k0 - 1), value);
        }
        const std::string* c = p.find("provenance_complete");
        p.complete = c && *c == "true";
        return p;
    }
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
                      const KGeoMeta& meta, const KGeoProvenance& provenance) {
    if (checked_pixel_count(meta.W, meta.H) != geo.size())
        throw std::runtime_error("KGEO dimensions do not match the pixel buffer");
    KGeoProvenance prov;   // format keys first, then what the caller knows
    prov.set("kgeo_version", int(KGEO_VERSION));
    const uint32_t tag = KGEO_BYTE_ORDER_TAG;
    prov.set("byte_order", reinterpret_cast<const unsigned char*>(&tag)[0] == 0x04 ? "little" : "big");
    prov.set("record", "GeoPixel");
    prov.set("record_size", int(sizeof(GeoPixel)));
    prov.set("fields", kgeo_geopixel_schema());
    prov.set("provenance_complete", provenance.complete);
    for (const auto& e : provenance.entries) prov.set_raw(e.first, e.second);
    const std::string json = prov.to_json();

    const uint32_t ver = KGEO_VERSION;
    const uint32_t json_len = uint32_t(json.size());
    const uint32_t header_bytes = 4 + 5*4 + uint32_t(sizeof(KGeoMeta)) + 4 + json_len;
    const uint32_t record_size = sizeof(GeoPixel), meta_size = sizeof(KGeoMeta);
    std::ofstream f(path, std::ios::binary);
    f.write(KGEO_MAGIC, 4);
    for (uint32_t v : {ver, tag, header_bytes, record_size, meta_size})
        f.write(reinterpret_cast<const char*>(&v), sizeof(v));
    f.write(reinterpret_cast<const char*>(&meta), sizeof(meta));
    f.write(reinterpret_cast<const char*>(&json_len), sizeof(json_len));
    f.write(json.data(), std::streamsize(json.size()));
    f.write(reinterpret_cast<const char*>(geo.data()),
            std::streamsize(geo.size()*sizeof(GeoPixel)));
    f.close();
    if (!f) throw std::runtime_error(std::string("Cannot write KGEO file: ") + path);
}

// Version 5, or version 4 (same records, no provenance: complete = false).
inline bool load_kgeo(const char* path, std::vector<GeoPixel>& geo, KGeoMeta& meta,
                      KGeoProvenance* provenance = nullptr) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const uint64_t size = uint64_t(std::streamoff(f.tellg()));
    f.seekg(0);
    char magic[4] = {};
    uint32_t version = 0;
    f.read(magic, sizeof(magic));
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!f || std::memcmp(magic, KGEO_MAGIC, 4) != 0) return false;

    uint64_t header_bytes = 0;
    KGeoProvenance prov;
    KGeoMeta candidate{};
    if (version == 4) {
        header_bytes = 8 + sizeof(KGeoMeta);
        if (!f.read(reinterpret_cast<char*>(&candidate), sizeof(candidate))) return false;
        prov.set("kgeo_version", 4);
        prov.set("provenance_complete", false);
    } else if (version == 5) {
        uint32_t tag = 0, hdr = 0, record_size = 0, meta_size = 0, json_len = 0;
        for (uint32_t* v : {&tag, &hdr, &record_size, &meta_size})
            f.read(reinterpret_cast<char*>(v), sizeof(*v));
        if (!f || tag != KGEO_BYTE_ORDER_TAG || record_size != sizeof(GeoPixel)
            || meta_size != sizeof(KGeoMeta)) return false;
        if (!f.read(reinterpret_cast<char*>(&candidate), sizeof(candidate))) return false;
        if (!f.read(reinterpret_cast<char*>(&json_len), sizeof(json_len))) return false;
        header_bytes = hdr;
        if (json_len > (1u << 20) || uint64_t(4 + 5*4 + sizeof(KGeoMeta) + 4) + json_len != header_bytes
            || header_bytes > size) return false;
        std::string json(json_len, '\0');
        if (!f.read(json.data(), json_len)) return false;
        prov = KGeoProvenance::from_json(json);
        const std::string* fields = prov.find("fields");
        if (!fields || *fields != "\"" + kgeo_geopixel_schema() + "\"") return false;
    } else {
        return false;
    }
    const double values[] = {candidate.M_bh, candidate.a_bh, candidate.Q_bh, candidate.Lam,
        candidate.r_isco, candidate.r_disk_in, candidate.r_disk_out,
        candidate.theta_obs, candidate.phi_obs, candidate.r_obs};
    for (double value : values) if (!std::isfinite(value)) return false;
    try {
        const size_t count = checked_pixel_count(candidate.W, candidate.H);
        if (size != header_bytes + uint64_t(count)*sizeof(GeoPixel)) return false;
        std::vector<GeoPixel> pixels(count);
        f.seekg(std::streamoff(header_bytes));
        if (!f.read(reinterpret_cast<char*>(pixels.data()), std::streamsize(count*sizeof(GeoPixel))))
            return false;
        geo.swap(pixels);
        meta = candidate;
        if (provenance) *provenance = std::move(prov);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}
