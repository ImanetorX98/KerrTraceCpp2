// kgeo_to_npy <in.kgeo> <out-prefix>
//
// Writes <out-prefix>.npy, a structured NumPy array of shape (H, W) whose dtype is
// generated from the Reflection description of GeoPixel, and <out-prefix>.json
// with KGeoMeta. In Python:   geo = numpy.load("out.npy"); geo["redshift"]
#include "kgeo_reflection.hpp"

#include <cstdio>
#include <string>
#include <type_traits>

// NumPy type string for one primitive member. The record is written in the host's
// byte order, which main() checks is little-endian.
template <typename R> constexpr const char* numpy_code() {
    static_assert(std::is_arithmetic<R>::value, "only primitive fields map to a dtype code");
    if (std::is_floating_point<R>::value) return sizeof(R) == 4 ? "<f4" : "<f8";
    if (sizeof(R) == 1) return std::is_signed<R>::value ? "|i1" : "|u1";
    if (std::is_signed<R>::value) return sizeof(R) == 2 ? "<i2" : sizeof(R) == 4 ? "<i4" : "<i8";
    return sizeof(R) == 2 ? "<u2" : sizeof(R) == 4 ? "<u4" : "<u8";
}

// Reflection visitor: one "('name', 'code'[, (n,)])" entry per member, in offset order.
template <typename T>
struct NumpyDescr {
    std::string descr = "[";
    size_t expected_offset = 0;
    bool in_order = true;

    template <typename R, int N>
    bool operator()(int, R T::*, const char (&name)[N], size_t offset) {
        in_order = in_order && offset == expected_offset;   // list form relies on it
        expected_offset = offset + sizeof(R);
        descr += std::string(descr.size() > 1 ? ", " : "") + "('" + name + "', ";
        using Elem = typename std::remove_all_extents<R>::type;
        descr += std::string("'") + numpy_code<Elem>() + "'";
        if (std::is_array<R>::value) descr += ", (" + std::to_string(std::extent<R>::value) + ",)";
        descr += ")";
        return true;
    }
};

// Reflection visitor: "name": value lines for the metadata sidecar.
template <typename T>
struct JsonFields {
    const T& obj;
    std::string out;
    template <typename R, int N>
    bool operator()(int, R T::* member, const char (&name)[N], size_t) {
        char value[64];
        if (std::is_floating_point<R>::value) std::snprintf(value, sizeof value, "%.17g", double(obj.*member));
        else std::snprintf(value, sizeof value, "%llu", (unsigned long long)(obj.*member));
        out += std::string(out.empty() ? "" : ",\n") + "  \"" + name + "\": " + value;
        return true;
    }
};

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: kgeo_to_npy <in.kgeo> <out-prefix>\n"); return 2; }
    const uint16_t probe = 1;
    if (*reinterpret_cast<const uint8_t*>(&probe) != 1) { std::fprintf(stderr, "big-endian host not supported\n"); return 1; }

    std::vector<GeoPixel> geo;
    KGeoMeta meta{};
    if (!load_kgeo(argv[1], geo, meta)) { std::fprintf(stderr, "cannot read %s (KGEO v%u)\n", argv[1], KGEO_VERSION); return 1; }

    NumpyDescr<GeoPixel> d;
    SC::Reflection::Reflect<GeoPixel>::visit(d);
    if (!d.in_order || d.expected_offset != sizeof(GeoPixel)) { std::fprintf(stderr, "GeoPixel layout has gaps\n"); return 1; }
    d.descr += "]";

    // .npy v1.0: magic, version, little-endian header length, then a Python dict
    // literal padded with spaces so the data starts on a 64-byte boundary.
    std::string header = "{'descr': " + d.descr + ", 'fortran_order': False, 'shape': (" +
                         std::to_string(meta.H) + ", " + std::to_string(meta.W) + "), }";
    const size_t preamble = 10;
    header.append((64 - (preamble + header.size() + 1) % 64) % 64, ' ');
    header += '\n';
    if (header.size() > 0xFFFF) { std::fprintf(stderr, "header too long for .npy v1\n"); return 1; }

    const std::string npy = std::string(argv[2]) + ".npy";
    std::FILE* f = std::fopen(npy.c_str(), "wb");
    if (!f) { std::perror(npy.c_str()); return 1; }
    const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    const uint16_t hlen = uint16_t(header.size());
    std::fwrite(magic, 1, 8, f);
    std::fwrite(&hlen, 2, 1, f);
    std::fwrite(header.data(), 1, header.size(), f);
    std::fwrite(geo.data(), sizeof(GeoPixel), geo.size(), f);
    if (std::fclose(f) != 0) { std::perror(npy.c_str()); return 1; }

    JsonFields<KGeoMeta> j{meta, {}};
    SC::Reflection::Reflect<KGeoMeta>::visit(j);
    const std::string json = std::string(argv[2]) + ".json";
    std::FILE* g = std::fopen(json.c_str(), "w");
    if (!g) { std::perror(json.c_str()); return 1; }
    std::fprintf(g, "{\n  \"kgeo_version\": %u,\n%s\n}\n", KGEO_VERSION, j.out.c_str());
    std::fclose(g);

    std::printf("wrote %s (%ux%u, %zu bytes/record) and %s\ndtype %s\n",
                npy.c_str(), meta.W, meta.H, sizeof(GeoPixel), json.c_str(), d.descr.c_str());
    return 0;
}
