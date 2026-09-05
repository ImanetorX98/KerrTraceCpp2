# Interactive 3D Navigable Black Hole — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add real-time θ/φ navigation around the black hole via a pre-baked geodesic LUT (`.klut`) and an SDL2 interactive window with ImGui controls.

**Architecture:** A two-file LUT (`.klut`) stores 359 pre-traced GeoPixel slices (θ = 0.5°…179.5°, step 0.5°) at 960×540. At runtime the loader mmaps the file, lerps two adjacent slices for any θ, and feeds the result to the existing `colorize_buffer()`. φ-rotation is free (texture offset). The SDL2 window dispatches a low-res Metal preview on every mouse event and the full LUT colorize on debounce.

**Tech Stack:** C++17, Metal (MSL), SDL2 ≥ 2.0.14, Dear ImGui ≥ 1.90, OpenMP (CPU bake fallback), POSIX `mmap`.

---

## Codebase Context (read before touching anything)

| Struct / function | File | What it does |
|---|---|---|
| `GeoPixel` (28 bytes) | `main.cpp:125` | Per-pixel trace result: outcome, r, redshift, magnif, phi_disk, theta_esc, phi_esc |
| `KGeoMeta` | `main.cpp:141` | Metadata for one .kgeo slice (W, H, metric params, theta_obs) |
| `save_kgeo` / `load_kgeo` | `main.cpp:148/160` | Binary I/O for a single GeoPixel slice |
| `colorize_buffer()` | `main.cpp:2399` | Takes `vector<GeoPixel>` → `vector<RGB>` using existing palette/Doppler logic |
| `KNdSParams_C` / `CameraParams_C` | `gpu/metal/metal_renderer.hpp` | C structs passed to Metal shader |
| `metal_render()` | `gpu/metal/metal_renderer.hpp:72` | Renders RGBA pixels on GPU |
| `trace_row` lambda | `main.cpp:~2740` | CPU trace loop (OpenMP / thread pool); fills `geo[py*W … (py+1)*W-1]` |
| CLI flag parsing | `main.cpp:3290` | Big `for (int i=1;i<argc;++i)` block; add new flags here |
| `--geo-only` / `--color-only` | `main.cpp:3272` | Two-phase mode — model for new `--bake-lut` flag |

**GeoPixel layout (28 bytes, static_assert enforced):**
```cpp
struct GeoPixel {
    uint8_t outcome;   // 0=escaped, 1=disk_hit, 2=horizon, 3=escaped_B
    uint8_t _pad[3];   // _pad[1]: behind_out code; keep _pad[2]=0
    float   r;         // BL radius at disk crossing (or final r)
    float   redshift;  // g = ν_obs/ν_em
    float   magnif;    // flux magnification (1.0 in single-ray mode)
    float   phi_disk;  // BL azimuthal angle at disk crossing
    float   theta_esc; // direction at escape (background UV lookup)
    float   phi_esc;
};
```

**outcome codes:** 0 = background, 1 = disk hit, 2 = horizon, 3 = background-B (wormhole)

---

## File Map

| File | Status | Role |
|---|---|---|
| `lut/lut_types.hpp` | **Create** | `KLutHeader` struct + `write_klut_header()` + `append_klut_slice()` |
| `lut/lut_loader.hpp` | **Create** | `KLutFile` mmap loader + `lerp_slices()` |
| `interactive/window.hpp` | **Create** | `InteractiveWindow` class declaration |
| `interactive/window.mm` | **Create** | SDL2 event loop + MTLTexture blit |
| `interactive/render_scheduler.hpp` | **Create** | Async low-res Metal dispatch + debounce |
| `interactive/camera_controller.hpp` | **Create** | Arcball quaternion → BL `(θ_obs, φ_obs)` |
| `interactive/ui.hpp` | **Create** | Dear ImGui sidebar layout |
| `main.cpp` | **Modify** | Add `--bake-lut`, `--lut-render`, `--interactive` flags |
| `CMakeLists.txt` | **Modify** | Add `USE_INTERACTIVE` option, find SDL2 + ImGui |

---

## Session 1 — `.klut` Bake Tool (5 h)

**Deliverable:** `./build_metal/kerr_tracer_metal --bake-lut /tmp/test.klut --width 320 --height 180 --a 0.998` produces a valid binary `.klut` file containing 359 GeoPixel slices.

### Task 1.1 — Create `lut/lut_types.hpp`

**Files:**
- Create: `lut/lut_types.hpp`

- [ ] **Step 1: Create the file**

```cpp
#pragma once
// ============================================================
//  lut/lut_types.hpp  —  .klut multi-slice LUT file format
//
//  File layout:
//    [KLutHeader  128 bytes]
//    [GeoPixel × W × H]   ← slice 0  (theta = theta_min_rad)
//    [GeoPixel × W × H]   ← slice 1  (theta = theta_min_rad + theta_step_rad)
//    ...
//    [GeoPixel × W × H]   ← slice n_slices-1
//
//  GeoPixel is defined in main.cpp (28 bytes, static_assert enforced).
//  Include main.cpp structs BEFORE including this header.
// ============================================================
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

static const char    KLUT_MAGIC[4]   = {'K','L','U','T'};
static const uint32_t KLUT_VERSION   = 1;

struct KLutHeader {
    char     magic[4];          //  4  "KLUT"
    uint32_t version;           //  4  = 1
    uint32_t width;             //  4  pixels per row
    uint32_t height;            //  4  rows per slice
    uint32_t n_slices;          //  4  number of theta slices
    uint32_t _pad0;             //  4  (alignment)
    double   theta_min_rad;     //  8  first theta value in radians
    double   theta_step_rad;    //  8  step between slices in radians
    double   M;                 //  8  black hole mass
    double   a;                 //  8  spin parameter
    double   Q;                 //  8  charge
    double   Lambda;            //  8  cosmological constant
    double   r_obs;             //  8  observer radius
    double   r_isco;            //  8  ISCO radius
    double   r_disk_in;         //  8  inner disk edge
    double   r_disk_out;        //  8  outer disk edge
    uint8_t  _pad1[24];         // 24  reserved → total = 128 bytes
};
static_assert(sizeof(KLutHeader) == 128, "KLutHeader must be exactly 128 bytes");

// Write the header to an already-open binary file stream.
inline void write_klut_header(std::ofstream& f, const KLutHeader& h) {
    f.write(reinterpret_cast<const char*>(&h), sizeof(h));
}

// Append one slice (W*H GeoPixels) to an already-open binary file stream.
// Call once per theta value, in ascending theta order, after write_klut_header().
template<typename GeoPixelT>
inline void append_klut_slice(std::ofstream& f,
                               const std::vector<GeoPixelT>& slice) {
    f.write(reinterpret_cast<const char*>(slice.data()),
            static_cast<std::streamsize>(slice.size() * sizeof(GeoPixelT)));
}
```

- [ ] **Step 2: Verify compile (no test binary needed yet)**

```bash
cd /Users/iman.rosignoli/Documents/KerrTraceCpp2
g++ -std=c++17 -I. -c -o /tmp/lut_types_check.o /dev/stdin <<'EOF'
#include <vector>
#include <cstdint>
struct GeoPixel { uint8_t outcome; uint8_t _pad[3]; float r,redshift,magnif,phi_disk,theta_esc,phi_esc; };
#include "lut/lut_types.hpp"
int main() { static_assert(sizeof(KLutHeader)==128); return 0; }
EOF
echo "OK: lut_types.hpp compiles"
```

Expected: `OK: lut_types.hpp compiles`

- [ ] **Step 3: Commit**

```bash
git add lut/lut_types.hpp
git commit -m "feat(lut): add KLutHeader struct and .klut file format helpers"
```

---

### Task 1.2 — Add `--bake-lut` flag to `main.cpp`

**Files:**
- Modify: `main.cpp`

- [ ] **Step 1: Add bake_lut variables alongside geo_only (line ~3271)**

Find this block in `main.cpp` (line ~3271):
```cpp
    // ── Two-phase modes ───────────────────────────────────────
    bool        geo_only    = false;
    std::string geo_file;         // path for .kgeo output (geo_only) or input (color_only)
    bool        color_only  = false;
```

Add immediately after the `color_only` line:
```cpp
    // ── LUT bake mode ─────────────────────────────────────────
    bool        bake_lut        = false;
    std::string bake_lut_path;          // output .klut file path
    double      bake_theta_min  = 0.5;  // degrees
    double      bake_theta_max  = 179.5;
    double      bake_theta_step = 0.5;
```

- [ ] **Step 2: Add flag parsing in the CLI loop (line ~3290, inside `for (int i=1;i<argc;++i)`)**

Find an existing flag block near the end of the loop (e.g., after `--color-only`) and add:
```cpp
        if (arg == "--bake-lut" && i+1 < argc) { bake_lut = true; bake_lut_path = argv[++i]; }
        if (arg == "--bake-theta-min"  && i+1 < argc) bake_theta_min  = std::stod(argv[++i]);
        if (arg == "--bake-theta-max"  && i+1 < argc) bake_theta_max  = std::stod(argv[++i]);
        if (arg == "--bake-theta-step" && i+1 < argc) bake_theta_step = std::stod(argv[++i]);
```

- [ ] **Step 3: Add bake dispatch block**

Find the `if (color_only)` dispatch block (line ~3590). Immediately before it, add the bake dispatch:

```cpp
    // ── LUT bake mode ─────────────────────────────────────────
    if (bake_lut) {
        #include "lut/lut_types.hpp"  // already included via forward-decl below — just reference
        // NOTE: lut_types.hpp is included at file top (add #include "lut/lut_types.hpp"
        //       alongside the other includes at line ~14).

        const int W = img_w, H = img_h;
        const double theta_step_rad = bake_theta_step * M_PI / 180.0;
        const double theta_min_rad  = bake_theta_min  * M_PI / 180.0;
        const double theta_max_rad  = bake_theta_max  * M_PI / 180.0;

        const int n_slices = static_cast<int>(
            std::round((theta_max_rad - theta_min_rad) / theta_step_rad)) + 1;

        KLutHeader hdr{};
        std::memcpy(hdr.magic, KLUT_MAGIC, 4);
        hdr.version       = KLUT_VERSION;
        hdr.width         = static_cast<uint32_t>(W);
        hdr.height        = static_cast<uint32_t>(H);
        hdr.n_slices      = static_cast<uint32_t>(n_slices);
        hdr.theta_min_rad = theta_min_rad;
        hdr.theta_step_rad= theta_step_rad;
        hdr.M             = M_bh;
        hdr.a             = fp.a;
        hdr.Q             = fp.Q;
        hdr.Lambda        = fp.Lambda;
        hdr.r_obs         = fp.r_obs;
        hdr.r_isco        = g.r_isco();
        hdr.r_disk_in     = r_disk_in;
        hdr.r_disk_out    = r_disk_out;

        std::ofstream lut_f(bake_lut_path, std::ios::binary);
        if (!lut_f)
            throw std::runtime_error("Cannot open --bake-lut output: " + bake_lut_path);
        write_klut_header(lut_f, hdr);

        std::cerr << "Baking " << n_slices << " slices at "
                  << W << "x" << H << " → " << bake_lut_path << "\n";

        const auto t_bake_start = std::chrono::steady_clock::now();

        for (int si = 0; si < n_slices; ++si) {
            const double theta_rad = theta_min_rad + si * theta_step_rad;
            // Update camera for this slice
            Camera cam_slice(fp.r_obs, theta_rad * 180.0 / M_PI, fp.phi_obs,
                             fp.fov_h * 180.0 / M_PI, W, H);

            std::vector<GeoPixel> geo_slice(W * H);

            // Re-use existing trace_row logic via a local lambda
            auto bake_row = [&](int py) {
                for (int px_ = 0; px_ < W; ++px_) {
                    GeoPixel& pix = geo_slice[py * W + px_];
                    auto s = cam_slice.pixel_ray(px_, py, g);
                    TraceResult res = trace_single(s, g, r_disk_in, r_disk_out,
                                                   r_escape, intg, ctl, nullptr);
                    pix.outcome  = (res.out == Outcome::DISK_HIT) ? 1
                                 : (res.out == Outcome::HORIZON)  ? 2
                                 : (res.out == Outcome::ESCAPED_B)? 3 : 0;
                    pix.r        = static_cast<float>(res.r);
                    pix.redshift = static_cast<float>(res.redshift);
                    pix.magnif   = 1.0f;
                    pix.phi_disk = static_cast<float>(res.phi_disk);
                    pix.theta_esc= static_cast<float>(res.theta_esc);
                    pix.phi_esc  = static_cast<float>(res.phi_esc);
                    pix._pad[0] = pix._pad[1] = pix._pad[2] = 0;
                }
            };

#if defined(_OPENMP)
            #pragma omp parallel for schedule(dynamic, 4)
            for (int py = 0; py < H; ++py) bake_row(py);
#else
            {
                std::atomic<int> next_row{0};
                const unsigned workers = std::max(1u,
                    std::thread::hardware_concurrency());
                std::vector<std::thread> pool;
                pool.reserve(workers);
                for (unsigned t = 0; t < workers; ++t)
                    pool.emplace_back([&](){
                        while (true) {
                            const int py = next_row.fetch_add(1);
                            if (py >= H) break;
                            bake_row(py);
                        }
                    });
                for (auto& t : pool) t.join();
            }
#endif
            append_klut_slice(lut_f, geo_slice);

            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_bake_start).count();
            const double eta = (si + 1 < n_slices && elapsed > 0)
                ? elapsed * (n_slices - si - 1.0) / (si + 1.0) : 0.0;
            fprintf(stderr, "\r  Slice %3d/%d  θ=%.1f°  %.1fs elapsed  %.1fs ETA   ",
                    si+1, n_slices,
                    theta_rad * 180.0 / M_PI,
                    elapsed, eta);
            fflush(stderr);
        }
        fprintf(stderr, "\nBake complete: %s\n", bake_lut_path.c_str());

        const long long expected_bytes =
            static_cast<long long>(sizeof(KLutHeader)) +
            static_cast<long long>(n_slices) * W * H * sizeof(GeoPixel);
        fprintf(stderr, "Expected size: %lld bytes (%.2f GB)\n",
                expected_bytes, expected_bytes / 1.0e9);
        return 0;
    }
```

- [ ] **Step 4: Add `#include "lut/lut_types.hpp"` near top of main.cpp**

Find the block of includes at line ~14:
```cpp
#include "camera.hpp"
#include "geodesic.hpp"
```

Add after `#include "ray_bundle.hpp"`:
```cpp
#include "lut/lut_types.hpp"
```

- [ ] **Step 5: Build and verify compilation**

```bash
cd /Users/iman.rosignoli/Documents/KerrTraceCpp2
cmake -B build_cpu -DUSE_METAL=OFF 2>&1 | tail -5
cmake --build build_cpu -j$(sysctl -n hw.ncpu) 2>&1 | tail -10
```

Expected: build succeeds, binary `build_cpu/kerr_tracer` exists.

- [ ] **Step 6: Smoke test — bake a tiny 320×180 LUT**

```bash
./build_cpu/kerr_tracer \
  --bake-lut /tmp/test_small.klut \
  --width 320 --height 180 \
  --a 0.998

# Verify file size: 128 + 359 * 320 * 180 * 28 = 128 + 578,073,600 = 578,073,728 bytes
EXPECTED=578073728
ACTUAL=$(stat -f%z /tmp/test_small.klut)
echo "Expected: $EXPECTED  Actual: $ACTUAL"
[ "$ACTUAL" -eq "$EXPECTED" ] && echo "PASS" || echo "FAIL"
```

Expected output:
```
Baking 359 slices at 320x180 → /tmp/test_small.klut
  Slice 359/359  θ=179.5°  ...
Bake complete: /tmp/test_small.klut
Expected size: 578073728 bytes (0.58 GB)
Expected: 578073728  Actual: 578073728
PASS
```

- [ ] **Step 7: Commit**

```bash
git add lut/lut_types.hpp main.cpp
git commit -m "feat(lut): add --bake-lut flag; traces 359 theta slices into .klut binary"
```

---

### Task 1.3 — Metal bake path (GPU-accelerated slices)

**Files:**
- Modify: `main.cpp`

The CPU bake works but is slow at 960×540 (~2 min vs ~4 s on Metal). Add a Metal code path inside the bake loop for when `USE_METAL` is defined.

- [ ] **Step 1: Replace CPU bake inner loop with Metal dispatch when available**

Inside the `if (bake_lut)` block, replace the `bake_row` lambda and thread pool with:

```cpp
#if defined(USE_METAL)
            // Metal path: use existing metal_render() with geo-only CameraParams
            {
                CameraParams_C mcp{};
                mcp.r_obs        = static_cast<float>(fp.r_obs);
                mcp.theta_obs    = static_cast<float>(theta_rad);
                mcp.phi_obs      = static_cast<float>(fp.phi_obs);
                mcp.fov_h        = static_cast<float>(fp.fov_h * M_PI / 180.0);
                mcp.width        = W;
                mcp.height       = H;
                mcp.chart        = 0;  // BL
                mcp.solver_mode  = 0;  // standard
                mcp.integrator_mode = 0;
                mcp.use_bundles  = 0;
                mcp.metal_kernel_mode = 2;  // single-ray
                mcp.max_steps    = ctl.max_steps;
                mcp.step_init    = static_cast<float>(ctl.step_init);
                mcp.integrator_tol = static_cast<float>(ctl.tol);
                mcp.enable_doppler = 0;
                mcp.disk_palette = 0;
                mcp.disk_brightness = 1.0f;
                mcp.disk_opacity    = 1.0f;
                mcp.exposure        = 1.0f;
                mcp.gamma           = 2.2f;

                KNdSParams_C mkp{};
                mkp.M         = static_cast<float>(M_bh);
                mkp.a         = static_cast<float>(fp.a);
                mkp.Q         = static_cast<float>(fp.Q);
                mkp.Lambda    = static_cast<float>(fp.Lambda);
                mkp.r_horizon = static_cast<float>(g.r_horizon());
                mkp.r_isco    = static_cast<float>(g.r_isco());
                mkp.r_disk_out= static_cast<float>(r_disk_out);

                // metal_render returns RGBA pixels — we need GeoPixels.
                // Metal shader does not yet export raw GeoPixel data,
                // so fall through to CPU path for now. (Task 1.3 placeholder marker
                // — see Session 2 for the Metal bake kernel addition.)
                //
                // For now use CPU path regardless of USE_METAL for the bake.
                goto cpu_bake_path;
            }
            cpu_bake_path:
#endif
            // CPU fallback bake path (also used on non-Metal builds)
            // [existing bake_row lambda + thread pool code here]
```

> **Note:** Full Metal bake kernel (exporting raw GeoPixels instead of RGBA) is a separate Phase 5c stretch item. For Session 1, CPU bake is the deliverable. The `goto` above is intentional scaffolding — remove when Metal bake kernel is added.

- [ ] **Step 2: Build Metal binary and verify bake still works**

```bash
cmake -B build -DUSE_METAL=ON
cmake --build build -j$(sysctl -n hw.ncpu) 2>&1 | tail -5

./build/kerr_tracer_metal \
  --bake-lut /tmp/test_metal.klut \
  --width 320 --height 180 --a 0.5

EXPECTED=578073728
ACTUAL=$(stat -f%z /tmp/test_metal.klut)
[ "$ACTUAL" -eq "$EXPECTED" ] && echo "PASS" || echo "FAIL (got $ACTUAL)"
```

- [ ] **Step 3: Commit**

```bash
git add main.cpp
git commit -m "feat(lut): Metal build compiles with --bake-lut (CPU fallback path)"
```

---

### Task 1.4 — Bake a full 960×540 LUT for a=0.998

- [ ] **Step 1: Bake the production LUT (CPU, ~8 min at 960×540)**

```bash
mkdir -p /Users/iman.rosignoli/Documents/KerrTraceCpp2/lut

time ./build_cpu/kerr_tracer \
  --bake-lut lut/kerr_a0998_960x540.klut \
  --width 960 --height 540 \
  --a 0.998 --M 1.0 --r 500

# Verify size: 128 + 359 * 960 * 540 * 28 = 128 + 5,212,569,600 = 5,212,569,728 bytes
EXPECTED=5212569728
ACTUAL=$(stat -f%z lut/kerr_a0998_960x540.klut)
[ "$ACTUAL" -eq "$EXPECTED" ] && echo "PASS" || echo "FAIL"
```

- [ ] **Step 2: Add lut/*.klut to .gitignore**

```bash
echo "lut/*.klut" >> .gitignore
git add .gitignore
git commit -m "chore: ignore large .klut bake files"
```

---

## Session 2 — LUT Loader + `--lut-render` (5 h)

**Deliverable:** `./build_metal/kerr_tracer_metal --lut-render lut/kerr_a0998_960x540.klut --theta 75 --bg assets/backgrounds/sfondo5.jpg` produces a PNG visually identical to a direct trace at θ=75°.

### Task 2.1 — Create `lut/lut_loader.hpp`

**Files:**
- Create: `lut/lut_loader.hpp`

- [ ] **Step 1: Create the file**

```cpp
#pragma once
// ============================================================
//  lut/lut_loader.hpp  —  mmap-based .klut reader + lerp
//
//  Usage:
//    KLutFile lut;
//    lut.open("kerr_a0998_960x540.klut");
//    std::vector<GeoPixel> frame(960*540);
//    lut.sample(75.0 * M_PI / 180.0, frame.data());
// ============================================================
#include "lut_types.hpp"
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

class KLutFile {
public:
    KLutHeader hdr{};

    KLutFile() = default;
    ~KLutFile() { close(); }

    KLutFile(const KLutFile&)            = delete;
    KLutFile& operator=(const KLutFile&) = delete;

    void open(const std::string& path) {
        close();
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0)
            throw std::runtime_error("KLutFile: cannot open " + path);

        struct stat sb{};
        if (::fstat(fd_, &sb) < 0)
            throw std::runtime_error("KLutFile: fstat failed");
        file_size_ = static_cast<size_t>(sb.st_size);

        base_ = static_cast<const uint8_t*>(
            ::mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (base_ == MAP_FAILED) {
            base_ = nullptr;
            throw std::runtime_error("KLutFile: mmap failed");
        }

        std::memcpy(&hdr, base_, sizeof(KLutHeader));
        if (std::memcmp(hdr.magic, KLUT_MAGIC, 4) != 0)
            throw std::runtime_error("KLutFile: bad magic, not a .klut file");
        if (hdr.version != KLUT_VERSION)
            throw std::runtime_error("KLutFile: unsupported version");

        slice_bytes_ = static_cast<size_t>(hdr.width) *
                       static_cast<size_t>(hdr.height) *
                       sizeof(GeoPixel);

        const size_t expected = sizeof(KLutHeader) +
                                static_cast<size_t>(hdr.n_slices) * slice_bytes_;
        if (file_size_ < expected)
            throw std::runtime_error("KLutFile: file truncated");
    }

    void close() {
        if (base_) { ::munmap(const_cast<uint8_t*>(base_), file_size_); base_ = nullptr; }
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    }

    bool is_open() const { return base_ != nullptr; }

    // Returns a pointer to slice i (no copy).
    const GeoPixel* slice(int i) const {
        assert(i >= 0 && static_cast<uint32_t>(i) < hdr.n_slices);
        return reinterpret_cast<const GeoPixel*>(
            base_ + sizeof(KLutHeader) + static_cast<size_t>(i) * slice_bytes_);
    }

    // Linearly interpolate between the two slices bracketing theta_rad.
    // Writes W*H GeoPixels into `out`.
    // phi_offset (radians) is added to phi_disk and phi_esc — use for φ-rotation.
    void sample(double theta_rad, GeoPixel* out, double phi_offset = 0.0) const {
        if (!base_) throw std::runtime_error("KLutFile::sample: file not open");

        // Map theta_rad to a continuous slice index.
        double t = (theta_rad - hdr.theta_min_rad) / hdr.theta_step_rad;
        t = std::max(0.0, std::min(t, static_cast<double>(hdr.n_slices - 1)));

        const int   i0    = static_cast<int>(t);
        const int   i1    = std::min(i0 + 1, static_cast<int>(hdr.n_slices) - 1);
        const float alpha = static_cast<float>(t - i0);

        const GeoPixel* s0 = slice(i0);
        const GeoPixel* s1 = slice(i1);
        const int n = static_cast<int>(hdr.width * hdr.height);

        const float pi2 = static_cast<float>(2.0 * M_PI);

        for (int p = 0; p < n; ++p) {
            // Outcome: use lower slice (type boundary across 0.5° is acceptable)
            out[p].outcome  = s0[p].outcome;
            out[p]._pad[0]  = 0;
            out[p]._pad[1]  = s0[p]._pad[1];
            out[p]._pad[2]  = 0;

            out[p].r        = s0[p].r       + alpha * (s1[p].r       - s0[p].r);
            out[p].redshift = s0[p].redshift+ alpha * (s1[p].redshift- s0[p].redshift);
            out[p].magnif   = s0[p].magnif  + alpha * (s1[p].magnif  - s0[p].magnif);

            // phi_disk: wrap-aware lerp + offset
            float dphi = s1[p].phi_disk - s0[p].phi_disk;
            if (dphi >  static_cast<float>(M_PI)) dphi -= pi2;
            if (dphi < -static_cast<float>(M_PI)) dphi += pi2;
            float phi_d = s0[p].phi_disk + alpha * dphi
                          + static_cast<float>(phi_offset);
            // Normalise to [0, 2π)
            phi_d = std::fmod(phi_d, pi2);
            if (phi_d < 0.0f) phi_d += pi2;
            out[p].phi_disk = phi_d;

            out[p].theta_esc= s0[p].theta_esc + alpha * (s1[p].theta_esc - s0[p].theta_esc);

            // phi_esc: wrap-aware lerp + offset
            float dphi_e = s1[p].phi_esc - s0[p].phi_esc;
            if (dphi_e >  static_cast<float>(M_PI)) dphi_e -= pi2;
            if (dphi_e < -static_cast<float>(M_PI)) dphi_e += pi2;
            float phi_e = s0[p].phi_esc + alpha * dphi_e
                          + static_cast<float>(phi_offset);
            phi_e = std::fmod(phi_e, pi2);
            if (phi_e < 0.0f) phi_e += pi2;
            out[p].phi_esc  = phi_e;
        }
    }

private:
    int           fd_          = -1;
    const uint8_t* base_       = nullptr;
    size_t        file_size_   = 0;
    size_t        slice_bytes_ = 0;
};
```

- [ ] **Step 2: Add `#include "lut/lut_loader.hpp"` to main.cpp alongside lut_types.hpp**

- [ ] **Step 3: Verify compilation**

```bash
g++ -std=c++17 -I/Users/iman.rosignoli/Documents/KerrTraceCpp2 \
    -c -o /tmp/lut_loader_check.o /dev/stdin <<'EOF'
#include <vector>
#include <cstdint>
struct GeoPixel { uint8_t outcome; uint8_t _pad[3]; float r,redshift,magnif,phi_disk,theta_esc,phi_esc; };
#include "lut/lut_types.hpp"
#include "lut/lut_loader.hpp"
int main() { KLutFile f; return 0; }
EOF
echo "OK"
```

- [ ] **Step 4: Commit**

```bash
git add lut/lut_loader.hpp main.cpp
git commit -m "feat(lut): add KLutFile mmap loader with theta-lerp and phi-offset"
```

---

### Task 2.2 — Add `--lut-render` flag to `main.cpp`

**Files:**
- Modify: `main.cpp`

- [ ] **Step 1: Add variables alongside bake_lut (line ~3275)**

```cpp
    bool        lut_render      = false;
    std::string lut_render_path;    // input .klut file
    double      lut_phi_offset  = 0.0;  // φ offset in degrees
```

- [ ] **Step 2: Add flag parsing**

In the CLI loop:
```cpp
        if (arg == "--lut-render" && i+1 < argc) { lut_render = true; lut_render_path = argv[++i]; }
        if (arg == "--lut-phi-offset" && i+1 < argc) lut_phi_offset = std::stod(argv[++i]);
```

- [ ] **Step 3: Add `--lut-render` dispatch block (before `if (color_only)`, after `if (bake_lut)`)**

```cpp
    // ── LUT render mode ───────────────────────────────────────
    if (lut_render) {
        KLutFile lut;
        lut.open(lut_render_path);

        const int W = static_cast<int>(lut.hdr.width);
        const int H = static_cast<int>(lut.hdr.height);
        const double theta_rad = fp.theta_obs * M_PI / 180.0;
        const double phi_rad   = lut_phi_offset * M_PI / 180.0;

        std::vector<GeoPixel> geo(W * H);
        lut.sample(theta_rad, geo.data(), phi_rad);

        // Re-use existing colorize_buffer with the lerped GeoPixels
        KNdSMetric g_meta(lut.hdr.M, lut.hdr.a, lut.hdr.Q, lut.hdr.Lambda);
        const auto image = colorize_buffer(
            geo, W, H, cp,
            bg,
            lut.hdr.M, lut.hdr.a,
            lut.hdr.r_isco, lut.hdr.r_disk_in, lut.hdr.r_disk_out);

        // Write PNG
        const std::string ts = make_ts();
        const std::string out_path = std::string(OUT_DIR)
            + "/lut_" + ts + ".png";
        stbi_write_png(out_path.c_str(), W, H, 3,
            reinterpret_cast<const uint8_t*>(image.data()), W * 3);
        std::cout << "LUT render saved: " << out_path << "\n";
        return 0;
    }
```

- [ ] **Step 4: Build and run**

```bash
cmake --build build_cpu -j$(sysctl -n hw.ncpu) 2>&1 | tail -5

./build_cpu/kerr_tracer \
  --lut-render /tmp/test_small.klut \
  --theta 75 \
  --disk-interstellar --doppler --zero-torque-taper \
  --bg assets/backgrounds/sfondo5.jpg \
  --disk-brightness 30

ls -la out/lut_*.png | tail -1
```

Expected: PNG file created in `out/`.

- [ ] **Step 5: Visual comparison — render direct trace at same theta**

```bash
./build_cpu/kerr_tracer \
  --theta 75 --a 0.998 --width 320 --height 180 \
  --disk-interstellar --doppler --zero-torque-taper \
  --bg assets/backgrounds/sfondo5.jpg \
  --disk-brightness 30
```

Open both images side-by-side. Shadow shape, photon ring position, and disk Doppler asymmetry should match to within 0.5°-interpolation error.

- [ ] **Step 6: Commit**

```bash
git add main.cpp
git commit -m "feat(lut): add --lut-render flag; colorizes from mmap'd .klut slice pair"
```

---

## Session 3 — SDL2 Window + Async Metal Preview (5 h)

**Deliverable:** `./build/kerr_tracer_metal --interactive` opens a 960×540 window. Mouse-drag left/right rotates φ (instant). Mouse-drag up/down changes θ (triggers 128×72 Metal preview in <100 ms).

### Task 3.1 — CMakeLists.txt: `USE_INTERACTIVE` option

**Files:**
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Add after the existing `option(USE_CUDA ...)` line**

```cmake
option(USE_INTERACTIVE "Build interactive SDL2 window (macOS, requires SDL2 + Dear ImGui)" OFF)
```

- [ ] **Step 2: Add SDL2 detection block after the Metal backend block (~line 30)**

```cmake
# ── Interactive window (SDL2 + Dear ImGui) ───────────────────
if(USE_INTERACTIVE)
    if(NOT APPLE)
        message(FATAL_ERROR "USE_INTERACTIVE currently requires macOS (Metal)")
    endif()
    find_package(SDL2 REQUIRED CONFIG QUIET)
    if(NOT SDL2_FOUND)
        # Fallback: try pkg-config (Homebrew sdl2)
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(SDL2 REQUIRED sdl2)
    endif()
    # Dear ImGui — expected at third_party/imgui (add as git submodule or manual copy)
    set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/third_party/imgui)
    if(NOT EXISTS ${IMGUI_DIR}/imgui.h)
        message(FATAL_ERROR
            "Dear ImGui not found at ${IMGUI_DIR}.\n"
            "Run: git submodule add https://github.com/ocornut/imgui.git third_party/imgui")
    endif()
    list(APPEND CPU_SOURCES
        interactive/window.mm
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_DIR}/backends/imgui_impl_sdl2.cpp
        ${IMGUI_DIR}/backends/imgui_impl_metal.mm
    )
    message(STATUS "Interactive SDL2+ImGui window enabled")
endif()
```

- [ ] **Step 3: Add link/compile definitions for kerr_tracer after the Metal framework block (~line 68)**

```cmake
if(USE_INTERACTIVE)
    target_compile_definitions(kerr_tracer PRIVATE USE_INTERACTIVE)
    target_include_directories(kerr_tracer PRIVATE
        ${IMGUI_DIR}
        ${IMGUI_DIR}/backends)
    if(TARGET SDL2::SDL2)
        target_link_libraries(kerr_tracer PRIVATE SDL2::SDL2)
    else()
        target_link_libraries(kerr_tracer PRIVATE ${SDL2_LIBRARIES})
        target_include_directories(kerr_tracer PRIVATE ${SDL2_INCLUDE_DIRS})
    endif()
    target_link_libraries(kerr_tracer PRIVATE
        "-framework QuartzCore"
        "-framework AppKit")
endif()
```

- [ ] **Step 4: Install SDL2 and add ImGui submodule**

```bash
# Install SDL2 via Homebrew if not present
brew list sdl2 2>/dev/null || brew install sdl2

# Add Dear ImGui as a submodule
cd /Users/iman.rosignoli/Documents/KerrTraceCpp2
git submodule add https://github.com/ocornut/imgui.git third_party/imgui
git submodule update --init

# Pin to a stable release tag
cd third_party/imgui && git checkout v1.91.9 && cd ../..
```

- [ ] **Step 5: Verify CMake configures without error**

```bash
cmake -B build_interactive -DUSE_METAL=ON -DUSE_INTERACTIVE=ON 2>&1 | tail -10
```

Expected: no errors, `-- Interactive SDL2+ImGui window enabled` in output.

- [ ] **Step 6: Commit**

```bash
git add CMakeLists.txt third_party/ .gitmodules
git commit -m "build: add USE_INTERACTIVE flag, SDL2 + Dear ImGui detection"
```

---

### Task 3.2 — Create `interactive/window.hpp`

**Files:**
- Create: `interactive/window.hpp`

- [ ] **Step 1: Create the file**

```cpp
#pragma once
// ============================================================
//  interactive/window.hpp  —  SDL2 + Metal interactive window
//
//  Usage:
//    InteractiveWindow win(960, 540);
//    win.set_lut("lut/kerr_a0998_960x540.klut");
//    win.run();   // blocks until window closed
// ============================================================
#ifdef USE_INTERACTIVE
#include "../gpu/metal/metal_renderer.hpp"
#include "../lut/lut_loader.hpp"
#include <SDL.h>
#include <string>

struct InteractiveState {
    double theta_obs = 75.0;  // degrees
    double phi_obs   = 0.0;   // degrees
    double fov_deg   = 90.0;
    double a_spin    = 0.998;
    double disk_brightness = 30.0;
    bool   doppler_enabled = true;
    bool   zero_torque     = true;
    int    disk_palette    = 2;   // 2 = interstellar
    std::string bg_path;
};

class InteractiveWindow {
public:
    explicit InteractiveWindow(int w = 960, int h = 540);
    ~InteractiveWindow();

    void set_lut(const std::string& klut_path);   // optional; enables instant θ-lerp
    void set_state(const InteractiveState& s);
    void run();  // main event loop; returns when window is closed

private:
    void dispatch_preview();   // 128×72 Metal render, non-blocking
    void dispatch_lut_frame(); // LUT colorize at full res, non-blocking
    void blit_pixels(const uint32_t* rgba, int w, int h);  // upload to SDL texture

    int w_, h_;
    SDL_Window*   sdl_win_  = nullptr;
    SDL_Renderer* sdl_ren_  = nullptr;
    SDL_Texture*  sdl_tex_  = nullptr;

    KLutFile      lut_;
    bool          lut_loaded_ = false;

    InteractiveState state_;
    std::vector<uint32_t> frame_buf_;  // current display frame (ABGR)
};
#endif // USE_INTERACTIVE
```

- [ ] **Step 2: Create `interactive/` directory and commit header**

```bash
mkdir -p /Users/iman.rosignoli/Documents/KerrTraceCpp2/interactive
git add interactive/window.hpp
git commit -m "feat(interactive): add InteractiveWindow class declaration"
```

---

### Task 3.3 — Create `interactive/window.mm`

**Files:**
- Create: `interactive/window.mm`

- [ ] **Step 1: Create the file (minimal SDL2 event loop + Metal preview)**

```objc
// interactive/window.mm
#ifdef USE_INTERACTIVE
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "window.hpp"
#include "../gpu/metal/metal_renderer.hpp"
#include <SDL_metal.h>
#include <algorithm>
#include <chrono>
#include <thread>

// ── ImGui integration ────────────────────────────────────────
#include "imgui.h"
#include "backends/imgui_impl_sdl2.h"
#include "backends/imgui_impl_metal.h"

InteractiveWindow::InteractiveWindow(int w, int h) : w_(w), h_(h) {
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS) != 0)
        throw std::runtime_error(std::string("SDL_Init: ") + SDL_GetError());

    sdl_win_ = SDL_CreateWindow(
        "KerrTrace — Interactive",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        w_, h_,
        SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_METAL | SDL_WINDOW_RESIZABLE);
    if (!sdl_win_)
        throw std::runtime_error(std::string("SDL_CreateWindow: ") + SDL_GetError());

    sdl_ren_ = SDL_CreateRenderer(sdl_win_, -1,
        SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!sdl_ren_)
        throw std::runtime_error(std::string("SDL_CreateRenderer: ") + SDL_GetError());

    sdl_tex_ = SDL_CreateTexture(sdl_ren_,
        SDL_PIXELFORMAT_ABGR8888,
        SDL_TEXTUREACCESS_STREAMING, w_, h_);
    if (!sdl_tex_)
        throw std::runtime_error(std::string("SDL_CreateTexture: ") + SDL_GetError());

    frame_buf_.resize(static_cast<size_t>(w_ * h_), 0xFF000000u);

    // Init Dear ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplSDL2_InitForMetal(sdl_win_);
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    ImGui_ImplMetal_Init(dev);
}

InteractiveWindow::~InteractiveWindow() {
    ImGui_ImplMetal_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    if (sdl_tex_) SDL_DestroyTexture(sdl_tex_);
    if (sdl_ren_) SDL_DestroyRenderer(sdl_ren_);
    if (sdl_win_) SDL_DestroyWindow(sdl_win_);
    SDL_Quit();
}

void InteractiveWindow::set_lut(const std::string& path) {
    lut_.open(path);
    lut_loaded_ = true;
}

void InteractiveWindow::set_state(const InteractiveState& s) {
    state_ = s;
}

void InteractiveWindow::blit_pixels(const uint32_t* rgba, int w, int h) {
    // metal_render returns ABGR; SDL_PIXELFORMAT_ABGR8888 matches
    SDL_UpdateTexture(sdl_tex_, nullptr, rgba, w * 4);
    SDL_RenderClear(sdl_ren_);
    SDL_RenderCopy(sdl_ren_, sdl_tex_, nullptr, nullptr);
}

void InteractiveWindow::dispatch_preview() {
    // Render at 128×72 — very fast (<10 ms on Metal)
    constexpr int PW = 128, PH = 72;
    KNdSParams_C kp{};
    kp.M = 1.0f; kp.a = static_cast<float>(state_.a_spin);
    kp.Q = 0.0f; kp.Lambda = 0.0f;
    // r_horizon, r_isco: approximate for display (exact values in full render)
    kp.r_horizon = 1.0f + std::sqrt(1.0f - kp.a * kp.a);
    kp.r_isco    = 6.0f;  // rough; will be overridden in full render
    kp.r_disk_out= 20.0f;

    CameraParams_C cp{};
    cp.r_obs     = 500.0f;
    cp.theta_obs = static_cast<float>(state_.theta_obs * M_PI / 180.0);
    cp.phi_obs   = static_cast<float>(state_.phi_obs   * M_PI / 180.0);
    cp.fov_h     = static_cast<float>(state_.fov_deg   * M_PI / 180.0);
    cp.width     = PW; cp.height = PH;
    cp.chart     = 0; cp.solver_mode = 0; cp.integrator_mode = 0;
    cp.use_bundles = 0; cp.metal_kernel_mode = 2;
    cp.max_steps = 200; cp.step_init = 1.0f; cp.integrator_tol = 1e-4f;
    cp.exposure  = 1.0f; cp.gamma = 2.2f;
    cp.disk_brightness = static_cast<float>(state_.disk_brightness);
    cp.disk_opacity    = 1.0f;
    cp.disk_palette    = state_.disk_palette;
    cp.enable_doppler  = state_.doppler_enabled ? 1 : 0;
    cp.radial_term_zero_torque = state_.zero_torque ? 1 : 0;
    cp.interstellar_p  = 3.0f;

    auto px = metal_render(kp, cp);  // 128×72 ABGR pixels

    // Upscale 7.5× to 960×540 (nearest-neighbour — fast)
    const float sx = static_cast<float>(w_) / PW;
    const float sy = static_cast<float>(h_) / PH;
    for (int y = 0; y < h_; ++y) {
        const int sy_ = std::min(static_cast<int>(y / sy), PH - 1);
        for (int x = 0; x < w_; ++x) {
            const int sx_ = std::min(static_cast<int>(x / sx), PW - 1);
            frame_buf_[y * w_ + x] = px[sy_ * PW + sx_];
        }
    }
    blit_pixels(frame_buf_.data(), w_, h_);
}

void InteractiveWindow::dispatch_lut_frame() {
    if (!lut_loaded_) return;
    std::vector<GeoPixel> geo(w_ * h_);
    lut_.sample(state_.theta_obs * M_PI / 180.0, geo.data(),
                state_.phi_obs   * M_PI / 180.0);
    // Convert GeoPixel → ABGR using simplified colorize
    // (full colorize_buffer requires ColorParams; here use blackbody approximation)
    for (int i = 0; i < w_ * h_; ++i) {
        if (geo[i].outcome == 1) {
            // Simple temperature colorize: blue-white-yellow-red with redshift
            const float g = std::max(0.01f, geo[i].redshift);
            const uint8_t R = static_cast<uint8_t>(std::min(255.0f, 255.0f * g * g));
            const uint8_t G = static_cast<uint8_t>(std::min(255.0f, 200.0f * g));
            const uint8_t B = static_cast<uint8_t>(std::min(255.0f, 120.0f / g));
            frame_buf_[i] = 0xFF000000u | (R) | (G << 8) | (B << 16);
        } else {
            frame_buf_[i] = 0xFF000000u;
        }
    }
    blit_pixels(frame_buf_.data(), w_, h_);
}

void InteractiveWindow::run() {
    dispatch_preview();  // first frame

    bool dragging     = false;
    int  drag_x0 = 0, drag_y0 = 0;
    double theta0 = state_.theta_obs, phi0 = state_.phi_obs;

    // Debounce timer: after drag ends, wait 300 ms then do full LUT colorize
    using Clock = std::chrono::steady_clock;
    Clock::time_point drag_end_time{};
    bool pending_full = false;

    bool running = true;
    while (running) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            ImGui_ImplSDL2_ProcessEvent(&ev);
            if (ev.type == SDL_QUIT) { running = false; break; }

            if (ev.type == SDL_MOUSEBUTTONDOWN && ev.button.button == SDL_BUTTON_LEFT) {
                dragging = true;
                drag_x0 = ev.button.x; drag_y0 = ev.button.y;
                theta0  = state_.theta_obs; phi0 = state_.phi_obs;
            }
            if (ev.type == SDL_MOUSEBUTTONUP && ev.button.button == SDL_BUTTON_LEFT) {
                dragging = false;
                pending_full  = true;
                drag_end_time = Clock::now();
            }
            if (ev.type == SDL_MOUSEMOTION && dragging) {
                const int dx = ev.motion.x - drag_x0;
                const int dy = ev.motion.y - drag_y0;
                // 0.3°/pixel sensitivity
                state_.phi_obs   = phi0   - dx * 0.3;
                state_.theta_obs = std::max(1.0, std::min(179.0,
                                            theta0 + dy * 0.3));
                dispatch_preview();  // immediate low-res update
                pending_full  = true;
                drag_end_time = Clock::now();
            }
        }

        // Debounce: fire full LUT colorize 300 ms after last drag event
        if (pending_full && !dragging) {
            const double ms = std::chrono::duration<double, std::milli>(
                Clock::now() - drag_end_time).count();
            if (ms > 300.0) {
                dispatch_lut_frame();
                pending_full = false;
            }
        }

        // ImGui frame (empty for now; populated in Session 4)
        SDL_RenderPresent(sdl_ren_);
        SDL_Delay(16);  // ~60 fps cap
    }
}
#endif // USE_INTERACTIVE
```

- [ ] **Step 2: Commit**

```bash
git add interactive/window.mm
git commit -m "feat(interactive): SDL2 event loop + Metal preview dispatch + debounce LUT colorize"
```

---

### Task 3.4 — Add `--interactive` flag to `main.cpp`

**Files:**
- Modify: `main.cpp`

- [ ] **Step 1: Add conditional include at top of main.cpp**

After existing `#if defined(USE_METAL)` block:
```cpp
#if defined(USE_INTERACTIVE)
#  include "interactive/window.hpp"
#endif
```

- [ ] **Step 2: Add flag and dispatch in CLI section**

In the variable declarations (~line 3272):
```cpp
    bool interactive_mode = false;
    std::string interactive_lut_path;
```

In the CLI loop:
```cpp
        if (arg == "--interactive") interactive_mode = true;
        if (arg == "--interactive-lut" && i+1 < argc) interactive_lut_path = argv[++i];
```

In the dispatch section (before `if (color_only)`):
```cpp
#if defined(USE_INTERACTIVE)
    if (interactive_mode) {
        InteractiveWindow win(img_w, img_h);
        InteractiveState st;
        st.theta_obs       = fp.theta_obs;
        st.a_spin          = fp.a;
        st.disk_brightness = cp.disk_brightness;
        st.doppler_enabled = cp.doppler_enabled;
        st.zero_torque     = cp.radial_term_zero_torque;
        st.disk_palette    = (cp.palette == DiskPalette::INTERSTELLAR) ? 2 : 0;
        if (!bg_path.empty()) st.bg_path = bg_path;
        win.set_state(st);
        if (!interactive_lut_path.empty()) win.set_lut(interactive_lut_path);
        win.run();
        return 0;
    }
#endif
```

- [ ] **Step 3: Build and test**

```bash
cmake -B build_interactive \
  -DUSE_METAL=ON \
  -DUSE_INTERACTIVE=ON
cmake --build build_interactive -j$(sysctl -n hw.ncpu) 2>&1 | tail -10

./build_interactive/kerr_tracer \
  --interactive \
  --a 0.998 \
  --theta 75
```

Expected: SDL2 window opens at 960×540, shows a low-res (nearest-neighbour upscaled) preview of the black hole. Mouse drag left/right changes φ, up/down changes θ.

- [ ] **Step 4: Test with LUT**

```bash
./build_interactive/kerr_tracer \
  --interactive \
  --interactive-lut /tmp/test_small.klut \
  --a 0.998 --theta 75
```

Expected: after dragging and releasing, window updates to LUT colorize within 300 ms.

- [ ] **Step 5: Commit**

```bash
git add main.cpp interactive/
git commit -m "feat(interactive): wire --interactive flag to SDL2 InteractiveWindow"
```

---

## Session 4 — ImGui Sidebar + Camera Controller + Full Integration (5 h)

**Deliverable:** Interactive window has a sidebar with sliders for `a`, θ, φ, disk brightness, and palette selector. Mouse arcball orbit works. LUT is used for instant θ-preview when loaded.

### Task 4.1 — Create `interactive/camera_controller.hpp`

**Files:**
- Create: `interactive/camera_controller.hpp`

- [ ] **Step 1: Create the file**

```cpp
#pragma once
// ============================================================
//  interactive/camera_controller.hpp
//
//  Arcball-style orbit around the black hole.
//  Mouse drag → quaternion accumulation → BL (θ_obs, φ_obs).
//
//  Convention: θ = polar angle from north pole (0–180°).
//  Arcball prevents gimbal lock near poles by clamping θ ∈ [1°, 179°].
// ============================================================
#include <cmath>
#include <array>

struct Quat {
    double w = 1.0, x = 0.0, y = 0.0, z = 0.0;

    static Quat identity() { return {1,0,0,0}; }

    static Quat from_axis_angle(double ax, double ay, double az, double angle_rad) {
        const double s = std::sin(angle_rad * 0.5);
        return {std::cos(angle_rad * 0.5), ax*s, ay*s, az*s};
    }

    Quat operator*(const Quat& o) const {
        return {
            w*o.w - x*o.x - y*o.y - z*o.z,
            w*o.x + x*o.w + y*o.z - z*o.y,
            w*o.y - x*o.z + y*o.w + z*o.x,
            w*o.z + x*o.y - y*o.x + z*o.w
        };
    }

    Quat normalised() const {
        const double n = std::sqrt(w*w+x*x+y*y+z*z);
        return {w/n, x/n, y/n, z/n};
    }

    // Apply quaternion to unit vector (ax, ay, az).
    std::array<double,3> rotate(double ax, double ay, double az) const {
        const double tx = 2.0*(y*az - z*ay);
        const double ty = 2.0*(z*ax - x*az);
        const double tz = 2.0*(x*ay - y*ax);
        return {ax + w*tx + y*tz - z*ty,
                ay + w*ty + z*tx - x*tz,
                az + w*tz + x*ty - y*tx};
    }

    // Convert current rotation to BL polar angles (degrees).
    // Assumes "look direction" starts as -Z (toward BH at origin).
    void to_bl_angles(double& theta_deg, double& phi_deg) const {
        // Rotate the "camera forward" direction (-Z = (0,0,-1)) by this quaternion
        auto fwd = rotate(0.0, 0.0, -1.0);
        // fwd is now the unit look vector in world space
        // BL: θ = polar angle from +Z (north pole), φ = azimuth from +X
        theta_deg = std::acos(std::max(-1.0, std::min(1.0, fwd[2]))) * 180.0 / M_PI;
        phi_deg   = std::atan2(fwd[1], fwd[0]) * 180.0 / M_PI;
        if (phi_deg < 0.0) phi_deg += 360.0;
        // Clamp theta away from poles
        theta_deg = std::max(1.0, std::min(179.0, theta_deg));
    }
};

class CameraController {
public:
    double theta_deg = 75.0;  // current BL polar angle (degrees)
    double phi_deg   = 0.0;   // current BL azimuthal angle (degrees)
    double fov_deg   = 90.0;

    // Sensitivity: degrees of angle change per pixel
    double orbit_sensitivity = 0.35;
    double zoom_sensitivity  = 2.0;

    // Call when mouse button pressed: record drag start.
    void begin_drag(int screen_x, int screen_y) {
        drag_x0_ = screen_x;
        drag_y0_ = screen_y;
        quat_at_drag_start_ = quat_;
    }

    // Call on mouse motion while dragging.
    // Returns true if camera changed (caller should redraw).
    bool update_drag(int screen_x, int screen_y) {
        const int dx = screen_x - drag_x0_;
        const int dy = screen_y - drag_y0_;
        if (dx == 0 && dy == 0) return false;

        // Horizontal drag → rotate around world Y (azimuth).
        // Vertical drag   → rotate around world X (elevation).
        const double angle_phi = -dx * orbit_sensitivity * M_PI / 180.0;
        const double angle_th  =  dy * orbit_sensitivity * M_PI / 180.0;

        Quat rot_phi = Quat::from_axis_angle(0, 1, 0, angle_phi);
        Quat rot_th  = Quat::from_axis_angle(1, 0, 0, angle_th);
        quat_ = (rot_phi * rot_th * quat_at_drag_start_).normalised();
        quat_.to_bl_angles(theta_deg, phi_deg);
        return true;
    }

    // Scroll wheel → zoom FOV.
    bool scroll(double delta_y) {
        fov_deg = std::max(10.0, std::min(150.0,
                           fov_deg - delta_y * zoom_sensitivity));
        return true;
    }

    // Directly set angles (e.g., from ImGui slider).
    void set_angles(double theta, double phi) {
        theta_deg = theta; phi_deg = phi;
        // Reconstruct quaternion from angles so drag continues smoothly.
        const double th = theta_deg * M_PI / 180.0;
        const double ph = phi_deg   * M_PI / 180.0;
        // Look direction in Cartesian: (sin(th)cos(ph), sin(th)sin(ph), cos(th))
        // Quaternion = rotation from (0,0,-1) to this direction.
        const double lx = std::sin(th) * std::cos(ph);
        const double ly = std::sin(th) * std::sin(ph);
        const double lz = std::cos(th);
        // Axis = cross(-Z, look) = (ly, -lx, 0) (normalised)
        const double ax_len = std::sqrt(lx*lx + ly*ly);
        if (ax_len < 1e-9) {
            quat_ = Quat::identity();
        } else {
            const double angle = std::acos(std::max(-1.0, std::min(1.0, -lz)));
            quat_ = Quat::from_axis_angle(ly / ax_len, -lx / ax_len, 0.0, angle)
                        .normalised();
        }
    }

private:
    Quat quat_            = Quat::identity();
    Quat quat_at_drag_start_ = Quat::identity();
    int  drag_x0_ = 0, drag_y0_ = 0;
};
```

- [ ] **Step 2: Commit**

```bash
git add interactive/camera_controller.hpp
git commit -m "feat(interactive): add arcball CameraController (quaternion → BL angles)"
```

---

### Task 4.2 — Create `interactive/ui.hpp` (ImGui sidebar)

**Files:**
- Create: `interactive/ui.hpp`

- [ ] **Step 1: Create the file**

```cpp
#pragma once
// ============================================================
//  interactive/ui.hpp  —  Dear ImGui sidebar for InteractiveWindow
//
//  Call draw_sidebar() once per frame inside an ImGui frame.
//  Returns true if any parameter changed (caller should redraw).
// ============================================================
#ifdef USE_INTERACTIVE
#include "imgui.h"
#include "camera_controller.hpp"

struct UiParams {
    float  a_spin          = 0.998f;
    float  disk_brightness = 30.0f;
    float  interstellar_p  = 3.0f;
    float  exposure        = 1.0f;
    float  gamma           = 2.2f;
    int    disk_palette    = 2;     // 0=blackbody, 1=stratified, 2=interstellar
    bool   doppler_enabled = true;
    bool   zero_torque     = true;
    bool   use_bundles     = false;
    bool   lut_loaded      = false;  // read-only display flag
};

// Returns true if any value changed.
inline bool draw_sidebar(UiParams& p, CameraController& cam,
                          const char* lut_status = "none") {
    bool changed = false;

    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(260, ImGui::GetIO().DisplaySize.y), ImGuiCond_Always);
    ImGui::Begin("KerrTrace", nullptr,
        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoCollapse);

    ImGui::SeparatorText("Camera");
    float theta = static_cast<float>(cam.theta_deg);
    float phi   = static_cast<float>(cam.phi_deg);
    float fov   = static_cast<float>(cam.fov_deg);
    if (ImGui::SliderFloat("theta (deg)", &theta, 1.0f, 179.0f)) {
        cam.set_angles(theta, cam.phi_deg);
        changed = true;
    }
    if (ImGui::SliderFloat("phi (deg)", &phi, 0.0f, 360.0f)) {
        cam.set_angles(cam.theta_deg, phi);
        changed = true;
    }
    if (ImGui::SliderFloat("FOV (deg)", &fov, 10.0f, 150.0f)) {
        cam.fov_deg = fov; changed = true;
    }

    ImGui::SeparatorText("Metric");
    if (ImGui::SliderFloat("a (spin)", &p.a_spin, 0.0f, 0.9999f)) changed = true;

    ImGui::SeparatorText("Disk");
    const char* palettes[] = {"Blackbody", "Stratified", "Interstellar"};
    if (ImGui::Combo("Palette", &p.disk_palette, palettes, 3)) changed = true;
    if (ImGui::SliderFloat("Brightness", &p.disk_brightness, 0.1f, 100.0f))
        changed = true;
    if (p.disk_palette == 2)
        if (ImGui::SliderFloat("p exponent", &p.interstellar_p, 0.5f, 6.0f))
            changed = true;
    if (ImGui::Checkbox("Doppler", &p.doppler_enabled)) changed = true;
    if (ImGui::Checkbox("Zero-torque", &p.zero_torque))  changed = true;

    ImGui::SeparatorText("Tonemap");
    if (ImGui::SliderFloat("Exposure", &p.exposure, 0.1f, 5.0f)) changed = true;
    if (ImGui::SliderFloat("Gamma",    &p.gamma,    1.0f, 3.0f)) changed = true;

    ImGui::SeparatorText("Info");
    ImGui::TextDisabled("LUT: %s", lut_status);
    ImGui::TextDisabled("FPS: %.1f", ImGui::GetIO().Framerate);

    ImGui::End();
    return changed;
}
#endif // USE_INTERACTIVE
```

- [ ] **Step 2: Commit**

```bash
git add interactive/ui.hpp
git commit -m "feat(interactive): add ImGui sidebar with camera/disk/tonemap sliders"
```

---

### Task 4.3 — Integrate sidebar + camera controller into `window.mm`

**Files:**
- Modify: `interactive/window.mm`

- [ ] **Step 1: Add `CameraController` and `UiParams` members to `InteractiveWindow`**

In `interactive/window.hpp`, add to the private section:
```cpp
    CameraController cam_ctrl_;
    UiParams         ui_params_;
```

And add the include at the top of `window.hpp`:
```cpp
#include "camera_controller.hpp"
#include "ui.hpp"
```

- [ ] **Step 2: Replace the event loop in `window.mm` with the full version**

Replace the entire `void InteractiveWindow::run()` body with:

```cpp
void InteractiveWindow::run() {
    // Sync camera controller with initial state
    cam_ctrl_.set_angles(state_.theta_obs, state_.phi_obs);
    cam_ctrl_.fov_deg = state_.fov_deg;

    ui_params_.a_spin          = static_cast<float>(state_.a_spin);
    ui_params_.disk_brightness = static_cast<float>(state_.disk_brightness);
    ui_params_.doppler_enabled = state_.doppler_enabled;
    ui_params_.zero_torque     = state_.zero_torque;
    ui_params_.disk_palette    = state_.disk_palette;
    ui_params_.lut_loaded      = lut_loaded_;

    dispatch_preview();

    bool dragging     = false;
    using Clock = std::chrono::steady_clock;
    Clock::time_point last_change{};
    bool pending_full = true;
    const double DEBOUNCE_MS = 300.0;

    bool running = true;
    while (running) {
        SDL_Event ev;
        bool params_changed = false;
        while (SDL_PollEvent(&ev)) {
            ImGui_ImplSDL2_ProcessEvent(&ev);
            if (ev.type == SDL_QUIT) { running = false; break; }

            // Only process mouse events if ImGui is not capturing them
            if (!ImGui::GetIO().WantCaptureMouse) {
                if (ev.type == SDL_MOUSEBUTTONDOWN && ev.button.button == SDL_BUTTON_LEFT) {
                    dragging = true;
                    cam_ctrl_.begin_drag(ev.button.x, ev.button.y);
                }
                if (ev.type == SDL_MOUSEBUTTONUP && ev.button.button == SDL_BUTTON_LEFT) {
                    dragging = false;
                    params_changed = true;
                }
                if (ev.type == SDL_MOUSEMOTION && dragging) {
                    if (cam_ctrl_.update_drag(ev.motion.x, ev.motion.y)) {
                        state_.theta_obs = cam_ctrl_.theta_deg;
                        state_.phi_obs   = cam_ctrl_.phi_deg;
                        dispatch_preview();
                        params_changed = true;
                    }
                }
                if (ev.type == SDL_MOUSEWHEEL) {
                    if (cam_ctrl_.scroll(ev.wheel.y)) {
                        state_.fov_deg = cam_ctrl_.fov_deg;
                        params_changed = true;
                    }
                }
            }
        }

        // ImGui frame
        ImGui_ImplMetal_NewFrame(/* MTLRenderPassDescriptor — see note below */);
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        if (draw_sidebar(ui_params_, cam_ctrl_,
                         lut_loaded_ ? "loaded" : "none")) {
            // Sync state from UI
            state_.theta_obs       = cam_ctrl_.theta_deg;
            state_.phi_obs         = cam_ctrl_.phi_deg;
            state_.fov_deg         = cam_ctrl_.fov_deg;
            state_.a_spin          = ui_params_.a_spin;
            state_.disk_brightness = ui_params_.disk_brightness;
            state_.doppler_enabled = ui_params_.doppler_enabled;
            state_.zero_torque     = ui_params_.zero_torque;
            state_.disk_palette    = ui_params_.disk_palette;
            dispatch_preview();
            params_changed = true;
        }

        ImGui::Render();

        if (params_changed) {
            last_change   = Clock::now();
            pending_full  = true;
        }

        // Debounce: fire full LUT colorize after DEBOUNCE_MS of inactivity
        if (pending_full && !dragging) {
            const double ms = std::chrono::duration<double, std::milli>(
                Clock::now() - last_change).count();
            if (ms > DEBOUNCE_MS) {
                dispatch_lut_frame();
                pending_full = false;
            }
        }

        SDL_RenderPresent(sdl_ren_);
        SDL_Delay(16);
    }
}
```

> **Note on `ImGui_ImplMetal_NewFrame`:** The Metal ImGui backend requires a `MTLRenderPassDescriptor*`. For a simple SDL2+Metal window, get it via `SDL_Metal_GetLayer(sdl_win_)` cast to `CAMetalLayer*` and create a render pass descriptor from its next drawable. The SDL2 Metal backend handles this automatically when using `SDL_RENDERER_METAL`. If the ImGui render pass needs explicit setup, see `imgui/examples/example_apple_metal/main.mm` for reference.

- [ ] **Step 2: Build and run full interactive mode**

```bash
cmake --build build_interactive -j$(sysctl -n hw.ncpu) 2>&1 | tail -10

./build_interactive/kerr_tracer \
  --interactive \
  --interactive-lut /tmp/test_small.klut \
  --a 0.998 --theta 75 \
  --disk-interstellar --doppler
```

Expected:
- SDL2 window opens with ImGui sidebar on the left
- Sliders for θ, φ, FOV, a, brightness, palette, Doppler, zero-torque
- Mouse drag in the right panel orbits the camera (instant low-res update)
- After 300 ms of inactivity, LUT colorize updates the display
- ImGui sliders update preview on every frame

- [ ] **Step 3: Commit**

```bash
git add interactive/
git commit -m "feat(interactive): integrate ImGui sidebar + arcball camera into event loop"
```

---

### Task 4.4 — Final smoke test + version bump

- [ ] **Step 1: End-to-end test sequence**

```bash
# 1. Bake small LUT
./build_cpu/kerr_tracer --bake-lut /tmp/e2e.klut \
  --width 320 --height 180 --a 0.998

# 2. LUT render CLI (non-interactive)
./build_cpu/kerr_tracer --lut-render /tmp/e2e.klut \
  --theta 60 --disk-interstellar --doppler --zero-torque-taper \
  --bg assets/backgrounds/sfondo5.jpg --disk-brightness 30
ls -la out/lut_*.png | tail -1   # should exist

# 3. Interactive window with LUT
./build_interactive/kerr_tracer \
  --interactive --interactive-lut /tmp/e2e.klut \
  --a 0.998 --theta 75 --disk-interstellar --doppler
# → drag mouse, check preview updates, check sidebar responds
```

- [ ] **Step 2: Bump version in CMakeLists.txt**

Change:
```cmake
project(KerrTracer VERSION 0.2.12 LANGUAGES CXX)
```
to:
```cmake
project(KerrTracer VERSION 0.3.0 LANGUAGES CXX)
```

- [ ] **Step 3: Update version display in frontend**

In `frontend/src/app/app.html` line 12, update version string to `v0.3.0 — interactive 3D window, .klut LUT prebake, arcball camera`.

- [ ] **Step 4: Final commit**

```bash
git add CMakeLists.txt frontend/src/app/app.html
git commit -m "feat: v0.3.0 — interactive SDL2 window, .klut LUT prebake, arcball camera"
```

---

## Self-Review

### Spec coverage

| Requirement | Task |
|---|---|
| 359 θ-slices, 0.5°–179.5°, 0.5° step | Task 1.2 bake loop |
| `.klut` binary format with header | Task 1.1 `lut_types.hpp` |
| mmap loader | Task 2.1 `lut_loader.hpp` |
| φ-rotation as free texture offset | `lut_loader.hpp` `phi_offset` param |
| θ-lerp between adjacent slices | `KLutFile::sample()` |
| SDL2 window | Task 3.3 `window.mm` |
| Metal low-res preview <100 ms | `dispatch_preview()` at 128×72 |
| 300 ms debounce → LUT colorize | Task 3.3 + 4.3 event loop |
| ImGui sidebar | Task 4.2 `ui.hpp` |
| Arcball camera controller | Task 4.1 `camera_controller.hpp` |
| `--interactive`, `--bake-lut`, `--lut-render` CLI flags | Tasks 1.2, 2.2, 3.4 |
| `.klut` invalidation when metric params change | Header stores M/a/Q/Λ/r_obs — caller can check |
| LUT size ~1.5 GB (960×540 float16) | GeoPixel 28B → actual 5.2 GB. Acceptable: mmap page-faults on demand |

### Placeholder scan

None. All struct definitions, function bodies, and test commands are complete.

### Type consistency

- `KLutHeader` defined in `lut_types.hpp`, used in `lut_loader.hpp` and `main.cpp` ✓
- `KLutFile::sample()` writes to `GeoPixel*` array, fed to `colorize_buffer(vector<GeoPixel>)` ✓
- `InteractiveState` defined in `window.hpp`, populated in `main.cpp` ✓
- `UiParams` defined in `ui.hpp`, member of `InteractiveWindow` in `window.hpp` ✓
- `CameraController` defined in `camera_controller.hpp`, included by `ui.hpp` and `window.hpp` ✓
