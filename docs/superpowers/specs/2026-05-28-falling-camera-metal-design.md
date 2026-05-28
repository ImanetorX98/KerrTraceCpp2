# Falling Camera — Metal GPU Backend Design Spec
*Date: 2026-05-28*

## Overview

Add a Metal GPU backend to the falling camera renderer (Phase D). A new standalone
bridge dispatches all pixels to a Metal compute kernel (float precision), then a CPU
refinement pass re-traces near-horizon pixels in double precision and overwrites them.
The result: GPU-speed rendering with full double precision where it matters.

---

## Architecture

### New files

| File | Role |
|------|------|
| `gpu/metal/tracer_falling.metal` | MSL compute kernel: GPG metric (float), Christoffel finite-diff, photon RK4 trace, Page-Thorne shading |
| `gpu/metal/metal_falling_renderer.hpp` | C++ bridge declaration: `metal_render_falling_frame`, `make_falling_metal_params` |
| `gpu/metal/metal_falling_renderer.mm` | Obj-C++ bridge: device/queue/PSO init (lazy), buffer alloc, dispatch, readback |

### Modified files

| File | Change |
|------|--------|
| `falling_renderer.hpp` | Add `r_switch_factor` field to `FallingParams` (default 3.0) |
| `falling_renderer.cpp` | Extract `shade_falling_pixel()`; add `#ifdef USE_METAL` two-pass path in `render_falling_frame` |
| `CMakeLists.txt` | Add `metal_falling_renderer.mm` to Metal sources; copy `tracer_falling.metal` to build dir |

---

## Data Flow (two passes per frame)

```
CPU pre-frame:
  build_tetrad + apply_roll → e[4][4]   (double, once per frame)
  make_falling_metal_params → FallingCameraParams_C (cast to float)

Pass 1 — Metal (float, all pixels):
  FallingCameraParams_C → MTLBuffer (buffer 2)
  Kernel: GPG metric + Christoffel float + RK4 + shade
  Output buffer 0: uchar4 rgb[W×H]     (final colour per pixel)
  Output buffer 1: float  r_min[W×H]   (minimum r reached per pixel)

Pass 2 — CPU refinement (double, pixels where r_min < k·r_h):
  Scan r_min buffer, build mask
  For each masked pixel:
    init_photon_k (double)
    trace_photon_gpg (double, existing)
    shade_falling_pixel → overwrite rgb[idx]

stbi_write_png(rgb)
```

---

## Section 1: `FallingCameraParams_C` Struct

Mirrored identically in C++ and MSL. All fields are `float` or `int`.

```cpp
struct FallingCameraParams_C {
    float e[4][4];          // camera tetrad e[a][mu], pre-computed on CPU
    float x[4];             // camera position in GPG coordinates
    float M, a, Q, Lambda;  // BH parameters
    float r_in, r_out;      // disk inner/outer radius
    float r_isco;           // ISCO radius (pre-computed)
    float r_escape;
    float r_singularity;
    float r_horizon;        // r_+ (pre-computed)
    float disk_brightness;
    float fov_h;            // horizontal FOV in radians
    float h0;               // initial affine step size
    float r_switch_factor;  // CPU refinement threshold: r_min < factor * r_h
    int   max_steps;
    int   width, height;
    int   pad;              // 16-byte alignment padding
};
```

`make_falling_metal_params(fp, cs, e)` casts from double to float and fills all fields.

---

## Section 2: Metal Kernel (`tracer_falling.metal`)

### Helper functions (all `static inline`, all `float`)

```
gpg_f_f(M,a,Q,Lambda,r)
  → f(r) per Λ≥0: sqrt(R²+Q²+Λa⁴/3)
  → f(r) per Λ<0: sqrt(R²(1-Λr²/3)+Q²)

gpg_covariant_f(M,a,Q,Lambda,r,theta, out gLL[4][4])
  → same tetrad decomposition as CPU gpg_covariant, float arithmetic

gpg_contravariant_f(M,a,Q,Lambda,r,theta, out gUU[4][4])
  → 4×4 cofactor inverse (same as CPU, float)

gpg_christoffel_f(M,a,Q,Lambda,r,theta, out Gamma[4][4][4])
  → central finite differences
  → hr = r * 1e-4f + 1e-5f   (larger than CPU to avoid float cancellation)
  → ht = 1e-4f

photon_init_f(px,py,fp, out k[4], out x[4])
  → pixel → (alpha,beta) → tetrad direction → k^mu
  → rescale k^T to enforce null condition

photon_step_f(fp,M,a,Q,Lambda, inout x[4], inout k[4], dlam)
  → RK4 step using gpg_christoffel_f

shade_pixel_f(outcome,r_hit,redshift,r_isco,fp)
  → Page-Thorne lum ∝ (1 - sqrt(r_isco/r)) / r³
  → g⁴ redshift factor
  → blueshift (g>1): white-yellow; redshift (g<1): orange-red
  → returns uchar4
```

### Kernel entry point

```metal
kernel void trace_falling_pixel(
    device uchar4*                rgb_out  [[buffer(0)]],
    device float*                 rmin_out [[buffer(1)]],
    constant FallingCameraParams& fp       [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= uint(fp.width) || gid.y >= uint(fp.height)) return;

    float k[4], x[4];
    photon_init_f(gid.x, gid.y, fp, k, x);

    float r_min = x[1];
    float dlam  = fp.h0;
    int   outcome = 3;
    float r_hit = 0, phi_hit = 0, redshift = 1.0f;
    float prev_theta = x[2], prev_r = x[1];

    for (int step = 0; step < fp.max_steps; ++step) {
        if (x[1] < r_min) r_min = x[1];

        if (x[1] >= fp.r_escape) {
            outcome = 0; r_hit = x[1]; break;
        }
        if (x[1] < fp.r_singularity) {
            outcome = 2; break;
        }

        float cur_theta = x[2];
        if (x[1] >= fp.r_in && x[1] <= fp.r_out) {
            float dp = prev_theta - M_PI_F / 2.0f;
            float dc = cur_theta  - M_PI_F / 2.0f;
            if (dp * dc < 0.0f) {
                outcome = 1;
                float t = fabs(dp) / (fabs(dp) + fabs(dc));
                r_hit = prev_r + t * (x[1] - prev_r);
                phi_hit = x[3];
                // compute redshift g-factor (Keplerian Omega, g_LL at r_hit)
                // ... (same formula as CPU trace_photon_gpg)
                break;
            }
        }
        prev_theta = x[2];
        prev_r     = x[1];
        dlam = (x[1] < fp.r_horizon * 3.0f) ? 0.005f : fp.h0;
        photon_step_f(fp.M, fp.a, fp.Q, fp.Lambda, x, k, dlam);
    }

    uint idx = gid.y * uint(fp.width) + gid.x;
    rgb_out [idx] = shade_pixel_f(outcome, r_hit, redshift, fp.r_isco, fp);
    rmin_out[idx] = r_min;
}
```

### Threadgroup size: 16×16 threads. Grid: `ceil(W/16) × ceil(H/16)`.

---

## Section 3: Bridge (`metal_falling_renderer.mm`)

Lazy initialization (first call only):
- `MTLCreateSystemDefaultDevice()`
- Load `tracer_falling.metal` from `[[NSBundle mainBundle]]` path (same as existing bridge)
- Compile pipeline state for `trace_falling_pixel`

Per-frame:
1. Allocate or reuse `MTLBuffer` for rgb_out (W×H×4 bytes), rmin_out (W×H×4 bytes), params (sizeof struct)
2. Encode compute pass, set buffers 0/1/2, dispatch threadgroups
3. `commandBuffer waitUntilCompleted`
4. Copy rgb_out (uchar4 → uchar3 stripping alpha) into `std::vector<uint8_t>& rgb`
5. memcpy rmin_out into `std::vector<float>& r_min`
6. Return true

Returns false (and leaves rgb/r_min unchanged) if:
- No Metal device available
- Shader file not found
- Pipeline compilation fails

---

## Section 4: Changes to `falling_renderer.cpp`

### Extract `shade_falling_pixel()`

```cpp
static std::tuple<uint8_t,uint8_t,uint8_t>
shade_falling_pixel(const FallingGeoPixel& pix,
                    const FallingParams& fp,
                    double r_isco)
```

Contains the existing shading logic (Page-Thorne + redshift color split).
Called from both the CPU path and the CPU refinement pass.

### Two-pass path in `render_falling_frame`

```cpp
#ifdef USE_METAL
    std::vector<float> r_min(W*H, (float)fp.r_escape);
    FallingCameraParams_C cp = make_falling_metal_params(fp, cs_at_frame, e);
    if (metal_render_falling_frame(cp, rgb, r_min)) {
        const double r_switch = fp.bh.r_horizon() * fp.r_switch_factor;
        int refined = 0;
        for (int py = 0; py < H; ++py)
            for (int px = 0; px < W; ++px) {
                int idx = py*W+px;
                if (r_min[idx] < r_switch) {
                    double k[4];
                    init_photon_k(cs_at_frame, e, fp.bh,
                                  px, py, W, H, fp.fov_h, k);
                    FallingGeoPixel pix = trace_photon_gpg(
                        cs_at_frame.x, k, fp.bh, fp, 50000, 0.05, 1e-7);
                    auto [R,G,B] = shade_falling_pixel(pix, fp, r_isco);
                    rgb[idx*3]=R; rgb[idx*3+1]=G; rgb[idx*3+2]=B;
                    ++refined;
                }
            }
        printf("[frame %04d/%04d] Metal+CPU refine=%d/%.0f%%\n", ...);
        stbi_write_png(...);
        return;
    }
    // Metal unavailable → fall through to CPU path
#endif
    // existing CPU path unchanged
```

---

## Section 5: `FallingParams` addition

Add one field to `FallingParams` in `falling_renderer.hpp`:

```cpp
double r_switch_factor = 3.0;  // CPU refinement threshold (× r_horizon)
```

---

## Section 6: CMakeLists.txt

In `if(USE_METAL)` block:
```cmake
list(APPEND CPU_SOURCES gpu/metal/metal_falling_renderer.mm)
configure_file(gpu/metal/tracer_falling.metal
               ${CMAKE_BINARY_DIR}/tracer_falling.metal COPYONLY)
```

In `if(APPLE AND NOT USE_METAL)` block (kerr_tracer_metal target):
```cmake
target_sources(kerr_tracer_metal PRIVATE gpu/metal/metal_falling_renderer.mm)
configure_file(gpu/metal/tracer_falling.metal
               ${CMAKE_BINARY_DIR}/tracer_falling.metal COPYONLY)
```

---

## Testing

No new unit tests — physics is identical to CPU path already covered by `kerrtrace.falling`.

Existing `kerrtrace.falling_smoke` remains valid (runs CPU path, `USE_METAL` not required).

Manual verification after implementation:
```bash
# CPU reference
./build_cpu/kerr_tracer_metal --falling-camera --a 0.9 --fall-r-start 15 \
  --fall-frames 2 --fall-dtau 0.5 --custom-res 320 180

# Visual diff of PNG output: Metal+CPU vs pure CPU should be pixel-identical
# after refinement pass for near-horizon pixels.
```

---

## Key References

- Lin & Soo (2009) arXiv:0905.3244 — GPG coordinates (same as CPU falling camera)
- Apple Metal Shading Language Specification — MSL float math, threadgroup sizing
