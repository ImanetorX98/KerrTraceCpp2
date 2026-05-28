# Falling Camera — Metal GPU Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Metal GPU backend to the falling camera renderer: all pixels traced in float on the GPU, then near-horizon pixels re-traced in double on the CPU and overwritten.

**Architecture:** New standalone Metal bridge (`metal_falling_renderer.hpp/.mm`) + new MSL kernel (`tracer_falling.metal`). `render_falling_frame` gains an `#ifdef USE_METAL` two-pass path: Pass 1 dispatches the Metal kernel (outputs RGBA + `r_min` per pixel); Pass 2 re-traces CPU-double for pixels with `r_min < r_switch_factor * r_horizon`. Falls back to CPU-only if Metal unavailable.

**Tech Stack:** C++17, Objective-C++, Metal Shading Language 2.4, existing GPG physics in `falling_camera.hpp`, existing `trace_photon_gpg` in `falling_renderer.cpp`.

**Spec:** `docs/superpowers/specs/2026-05-28-falling-camera-metal-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `falling_camera.hpp` | Modify line 112 | Add `r_switch_factor = 3.0` to `FallingParams` |
| `gpu/metal/metal_falling_renderer.hpp` | Create | `FallingCameraParams_C` struct, `make_falling_metal_params`, `metal_render_falling_frame` declarations |
| `gpu/metal/tracer_falling.metal` | Create | MSL helpers + `trace_falling_pixel` kernel |
| `gpu/metal/metal_falling_renderer.mm` | Create | Obj-C++ Metal bridge: lazy init, dispatch, readback |
| `falling_renderer.cpp` | Modify | Extract `shade_falling_pixel`; add `#ifdef USE_METAL` two-pass path |
| `CMakeLists.txt` | Modify | Add `.mm` to Metal sources, copy `tracer_falling.metal` |

---

## Task 1: Add `r_switch_factor` to `FallingParams`

**Files:**
- Modify: `falling_camera.hpp:112`

- [ ] **Step 1: Add the field**

In `falling_camera.hpp`, find the line:
```cpp
    double disk_brightness = 1.0;
};
```

Replace with:
```cpp
    double disk_brightness   = 1.0;
    double r_switch_factor   = 3.0;  // CPU refinement: r_min < factor*r_h → re-trace double
};
```

- [ ] **Step 2: Build to verify no breakage**

```bash
cd /Users/iman.rosignoli/Documents/KerrTraceCpp2
cmake -B build_cpu -DUSE_METAL=OFF 2>&1 | tail -2
cmake --build build_cpu -j$(sysctl -n hw.ncpu) 2>&1 | grep -E "error:|Error" | head -5
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add falling_camera.hpp
git commit -m "feat(falling-metal): add r_switch_factor to FallingParams"
```

---

## Task 2: Create `metal_falling_renderer.hpp`

**Files:**
- Create: `gpu/metal/metal_falling_renderer.hpp`

- [ ] **Step 1: Create the header**

```cpp
#pragma once
// metal_falling_renderer.hpp — C++ bridge for the falling-camera Metal GPU backend.
// Build only on macOS with -DUSE_METAL.
#include "falling_renderer.hpp"
#include <vector>

// ── FallingCameraParams_C ─────────────────────────────────────────────────────
// POD struct mirrored exactly in MSL as FallingCameraParams.
// All fields float/int. 16-byte aligned (pad at end).
struct FallingCameraParams_C {
    float e[4][4];          // camera tetrad e[a][mu], pre-computed on CPU (double→float)
    float x[4];             // camera GPG position (T,r,θ,φ)
    float M, a, Q, Lambda;  // BH parameters
    float r_in;             // disk inner radius (ISCO when FallingParams.r_disk_in<0)
    float r_out;            // disk outer radius
    float r_isco;           // ISCO pre-computed
    float r_escape;
    float r_singularity;
    float r_horizon;        // outer horizon r_+
    float disk_brightness;
    float fov_h;            // horizontal FOV in radians
    float h0;               // initial affine step (GPU path uses 0.05f)
    float r_switch_factor;  // pixels with r_min < factor*r_horizon → CPU re-trace
    int   max_steps;        // hard cap (GPU path: 20000)
    int   width, height;
    int   pad;              // 16-byte alignment
};

// Build FallingCameraParams_C from C++ types (casts double→float).
FallingCameraParams_C make_falling_metal_params(
    const FallingParams& fp,
    const CameraState&   cs,
    const double         e[4][4]);

// Pass 1: dispatch Metal kernel for all pixels.
// Fills rgb (W*H*3 bytes) and r_min (W*H floats, minimum r per pixel).
// Returns false if Metal is unavailable or initialisation fails.
bool metal_render_falling_frame(
    const FallingCameraParams_C& params,
    std::vector<uint8_t>&        rgb,
    std::vector<float>&          r_min);
```

- [ ] **Step 2: Build to verify it compiles (CPU build includes it transitively via falling_renderer.hpp)**

```bash
cd /Users/iman.rosignoli/Documents/KerrTraceCpp2
cmake --build build_cpu -j$(sysctl -n hw.ncpu) 2>&1 | grep -E "error:" | head -5
```

Expected: no errors (the header isn't compiled yet, just present).

- [ ] **Step 3: Commit**

```bash
git add gpu/metal/metal_falling_renderer.hpp
git commit -m "feat(falling-metal): FallingCameraParams_C struct and bridge declarations"
```

---

## Task 3: Create `tracer_falling.metal` — GPG helper functions

**Files:**
- Create: `gpu/metal/tracer_falling.metal`

This task writes all MSL helper functions. The kernel entry point is added in Task 4.

- [ ] **Step 1: Create the file with helpers**

```metal
// ============================================================
//  tracer_falling.metal — Falling-camera GPG ray tracer (MSL)
//
//  One thread = one pixel.
//  Threadgroup size: 16×16.
//  All arithmetic in float. Near-horizon pixels are refined by
//  the CPU pass in falling_renderer.cpp (double precision).
//
//  Physics reference: Lin & Soo 2009, arXiv:0905.3244
// ============================================================
#include <metal_stdlib>
using namespace metal;

// ── FallingCameraParams (mirrors FallingCameraParams_C in C++) ───────────────
struct FallingCameraParams {
    float e[4][4];
    float x[4];
    float M, a, Q, Lambda;
    float r_in, r_out, r_isco, r_escape, r_singularity, r_horizon;
    float disk_brightness, fov_h, h0, r_switch_factor;
    int   max_steps, width, height, pad;
};

// ── Flat 4×4 index helper ─────────────────────────────────────────────────────
#define G4(arr,r,c)         ((arr)[(r)*4+(c)])
// Flat 4×4×4 index helper (for Gamma)
#define G444(arr,mu,al,be)  ((arr)[(mu)*16+(al)*4+(be)])

// ── gpg_f_f: GPG gauge function f(r) ─────────────────────────────────────────
static float gpg_f_f(float M, float a, float Q, float Lambda, float r) {
    float R2 = r*r + a*a;
    float Q2 = Q*Q;
    if (Lambda >= 0.0f)
        return sqrt(R2 + Q2 + Lambda*a*a*a*a/3.0f);
    else
        return sqrt(R2*(1.0f - Lambda*r*r/3.0f) + Q2);
}

// ── gpg_covariant_f: g_μν in GPG coordinates (flat output, row-major) ────────
// gLL[mu*4+nu], float precision.
static void gpg_covariant_f(float M, float a, float Q, float Lambda,
                              float r, float theta, thread float* gLL)
{
    float R2   = r*r + a*a;
    float Q2   = Q*Q;
    float ct   = cos(theta), st = sin(theta);
    float cos2 = ct*ct, sin2 = st*st;
    float rho2 = r*r + a*a*cos2;
    float Xi   = 1.0f + Lambda*a*a/3.0f;
    float Xit  = 1.0f + Lambda*a*a*cos2/3.0f;
    float Del  = R2*(1.0f - Lambda*r*r/3.0f) - 2.0f*M*r + Q2;
    float f    = gpg_f_f(M, a, Q, Lambda, r);
    float f2   = f*f;

    float D    = f2 - a*a*Xit*sin2;
    float sqD  = sqrt(max(D, 0.0f));
    float sqFD = sqrt(max(f2 - Del, 0.0f));
    float sqXit= sqrt(max(Xit, 0.0f));
    float rho  = sqrt(rho2);

    float e0T  = sqD / (Xi * rho);
    float e0ph = (sqD > 1e-10f) ? a*sin2*(Xit*R2 - f2) / (Xi*rho*sqD) : 0.0f;
    float e1T  = sqFD / (Xi * rho);
    float e1r  = rho / max(f, 1e-10f);
    float e1ph = -a*sin2*sqFD / max(Xi*rho, 1e-10f);
    float e2th = rho / max(sqXit, 1e-10f);
    float e3T  = (sqD > 1e-10f) ? -a*rho*sqXit*st / (f*Xi*sqD) : 0.0f;
    float e3ph = (sqD > 1e-10f) ?  rho*R2*sqXit*st / (f*Xi*sqD) : 0.0f;

    for (int i=0;i<16;++i) gLL[i] = 0.0f;
    G4(gLL,0,0) = -e0T*e0T + e1T*e1T + e3T*e3T;
    G4(gLL,1,1) = e1r*e1r;
    G4(gLL,2,2) = e2th*e2th;
    G4(gLL,3,3) = -e0ph*e0ph + e1ph*e1ph + e3ph*e3ph;
    G4(gLL,0,1) = G4(gLL,1,0) = e1T*e1r;
    G4(gLL,0,3) = G4(gLL,3,0) = -e0T*e0ph + e1T*e1ph + e3T*e3ph;
    G4(gLL,1,3) = G4(gLL,3,1) = e1r*e1ph;
}

// ── cofactor3_f: 3×3 minor from a flat 4×4 matrix ────────────────────────────
static float cofactor3_f(thread float* m, int r0, int c0) {
    float sub[9]; int ri = 0;
    for (int i=0;i<4;++i) {
        if (i==r0) continue;
        int ci=0;
        for (int j=0;j<4;++j) {
            if (j==c0) continue;
            sub[ri*3+ci++] = m[i*4+j];
        }
        ++ri;
    }
    return sub[0]*(sub[4]*sub[8]-sub[5]*sub[7])
          -sub[1]*(sub[3]*sub[8]-sub[5]*sub[6])
          +sub[2]*(sub[3]*sub[7]-sub[4]*sub[6]);
}

// ── gpg_contravariant_f: g^μν via 4×4 cofactor inverse ───────────────────────
static void gpg_contravariant_f(float M, float a, float Q, float Lambda,
                                  float r, float theta, thread float* gUU)
{
    float gLL[16];
    gpg_covariant_f(M, a, Q, Lambda, r, theta, gLL);

    float det = 0.0f;
    for (int j=0;j<4;++j)
        det += gLL[j] * cofactor3_f(gLL,0,j) * ((j%2==0) ? 1.0f : -1.0f);
    float inv_det = (abs(det) > 1e-20f) ? 1.0f/det : 0.0f;

    for (int i=0;i<4;++i)
        for (int j=0;j<4;++j)
            gUU[i*4+j] = cofactor3_f(gLL,j,i)
                         * (((i+j)%2==0) ? 1.0f : -1.0f)
                         * inv_det;
}

// ── gpg_christoffel_f: Γ^μ_{αβ} via central finite differences ───────────────
// Gamma[mu*16 + al*4 + be] (flat 4×4×4).
// Step sizes larger than CPU to avoid float cancellation: hr=r*1e-4+1e-5, ht=1e-4.
static void gpg_christoffel_f(float M, float a, float Q, float Lambda,
                                float r, float theta,
                                thread float* Gamma)
{
    float hr = r * 1e-4f + 1e-5f;
    float ht = 1e-4f;

    float gp[16], gm[16], gtp[16], gtm[16];
    gpg_covariant_f(M, a, Q, Lambda, r+hr, theta,    gp);
    gpg_covariant_f(M, a, Q, Lambda, r-hr, theta,    gm);
    gpg_covariant_f(M, a, Q, Lambda, r,    theta+ht, gtp);
    gpg_covariant_f(M, a, Q, Lambda, r,    theta-ht, gtm);

    float dgr[16], dgt[16];
    for (int i=0;i<16;++i) {
        dgr[i] = (gp[i]  - gm[i])  / (2.0f*hr);
        dgt[i] = (gtp[i] - gtm[i]) / (2.0f*ht);
    }

    float gUU[16];
    gpg_contravariant_f(M, a, Q, Lambda, r, theta, gUU);

    for (int mu=0;mu<4;++mu)
        for (int al=0;al<4;++al)
            for (int be=0;be<4;++be) {
                float s = 0.0f;
                for (int nu=0;nu<4;++nu) {
                    // ∂_α g_νβ: coord=1→r, 2→θ, else 0 (stationary+axisymmetric)
                    float dg_al = (al==1) ? dgr[nu*4+be]
                                : (al==2) ? dgt[nu*4+be] : 0.0f;
                    float dg_be = (be==1) ? dgr[nu*4+al]
                                : (be==2) ? dgt[nu*4+al] : 0.0f;
                    float dg_nu = (nu==1) ? dgr[al*4+be]
                                : (nu==2) ? dgt[al*4+be] : 0.0f;
                    s += gUU[mu*4+nu] * (dg_al + dg_be - dg_nu);
                }
                G444(Gamma,mu,al,be) = 0.5f * s;
            }
}

// ── photon_step_f: single RK4 step for null geodesic ─────────────────────────
static void photon_step_f(float M, float a, float Q, float Lambda,
                           thread float* x, thread float* k, float dlam)
{
    float Gamma[64];
    float dx1[4],dk1[4], dx2[4],dk2[4], dx3[4],dk3[4], dx4[4],dk4[4];
    float xt[4], kt[4];

    // k1
    gpg_christoffel_f(M, a, Q, Lambda, x[1], x[2], Gamma);
    for (int mu=0;mu<4;++mu) {
        dx1[mu] = k[mu];
        float acc=0.0f;
        for (int al=0;al<4;++al) for (int be=0;be<4;++be)
            acc -= G444(Gamma,mu,al,be)*k[al]*k[be];
        dk1[mu] = acc;
    }
    // k2
    for (int i=0;i<4;++i){xt[i]=x[i]+0.5f*dlam*dx1[i]; kt[i]=k[i]+0.5f*dlam*dk1[i];}
    gpg_christoffel_f(M, a, Q, Lambda, xt[1], xt[2], Gamma);
    for (int mu=0;mu<4;++mu) {
        dx2[mu] = kt[mu];
        float acc=0.0f;
        for (int al=0;al<4;++al) for (int be=0;be<4;++be)
            acc -= G444(Gamma,mu,al,be)*kt[al]*kt[be];
        dk2[mu] = acc;
    }
    // k3
    for (int i=0;i<4;++i){xt[i]=x[i]+0.5f*dlam*dx2[i]; kt[i]=k[i]+0.5f*dlam*dk2[i];}
    gpg_christoffel_f(M, a, Q, Lambda, xt[1], xt[2], Gamma);
    for (int mu=0;mu<4;++mu) {
        dx3[mu] = kt[mu];
        float acc=0.0f;
        for (int al=0;al<4;++al) for (int be=0;be<4;++be)
            acc -= G444(Gamma,mu,al,be)*kt[al]*kt[be];
        dk3[mu] = acc;
    }
    // k4
    for (int i=0;i<4;++i){xt[i]=x[i]+dlam*dx3[i]; kt[i]=k[i]+dlam*dk3[i];}
    gpg_christoffel_f(M, a, Q, Lambda, xt[1], xt[2], Gamma);
    for (int mu=0;mu<4;++mu) {
        dx4[mu] = kt[mu];
        float acc=0.0f;
        for (int al=0;al<4;++al) for (int be=0;be<4;++be)
            acc -= G444(Gamma,mu,al,be)*kt[al]*kt[be];
        dk4[mu] = acc;
    }
    // Combine
    for (int i=0;i<4;++i) {
        x[i] += (dlam/6.0f)*(dx1[i]+2*dx2[i]+2*dx3[i]+dx4[i]);
        k[i] += (dlam/6.0f)*(dk1[i]+2*dk2[i]+2*dk3[i]+dk4[i]);
    }
}

// ── shade_pixel_f: Page-Thorne shading (mirrors shade_falling_pixel in C++) ──
static uchar4 shade_pixel_f(int outcome, float r_hit, float redshift,
                              float r_isco, float r_out, float disk_brightness)
{
    uint8_t R=0, G=0, B=0;
    if (outcome == 0) {
        R = G = B = 30;
    } else if (outcome == 1) {
        float lum = 0.0f;
        if (r_hit > r_isco && r_hit <= r_out) {
            float x_pt     = sqrt(r_isco / r_hit);
            lum = (1.0f - x_pt) / (r_hit * r_hit * r_hit);
            float r_peak   = 3.0f * r_isco;
            float x_peak   = sqrt(r_isco / r_peak);
            float lum_peak = (1.0f - x_peak) / (r_peak*r_peak*r_peak);
            if (lum_peak > 1e-30f) lum /= lum_peak;
            lum *= pow(max(redshift, 0.0f), 4.0f);
            lum  = min(lum * disk_brightness, 1.0f);
        }
        if (redshift > 1.0f) {
            R = (uint8_t)min(255.0f, 255.0f * lum);
            G = (uint8_t)min(255.0f, 210.0f * lum);
            B = (uint8_t)min(255.0f, 100.0f * lum);
        } else {
            R = (uint8_t)min(255.0f, 220.0f * lum);
            G = (uint8_t)min(255.0f, 100.0f * lum * redshift);
            B = 0;
        }
    }
    // outcome 2 (singularity), 3 (trapped) → black
    return uchar4(R, G, B, 255);
}
```

- [ ] **Step 2: Verify the file exists and has the expected line count**

```bash
wc -l /Users/iman.rosignoli/Documents/KerrTraceCpp2/gpu/metal/tracer_falling.metal
```

Expected: ≥ 180 lines.

- [ ] **Step 3: Commit**

```bash
git add gpu/metal/tracer_falling.metal
git commit -m "feat(falling-metal): tracer_falling.metal GPG helpers (metric, Christoffel, RK4, shading)"
```

---

## Task 4: Add kernel entry point to `tracer_falling.metal`

**Files:**
- Modify: `gpu/metal/tracer_falling.metal` (append)

- [ ] **Step 1: Append photon init helper and kernel**

Append to the end of `gpu/metal/tracer_falling.metal`:

```metal
// ── photon_init_f: null photon k^μ for pixel (px,py) from tetrad ──────────────
static void photon_init_f(uint px, uint py,
                           constant FallingCameraParams& fp,
                           thread float* k_out)
{
    float W    = float(fp.width);
    float H    = float(fp.height);
    float fov_v= fp.fov_h * H / W;
    float alpha= fp.fov_h * (float(px) - W*0.5f) / (W - 1.0f);
    float beta = fov_v   * (H*0.5f - float(py))  / (H - 1.0f);

    float nx = sin(beta)*cos(alpha);
    float ny = sin(beta)*sin(alpha);
    float nz = cos(beta);

    // k^μ = e[0]^μ + nx·e[1]^μ + ny·e[2]^μ + nz·e[3]^μ
    float k[4];
    for (int mu=0;mu<4;++mu)
        k[mu] = fp.e[0][mu] + nx*fp.e[1][mu] + ny*fp.e[2][mu] + nz*fp.e[3][mu];

    // Enforce null: rescale k^T so g_μν k^μ k^ν = 0
    float gLL[16];
    gpg_covariant_f(fp.M, fp.a, fp.Q, fp.Lambda, fp.x[1], fp.x[2], gLL);
    float A = G4(gLL,0,0);
    float B = 0.0f, C = 0.0f;
    for (int mu=1;mu<4;++mu) B += 2.0f*G4(gLL,0,mu)*k[mu];
    for (int mu=1;mu<4;++mu)
        for (int nu=1;nu<4;++nu)
            C += G4(gLL,mu,nu)*k[mu]*k[nu];
    float disc = B*B - 4.0f*A*C;
    if (disc >= 0.0f) {
        float kT1 = (-B + sqrt(disc))/(2.0f*A);
        float kT2 = (-B - sqrt(disc))/(2.0f*A);
        k[0] = (kT1 > kT2) ? kT1 : kT2;
    }
    for (int i=0;i<4;++i) k_out[i] = k[i];
}

// ── trace_falling_pixel: main kernel — one thread per pixel ───────────────────
kernel void trace_falling_pixel(
    device uchar4*                   rgb_out  [[buffer(0)]],
    device float*                    rmin_out [[buffer(1)]],
    constant FallingCameraParams&    fp       [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= uint(fp.width) || gid.y >= uint(fp.height)) return;

    // Initialise photon at camera position fp.x
    float k[4], x[4];
    for (int i=0;i<4;++i) x[i] = fp.x[i];
    photon_init_f(gid.x, gid.y, fp, k);

    float r_min      = x[1];
    float dlam       = fp.h0;
    int   outcome    = 3;       // trapped default
    float r_hit      = 0.0f;
    float redshift   = 1.0f;
    float prev_theta = x[2];
    float prev_r     = x[1];

    for (int step=0; step < fp.max_steps; ++step) {
        if (x[1] < r_min) r_min = x[1];

        // Escape
        if (x[1] >= fp.r_escape) {
            outcome = 0; r_hit = x[1]; break;
        }
        // Singularity
        if (x[1] < fp.r_singularity) {
            outcome = 2; break;
        }

        // Disk crossing: sign change of (θ − π/2) within [r_in, r_out]
        float cur_theta = x[2];
        if (x[1] >= fp.r_in && x[1] <= fp.r_out) {
            float dp = prev_theta - M_PI_F/2.0f;
            float dc = cur_theta  - M_PI_F/2.0f;
            if (dp * dc < 0.0f) {
                outcome = 1;
                float t = abs(dp) / (abs(dp) + abs(dc));
                r_hit = prev_r + t*(x[1] - prev_r);

                // Redshift g-factor: Keplerian Omega_K, g_μν at r_hit equatorial
                float gLL_d[16];
                gpg_covariant_f(fp.M, fp.a, fp.Q, fp.Lambda,
                                r_hit, M_PI_F/2.0f, gLL_d);
                float sqrtM = sqrt(fp.M);
                float Omega = sqrtM / (pow(r_hit, 1.5f) + fp.a * sqrtM);
                float N2 = -(G4(gLL_d,0,0)
                           + 2.0f*G4(gLL_d,0,3)*Omega
                           + G4(gLL_d,3,3)*Omega*Omega);
                if (N2 > 1e-20f) {
                    float ut_em  = 1.0f / sqrt(N2);
                    float uph_em = Omega * ut_em;
                    float k_low[4] = {0,0,0,0};
                    for (int mu=0;mu<4;++mu)
                        for (int nu=0;nu<4;++nu)
                            k_low[mu] += G4(gLL_d,mu,nu)*k[nu];
                    float k_u_emit = k_low[0]*ut_em + k_low[3]*uph_em;
                    float k_u_obs  = abs(k_low[0]);
                    if (abs(k_u_emit) > 1e-20f)
                        redshift = k_u_obs / abs(k_u_emit);
                }
                break;
            }
        }
        prev_theta = cur_theta;
        prev_r     = x[1];

        // Adaptive step: fine near horizon
        dlam = (x[1] < fp.r_horizon * 3.0f) ? 0.005f : fp.h0;
        photon_step_f(fp.M, fp.a, fp.Q, fp.Lambda, x, k, dlam);
    }

    uint idx = gid.y * uint(fp.width) + gid.x;
    rgb_out [idx] = shade_pixel_f(outcome, r_hit, redshift,
                                   fp.r_isco, fp.r_out, fp.disk_brightness);
    rmin_out[idx] = r_min;
}
```

- [ ] **Step 2: Verify total line count**

```bash
wc -l /Users/iman.rosignoli/Documents/KerrTraceCpp2/gpu/metal/tracer_falling.metal
```

Expected: ≥ 290 lines.

- [ ] **Step 3: Commit**

```bash
git add gpu/metal/tracer_falling.metal
git commit -m "feat(falling-metal): trace_falling_pixel kernel entry point"
```

---

## Task 5: Create `metal_falling_renderer.mm`

**Files:**
- Create: `gpu/metal/metal_falling_renderer.mm`

- [ ] **Step 1: Create the bridge**

```objc
// metal_falling_renderer.mm — Objective-C++ Metal bridge for falling-camera renderer.
// Loads tracer_falling.metal at runtime, dispatches one thread per pixel.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_falling_renderer.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

// ── Per-process singleton ─────────────────────────────────────────────────────
namespace {
struct FallingMetalCache {
    id<MTLDevice>               device = nil;
    id<MTLLibrary>              lib    = nil;
    id<MTLComputePipelineState> pso    = nil;
    id<MTLCommandQueue>         queue  = nil;
};

static FallingMetalCache g_fc;
static bool              g_fc_valid = false;

static bool ensure_falling_cache()
{
    if (g_fc_valid) return true;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return false;

    // Load tracer_falling.metal: try source path, then build-dir copy
    NSString* exeDir = [[[NSProcessInfo processInfo] arguments][0]
                         stringByDeletingLastPathComponent];
    NSString* src = nil;
    NSError*  err = nil;
    for (NSString* candidate in @[
        [exeDir stringByAppendingPathComponent:@"../gpu/metal/tracer_falling.metal"],
        [exeDir stringByAppendingPathComponent:@"tracer_falling.metal"],
        [[[NSBundle mainBundle] resourcePath]
             stringByAppendingPathComponent:@"tracer_falling.metal"]
    ]) {
        src = [NSString stringWithContentsOfFile:candidate
                                        encoding:NSUTF8StringEncoding
                                           error:&err];
        if (src) break;
    }
    if (!src) {
        fprintf(stderr, "[falling-metal] cannot load tracer_falling.metal\n");
        return false;
    }

    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.languageVersion = MTLLanguageVersion2_4;
    if (@available(macOS 15.0, *)) {
        opts.mathMode = MTLMathModeFast;
    } else {
        opts.fastMathEnabled = YES;
    }

    id<MTLLibrary> lib = [device newLibraryWithSource:src options:opts error:&err];
    if (!lib) {
        fprintf(stderr, "[falling-metal] compile error: %s\n",
                [err.localizedDescription UTF8String]);
        return false;
    }

    id<MTLFunction> fn = [lib newFunctionWithName:@"trace_falling_pixel"];
    if (!fn) {
        fprintf(stderr, "[falling-metal] kernel 'trace_falling_pixel' not found\n");
        return false;
    }

    NSError* e2 = nil;
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&e2];
    if (!pso) {
        fprintf(stderr, "[falling-metal] PSO creation failed: %s\n",
                [e2.localizedDescription UTF8String]);
        return false;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) return false;

    g_fc = { device, lib, pso, queue };
    g_fc_valid = true;
    return true;
}
} // namespace

// ── make_falling_metal_params ─────────────────────────────────────────────────
FallingCameraParams_C make_falling_metal_params(
    const FallingParams& fp,
    const CameraState&   cs,
    const double         e[4][4])
{
    FallingCameraParams_C c{};
    for (int a=0;a<4;++a)
        for (int mu=0;mu<4;++mu)
            c.e[a][mu] = float(e[a][mu]);
    for (int i=0;i<4;++i) c.x[i] = float(cs.x[i]);
    c.M      = float(fp.bh.M);
    c.a      = float(fp.bh.a);
    c.Q      = float(fp.bh.Q);
    c.Lambda = float(fp.bh.Lambda);
    const double r_isco_val = fp.bh.r_isco();
    c.r_in          = float((fp.r_disk_in < 0.0) ? r_isco_val : fp.r_disk_in);
    c.r_out         = float(fp.r_disk_out);
    c.r_isco        = float(r_isco_val);
    c.r_escape      = float(fp.r_escape);
    c.r_singularity = float(fp.r_singularity);
    c.r_horizon     = float(fp.bh.r_horizon());
    c.disk_brightness = float(fp.disk_brightness);
    c.fov_h         = float(fp.fov_h);
    c.h0            = 0.05f;
    c.r_switch_factor = float(fp.r_switch_factor);
    c.max_steps     = 20000;   // GPU cap (lower than CPU 50000; near-horizon pixels
                                // are refined by CPU pass anyway)
    c.width         = fp.width;
    c.height        = fp.height;
    c.pad           = 0;
    return c;
}

// ── metal_render_falling_frame ────────────────────────────────────────────────
bool metal_render_falling_frame(
    const FallingCameraParams_C& params,
    std::vector<uint8_t>&        rgb,
    std::vector<float>&          r_min)
{
    if (!ensure_falling_cache()) return false;

    auto& c = g_fc;
    const NSUInteger W    = (NSUInteger)params.width;
    const NSUInteger H    = (NSUInteger)params.height;
    const NSUInteger npix = W * H;

    id<MTLBuffer> rgbBuf  = [c.device
        newBufferWithLength:npix * 4
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> rminBuf = [c.device
        newBufferWithLength:npix * sizeof(float)
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> cpBuf   = [c.device
        newBufferWithBytes:&params
        length:sizeof(params)
        options:MTLResourceStorageModeShared];
    if (!rgbBuf || !rminBuf || !cpBuf) return false;

    // Tile by rows (16 rows/tile) to stay within GPU watchdog timeout.
    // Each tile dispatches full-width rows [y0, y0+tile_h).
    // The kernel guard (gid.x >= width || gid.y >= height) handles boundary threads.
    // We dispatch only the rows in the tile by adjusting the grid height and writing
    // with base offset y0*W into the shared buffers.
    //
    // Simplification: dispatch the full image in slices, but the kernel always
    // reads/writes absolute pixel index gid.y*W+gid.x.  We encode a 4-byte
    // y_start uniform per tile so the kernel can clamp to [y_start, y_start+tile_h).
    // To avoid adding a new kernel param, we instead just dispatch row slices:
    // for each tile, we create a modified params copy with height=tile_h and
    // offset the output buffers appropriately.
    //
    // Cleaner approach: dispatch full image in one shot for ≤ 640×360,
    // tile for larger.  For simplicity here we always do a single dispatch
    // (falling camera is typically ≤ 640×360 in practice).  Add tiling if
    // GPU watchdog triggers on large renders.
    const auto t0 = std::chrono::steady_clock::now();

    id<MTLCommandBuffer> cmd = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:c.pso];
    [enc setBuffer:rgbBuf  offset:0 atIndex:0];
    [enc setBuffer:rminBuf offset:0 atIndex:1];
    [enc setBuffer:cpBuf   offset:0 atIndex:2];

    MTLSize tg   = MTLSizeMake(16, 16, 1);
    MTLSize grid = MTLSizeMake(((W+15)/16)*16, ((H+15)/16)*16, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if (cmd.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "[falling-metal] command buffer failed: %s\n",
                cmd.error ? [cmd.error.localizedDescription UTF8String] : "unknown");
        return false;
    }

    const auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1-t0).count();
    fprintf(stderr, "[falling-metal] GPU pass: %.2fs\n", elapsed);

    // Copy RGBA → RGB
    const uint8_t* src = (const uint8_t*)[rgbBuf contents];
    rgb.resize(npix * 3);
    for (NSUInteger i=0; i<npix; ++i) {
        rgb[i*3+0] = src[i*4+0];
        rgb[i*3+1] = src[i*4+1];
        rgb[i*3+2] = src[i*4+2];
    }
    r_min.resize(npix);
    std::memcpy(r_min.data(), [rminBuf contents], npix * sizeof(float));
    return true;
}
```

- [ ] **Step 2: Check the file was created**

```bash
wc -l /Users/iman.rosignoli/Documents/KerrTraceCpp2/gpu/metal/metal_falling_renderer.mm
```

Expected: ≥ 150 lines.

- [ ] **Step 3: Commit**

```bash
git add gpu/metal/metal_falling_renderer.mm
git commit -m "feat(falling-metal): metal_falling_renderer.mm bridge — lazy init, dispatch, readback"
```

---

## Task 6: Extract `shade_falling_pixel` in `falling_renderer.cpp`

**Files:**
- Modify: `falling_renderer.cpp`

The shading logic currently inline in `render_falling_frame` needs to be a standalone function, reused by both the CPU path and the Metal refinement pass.

- [ ] **Step 1: Add `shade_falling_pixel` before `render_falling_frame`**

In `falling_renderer.cpp`, find the line:
```cpp
// ── render_falling_frame ──────────────────────────────────────────────────────
```

Insert before it:

```cpp
// ── shade_falling_pixel ───────────────────────────────────────────────────────
// Returns {R,G,B} for a traced pixel. Mirrors shade_pixel_f in tracer_falling.metal.
static std::tuple<uint8_t,uint8_t,uint8_t>
shade_falling_pixel(const FallingGeoPixel& pix, const FallingParams& fp,
                    double r_isco_val)
{
    uint8_t R=0, G=0, B=0;
    if (pix.outcome == 0) {
        R = G = B = 30;
    } else if (pix.outcome == 1) {
        const double r_h  = double(pix.r_hit);
        const double g_rs = double(pix.redshift);
        double lum = 0.0;
        if (r_h > r_isco_val && r_h <= fp.r_disk_out) {
            const double x_pt     = std::sqrt(r_isco_val / r_h);
            lum = (1.0 - x_pt) / (r_h * r_h * r_h);
            const double r_peak   = 3.0 * r_isco_val;
            const double x_peak   = std::sqrt(r_isco_val / r_peak);
            const double lum_peak = (1.0 - x_peak) / (r_peak*r_peak*r_peak);
            if (lum_peak > 1e-30) lum /= lum_peak;
            lum *= std::pow(std::max(g_rs, 0.0), 4.0);
            lum  = std::min(lum * fp.disk_brightness, 1.0);
        }
        const float fl = float(lum);
        if (g_rs > 1.0) {
            R = uint8_t(std::min(255.0f, 255.0f * fl));
            G = uint8_t(std::min(255.0f, 210.0f * fl));
            B = uint8_t(std::min(255.0f, 100.0f * fl));
        } else {
            R = uint8_t(std::min(255.0f, 220.0f * fl));
            G = uint8_t(std::min(255.0f, 100.0f * fl * float(g_rs)));
            B = 0;
        }
    }
    return {R, G, B};
}

```

- [ ] **Step 2: Replace the inline shading block inside `render_falling_frame`**

Find the inline shading inside the pixel loop in `render_falling_frame`:

```cpp
            uint8_t R = 0, G = 0, B = 0;

            if (pix.outcome == 0) {
                // Escaped — dark background (Phase A); Phase C will add HDRI skybox
                R = G = B = 30;

            } else if (pix.outcome == 1) {
                // Disk hit — Phase B shading
                const double r_h  = double(pix.r_hit);
                const double g_rs = double(pix.redshift);

                // Page-Thorne flux profile: f(r) ∝ (1 - sqrt(r_isco/r)) / r³
                double lum = 0.0;
                if (r_h > r_isco_val && r_h <= r_out) {
                    const double x_pt = std::sqrt(r_isco_val / r_h);
                    lum = (1.0 - x_pt) / (r_h * r_h * r_h);
                    // Normalize by approximate peak at 3*r_isco
                    const double r_peak = 3.0 * r_isco_val;
                    const double x_peak = std::sqrt(r_isco_val / r_peak);
                    const double lum_peak = (1.0 - x_peak) / (r_peak*r_peak*r_peak);
                    if (lum_peak > 1e-30) lum /= lum_peak;
                    // Apply redshift: I_obs = g^4 * I_emit
                    lum *= std::pow(std::max(g_rs, 0.0), 4.0);
                    lum  = std::min(lum * fp.disk_brightness, 1.0);
                }

                // Color: blueshift side (g>1) → white-yellow; redshift side → orange-red
                const float fl = float(lum);
                if (g_rs > 1.0) {
                    R = uint8_t(std::min(255.0f, 255.0f * fl));
                    G = uint8_t(std::min(255.0f, 210.0f * fl));
                    B = uint8_t(std::min(255.0f, 100.0f * fl));
                } else {
                    R = uint8_t(std::min(255.0f, 220.0f * fl));
                    G = uint8_t(std::min(255.0f, 100.0f * fl * float(g_rs)));
                    B = 0;
                }
                // outcome 2 (singularity), 3 (trapped) → remain black
            }
```

Replace with:

```cpp
            auto [R, G, B] = shade_falling_pixel(pix, fp, r_isco_val);
```

- [ ] **Step 3: Remove the now-unused `r_out` variable (it was `(void)r_in` guarded already)**

Find:
```cpp
    const double r_in  = (fp.r_disk_in < 0.0) ? r_isco_val : fp.r_disk_in;
    const double r_out = fp.r_disk_out;
    (void)r_in;  // used indirectly via FallingParams in trace_photon_gpg
```

Replace with:
```cpp
    (void)fp.r_disk_in;  // used indirectly via FallingParams in trace_photon_gpg
```

- [ ] **Step 4: Build CPU binary to verify**

```bash
cd /Users/iman.rosignoli/Documents/KerrTraceCpp2
cmake --build build_cpu -j$(sysctl -n hw.ncpu) 2>&1 | grep -E "error:" | head -10
```

Expected: no errors.

- [ ] **Step 5: Run existing falling smoke test to confirm CPU path unchanged**

```bash
cd build_cpu && ctest -R falling_smoke --output-on-failure 2>&1 | tail -5
```

Expected: `Passed`.

- [ ] **Step 6: Commit**

```bash
git add falling_renderer.cpp
git commit -m "refactor(falling): extract shade_falling_pixel helper"
```

---

## Task 7: Add Metal two-pass path to `render_falling_frame`

**Files:**
- Modify: `falling_renderer.cpp`

- [ ] **Step 1: Add `#include` guard at the top of `falling_renderer.cpp`**

After the existing includes, add:

```cpp
#ifdef USE_METAL
#include "gpu/metal/metal_falling_renderer.hpp"
#endif
```

Find:
```cpp
#include "stb_image_write.h"
#include <cmath>
#include <chrono>
#include <cstdio>
#include <vector>
```

Replace with:
```cpp
#include "stb_image_write.h"
#include <cmath>
#include <chrono>
#include <cstdio>
#include <vector>
#ifdef USE_METAL
#include "gpu/metal/metal_falling_renderer.hpp"
#endif
```

- [ ] **Step 2: Add the two-pass Metal block at the start of `render_falling_frame`**

In `render_falling_frame`, find:
```cpp
    // Build local tetrad and apply HorizonFlip roll
    double e[4][4];
    build_tetrad(cs_at_frame, fp.bh, e);
    const double psi = horizon_flip_psi(cs_at_frame.x[1], fp.bh.r_horizon());
    apply_roll(e, psi);

    // Pre-compute disk ISCO for Phase B shading
    const double r_isco_val = fp.bh.r_isco();
    (void)fp.r_disk_in;  // used indirectly via FallingParams in trace_photon_gpg
```

Replace with:
```cpp
    // Build local tetrad and apply HorizonFlip roll
    double e[4][4];
    build_tetrad(cs_at_frame, fp.bh, e);
    const double psi = horizon_flip_psi(cs_at_frame.x[1], fp.bh.r_horizon());
    apply_roll(e, psi);

    // Pre-compute disk ISCO for Phase B shading
    const double r_isco_val = fp.bh.r_isco();
    (void)fp.r_disk_in;  // used indirectly via FallingParams in trace_photon_gpg

#ifdef USE_METAL
    {
        std::vector<float>   r_min(W * H, float(fp.r_escape));
        FallingCameraParams_C cp = make_falling_metal_params(fp, cs_at_frame, e);
        if (metal_render_falling_frame(cp, rgb, r_min)) {
            // Pass 2: CPU double refinement for near-horizon pixels
            const double r_switch = fp.bh.r_horizon() * fp.r_switch_factor;
            int refined = 0;
            for (int py = 0; py < H; ++py) {
                for (int px = 0; px < W; ++px) {
                    const int idx = py * W + px;
                    if (r_min[idx] < float(r_switch)) {
                        double k[4];
                        init_photon_k(cs_at_frame, e, fp.bh,
                                      px, py, W, H, fp.fov_h, k);
                        FallingGeoPixel pix = trace_photon_gpg(
                            cs_at_frame.x, k, fp.bh, fp, 50000, 0.05, 1e-7);
                        auto [R, G, B] = shade_falling_pixel(pix, fp, r_isco_val);
                        rgb[idx*3+0] = R;
                        rgb[idx*3+1] = G;
                        rgb[idx*3+2] = B;
                        ++refined;
                    }
                }
            }
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - t0).count();
            std::printf("[frame %04d/%04d] Metal+CPU refine=%d (%.0f%%)  %.1fs elapsed\n",
                        frame_idx + 1, total_frames,
                        refined,
                        refined * 100.0 / double(W * H),
                        elapsed);
            stbi_write_png(out_path.c_str(), W, H, 3, rgb.data(), W * 3);
            return;
        }
        // Metal failed → fall through to CPU path
    }
#endif
```

Note: `t0` is declared after the tetrad build block. Move its declaration before the `#ifdef USE_METAL` block.

Find:
```cpp
    auto t0 = std::chrono::steady_clock::now();
```

This line is inside the CPU path. Move it up so it appears before the `#ifdef USE_METAL` block. The final order in `render_falling_frame` should be:

```
1. Declare W, H, rgb
2. build_tetrad + apply_roll
3. r_isco_val, (void)fp.r_disk_in
4. auto t0 = std::chrono::steady_clock::now();   ← move here
5. #ifdef USE_METAL ... two-pass block ... #endif
6. existing CPU loop (unchanged)
```

Find the existing `auto t0 = ...` line inside the CPU loop region and move it above the `#ifdef USE_METAL` block. The exact edit:

Find:
```cpp
    (void)fp.r_disk_in;  // used indirectly via FallingParams in trace_photon_gpg

#ifdef USE_METAL
```

Replace with:
```cpp
    (void)fp.r_disk_in;  // used indirectly via FallingParams in trace_photon_gpg

    auto t0 = std::chrono::steady_clock::now();

#ifdef USE_METAL
```

Then remove the duplicate `auto t0` that was in the CPU loop (it will now be a compile error or duplicate). Find and remove:
```cpp
    auto t0 = std::chrono::steady_clock::now();

    #ifdef _OPENMP
```

Replace with:
```cpp
    #ifdef _OPENMP
```

- [ ] **Step 3: Build CPU binary (no Metal) to verify**

```bash
cd /Users/iman.rosignoli/Documents/KerrTraceCpp2
cmake --build build_cpu -j$(sysctl -n hw.ncpu) 2>&1 | grep -E "error:" | head -10
```

Expected: no errors.

- [ ] **Step 4: Run falling smoke test**

```bash
cd build_cpu && ctest -R falling_smoke --output-on-failure 2>&1 | tail -5
```

Expected: `Passed`.

- [ ] **Step 5: Build Metal binary**

```bash
cd /Users/iman.rosignoli/Documents/KerrTraceCpp2
cmake --build build -j$(sysctl -n hw.ncpu) 2>&1 | grep -E "error:" | head -10
```

Expected: no errors. (CMakeLists changes are in Task 8 — if this step fails because `.mm` isn't added yet, do Task 8 first.)

- [ ] **Step 6: Commit**

```bash
git add falling_renderer.cpp
git commit -m "feat(falling-metal): two-pass Metal+CPU path in render_falling_frame"
```

---

## Task 8: Update `CMakeLists.txt`

**Files:**
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Add `metal_falling_renderer.mm` and copy `tracer_falling.metal` in the `USE_METAL` block**

Find:
```cmake
if(USE_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "USE_METAL requires macOS")
    endif()
    list(APPEND CPU_SOURCES gpu/metal/metal_renderer.mm)
    # Copy the .metal shader next to the binary so the renderer can find it
    configure_file(gpu/metal/tracer.metal
                   ${CMAKE_BINARY_DIR}/tracer.metal COPYONLY)
    message(STATUS "Metal GPU backend enabled")
endif()
```

Replace with:
```cmake
if(USE_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "USE_METAL requires macOS")
    endif()
    list(APPEND CPU_SOURCES gpu/metal/metal_renderer.mm)
    list(APPEND CPU_SOURCES gpu/metal/metal_falling_renderer.mm)
    configure_file(gpu/metal/tracer.metal
                   ${CMAKE_BINARY_DIR}/tracer.metal COPYONLY)
    configure_file(gpu/metal/tracer_falling.metal
                   ${CMAKE_BINARY_DIR}/tracer_falling.metal COPYONLY)
    message(STATUS "Metal GPU backend enabled")
endif()
```

- [ ] **Step 2: Add the same to the `kerr_tracer_metal` always-build block**

Find:
```cmake
if(APPLE AND NOT USE_METAL)
    add_executable(kerr_tracer_metal main.cpp falling_renderer.cpp gpu/metal/metal_renderer.mm)
    configure_file(gpu/metal/tracer.metal
                   ${CMAKE_BINARY_DIR}/tracer.metal COPYONLY)
```

Replace with:
```cmake
if(APPLE AND NOT USE_METAL)
    add_executable(kerr_tracer_metal main.cpp falling_renderer.cpp
                   gpu/metal/metal_renderer.mm
                   gpu/metal/metal_falling_renderer.mm)
    configure_file(gpu/metal/tracer.metal
                   ${CMAKE_BINARY_DIR}/tracer.metal COPYONLY)
    configure_file(gpu/metal/tracer_falling.metal
                   ${CMAKE_BINARY_DIR}/tracer_falling.metal COPYONLY)
```

- [ ] **Step 3: Full build (both CPU and Metal binaries)**

```bash
cd /Users/iman.rosignoli/Documents/KerrTraceCpp2
cmake -B build_cpu -DUSE_METAL=OFF && cmake --build build_cpu -j$(sysctl -n hw.ncpu) 2>&1 | grep -E "error:" | head -10
cmake -B build    -DUSE_METAL=ON  && cmake --build build    -j$(sysctl -n hw.ncpu) 2>&1 | grep -E "error:" | head -10
```

Expected: no errors for either build.

- [ ] **Step 4: Commit**

```bash
git add CMakeLists.txt
git commit -m "build: add metal_falling_renderer.mm and tracer_falling.metal to Metal targets"
```

---

## Task 9: Integration test and final validation

**Files:**
- No new files.

- [ ] **Step 1: Verify `tracer_falling.metal` is copied to the build directory**

```bash
ls /Users/iman.rosignoli/Documents/KerrTraceCpp2/build/tracer_falling.metal
```

Expected: file exists.

- [ ] **Step 2: Run Metal smoke test — 2 frames at 64×36**

```bash
cd /Users/iman.rosignoli/Documents/KerrTraceCpp2
./build/kerr_tracer_metal --falling-camera \
  --a 0.9 --fall-r-start 15 --fall-E 1.0 \
  --fall-frames 2 --fall-dtau 0.5 \
  --custom-res 64 36 2>&1
```

Expected output (to stderr + stdout):
```
[falling-metal] GPU pass: X.XXs
[frame 0001/0002] Metal+CPU refine=N (X%) X.Xs elapsed
[falling-metal] GPU pass: X.XXs
[frame 0002/0002] Metal+CPU refine=N (X%) X.Xs elapsed
Falling render complete: .../out/falling/...
```

- [ ] **Step 3: Verify PNG files were written**

```bash
ls out/falling/*/frame_*.png | head -4
```

Expected: at least `frame_0000.png` and `frame_0001.png`.

- [ ] **Step 4: Run at 320×180 for visual check**

```bash
./build/kerr_tracer_metal --falling-camera \
  --a 0.9 --fall-r-start 15 --fall-E 1.0 \
  --fall-frames 4 --fall-dtau 0.5 \
  --custom-res 320 180 2>&1
open out/falling/*/frame_0001.png
```

Expected: a frame with dark background (grey ~30 for escaped photons) and an orange disk glow. The image should look identical to the CPU path.

- [ ] **Step 5: Compare Metal vs CPU output (optional sanity check)**

```bash
# CPU
./build_cpu/kerr_tracer --falling-camera \
  --a 0.9 --fall-r-start 15 --fall-E 1.0 \
  --fall-frames 1 --fall-dtau 0.5 \
  --custom-res 64 36
CPU_FRAME=$(ls -t out/falling/*/frame_0000.png | head -1)

# Metal
./build/kerr_tracer_metal --falling-camera \
  --a 0.9 --fall-r-start 15 --fall-E 1.0 \
  --fall-frames 1 --fall-dtau 0.5 \
  --custom-res 64 36
METAL_FRAME=$(ls -t out/falling/*/frame_0000.png | head -1)

# Diff (expect near-identical; small float vs double differences are acceptable)
python3 -c "
from PIL import Image
import numpy as np
a = np.array(Image.open('$CPU_FRAME'))
b = np.array(Image.open('$METAL_FRAME'))
diff = np.abs(a.astype(int)-b.astype(int))
print(f'max_diff={diff.max()} mean_diff={diff.mean():.2f}')
"
```

Expected: `max_diff` ≤ 5 (float vs double rounding), `mean_diff` < 0.5. Near-horizon pixels (refined by CPU double) should be identical.

- [ ] **Step 6: Run full CTest suite**

```bash
cd /Users/iman.rosignoli/Documents/KerrTraceCpp2
cmake -B build_cpu -DUSE_METAL=OFF -DBUILD_TESTING=ON
cmake --build build_cpu -j$(sysctl -n hw.ncpu)
cd build_cpu && ctest --output-on-failure 2>&1 | tail -10
```

Expected: `kerrtrace.falling` and `kerrtrace.falling_smoke` pass. `kerrtrace.spin_orientation` failure is pre-existing and unrelated.

- [ ] **Step 7: Commit**

```bash
git add -u
git commit -m "feat(falling-metal): v0.3.0 — Metal GPU backend for falling camera (Phase D)"
```

---

## Self-Review

**Spec coverage check:**
- [x] New standalone bridge (`metal_falling_renderer.hpp/.mm`): Tasks 2, 5
- [x] `tracer_falling.metal` with GPG float metric, Christoffel, RK4, shading: Tasks 3, 4
- [x] `FallingCameraParams_C` struct (C++ ↔ MSL mirror): Task 2
- [x] `make_falling_metal_params` conversion: Task 5
- [x] Two-pass path in `render_falling_frame`: Task 7
- [x] `r_switch_factor` in `FallingParams`: Task 1
- [x] `shade_falling_pixel` extracted (single shading logic): Task 6
- [x] CMakeLists updated for both `USE_METAL=ON` and `kerr_tracer_metal`: Task 8
- [x] CPU fallback if Metal unavailable: Task 7 (`if (!metal_render_falling_frame(...))`)
- [x] `tracer_falling.metal` copied to build dir: Task 8
- [x] Finite difference steps larger in float (1e-4 vs 1e-5): Task 3 code
- [x] max_steps=20000 for GPU (vs 50000 CPU): Task 5 code

**Type consistency check:**
- `FallingCameraParams_C` defined in Task 2, used in Tasks 5 and 7 ✓
- `make_falling_metal_params` declared in Task 2, implemented in Task 5 ✓
- `metal_render_falling_frame` declared in Task 2, implemented in Task 5 ✓
- `shade_falling_pixel` defined in Task 6, called in Tasks 6 and 7 ✓
- `G4(arr,r,c)` macro defined at top of metal file, used throughout Tasks 3 and 4 ✓
- `G444(arr,mu,al,be)` macro defined at top, used in `photon_step_f` and kernel ✓

**Placeholder scan:** no TBD, no incomplete steps.
