# Metal Jacobi-Field Ray Bundle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 5-ray finite-difference magnification proxy in `tracer.metal` with the physically correct Jacobi-field (variational-equation) approach, matching the CPU implementation in `ray_bundle.hpp`.

**Architecture:** Port `hessian_H`, `build_M`, `bundle_rhs`, `bundle_rk4`, and `bundle_adaptive` from `ray_bundle.hpp` to MSL (float precision, adjusted finite-diff epsilons). Add a new `trace_bundle_jacobi_f()` tracer and a `trace_pixel_jacobi` kernel (mode 4). Wire up `--jacobi-bundle` CLI flag in `main.cpp` and keep the existing 5-ray proxy on mode 3 for speed comparison.

**Tech Stack:** Metal Shading Language 2.4, C++17, existing `hamiltonian()` + `geodesic_rhs()` + `adaptive_step_bl()` already in `tracer.metal`.

---

## File Structure

| File | Change |
|------|--------|
| `gpu/metal/tracer.metal` | Add ~400 lines: Hessian, M-matrix, bundle RK4/adaptive, `trace_bundle_jacobi_f()`, `trace_pixel_jacobi` kernel |
| `gpu/metal/metal_renderer.mm` | Add mode-4 kernel name selection |
| `main.cpp` | Add `--jacobi-bundle` flag → sets `metal_kernel_mode=4` |
| `frontend/src/app/app.ts` | Add "Jacobi bundle" option in bundle mode selector |

---

## Background — Float precision for Hessian

CPU uses `double` with `eps = 1e-5`. In `float` (machine-epsilon ≈ 1.2e-7), optimal
step for second derivatives is `√(eps_mach) ≈ 3.5e-4`. We use:

```
eps_r   = 3e-4 * (|r|   + 0.1)
eps_th  = 3e-4
eps_pr  = 3e-4 * (|pr|  + 1e-3)
eps_pth = 3e-4 * (|pth| + 1e-3)
```

Hessian cost: 9 (diagonal) + 24 (off-diagonal) + 1 (center) = **34 Hamiltonian evals/call**.  
Per `bundle_rhs`: 34 + 4 (geodesic_rhs) ≈ 38 evals.  
Per `bundle_rk4` (4 stages): ≈ 152 evals.  
Per adaptive step (3 full RK4 for step-doubling): ≈ 456 evals.  
GPU parallelism absorbs this; expect ~20–50× slower than single-ray GPU, but much faster than CPU Jacobi.

---

## Task 1: Jacobi helper functions in tracer.metal

**Files:**
- Modify: `gpu/metal/tracer.metal` — insert after line ~332 (after `adaptive_step_bl`)

- [ ] **Step 1: Insert `hessian_H_f` function**

Add after the `adaptive_step_bl` block (around line 332), before the `sign_change_f` block:

```metal
// ── Jacobi field — Hessian of H in 4D phase space ──────────────
// z = (r, theta, pr, ptheta); conserved (pt, pphi) unchanged.
// Uses float-appropriate finite-diff steps: eps ≈ sqrt(float_eps) ≈ 3e-4.
static void hessian_H_f(float r,   float theta,
                         float pr,  float pth,
                         float pt,  float pphi,
                         float M,   float a, float Q, float L,
                         thread float H4[4][4]) {
    const float e0 = 3e-4f * (abs(r)   + 0.1f);
    const float e1 = 3e-4f;
    const float e2 = 3e-4f * (abs(pr)  + 1e-3f);
    const float e3 = 3e-4f * (abs(pth) + 1e-3f);
    const float e[4] = {e0, e1, e2, e3};

    // z-components as a flat array for indexing
    float z[4] = {r, theta, pr, pth};

    // Helper: hamiltonian at z + delta
    // delta encoded as (d0,d1,d2,d3)
    #define HAM_D(d0,d1,d2,d3) \
        hamiltonian(z[0]+(d0), z[1]+(d1), z[2]+(d2), z[3]+(d3), pt, pphi, M, a, Q, L)

    const float H0 = HAM_D(0,0,0,0);

    // Diagonal: H_aa = (H(+ea) - 2H0 + H(-ea)) / ea^2
    // a=0: r
    { const float p=HAM_D(e0,0,0,0), m=HAM_D(-e0,0,0,0);
      H4[0][0] = (p - 2.0f*H0 + m) / (e0*e0); }
    // a=1: theta
    { const float p=HAM_D(0,e1,0,0), m=HAM_D(0,-e1,0,0);
      H4[1][1] = (p - 2.0f*H0 + m) / (e1*e1); }
    // a=2: pr
    { const float p=HAM_D(0,0,e2,0), m=HAM_D(0,0,-e2,0);
      H4[2][2] = (p - 2.0f*H0 + m) / (e2*e2); }
    // a=3: pth
    { const float p=HAM_D(0,0,0,e3), m=HAM_D(0,0,0,-e3);
      H4[3][3] = (p - 2.0f*H0 + m) / (e3*e3); }

    // Off-diagonal: H_ab = (H(+ea+eb)-H(+ea-eb)-H(-ea+eb)+H(-ea-eb)) / (4 ea eb)
    // (0,1): r,theta
    { const float pp=HAM_D( e0, e1,0,0), pm=HAM_D( e0,-e1,0,0),
                  mp=HAM_D(-e0, e1,0,0), mm=HAM_D(-e0,-e1,0,0);
      H4[0][1]=H4[1][0]=(pp-pm-mp+mm)/(4.0f*e0*e1); }
    // (0,2): r,pr
    { const float pp=HAM_D( e0,0, e2,0), pm=HAM_D( e0,0,-e2,0),
                  mp=HAM_D(-e0,0, e2,0), mm=HAM_D(-e0,0,-e2,0);
      H4[0][2]=H4[2][0]=(pp-pm-mp+mm)/(4.0f*e0*e2); }
    // (0,3): r,pth
    { const float pp=HAM_D( e0,0,0, e3), pm=HAM_D( e0,0,0,-e3),
                  mp=HAM_D(-e0,0,0, e3), mm=HAM_D(-e0,0,0,-e3);
      H4[0][3]=H4[3][0]=(pp-pm-mp+mm)/(4.0f*e0*e3); }
    // (1,2): theta,pr
    { const float pp=HAM_D(0, e1, e2,0), pm=HAM_D(0, e1,-e2,0),
                  mp=HAM_D(0,-e1, e2,0), mm=HAM_D(0,-e1,-e2,0);
      H4[1][2]=H4[2][1]=(pp-pm-mp+mm)/(4.0f*e1*e2); }
    // (1,3): theta,pth
    { const float pp=HAM_D(0, e1,0, e3), pm=HAM_D(0, e1,0,-e3),
                  mp=HAM_D(0,-e1,0, e3), mm=HAM_D(0,-e1,0,-e3);
      H4[1][3]=H4[3][1]=(pp-pm-mp+mm)/(4.0f*e1*e3); }
    // (2,3): pr,pth
    { const float pp=HAM_D(0,0, e2, e3), pm=HAM_D(0,0, e2,-e3),
                  mp=HAM_D(0,0,-e2, e3), mm=HAM_D(0,0,-e2,-e3);
      H4[2][3]=H4[3][2]=(pp-pm-mp+mm)/(4.0f*e2*e3); }

    #undef HAM_D
}
```

- [ ] **Step 2: Insert `build_M_f` function** (immediately after `hessian_H_f`)

```metal
// M = J_symplectic · Hess(H)
// z = (r=0, theta=1, pr=2, pth=3)
// Row 0,1: come from Hess rows 2,3  (dq/dlam part)
// Row 2,3: come from -Hess rows 0,1 (dp/dlam part)
static void build_M_f(thread const float Hess[4][4],
                       thread float Mout[4][4]) {
    for (int j = 0; j < 4; ++j) {
        Mout[0][j] =  Hess[2][j];
        Mout[1][j] =  Hess[3][j];
        Mout[2][j] = -Hess[0][j];
        Mout[3][j] = -Hess[1][j];
    }
}
```

- [ ] **Step 3: Build and verify compile**

```bash
cmake -B build -DUSE_METAL=ON && cmake --build build -j$(sysctl -n hw.logicalcpu) 2>&1 | grep -E "error:|warning:|Built"
```

Expected: `[100%] Built target kerr_tracer_metal` — no errors. (Warnings about unused variables are OK.)

- [ ] **Step 4: Commit**

```bash
git add gpu/metal/tracer.metal
git commit -m "feat(metal): add hessian_H_f and build_M_f for Jacobi bundle"
```

---

## Task 2: Bundle RHS and RK4 in tracer.metal

**Files:**
- Modify: `gpu/metal/tracer.metal` — insert after `build_M_f`

The bundle state has 13 floats: (r, theta, phi, pr, pth) + W[4][2].
We encode W as two float4s: `w0` (alpha deviations) and `w1` (beta deviations),
where `w0[i]` = W[i][0] and `w1[i]` = W[i][1].

- [ ] **Step 1: Insert `bundle_rhs_f` function**

```metal
// dz/dlam and dW/dlam for the coupled (geodesic + Jacobi) system.
// Inputs:  current (r,theta,pr,pth) and W encoded as w0,w1 (float4 each).
// Outputs: dz[4], dw0, dw1.
static void bundle_rhs_f(float r, float theta, float pr, float pth,
                          float pt, float pphi,
                          float M, float a, float Q, float L,
                          float4 w0, float4 w1,        // W[:,0], W[:,1]
                          thread float dz[4],
                          thread float4& dw0_out,
                          thread float4& dw1_out) {
    // 1. Geodesic RHS (reuse existing function, ignore phi derivative)
    float dr, dth, dphi_unused, dpr, dpth;
    geodesic_rhs(r, theta, pr, pth, pt, pphi, M, a, Q, L,
                 dr, dth, dphi_unused, dpr, dpth);
    dz[0] = dr; dz[1] = dth; dz[2] = dpr; dz[3] = dpth;

    // 2. Hessian
    float Hess[4][4];
    hessian_H_f(r, theta, pr, pth, pt, pphi, M, a, Q, L, Hess);

    // 3. M = J_s * Hess
    float Mmat[4][4];
    build_M_f(Hess, Mmat);

    // 4. dW/dlam = M * W  (separately for each column)
    float4 dw0 = float4(0.0f);
    float4 dw1 = float4(0.0f);
    // Unroll: dw[i] = sum_j Mmat[i][j] * w[j]
    for (int i = 0; i < 4; ++i) {
        dw0[i] = Mmat[i][0]*w0[0] + Mmat[i][1]*w0[1]
               + Mmat[i][2]*w0[2] + Mmat[i][3]*w0[3];
        dw1[i] = Mmat[i][0]*w1[0] + Mmat[i][1]*w1[1]
               + Mmat[i][2]*w1[2] + Mmat[i][3]*w1[3];
    }
    dw0_out = dw0;
    dw1_out = dw1;
}
```

- [ ] **Step 2: Insert `bundle_rk4_f` function** (immediately after `bundle_rhs_f`)

```metal
// Single RK4 step for the 13-component bundle state.
// (r,theta,phi) in the geo part; phi integrated separately (no Jacobi needed).
// w0, w1 are the two columns of W.
static void bundle_rk4_f(thread float& r,  thread float& theta,
                          thread float& phi,
                          thread float& pr, thread float& pth,
                          float pt, float pphi,
                          float M, float a, float Q, float L,
                          thread float4& w0, thread float4& w1,
                          float dlam) {
    // Stage 1
    float dz1[4]; float4 dw0_1, dw1_1;
    bundle_rhs_f(r, theta, pr, pth, pt, pphi, M, a, Q, L,
                 w0, w1, dz1, dw0_1, dw1_1);

    // Stage 2
    float r2  = r   + 0.5f*dlam*dz1[0];
    float th2 = theta + 0.5f*dlam*dz1[1];
    float pr2 = pr  + 0.5f*dlam*dz1[2];
    float pt2 = pth + 0.5f*dlam*dz1[3];
    float4 w0_2 = w0 + 0.5f*dlam*dw0_1;
    float4 w1_2 = w1 + 0.5f*dlam*dw1_1;
    float dz2[4]; float4 dw0_2, dw1_2;
    bundle_rhs_f(r2, th2, pr2, pt2, pt, pphi, M, a, Q, L,
                 w0_2, w1_2, dz2, dw0_2, dw1_2);

    // Stage 3
    float r3  = r   + 0.5f*dlam*dz2[0];
    float th3 = theta + 0.5f*dlam*dz2[1];
    float pr3 = pr  + 0.5f*dlam*dz2[2];
    float pt3 = pth + 0.5f*dlam*dz2[3];
    float4 w0_3 = w0 + 0.5f*dlam*dw0_2;
    float4 w1_3 = w1 + 0.5f*dlam*dw1_2;
    float dz3[4]; float4 dw0_3, dw1_3;
    bundle_rhs_f(r3, th3, pr3, pt3, pt, pphi, M, a, Q, L,
                 w0_3, w1_3, dz3, dw0_3, dw1_3);

    // Stage 4
    float r4  = r   + dlam*dz3[0];
    float th4 = theta + dlam*dz3[1];
    float pr4 = pr  + dlam*dz3[2];
    float pt4 = pth + dlam*dz3[3];
    float4 w0_4 = w0 + dlam*dw0_3;
    float4 w1_4 = w1 + dlam*dw1_3;
    float dz4[4]; float4 dw0_4, dw1_4;
    bundle_rhs_f(r4, th4, pr4, pt4, pt, pphi, M, a, Q, L,
                 w0_4, w1_4, dz4, dw0_4, dw1_4);

    // φ at each stage (independent of W)
    const float dphi1 = gUU_phi_vel(r, theta,        pt, pphi, M, a, Q, L);
    const float dphi2 = gUU_phi_vel(r2, th2,         pt, pphi, M, a, Q, L);
    const float dphi3 = gUU_phi_vel(r3, th3,         pt, pphi, M, a, Q, L);
    const float dphi4 = gUU_phi_vel(r4, th4,         pt, pphi, M, a, Q, L);

    r     += dlam/6.0f*(dz1[0]+2.0f*dz2[0]+2.0f*dz3[0]+dz4[0]);
    theta += dlam/6.0f*(dz1[1]+2.0f*dz2[1]+2.0f*dz3[1]+dz4[1]);
    phi   += dlam/6.0f*(dphi1 +2.0f*dphi2 +2.0f*dphi3 +dphi4);
    pr    += dlam/6.0f*(dz1[2]+2.0f*dz2[2]+2.0f*dz3[2]+dz4[2]);
    pth   += dlam/6.0f*(dz1[3]+2.0f*dz2[3]+2.0f*dz3[3]+dz4[3]);
    w0    += dlam/6.0f*(dw0_1 +2.0f*dw0_2 +2.0f*dw0_3 +dw0_4);
    w1    += dlam/6.0f*(dw1_1 +2.0f*dw1_2 +2.0f*dw1_3 +dw1_4);
}
```

`gUU_phi_vel` is already in tracer.metal as the φ-velocity helper used in `rk4()`.
Check its actual name:

```bash
grep -n "phi_vel\|dphi_vel\|gu\[3\]\[0\].*pt" gpu/metal/tracer.metal | head -10
```

If it's inlined inside `rk4()`, extract it as a `static float gUU_phi_vel(...)` helper **before** `rk4()`:

```metal
static float gUU_phi_vel(float r, float theta,
                          float pt, float pphi,
                          float M, float a, float Q, float L) {
    float gu[4][4];
    gUU(r, theta, M, a, Q, L, gu);
    return gu[3][0]*pt + gu[3][3]*pphi;
}
```

Then update `rk4()` to call `gUU_phi_vel(...)` for its dphi stages.

- [ ] **Step 3: Insert `bundle_adaptive_f` function** (immediately after `bundle_rk4_f`)

Uses step-doubling, same logic as `adaptive_step_bl` but for 13-component bundle state.

```metal
// Adaptive RK4 + step-doubling for bundle state.
// Returns true when step accepted; dlam updated in-place.
// Error estimated from geodesic component only (W error is secondary).
static bool bundle_adaptive_f(thread float& r,  thread float& theta,
                               thread float& phi,
                               thread float& pr, thread float& pth,
                               float pt, float pphi,
                               float M, float a, float Q, float L,
                               thread float4& w0, thread float4& w1,
                               thread float& dlam,
                               float tol) {
    // Save state
    const float r0=r, th0=theta, ph0=phi, pr0=pr, pth0=pth;
    const float4 w0_0=w0, w1_0=w1;

    // Path A: one full step
    float rA=r0,thA=th0,phA=ph0,prA=pr0,pthA=pth0;
    float4 w0A=w0_0, w1A=w1_0;
    bundle_rk4_f(rA,thA,phA,prA,pthA, pt,pphi, M,a,Q,L, w0A,w1A, dlam);

    // Path B: two half steps
    float rB=r0,thB=th0,phB=ph0,prB=pr0,pthB=pth0;
    float4 w0B=w0_0, w1B=w1_0;
    bundle_rk4_f(rB,thB,phB,prB,pthB, pt,pphi, M,a,Q,L, w0B,w1B, dlam*0.5f);
    bundle_rk4_f(rB,thB,phB,prB,pthB, pt,pphi, M,a,Q,L, w0B,w1B, dlam*0.5f);

    const float err = length(float4(rA-rB, thA-thB, prA-prB, pthA-pthB)) / 15.0f;

    if (!isfinite(err)) {
        // Restore and shrink
        r=r0; theta=th0; phi=ph0; pr=pr0; pth=pth0;
        w0=w0_0; w1=w1_0;
        dlam = max(dlam * 0.5f, ADAPT_H_MIN);
        return false;
    }

    const bool accepted = (err < tol || dlam < ADAPT_H_MIN);
    if (accepted) {
        r=rB; theta=thB; phi=phB; pr=prB; pth=pthB;
        w0=w0B; w1=w1B;
        const float sc = (err > 1e-10f) ? 0.9f*pow(tol/err, 0.2f) : 2.0f;
        dlam = clamp(dlam*sc, ADAPT_H_MIN, ADAPT_H_MAX);
    } else {
        r=r0; theta=th0; phi=ph0; pr=pr0; pth=pth0;
        w0=w0_0; w1=w1_0;
        dlam = clamp(dlam*0.9f*pow(tol/err, 0.25f), ADAPT_H_MIN, dlam*0.5f);
    }
    return accepted;
}
```

- [ ] **Step 4: Build and verify compile**

```bash
cmake --build build -j$(sysctl -n hw.logicalcpu) 2>&1 | grep -E "error:|Built"
```

Expected: `[100%] Built target kerr_tracer_metal` — no errors.

- [ ] **Step 5: Commit**

```bash
git add gpu/metal/tracer.metal
git commit -m "feat(metal): add bundle_rhs_f, bundle_rk4_f, bundle_adaptive_f"
```

---

## Task 3: `trace_bundle_jacobi_f` tracer function

**Files:**
- Modify: `gpu/metal/tracer.metal` — insert after `bundle_adaptive_f`, before the `RayTraceResultBL` struct

This function mirrors `trace_bundle()` in `ray_bundle.hpp`.
It returns a `RayTraceResultBL` plus fills an output `float& magnif_out`.

- [ ] **Step 1: Insert initial-conditions helper `init_bundle_f`**

Computes the two deviation vectors W[:,0] (alpha direction) and W[:,1] (beta direction)
by finite-differencing the initial conditions w.r.t. pixel angle.

```metal
// Compute initial Jacobi deviations by finite-differencing angle_ray w.r.t. alpha/beta.
// eps_pix = half-pixel angular size = fov_h / (2 * width).
// Returns w0 = dz/dalpha, w1 = dz/dbeta  (z = r, theta, pr, pth).
static void init_bundle_f(float alpha, float beta,
                           float eps_pix,
                           float r_obs, float theta_obs,
                           float M, float a, float Q, float L,
                           float pt, float pphi,
                           CameraParams cp,
                           thread float4& w0_out,
                           thread float4& w1_out) {
    // We need angle_ray at (alpha ± eps, beta) and (alpha, beta ± eps).
    // Rather than calling the full trace, we just compute the INITIAL state
    // (r,theta,pr,pth) for each perturbed angle.
    // The initial conditions are set by the tetrad construction in
    // trace_standard_chart_from_angles; replicate the relevant part here.
    // Since at the observer (large r, approx. flat spacetime) the mapping
    // angle -> initial momenta is nearly linear, a central difference suffices.

    // Helper: compute (pr, pth) from (alpha, beta) at observer.
    // Reuse the exact same tetrad logic as trace_standard_chart_from_angles.
    auto init_pr_pth = [&](float al, float be,
                            thread float& pr_out, thread float& pth_out) {
        float gll[4][4];
        gLL_BL(r_obs, theta_obs, M, a, Q, L, gll);
        float gu[4][4];
        gUU(r_obs, theta_obs, M, a, Q, L, gu);

        const float ca = cos(al), sa = sin(al);
        const float cb = cos(be), sb = sin(be);
        const float gtt   = gll[0][0];
        const float gtphi = gll[0][3];
        const float grr   = gll[1][1];
        const float gthth = gll[2][2];
        const float gphph = gll[3][3];
        const float sqrt_grr   = sqrt(abs(grr));
        const float sqrt_gthth = sqrt(abs(gthth));
        const float denom_phi  = gphph - (gtphi*gtphi)/gtt;

        float pUt = 1.0f, pUr, pUth, pUphi;
        const bool tetrad_ok =
            (gtt < -1e-14f) && isfinite(denom_phi) && (denom_phi > 1e-14f) &&
            isfinite(sqrt_grr) && (sqrt_grr > 1e-14f) &&
            isfinite(sqrt_gthth) && (sqrt_gthth > 1e-14f);
        if (tetrad_ok) {
            const float ut     = 1.0f / sqrt(-gtt);
            const float ephi_p = 1.0f / sqrt(denom_phi);
            const float ephi_t = -gtphi/gtt * ephi_p;
            pUt   = ut + (-sa*cb) * ephi_t;
            pUr   = (-ca*cb) / sqrt_grr;
            pUth  = (-sb)    / sqrt_gthth;
            pUphi = (-sa*cb) * ephi_p;
        } else {
            pUr   = -ca*cb / max(sqrt_grr, 1e-14f);
            pUth  = -sb    / max(sqrt_gthth, 1e-14f);
            pUphi = -sa*cb / sqrt(max(abs(gphph), 1e-14f));
        }
        pr_out  = grr   * pUr;
        pth_out = gthth * pUth;
    };

    float pr_ap, pth_ap, pr_am, pth_am;
    float pr_bp, pth_bp, pr_bm, pth_bm;
    init_pr_pth(alpha + eps_pix, beta,          pr_ap, pth_ap);
    init_pr_pth(alpha - eps_pix, beta,          pr_am, pth_am);
    init_pr_pth(alpha,           beta + eps_pix, pr_bp, pth_bp);
    init_pr_pth(alpha,           beta - eps_pix, pr_bm, pth_bm);

    // r and theta don't change with angle at the observer (fixed observer position)
    // Only pr and pth change. delta_r = 0, delta_theta = 0.
    const float two_eps = 2.0f * eps_pix;
    w0_out = float4(0.0f,
                    0.0f,
                    (pr_ap  - pr_am)  / two_eps,
                    (pth_ap - pth_am) / two_eps);
    w1_out = float4(0.0f,
                    0.0f,
                    (pr_bp  - pr_bm)  / two_eps,
                    (pth_bp - pth_bm) / two_eps);
}
```

- [ ] **Step 2: Insert `trace_bundle_jacobi_f` function** (immediately after `init_bundle_f`)

```metal
struct BundleTraceResult {
    int   outcome;      // 0=escape, 1=disk, 2=horizon
    float r_hit;
    float redshift;
    float phi_hit;
    float theta_esc;
    float phi_esc;
    float magnif;       // |det J| at disk crossing
};

static BundleTraceResult trace_bundle_jacobi_f(
        float alpha, float beta,
        float M, float a, float Q, float L,
        float r_obs, float theta_obs, float phi_obs,
        float r_horizon, float r_disk_in, float r_disk_out,
        float pt, float pphi,
        float step_init, float tol, int max_steps,
        CameraParams cp) {

    BundleTraceResult res{};
    res.outcome   = 2;
    res.r_hit     = 0.0f;
    res.redshift  = 1.0f;
    res.phi_hit   = phi_obs;
    res.theta_esc = theta_obs;
    res.phi_esc   = phi_obs;
    res.magnif    = 1.0f;

    // Initial geodesic state (same tetrad as trace_standard_chart_from_angles)
    float gll[4][4]; gLL_BL(r_obs, theta_obs, M, a, Q, L, gll);
    float gu[4][4];  gUU(r_obs, theta_obs, M, a, Q, L, gu);

    const float ca = cos(alpha), sa = sin(alpha);
    const float cb = cos(beta),  sb = sin(beta);
    const float gtt=gll[0][0], gtphi=gll[0][3];
    const float grr=gll[1][1], gthth=gll[2][2], gphph=gll[3][3];
    const float sqrt_grr   = sqrt(abs(grr));
    const float sqrt_gthth = sqrt(abs(gthth));
    const float denom_phi  = gphph - (gtphi*gtphi)/gtt;
    const bool tetrad_ok =
        (gtt < -1e-14f) && isfinite(denom_phi) && (denom_phi > 1e-14f) &&
        isfinite(sqrt_grr) && (sqrt_grr > 1e-14f) &&
        isfinite(sqrt_gthth) && (sqrt_gthth > 1e-14f);

    float pUt=1.0f, pUr, pUth, pUphi;
    if (tetrad_ok) {
        const float ut=1.0f/sqrt(-gtt), ephi_p=1.0f/sqrt(denom_phi);
        const float ephi_t=-gtphi/gtt*ephi_p;
        pUt  =ut+(-sa*cb)*ephi_t;
        pUr  =(-ca*cb)/sqrt_grr;
        pUth =(-sb)/sqrt_gthth;
        pUphi=(-sa*cb)*ephi_p;
    } else {
        pUr  =-ca*cb/max(sqrt_grr,1e-14f);
        pUth =-sb/max(sqrt_gthth,1e-14f);
        pUphi=-sa*cb/sqrt(max(abs(gphph),1e-14f));
    }
    float r   = r_obs,     theta = theta_obs, phi = phi_obs;
    float pr  = grr*pUr,   pth   = gthth*pUth;
    // pphi already provided as conserved parameter
    // Recompute pt enforcing null constraint:
    { const float A=gu[0][0], B=2.0f*gu[0][3]*pphi;
      const float C=gu[1][1]*pr*pr+gu[2][2]*pth*pth+gu[3][3]*pphi*pphi;
      const float disc=B*B-4.0f*A*C;
      if (disc>=0.0f && abs(A)>1e-15f) {
          const float sq=sqrt(disc);
          const float pt1=(-B-sq)/(2.0f*A), pt2=(-B+sq)/(2.0f*A);
          pt=(pt1<0.0f)?pt1:pt2;
          if(pt>0.0f) pt=min(pt1,pt2);
      }
    }

    // Initial Jacobi deviations
    const float eps_pix = cp.fov_h / float(max(cp.width, 1)) * 0.5f;
    float4 w0, w1;
    init_bundle_f(alpha, beta, eps_pix, r_obs, theta_obs,
                  M, a, Q, L, pt, pphi, cp, w0, w1);

    const float rh_cut   = r_horizon * 1.03f;
    const float r_escape = r_obs * 1.05f;
    float dlam = max(step_init, ADAPT_H_MIN);
    const int iter_cap = max(max_steps, 1);

    float prev_r=r, prev_theta=theta, prev_phi=phi, prev_pr=pr, prev_pth=pth;
    float4 prev_w0=w0, prev_w1=w1;
    float prev_dr, prev_dth, prev_dphi_u, prev_dpr, prev_dpth;
    geodesic_rhs(r, theta, pr, pth, pt, pphi, M, a, Q, L,
                 prev_dr, prev_dth, prev_dphi_u, prev_dpr, prev_dpth);

    for (int iter = 0; iter < iter_cap; ++iter) {
        // Save pre-step state (for Hermite interpolation)
        const float s_r=r, s_th=theta, s_ph=phi, s_pr=pr, s_pth=pth;
        const float4 s_w0=w0, s_w1=w1;
        const float s_dr=prev_dr, s_dth=prev_dth, s_dpr=prev_dpr, s_dpth=prev_dpth;
        const float step_used = dlam;

        // Advance (with adaptive rejection loop)
        int rejects = 0;
        while (true) {
            if (bundle_adaptive_f(r, theta, phi, pr, pth, pt, pphi, M, a, Q, L,
                                  w0, w1, dlam, tol)) break;
            if (!isfinite(dlam) || ++rejects > 32) {
                res.outcome = 2; return res;
            }
        }

        // Compute post-step RHS derivatives for Hermite crossing detection
        float cur_dr, cur_dth, cur_dphi_u, cur_dpr, cur_dpth;
        geodesic_rhs(r, theta, pr, pth, pt, pphi, M, a, Q, L,
                     cur_dr, cur_dth, cur_dphi_u, cur_dpr, cur_dpth);

        // ── Disk crossing? ──────────────────────────────────────
        const float q0 = s_th  - M_PI_2_F;
        const float q1 = theta - M_PI_2_F;
        if (sign_change_f(q0, q1) ||
            (min(abs(q0), abs(q1)) < 0.35f)) {
            float alpha_c;
            if (first_event_alpha_hermite_f(
                    s_th, theta, s_dth, cur_dth, step_used, M_PI_2_F,
                    alpha_c, 8, 8)) {
                const float r_hit = hermite_interp_f(s_r, r, s_dr, cur_dr, step_used, alpha_c);
                if (r_hit >= r_disk_in && r_hit <= r_disk_out) {
                    // Redshift (same formula as in trace_standard_chart_from_angles)
                    float gll2[4][4]; gLL_BL(r_hit, M_PI_2_F, M, a, Q, L, gll2);
                    const float Omega = keplerian_omega(r_hit, M, a);
                    const float b_imp = -pphi / (-pt);
                    const float d2 = -(gll2[0][0]+2.0f*gll2[0][3]*Omega+gll2[3][3]*Omega*Omega);
                    const float dv  = 1.0f - Omega*b_imp;
                    float red = (d2>0.0f && abs(dv)>1e-10f) ? sqrt(d2)/dv : 1.0f;
                    red = clamp(red, 0.0f, 20.0f);

                    // Interpolate W at crossing
                    const float4 w0_c = s_w0 + alpha_c*(w0 - s_w0);
                    const float4 w1_c = s_w1 + alpha_c*(w1 - s_w1);

                    // Jacobi map J: screen(alpha,beta) -> disk(r, phi)
                    // J[0][*] = w0_c[0], w1_c[0]  (dr deviations)
                    // J[1][*] = w0_c[1], w1_c[1]  ... but at equatorial crossing
                    // we map (r, theta) sub-block; theta-dev ~ dphi via disk geometry.
                    // Use (delta_r, delta_theta) as proxy for (delta_r, delta_phi_disk):
                    const float J00 = w0_c[0]; // dr/dalpha
                    const float J01 = w1_c[0]; // dr/dbeta
                    const float J10 = w0_c[1]; // dth/dalpha
                    const float J11 = w1_c[1]; // dth/dbeta
                    float det = abs(J00*J11 - J01*J10);
                    det = max(det, 1e-12f);

                    const float phi_c = s_ph + alpha_c*(phi - s_ph);
                    res.outcome  = 1;
                    res.r_hit    = r_hit;
                    res.redshift = red;
                    res.phi_hit  = phi_c;
                    res.magnif   = det;
                    return res;
                }
            }
        }

        // ── Horizon? ───────────────────────────────────────────
        if ((s_r > rh_cut && r <= rh_cut) || r <= rh_cut) {
            res.outcome = 2; return res;
        }

        // ── Escape? ────────────────────────────────────────────
        if ((s_r < r_escape && r >= r_escape) || r >= r_escape) {
            res.outcome   = 0;
            res.theta_esc = s_th + (r_escape-s_r)/(r-s_r+1e-12f) * (theta-s_th);
            res.phi_esc   = s_ph + (r_escape-s_r)/(r-s_r+1e-12f) * (phi-s_ph);
            return res;
        }

        // Update stored derivatives for next iteration
        prev_dr=cur_dr; prev_dth=cur_dth; prev_dpr=cur_dpr; prev_dpth=cur_dpth;
    }
    return res;
}
```

**Note:** `keplerian_omega(r, M, a)` and `M_PI_2_F` are already in tracer.metal; verify their names:

```bash
grep -n "keplerian_omega\|M_PI_2\|PI_2_F" gpu/metal/tracer.metal | head -10
```

If `keplerian_omega` is inlined in the disk color function, extract it as:

```metal
static float keplerian_omega(float r, float M, float a) {
    const float denom = pow(r, 1.5f) + a * sqrt(M);
    return (denom > 1e-12f) ? sqrt(M) / denom : 0.0f;
}
```

- [ ] **Step 3: Build and verify compile**

```bash
cmake --build build -j$(sysctl -n hw.logicalcpu) 2>&1 | grep -E "error:|Built"
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add gpu/metal/tracer.metal
git commit -m "feat(metal): add init_bundle_f and trace_bundle_jacobi_f (Jacobi-field tracer)"
```

---

## Task 4: New `trace_pixel_jacobi` kernel and metal_renderer.mm wiring

**Files:**
- Modify: `gpu/metal/tracer.metal` — add kernel after `trace_pixel_bundle`
- Modify: `gpu/metal/metal_renderer.mm` — add mode-4 kernel name

- [ ] **Step 1: Add `trace_pixel_jacobi` kernel to tracer.metal**

Insert at end of `tracer.metal`, after the `trace_pixel_bundle` kernel:

```metal
// ── Jacobi-field bundle kernel (mode 4) ─────────────────────────
// One thread = one pixel. Traces the Jacobi-field variational equations
// alongside the main geodesic to compute the physically correct magnification.
kernel void trace_pixel_jacobi(
    device uint32_t*          output   [[buffer(0)]],
    constant KNdSParams_C&    kp       [[buffer(1)]],
    constant CameraParams&    cp_cfg   [[buffer(2)]],
    constant RenderParams_C&  rp       [[buffer(3)]],
    texture2d<float>          bg_tex   [[texture(0)]],
    sampler                   bg_samp  [[sampler(0)]],
    uint2                     gid      [[thread_position_in_grid]])
{
    CameraParams cp = cp_cfg;

    const uint px_local = gid.x;
    const uint py_local = gid.y;
    if (px_local >= (uint)rp.tile_w || py_local >= (uint)rp.tile_h) return;
    const uint px = px_local + rp.x_offset;
    const uint py = py_local + rp.y_offset;
    const uint width  = (uint)cp.width;
    const uint height = (uint)cp.height;
    if (px >= width || py >= height) return;

    const float M = kp.M_bh, a = kp.a_bh, Q = kp.Q_bh, L = kp.Lambda;
    const float span = float(max(cp.width, 1) - 1 > 0 ? cp.width - 1 : 1);
    const float pxf  = float(px) + cp.pixel_offset_x;
    const float pyf  = float(py) + cp.pixel_offset_y;
    const float alpha = cp.fov_h * (pxf - 0.5f*(cp.width -1)) / span;
    const float beta  = cp.fov_h * (0.5f*(cp.height-1) - pyf) / span;

    // Conserved quantities: compute pphi from observer angle (same logic as single-ray)
    float gll[4][4]; gLL_BL(cp.r_obs, cp.theta_obs, M, a, Q, L, gll);
    const float ca=cos(alpha), sa=sin(alpha), cb=cos(beta), sb=sin(beta);
    const float gphph=gll[3][3], gtphi=gll[0][3], gtt=gll[0][0], gthth=gll[2][2];
    const float denom_phi = gphph-(gtphi*gtphi)/gtt;
    float pUphi = (isfinite(denom_phi)&&denom_phi>1e-14f)
                  ? (-sa*cb)/sqrt(denom_phi) : (-sa*cb)/sqrt(max(abs(gphph),1e-14f));
    const float pphi = gtphi*(1.0f/sqrt(max(-gtt,1e-14f))) + gphph*pUphi;
    const float pt   = -1.0f;  // placeholder; corrected inside trace_bundle_jacobi_f

    BundleTraceResult btr = trace_bundle_jacobi_f(
        alpha, beta, M, a, Q, L,
        cp.r_obs, cp.theta_obs, cp.phi_obs,
        kp.r_horizon, kp.r_isco, kp.r_disk_out,
        pt, pphi,
        cp.step_init, cp.integrator_tol, cp.max_steps,
        cp);

    uint32_t colour = 0xFF000000u;
    if (btr.outcome == 0) {
        const float4 bgc = clamp(sample_background(bg_tex, bg_samp, btr.theta_esc, btr.phi_esc), 0.0f, 1.0f);
        colour = pack_abgr(bgc.r, bgc.g, bgc.b);
    } else if (btr.outcome == 1) {
        colour = disk_color_abgr(btr.r_hit, btr.phi_hit, btr.redshift, btr.magnif,
                                  M, kp.r_isco, kp.r_disk_out, cp);
    }
    output[py * width + px] = colour;
}
```

- [ ] **Step 2: Add mode 4 to `metal_renderer.mm`**

In `metal_renderer.mm`, find the kernel name selection block (around line 73–81):

```objc
    NSString* kernelName = @"trace_pixel_single";
    switch (cp.metal_kernel_mode) {
        case 1:  kernelName = @"trace_pixel";        break;
        case 2:  kernelName = @"trace_pixel_single"; break;
        case 3:  kernelName = @"trace_pixel_bundle"; break;
        default: kernelName = (cp.use_bundles != 0) ? @"trace_pixel_bundle"
                                                    : @"trace_pixel_single";
                 break;
    }
```

Add case 4:

```objc
    NSString* kernelName = @"trace_pixel_single";
    switch (cp.metal_kernel_mode) {
        case 1:  kernelName = @"trace_pixel";        break;
        case 2:  kernelName = @"trace_pixel_single"; break;
        case 3:  kernelName = @"trace_pixel_bundle"; break;
        case 4:  kernelName = @"trace_pixel_jacobi"; break;
        default: kernelName = (cp.use_bundles != 0) ? @"trace_pixel_bundle"
                                                    : @"trace_pixel_single";
                 break;
    }
```

- [ ] **Step 3: Build and verify compile**

```bash
cmake --build build -j$(sysctl -n hw.logicalcpu) 2>&1 | grep -E "error:|Built"
```

Expected: no errors. (May take longer — the Metal shader is now larger.)

- [ ] **Step 4: Smoke-test via command line**

```bash
KERR_METAL_TILE_ROWS=2 KERR_METAL_TILE_COLS=128 \
./build/kerr_tracer_metal --720p --spin 0.65 --interstellar \
  --metal-kernel-mode 4 2>&1 | tail -5
ls -lh out/*.png | tail -1
```

Expected: renders without crash, produces a PNG file, progress bar shows 100%.

- [ ] **Step 5: Commit**

```bash
git add gpu/metal/tracer.metal gpu/metal/metal_renderer.mm
git commit -m "feat(metal): add trace_pixel_jacobi kernel (mode 4, Jacobi-field bundle)"
```

---

## Task 5: CLI flag + frontend wiring

**Files:**
- Modify: `main.cpp` — add `--jacobi-bundle` flag
- Modify: `frontend/src/app/app.ts` — add UI option

- [ ] **Step 1: Add `--jacobi-bundle` flag in main.cpp**

Search for the `--bundles` flag parsing block:

```bash
grep -n "bundles\|metal.kernel.mode\|metal_kernel_mode" main.cpp | head -20
```

In the CLI argument loop, add after the `--bundles` handling:

```cpp
} else if (arg == "--jacobi-bundle") {
    fp.use_bundles = 1;
    fp.metal_kernel_mode = 4;   // Jacobi-field mode
```

- [ ] **Step 2: Rebuild CPU binary to include new flag**

```bash
cmake --build build_cpu -j$(sysctl -n hw.logicalcpu) 2>&1 | grep -E "error:|Built"
```

Expected: no errors. (CPU binary will ignore `metal_kernel_mode=4` and fall back to CPU Jacobi bundle, which is correct.)

- [ ] **Step 3: Test `--jacobi-bundle` end-to-end**

```bash
KERR_METAL_TILE_ROWS=2 \
./build/kerr_tracer_metal --720p --spin 0.65 --interstellar --jacobi-bundle 2>&1 | tail -5
```

Expected: produces a PNG, same visual quality or better than `--bundles`.

- [ ] **Step 4: Add "Jacobi bundle" option in frontend**

In `frontend/src/app/app.ts`, find the bundle mode signal/options:

```bash
grep -n "bundle\|Bundle\|use_bundles\|metalKernelMode" frontend/src/app/app.ts | head -20
```

Add a new option in the bundle selector. For example, if there's a `bundleOptions` array:

```typescript
readonly bundleOptions = [
  { value: 0, label: 'Off' },
  { value: 1, label: 'GPU proxy (fast)' },
  { value: 4, label: 'Jacobi field (accurate)' },
];
```

And ensure that when mode 4 is selected, `metal_kernel_mode: 4` is sent in the render params.
The exact code change depends on the current structure — grep for `use_bundles` to find the right place.

- [ ] **Step 5: Commit**

```bash
git add main.cpp frontend/src/app/app.ts
git commit -m "feat: add --jacobi-bundle flag and frontend Jacobi bundle option"
```

---

## Task 6: Visual validation and version bump

**Files:**
- Modify: `frontend/src/app/app.html` — version bump

- [ ] **Step 1: Side-by-side comparison at 720p, a=0.65**

```bash
# 5-ray proxy (existing)
./build/kerr_tracer_metal --720p --spin 0.65 --interstellar --bundles
# Jacobi field (new)
./build/kerr_tracer_metal --720p --spin 0.65 --interstellar --jacobi-bundle
ls -lh out/*.png | tail -2
```

Expected: two PNGs produced. Jacobi version should show smoother magnification
near the photon ring (no finite-diff artifacts) and identical disk structure.

- [ ] **Step 2: Test at a=0.998 with small tile sizes (avoid GPU watchdog)**

```bash
KERR_METAL_TILE_ROWS=1 KERR_METAL_TILE_COLS=64 \
./build/kerr_tracer_metal --720p --spin 0.998 --interstellar --jacobi-bundle 2>&1 | tail -5
```

Expected: completes without crash (may be slow; progress bar should advance).

- [ ] **Step 3: Bump version to v0.2.11**

In `frontend/src/app/app.html`, line 12:

```html
<div><strong>Version</strong> v0.2.11 — Jacobi bundle GPU</div>
```

- [ ] **Step 4: Final commit and push**

```bash
git add frontend/src/app/app.html
git commit -m "feat(v0.2.11): Jacobi-field bundle on Metal GPU (mode 4)"
git push
```

---

## Self-Review

### Spec coverage
- ✅ Hessian_H_f ported with float-appropriate epsilons
- ✅ build_M_f (symplectic gradient matrix)
- ✅ bundle_rhs_f (geo + Jacobi RHS in one function)
- ✅ bundle_rk4_f (single RK4 for 20 components)
- ✅ bundle_adaptive_f (step-doubling, saves/restores full 13-component state)
- ✅ trace_bundle_jacobi_f (full trace loop with Hermite crossing detection)
- ✅ trace_pixel_jacobi kernel (mode 4)
- ✅ metal_renderer.mm case 4
- ✅ --jacobi-bundle CLI flag
- ✅ Frontend option
- ✅ Version bump

### Type consistency
- All new Metal functions use `float` (not double)
- `float4` used for the 4-component Jacobi column vectors (w0, w1)
- `BundleTraceResult` struct distinct from `RayTraceResultBL` (adds `magnif` field)
- `gUU_phi_vel` must be added before `rk4()` if it's currently inlined

### Potential issue: `pt` initialization in `trace_pixel_jacobi`
The kernel computes `pphi` from the tetrad but passes `pt=-1` as placeholder. The actual `pt` is computed inside `trace_bundle_jacobi_f` via the null-constraint quadratic. Verify that the `pphi` passed in is consistent with the `pphi` that would be computed inside the function — if both use the same tetrad logic, they will be.
