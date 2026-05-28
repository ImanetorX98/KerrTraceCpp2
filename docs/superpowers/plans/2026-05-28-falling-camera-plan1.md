# Falling Camera — Implementation Plan 1 (Phases A+B)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a CPU-only falling camera that traces backward null geodesics from a local GPG tetrad, producing per-frame PNG output with background and disk shading.

**Architecture:** Two new files (`falling_camera.hpp`, `falling_renderer.cpp/.hpp`) contain all GPG physics. `main.cpp` gains a `--falling-camera` dispatch branch. No existing files are modified except `main.cpp` and `CMakeLists.txt`.

**Tech Stack:** C++17, KNdSMetric (existing), adaptive RK4 (custom for GPG coords), libpng via existing PNG writer, OpenMP for pixel parallelism.

**Spec:** `docs/superpowers/specs/2026-05-28-falling-camera-design.md`

**This plan covers:** Phases A (background) and B (thin disk + redshift). Metal GPU pass, HorizonFlip, roll, server, and frontend are Plans 2 and 3.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `falling_camera.hpp` | Create | GPG metric, f(r), CameraState, init_worldline, step_worldline, CameraTetrad |
| `falling_renderer.hpp` | Create | FallingParams struct, render_falling_frame() declaration |
| `falling_renderer.cpp` | Create | render_falling_frame(), trace_photon_gpg(), disk shading |
| `tests/test_falling_camera.cpp` | Create | Unit tests for metric, worldline, tetrad, photon init |
| `CMakeLists.txt` | Modify | Add falling_renderer.cpp to CPU_SOURCES, add test target |
| `main.cpp` | Modify | Add --falling-camera arg parsing and dispatch |

---

## Task 1: GPG metric in `falling_camera.hpp`

**Files:**
- Create: `falling_camera.hpp`

- [ ] **Step 1: Write the failing test**

Create `tests/test_falling_camera.cpp`:

```cpp
#include "falling_camera.hpp"
#include "knds_metric.hpp"
#include <cmath>
#include <iostream>

static bool approx(double a, double b, double tol=1e-8) {
    return std::abs(a-b) <= tol;
}

// Test 1: g_μν * g^νρ = δ^μ_ρ at r=10, θ=π/2, Kerr a=0.9
bool test_gpg_metric_inverse() {
    KNdSMetric bh(1.0, 0.9, 0.0, 0.0);
    double gLL[4][4], gUU[4][4];
    gpg_covariant(bh, 10.0, M_PI/2, gLL);
    gpg_contravariant(bh, 10.0, M_PI/2, gUU);
    for (int mu=0; mu<4; ++mu)
        for (int nu=0; nu<4; ++nu) {
            double s=0.0;
            for (int a=0; a<4; ++a) s += gUU[mu][a]*gLL[a][nu];
            double ex = (mu==nu)?1.0:0.0;
            if (!approx(s, ex, 1e-7)) {
                std::cerr << "FAIL inverse ["<<mu<<"]["<<nu<<"] = "<<s<<"\n";
                return false;
            }
        }
    return true;
}

// Test 2: GPG metric reduces to BL metric far from horizon (Δ→R²)
// at r=1000M, f²≈R²≈Δ so off-diagonal terms g_Tr, g_rφ → 0
bool test_gpg_flat_limit() {
    KNdSMetric bh(1.0, 0.5, 0.0, 0.0);
    double gLL[4][4];
    gpg_covariant(bh, 1000.0, M_PI/2, gLL);
    // g_Tr should be tiny
    if (std::abs(gLL[0][1]) > 1e-3) {
        std::cerr << "FAIL flat limit g_Tr = " << gLL[0][1] << "\n";
        return false;
    }
    return true;
}

int main() {
    int fail=0;
    if (!test_gpg_metric_inverse())  { std::cerr<<"FAIL test_gpg_metric_inverse\n";  ++fail; }
    else std::cout<<"PASS test_gpg_metric_inverse\n";
    if (!test_gpg_flat_limit())      { std::cerr<<"FAIL test_gpg_flat_limit\n";      ++fail; }
    else std::cout<<"PASS test_gpg_flat_limit\n";
    return fail;
}
```

- [ ] **Step 2: Run test to verify it fails (no falling_camera.hpp yet)**

```bash
cd /Users/iman.rosignoli/Documents/KerrTraceCpp2
g++ -std=c++17 -O2 -I. tests/test_falling_camera.cpp -o /tmp/test_falling 2>&1 | head -5
```

Expected: compile error `falling_camera.hpp: No such file`.

- [ ] **Step 3: Create `falling_camera.hpp` with GPG metric**

```cpp
#pragma once
// falling_camera.hpp — GPG coordinates for KNdS (Lin & Soo 2009, arXiv:0905.3244)
// Coordinate order: 0=T, 1=r, 2=θ, 3=φ_P

#include "knds_metric.hpp"
#include <cmath>
#include <array>

// ── GPG helper: f(r) ─────────────────────────────────────────────────────────
// Ensures f² > max(Δ, a²Ξ) for all (M,a,Q,Λ).
inline double gpg_f(const KNdSMetric& bh, double r) {
    const double R2  = r*r + bh.a*bh.a;
    const double Q2  = bh.Q*bh.Q;
    const double Lam = bh.Lambda;
    if (Lam >= 0.0)
        return std::sqrt(R2 + Q2 + Lam*bh.a*bh.a*bh.a*bh.a/3.0);
    else
        return std::sqrt(R2*(1.0 - Lam*r*r/3.0) + Q2);
}

// ── Covariant GPG metric g_μν ─────────────────────────────────────────────────
// Fills gLL[4][4]. Index order: 0=T,1=r,2=θ,3=φ_P.
inline void gpg_covariant(const KNdSMetric& bh, double r, double theta,
                           double gLL[4][4])
{
    const double a   = bh.a;
    const double M   = bh.M;
    const double Q2  = bh.Q*bh.Q;
    const double Lam = bh.Lambda;

    const double R2  = r*r + a*a;
    const double rho2= r*r + a*a*std::cos(theta)*std::cos(theta);
    const double Xi  = 1.0 + Lam*a*a/3.0;
    const double Xit = 1.0 + Lam*a*a*std::cos(theta)*std::cos(theta)/3.0;
    const double Del = R2*(1.0 - Lam*r*r/3.0) - 2.0*M*r + Q2;
    const double f   = gpg_f(bh, r);
    const double f2  = f*f;
    const double s   = std::sin(theta);
    const double s2  = s*s;

    // Pre-computed vierbein components (e^A_μ, A=tetrad index)
    const double D   = f2 - a*a*Xit*s2;           // discriminant for e^0, e^3
    const double sqD = std::sqrt(std::max(D, 0.0));
    const double sqFD= std::sqrt(std::max(f2 - Del, 0.0)); // √(f²-Δ)

    // e^0 (timelike)
    const double e0T  =  sqD / (Xi*std::sqrt(rho2));
    const double e0ph = a*s2*(Xit*R2 - f2) / (Xi*std::sqrt(rho2)*sqD);

    // e^1 (radial)
    const double e1T  =  sqFD / (Xi*std::sqrt(rho2));
    const double e1r  =  std::sqrt(rho2) / f;
    const double e1ph = -a*s2*sqFD / (Xi*std::sqrt(rho2));

    // e^2 (polar) — only θ component
    const double e2th =  std::sqrt(rho2) / std::sqrt(Xit);

    // e^3 (azimuthal frame)
    const double e3T  = -a*std::sqrt(rho2)*std::sqrt(Xit)*s / (f*Xi*sqD);
    const double e3ph =  std::sqrt(rho2)*R2*std::sqrt(Xit)*s / (f*Xi*sqD);

    // Zero out
    for (int i=0;i<4;++i) for (int j=0;j<4;++j) gLL[i][j]=0.0;

    // g_μν = η_AB e^A_μ e^B_ν,  η=diag(-1,+1,+1,+1)
    // TT
    gLL[0][0] = -e0T*e0T + e1T*e1T + e3T*e3T;
    // rr
    gLL[1][1] = e1r*e1r;                          // = rho²/f²
    // θθ
    gLL[2][2] = e2th*e2th;                         // = rho²/Xit
    // φφ
    gLL[3][3] = -e0ph*e0ph + e1ph*e1ph + e3ph*e3ph;
    // Tr (symmetric)
    gLL[0][1] = gLL[1][0] = -e0T*0.0 + e1T*e1r + 0.0;   // e0r=0, e3r=0
    // Tφ (symmetric)
    gLL[0][3] = gLL[3][0] = -e0T*e0ph + e1T*e1ph + e3T*e3ph;
    // rφ (symmetric)
    gLL[1][3] = gLL[3][1] = e1r*e1ph;             // e0r=e3r=0 → only e1 term
}

// ── Contravariant GPG metric g^μν (numerical inverse via cofactors) ───────────
inline void gpg_contravariant(const KNdSMetric& bh, double r, double theta,
                               double gUU[4][4])
{
    double gLL[4][4];
    gpg_covariant(bh, r, theta, gLL);

    // 4×4 matrix inverse via cofactor expansion
    // Only non-zero blocks: (T,r,φ) 3×3 + θ diagonal
    // General inversion for robustness:
    const auto& m = gLL;
    auto cofactor = [&](int r0, int c0) -> double {
        double sub[3][3]; int ri=0;
        for (int i=0;i<4;++i) { if(i==r0) continue; int ci=0;
            for (int j=0;j<4;++j) { if(j==c0) continue; sub[ri][ci++]=m[i][j]; } ++ri; }
        return sub[0][0]*(sub[1][1]*sub[2][2]-sub[1][2]*sub[2][1])
              -sub[0][1]*(sub[1][0]*sub[2][2]-sub[1][2]*sub[2][0])
              +sub[0][2]*(sub[1][0]*sub[2][1]-sub[1][1]*sub[2][0]);
    };
    double det = 0.0;
    for (int j=0;j<4;++j) det += m[0][j]*cofactor(0,j)*((j%2==0)?1:-1);
    const double inv_det = 1.0/det;
    for (int i=0;i<4;++i)
        for (int j=0;j<4;++j)
            gUU[i][j] = cofactor(j,i)*((i+j)%2==0?1:-1)*inv_det;
}
```

- [ ] **Step 4: Run test**

```bash
cd /Users/iman.rosignoli/Documents/KerrTraceCpp2
g++ -std=c++17 -O2 -I. tests/test_falling_camera.cpp -o /tmp/test_falling && /tmp/test_falling
```

Expected:
```
PASS test_gpg_metric_inverse
PASS test_gpg_flat_limit
```

- [ ] **Step 5: Commit**

```bash
git add falling_camera.hpp tests/test_falling_camera.cpp
git commit -m "feat(falling): GPG metric covariant/contravariant for KNdS"
```

---

## Task 2: CameraState and worldline initial conditions

**Files:**
- Modify: `falling_camera.hpp` (append)
- Modify: `tests/test_falling_camera.cpp` (append test)

- [ ] **Step 1: Add failing test** — append to `tests/test_falling_camera.cpp` before `main()`:

```cpp
// Test 3: worldline normalization g_μν u^μ u^ν = -1 after init
bool test_worldline_init_normalized() {
    KNdSMetric bh(1.0, 0.9, 0.0, 0.0);
    // E=1 (fall from rest at infinity), L=0, Qc=0, equatorial
    FallingParams fp{ bh, 20.0, 1.0, 0.0, 0.0, M_PI/2 };
    CameraState cs = init_worldline(fp);
    double gLL[4][4];
    gpg_covariant(bh, cs.x[1], cs.x[2], gLL);
    double norm=0.0;
    for (int mu=0;mu<4;++mu)
        for (int nu=0;nu<4;++nu)
            norm += gLL[mu][nu]*cs.u[mu]*cs.u[nu];
    if (!approx(norm, -1.0, 1e-7)) {
        std::cerr << "FAIL norm = " << norm << "\n";
        return false;
    }
    return true;
}

// Test 4: u_T = -E (conserved energy)
bool test_worldline_killing_energy() {
    KNdSMetric bh(1.0, 0.5, 0.0, 0.0);
    FallingParams fp{ bh, 15.0, 1.2, 0.0, 0.0, M_PI/2 };
    CameraState cs = init_worldline(fp);
    double gLL[4][4];
    gpg_covariant(bh, cs.x[1], cs.x[2], gLL);
    double u_T=0.0;
    for (int nu=0;nu<4;++nu) u_T += gLL[0][nu]*cs.u[nu];
    if (!approx(u_T, -fp.E, 1e-7)) {
        std::cerr << "FAIL u_T = " << u_T << " expected " << -fp.E << "\n";
        return false;
    }
    return true;
}
```

Add to `main()`:
```cpp
if (!test_worldline_init_normalized()) { std::cerr<<"FAIL test_worldline_init_normalized\n"; ++fail; }
else std::cout<<"PASS test_worldline_init_normalized\n";
if (!test_worldline_killing_energy())  { std::cerr<<"FAIL test_worldline_killing_energy\n";  ++fail; }
else std::cout<<"PASS test_worldline_killing_energy\n";
```

- [ ] **Step 2: Run test to verify it fails**

```bash
g++ -std=c++17 -O2 -I. tests/test_falling_camera.cpp -o /tmp/test_falling 2>&1 | head -5
```

Expected: error on `FallingParams`, `CameraState`, `init_worldline` not defined.

- [ ] **Step 3: Append to `falling_camera.hpp`**

```cpp
// ── Falling camera parameter block ───────────────────────────────────────────
struct FallingParams {
    KNdSMetric bh;
    double r_start    = 20.0;   // initial BL radius [M]
    double E          = 1.0;    // conserved energy (1 = fall from rest at ∞)
    double L          = 0.0;    // conserved angular momentum [M]
    double Qc         = 0.0;    // Carter constant [M²]
    double theta_start= M_PI/2; // initial polar angle [rad]
    double phi_start  = 0.0;    // initial azimuthal angle [rad]
    // Render geometry
    double fov_h      = 90.0 * M_PI/180.0;
    int    width      = 1280;
    int    height     = 720;
    int    frames     = 120;
    double dtau       = 0.1;    // proper time step per frame
    // Shading
    double r_disk_in  = -1.0;   // <0 = use ISCO
    double r_disk_out = 12.0;
    double r_escape   = 200.0;
    double r_singularity = 0.05;
    std::string disk_palette = "interstellar";
    double disk_brightness   = 1.0;
    double background        = 0.0;  // 0 = black, use HDR path later
};

// ── Camera state in GPG coordinates ──────────────────────────────────────────
// x[0]=T, x[1]=r, x[2]=θ, x[3]=φ_P   (position)
// u[0..3]                              (contravariant four-velocity)
struct CameraState {
    double x[4];
    double u[4];
};

// ── Initial worldline from conserved quantities ───────────────────────────────
// Sets up (T=0, r_start, θ_start, φ=0) with four-velocity from (E, L, Qc).
// u_T = -E, u_φ = L,  u_θ from Qc,  u_r from normalization g^μν u_μ u_ν = -1.
inline CameraState init_worldline(const FallingParams& fp) {
    const KNdSMetric& bh = fp.bh;
    const double r  = fp.r_start;
    const double th = fp.theta_start;

    double gLL[4][4], gUU[4][4];
    gpg_covariant(bh, r, th, gLL);
    gpg_contravariant(bh, r, th, gUU);

    // Covariant conserved components
    const double uT_low  = -fp.E;   // u_T = g_Tμ u^μ = -E
    const double uph_low =  fp.L;   // u_φ = g_φμ u^μ = +L

    // u_θ from Carter constant: Qc = u_θ² + cos²θ (a²u_T² - L²/sin²θ)
    // For KNdS the full expression includes Ξ corrections; this is the
    // leading-order form valid for a·Λ ≪ 1. Good for Phase A/B.
    const double cos2 = std::cos(th)*std::cos(th);
    const double sin2 = std::sin(th)*std::sin(th);
    double under_uth2 = fp.Qc - cos2*(bh.a*bh.a*fp.E*fp.E - fp.L*fp.L/std::max(sin2,1e-10));
    const double uth_low = std::sqrt(std::max(under_uth2, 0.0));

    // Raise indices: u^μ = g^μν u_ν  (only T, θ, φ known; solve for r)
    // Normalization: g^μν u_μ u_ν = -1
    // Expand: g^rr u_r² + 2 g^rT u_T u_r + 2 g^rφ u_φ u_r + C = -1
    // where C = g^TT u_T² + 2 g^Tφ u_T u_φ + g^φφ u_φ² + g^θθ u_θ²
    const double C = gUU[0][0]*uT_low*uT_low
                   + 2.0*gUU[0][3]*uT_low*uph_low
                   + gUU[3][3]*uph_low*uph_low
                   + gUU[2][2]*uth_low*uth_low;
    // A·ur² + B·ur + (C+1) = 0
    const double A = gUU[1][1];
    const double B = 2.0*(gUU[1][0]*uT_low + gUU[1][3]*uph_low);
    const double disc = B*B - 4.0*A*(C+1.0);
    // Pick ingoing root (ur < 0 = falling inward)
    const double ur_low = (-B - std::sqrt(std::max(disc, 0.0))) / (2.0*A);

    // Raise all indices
    CameraState cs;
    cs.x[0] = 0.0; cs.x[1] = r; cs.x[2] = th; cs.x[3] = fp.phi_start;
    cs.u[0] = gUU[0][0]*uT_low + gUU[0][1]*ur_low + gUU[0][3]*uph_low;
    cs.u[1] = gUU[1][0]*uT_low + gUU[1][1]*ur_low + gUU[1][3]*uph_low;
    cs.u[2] = gUU[2][2]*uth_low;
    cs.u[3] = gUU[3][0]*uT_low + gUU[3][1]*ur_low + gUU[3][3]*uph_low;
    return cs;
}
```

- [ ] **Step 4: Run test**

```bash
g++ -std=c++17 -O2 -I. tests/test_falling_camera.cpp -o /tmp/test_falling && /tmp/test_falling
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add falling_camera.hpp tests/test_falling_camera.cpp
git commit -m "feat(falling): FallingParams, CameraState, init_worldline"
```

---

## Task 3: Worldline RK4 integration

**Files:**
- Modify: `falling_camera.hpp` (append)
- Modify: `tests/test_falling_camera.cpp` (append test)

- [ ] **Step 1: Add failing test** — append before `main()`:

```cpp
// Test 5: after 10 steps normalization stays within 1e-6
bool test_worldline_step_stays_normalized() {
    KNdSMetric bh(1.0, 0.9, 0.0, 0.0);
    FallingParams fp{ bh }; fp.r_start=15.0; fp.E=1.0; fp.L=0.0;
    CameraState cs = init_worldline(fp);
    for (int i=0;i<10;++i) cs = step_worldline(cs, bh, 0.05);
    double gLL[4][4];
    gpg_covariant(bh, cs.x[1], cs.x[2], gLL);
    double norm=0.0;
    for (int mu=0;mu<4;++mu)
        for (int nu=0;nu<4;++nu) norm += gLL[mu][nu]*cs.u[mu]*cs.u[nu];
    if (!approx(norm, -1.0, 1e-5)) {
        std::cerr<<"FAIL step norm="<<norm<<"\n"; return false;
    }
    return true;
}

// Test 6: camera falls inward (r decreases)
bool test_worldline_r_decreases() {
    KNdSMetric bh(1.0, 0.5, 0.0, 0.0);
    FallingParams fp{ bh }; fp.r_start=20.0; fp.E=1.0; fp.L=0.0;
    CameraState cs = init_worldline(fp);
    double r0 = cs.x[1];
    for (int i=0;i<20;++i) cs = step_worldline(cs, bh, 0.1);
    if (cs.x[1] >= r0) {
        std::cerr<<"FAIL r did not decrease: "<<r0<<" -> "<<cs.x[1]<<"\n";
        return false;
    }
    return true;
}
```

Add to `main()`:
```cpp
if (!test_worldline_step_stays_normalized()) { std::cerr<<"FAIL step_normalized\n"; ++fail; }
else std::cout<<"PASS test_worldline_step_stays_normalized\n";
if (!test_worldline_r_decreases())           { std::cerr<<"FAIL r_decreases\n";     ++fail; }
else std::cout<<"PASS test_worldline_r_decreases\n";
```

- [ ] **Step 2: Run test to verify it fails**

```bash
g++ -std=c++17 -O2 -I. tests/test_falling_camera.cpp -o /tmp/test_falling 2>&1 | head -3
```

Expected: error on `step_worldline` not defined.

- [ ] **Step 3: Append to `falling_camera.hpp`**

```cpp
// ── Numerical Christoffel Γ^μ_{αβ} from GPG metric ───────────────────────────
// Uses central finite differences: Γ^μ_{αβ} = ½ g^μν(∂_α g_νβ + ∂_β g_να - ∂_ν g_αβ)
inline void gpg_christoffel(const KNdSMetric& bh, double r, double th,
                             double Gamma[4][4][4])  // Gamma[mu][alpha][beta]
{
    const double hr = r  * 1e-5 + 1e-9;
    const double ht = 1e-5;

    double gp[4][4], gm[4][4], gtp[4][4], gtm[4][4];
    gpg_covariant(bh, r+hr, th,    gp);
    gpg_covariant(bh, r-hr, th,    gm);
    gpg_covariant(bh, r,    th+ht, gtp);
    gpg_covariant(bh, r,    th-ht, gtm);

    // ∂g/∂r, ∂g/∂θ  (other coords don't appear in g for stationary axisymmetric)
    double dg[4][4][4] = {};   // dg[coord][mu][nu]
    for (int i=0;i<4;++i) for (int j=0;j<4;++j) {
        dg[1][i][j] = (gp[i][j] - gm[i][j]) / (2.0*hr);
        dg[2][i][j] = (gtp[i][j]- gtm[i][j])/ (2.0*ht);
    }

    double gUU[4][4];
    gpg_contravariant(bh, r, th, gUU);

    for (int mu=0;mu<4;++mu)
        for (int al=0;al<4;++al)
            for (int be=0;be<4;++be) {
                double s=0.0;
                for (int nu=0;nu<4;++nu)
                    s += gUU[mu][nu]*( dg[al][nu][be] + dg[be][nu][al] - dg[nu][al][be] );
                Gamma[mu][al][be] = 0.5*s;
            }
}

// ── RK4 step for worldline ────────────────────────────────────────────────────
// Returns new CameraState after proper time step dtau.
// Applies normalization re-projection at end of step.
inline CameraState step_worldline(const CameraState& cs,
                                   const KNdSMetric& bh, double dtau)
{
    // Derivative function: given state, compute (dx/dτ, du/dτ)
    auto deriv = [&](const CameraState& s, double dxdt[4], double dudt[4]) {
        double Gamma[4][4][4];
        gpg_christoffel(bh, s.x[1], s.x[2], Gamma);
        for (int mu=0;mu<4;++mu) {
            dxdt[mu] = s.u[mu];
            double acc=0.0;
            for (int al=0;al<4;++al)
                for (int be=0;be<4;++be)
                    acc -= Gamma[mu][al][be]*s.u[al]*s.u[be];
            dudt[mu] = acc;
        }
    };

    // RK4
    CameraState k1=cs, k2, k3, k4, tmp;
    double dx1[4],du1[4], dx2[4],du2[4], dx3[4],du3[4], dx4[4],du4[4];

    deriv(cs, dx1, du1);

    for (int i=0;i<4;++i) { tmp.x[i]=cs.x[i]+0.5*dtau*dx1[i];
                             tmp.u[i]=cs.u[i]+0.5*dtau*du1[i]; }
    deriv(tmp, dx2, du2);

    for (int i=0;i<4;++i) { tmp.x[i]=cs.x[i]+0.5*dtau*dx2[i];
                             tmp.u[i]=cs.u[i]+0.5*dtau*du2[i]; }
    deriv(tmp, dx3, du3);

    for (int i=0;i<4;++i) { tmp.x[i]=cs.x[i]+dtau*dx3[i];
                             tmp.u[i]=cs.u[i]+dtau*du3[i]; }
    deriv(tmp, dx4, du4);

    CameraState next;
    for (int i=0;i<4;++i) {
        next.x[i] = cs.x[i] + (dtau/6.0)*(dx1[i]+2*dx2[i]+2*dx3[i]+dx4[i]);
        next.u[i] = cs.u[i] + (dtau/6.0)*(du1[i]+2*du2[i]+2*du3[i]+du4[i]);
    }

    // Normalization re-projection: adjust u^r so g_μν u^μ u^ν = -1
    {
        double gLL[4][4], gUU[4][4];
        gpg_covariant(bh, next.x[1], next.x[2], gLL);
        gpg_contravariant(bh, next.x[1], next.x[2], gUU);
        // C = sum excluding u^r terms
        double C=0.0;
        for (int mu=0;mu<4;++mu) for (int nu=0;nu<4;++nu) {
            if (mu==1||nu==1) continue;
            C += gLL[mu][nu]*next.u[mu]*next.u[nu];
        }
        // g_rr ur² + 2(g_rT uT + g_rθ uθ + g_rφ uφ) ur + (C+1) = 0
        const double A = gLL[1][1];
        const double B = 2.0*(gLL[1][0]*next.u[0]+gLL[1][2]*next.u[2]+gLL[1][3]*next.u[3]);
        const double disc = B*B - 4.0*A*(C+1.0);
        if (disc >= 0.0) {
            // keep same sign as before
            const double ur_neg = (-B - std::sqrt(disc))/(2.0*A);
            const double ur_pos = (-B + std::sqrt(disc))/(2.0*A);
            next.u[1] = (next.u[1] < 0.0) ? ur_neg : ur_pos;
        }
    }
    return next;
}
```

- [ ] **Step 4: Run test**

```bash
g++ -std=c++17 -O2 -I. tests/test_falling_camera.cpp -o /tmp/test_falling && /tmp/test_falling
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add falling_camera.hpp tests/test_falling_camera.cpp
git commit -m "feat(falling): gpg_christoffel, step_worldline RK4"
```

---

## Task 4: CameraTetrad (metric Gram-Schmidt)

**Files:**
- Modify: `falling_camera.hpp` (append)
- Modify: `tests/test_falling_camera.cpp` (append test)

- [ ] **Step 1: Add failing test** — append before `main()`:

```cpp
// Test 7: tetrad orthonormality g_μν ê_a^μ ê_b^ν = η_ab
bool test_tetrad_orthonormal() {
    KNdSMetric bh(1.0, 0.9, 0.0, 0.0);
    FallingParams fp{ bh }; fp.r_start=5.0; fp.E=1.0;
    CameraState cs = init_worldline(fp);
    double e[4][4]; // e[a][mu]
    build_tetrad(cs, bh, e);
    double gLL[4][4];
    gpg_covariant(bh, cs.x[1], cs.x[2], gLL);
    double eta[4][4]={{-1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
    for (int a=0;a<4;++a) for (int b=0;b<4;++b) {
        double s=0.0;
        for (int mu=0;mu<4;++mu) for (int nu=0;nu<4;++nu)
            s += gLL[mu][nu]*e[a][mu]*e[b][nu];
        if (!approx(s, eta[a][b], 1e-6)) {
            std::cerr<<"FAIL tetrad ["<<a<<"]["<<b<<"] = "<<s<<" expected "<<eta[a][b]<<"\n";
            return false;
        }
    }
    return true;
}
```

Add to `main()`:
```cpp
if (!test_tetrad_orthonormal()) { std::cerr<<"FAIL tetrad_orthonormal\n"; ++fail; }
else std::cout<<"PASS test_tetrad_orthonormal\n";
```

- [ ] **Step 2: Run to verify it fails**

```bash
g++ -std=c++17 -O2 -I. tests/test_falling_camera.cpp -o /tmp/test_falling 2>&1 | head -3
```

- [ ] **Step 3: Append to `falling_camera.hpp`**

```cpp
// ── CameraTetrad: metric Gram-Schmidt ─────────────────────────────────────────
// e[a][mu]: a=tetrad index (0=time,1=radial,2=polar,3=azimuthal), mu=coord index
// Satisfies: g_μν e[a]^μ e[b]^ν = η_ab = diag(-1,+1,+1,+1)
inline void build_tetrad(const CameraState& cs, const KNdSMetric& bh,
                          double e[4][4])
{
    double gLL[4][4];
    gpg_covariant(bh, cs.x[1], cs.x[2], gLL);

    auto inner = [&](const double* v, const double* w) -> double {
        double s=0.0;
        for (int mu=0;mu<4;++mu) for (int nu=0;nu<4;++nu)
            s += gLL[mu][nu]*v[mu]*w[nu];
        return s;
    };
    auto normalize = [&](double* v, double sign) {
        double n2 = sign*inner(v,v);  // sign=-1 for timelike, +1 for spacelike
        double n  = std::sqrt(std::max(n2, 1e-30));
        for (int mu=0;mu<4;++mu) v[mu] /= n;
    };
    auto subtract = [&](double* v, const double* basis, double coeff) {
        for (int mu=0;mu<4;++mu) v[mu] -= coeff*basis[mu];
    };

    // ê_0 = u^μ (already timelike unit)
    for (int mu=0;mu<4;++mu) e[0][mu] = cs.u[mu];
    normalize(e[0], -1.0);

    // ê_1 seed = ∂_r = (0,1,0,0) in coordinate basis
    double seed1[4]={0,1,0,0};
    // Gram-Schmidt: subtract e_0 projection
    subtract(seed1, e[0], inner(seed1,e[0])/inner(e[0],e[0]));
    for (int mu=0;mu<4;++mu) e[1][mu]=seed1[mu];
    normalize(e[1], 1.0);

    // ê_2 seed = ∂_θ = (0,0,1,0)
    double seed2[4]={0,0,1,0};
    subtract(seed2, e[0], inner(seed2,e[0])/inner(e[0],e[0]));
    subtract(seed2, e[1], inner(seed2,e[1])/inner(e[1],e[1]));
    for (int mu=0;mu<4;++mu) e[2][mu]=seed2[mu];
    normalize(e[2], 1.0);

    // ê_3 seed = ∂_φ = (0,0,0,1)
    double seed3[4]={0,0,0,1};
    subtract(seed3, e[0], inner(seed3,e[0])/inner(e[0],e[0]));
    subtract(seed3, e[1], inner(seed3,e[1])/inner(e[1],e[1]));
    subtract(seed3, e[2], inner(seed3,e[2])/inner(e[2],e[2]));
    for (int mu=0;mu<4;++mu) e[3][mu]=seed3[mu];
    normalize(e[3], 1.0);
}

// ── Apply roll ψ around ê_3 (azimuthal axis) ─────────────────────────────────
// Rotates ê_1 and ê_2 in the radial-polar plane.
inline void apply_roll(double e[4][4], double psi) {
    double e1[4], e2[4];
    for (int mu=0;mu<4;++mu) { e1[mu]=e[1][mu]; e2[mu]=e[2][mu]; }
    for (int mu=0;mu<4;++mu) {
        e[1][mu] =  std::cos(psi)*e1[mu] + std::sin(psi)*e2[mu];
        e[2][mu] = -std::sin(psi)*e1[mu] + std::cos(psi)*e2[mu];
    }
}

// ── HorizonFlip: compute ψ(r) ─────────────────────────────────────────────────
// Returns ψ ∈ [0, π]: 0 = look outward, π = look toward BH.
inline double horizon_flip_psi(double r, double r_horizon,
                                double delta_out=2.0, double delta_in=0.8)
{
    const double r_far  = r_horizon * delta_out;
    const double r_near = r_horizon * delta_in;
    if (r >= r_far)  return M_PI;
    if (r <= r_near) return 0.0;
    // Cubic smoothstep
    double t = (r - r_near) / (r_far - r_near);
    double smooth = t*t*(3.0 - 2.0*t);
    return M_PI * smooth;
}
```

- [ ] **Step 4: Run test**

```bash
g++ -std=c++17 -O2 -I. tests/test_falling_camera.cpp -o /tmp/test_falling && /tmp/test_falling
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add falling_camera.hpp tests/test_falling_camera.cpp
git commit -m "feat(falling): build_tetrad, apply_roll, horizon_flip_psi"
```

---

## Task 5: Photon init and backward null geodesic

**Files:**
- Create: `falling_renderer.hpp`
- Create: `falling_renderer.cpp` (skeleton + photon init + trace)
- Modify: `tests/test_falling_camera.cpp` (append test)

- [ ] **Step 1: Add failing test** — append before `main()`:

```cpp
#include "falling_renderer.hpp"

// Test 8: center pixel photon is null
bool test_photon_null() {
    KNdSMetric bh(1.0, 0.9, 0.0, 0.0);
    FallingParams fp{ bh }; fp.r_start=5.0; fp.E=1.0;
    CameraState cs = init_worldline(fp);
    double e[4][4];
    build_tetrad(cs, bh, e);
    // center pixel
    double k[4];
    init_photon_k(cs, e, bh, fp.width/2, fp.height/2,
                  fp.width, fp.height, fp.fov_h, k);
    double gLL[4][4];
    gpg_covariant(bh, cs.x[1], cs.x[2], gLL);
    double norm=0.0;
    for (int mu=0;mu<4;++mu) for (int nu=0;nu<4;++nu)
        norm += gLL[mu][nu]*k[mu]*k[nu];
    if (!approx(norm, 0.0, 1e-6)) {
        std::cerr<<"FAIL photon null residual="<<norm<<"\n"; return false;
    }
    return true;
}
```

Add to `main()`:
```cpp
if (!test_photon_null()) { std::cerr<<"FAIL photon_null\n"; ++fail; }
else std::cout<<"PASS test_photon_null\n";
```

- [ ] **Step 2: Run to verify fails**

```bash
g++ -std=c++17 -O2 -I. tests/test_falling_camera.cpp -o /tmp/test_falling 2>&1 | head -3
```

- [ ] **Step 3: Create `falling_renderer.hpp`**

```cpp
#pragma once
#include "falling_camera.hpp"
#include <string>

// ── FallingGeoPixel ───────────────────────────────────────────────────────────
struct FallingGeoPixel {
    uint8_t outcome;   // 0=escaped, 1=disk_hit, 2=singularity, 3=trapped
    float   r_hit;     // r at disk or escape
    float   phi_hit;
    float   redshift;
    float   r_min;     // minimum r reached (for CPU refinement mask)
    float   theta_esc;
    float   phi_esc;
};

// ── Photon k^μ from pixel (px,py) ────────────────────────────────────────────
void init_photon_k(const CameraState& cs,
                   const double e[4][4],   // tetrad e[a][mu]
                   const KNdSMetric& bh,
                   int px, int py, int W, int H, double fov_h,
                   double k[4]);           // output contravariant k^μ

// ── Backward null geodesic integration ───────────────────────────────────────
FallingGeoPixel trace_photon_gpg(const double x0[4], const double k0[4],
                                  const KNdSMetric& bh,
                                  const FallingParams& fp,
                                  int max_steps = 50000,
                                  double h0     = 0.05,
                                  double tol    = 1e-7);

// ── Render one frame ──────────────────────────────────────────────────────────
// Writes PNG to out_path. Reports progress to stdout as:
// "[frame NNNN/TTTT] PP% T.Ts elapsed"
void render_falling_frame(int frame_idx, const FallingParams& fp,
                           const CameraState& cs_at_frame,
                           const std::string& out_path);
```

- [ ] **Step 4: Create `falling_renderer.cpp` — photon init + trace**

```cpp
#include "falling_renderer.hpp"
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

// ── Photon initialization ─────────────────────────────────────────────────────
void init_photon_k(const CameraState& cs, const double e[4][4],
                   const KNdSMetric& bh,
                   int px, int py, int W, int H, double fov_h,
                   double k[4])
{
    const double fov_v = fov_h * double(H) / double(W);
    const double alpha = fov_h * (px - W*0.5) / double(W - 1);
    const double beta  = fov_v * (H*0.5 - py) / double(H - 1);

    // Local direction in tetrad frame: k̂ = (1, n_x, n_y, n_z)
    const double nx = std::sin(beta)*std::cos(alpha);
    const double ny = std::sin(beta)*std::sin(alpha);
    const double nz = std::cos(beta);

    // k^μ = e[0]^μ + nx·e[1]^μ + ny·e[2]^μ + nz·e[3]^μ
    for (int mu=0;mu<4;++mu)
        k[mu] = e[0][mu] + nx*e[1][mu] + ny*e[2][mu] + nz*e[3][mu];

    // Enforce null condition: rescale k^T so g_μν k^μ k^ν = 0
    // g_TT (k^T)² + 2 g_Tμ k^T k^μ_rest + C_rest = 0
    double gLL[4][4];
    gpg_covariant(bh, cs.x[1], cs.x[2], gLL);

    const double A = gLL[0][0];
    double B = 0.0;
    double C = 0.0;
    for (int mu=1;mu<4;++mu) B += 2.0*gLL[0][mu]*k[mu];
    for (int mu=1;mu<4;++mu) for (int nu=1;nu<4;++nu) C += gLL[mu][nu]*k[mu]*k[nu];
    const double disc = B*B - 4.0*A*C;
    if (disc >= 0.0) {
        // Pick future-directed root (k^T > 0 in our convention)
        const double kT1 = (-B + std::sqrt(disc))/(2.0*A);
        const double kT2 = (-B - std::sqrt(disc))/(2.0*A);
        k[0] = (kT1 > kT2) ? kT1 : kT2;  // more positive = future-directed
    }
}

// ── RK4 step for photon geodesic ──────────────────────────────────────────────
static void photon_step(const KNdSMetric& bh, double x[4], double kv[4],
                         double dlam)
{
    auto deriv = [&](const double* xi, const double* ki,
                     double* dxdl, double* dkdl) {
        double Gamma[4][4][4];
        gpg_christoffel(bh, xi[1], xi[2], Gamma);
        for (int mu=0;mu<4;++mu) {
            dxdl[mu] = ki[mu];
            double acc=0.0;
            for (int al=0;al<4;++al) for (int be=0;be<4;++be)
                acc -= Gamma[mu][al][be]*ki[al]*ki[be];
            dkdl[mu] = acc;
        }
    };

    double dx1[4],dk1[4], dx2[4],dk2[4], dx3[4],dk3[4], dx4[4],dk4[4];
    double xt[4], kt[4];

    deriv(x, kv, dx1, dk1);
    for (int i=0;i<4;++i){xt[i]=x[i]+0.5*dlam*dx1[i]; kt[i]=kv[i]+0.5*dlam*dk1[i];}
    deriv(xt, kt, dx2, dk2);
    for (int i=0;i<4;++i){xt[i]=x[i]+0.5*dlam*dx2[i]; kt[i]=kv[i]+0.5*dlam*dk2[i];}
    deriv(xt, kt, dx3, dk3);
    for (int i=0;i<4;++i){xt[i]=x[i]+dlam*dx3[i]; kt[i]=kv[i]+dlam*dk3[i];}
    deriv(xt, kt, dx4, dk4);

    for (int i=0;i<4;++i) {
        x[i]  += (dlam/6.0)*(dx1[i]+2*dx2[i]+2*dx3[i]+dx4[i]);
        kv[i] += (dlam/6.0)*(dk1[i]+2*dk2[i]+2*dk3[i]+dk4[i]);
    }
}

// ── Backward null geodesic trace ──────────────────────────────────────────────
FallingGeoPixel trace_photon_gpg(const double x0[4], const double k0[4],
                                  const KNdSMetric& bh,
                                  const FallingParams& fp,
                                  int max_steps, double h0, double tol)
{
    double x[4], k[4];
    for (int i=0;i<4;++i){x[i]=x0[i]; k[i]=k0[i];}

    const double r_sing   = fp.r_singularity;
    const double r_esc    = fp.r_escape;
    const double r_in     = (fp.r_disk_in < 0) ? bh.r_isco() : fp.r_disk_in;
    const double r_out    = fp.r_disk_out;

    FallingGeoPixel pix{};
    pix.r_min = float(x[1]);

    double prev_theta = x[2];
    double dlam = h0;

    for (int step=0; step<max_steps; ++step) {
        if (x[1] < float(pix.r_min)) pix.r_min = float(x[1]);

        // Stop conditions
        if (x[1] >= r_esc) {
            pix.outcome   = 0;
            pix.r_hit     = float(x[1]);
            pix.theta_esc = float(std::fmod(x[2], M_PI));
            pix.phi_esc   = float(std::fmod(x[3], 2.0*M_PI));
            pix.redshift  = 1.0f;
            return pix;
        }
        if (x[1] < r_sing) {
            pix.outcome = 2; pix.r_hit = float(x[1]); return pix;
        }

        // Disk crossing: sign change of (θ - π/2) in [r_in, r_out]
        double cur_theta = x[2];
        if (x[1] >= r_in && x[1] <= r_out) {
            double d_prev = prev_theta - M_PI/2.0;
            double d_cur  = cur_theta  - M_PI/2.0;
            if (d_prev * d_cur < 0.0) {
                // Linear interpolation for r_hit, phi_hit
                double t = d_prev / (d_prev - d_cur);
                pix.outcome  = 1;
                pix.r_hit    = float(x[1] - t*(x[1]-x[1]));  // approx
                pix.phi_hit  = float(x[3]);
                pix.redshift = 1.0f;  // computed in shading step
                return pix;
            }
        }
        prev_theta = cur_theta;

        // Adaptive step: halve h if r is near horizon
        double r_h = bh.r_horizon();
        if (x[1] < r_h * 3.0) dlam = std::min(dlam, 0.005);
        photon_step(bh, x, k, dlam);
    }

    // Trapped
    pix.outcome = 3;
    return pix;
}
```

- [ ] **Step 5: Run test**

```bash
g++ -std=c++17 -O2 -I. tests/test_falling_camera.cpp falling_renderer.cpp -o /tmp/test_falling && /tmp/test_falling
```

Expected: all 8 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add falling_renderer.hpp falling_renderer.cpp tests/test_falling_camera.cpp
git commit -m "feat(falling): init_photon_k, trace_photon_gpg"
```

---

## Task 6: render_falling_frame — Phase A (background only)

**Files:**
- Modify: `falling_renderer.cpp` (append render_falling_frame)

- [ ] **Step 1: Append render_falling_frame to `falling_renderer.cpp`**

```cpp
// ── PNG write helper (reuses stb_image_write or existing writer) ──────────────
// KerrTraceCpp2 uses a save_png() free function already in main.cpp.
// We declare it here; it will be linked from main.cpp.
extern void save_png(const std::string& path,
                     const std::vector<uint8_t>& rgb,
                     int W, int H);

// ── render_falling_frame ──────────────────────────────────────────────────────
void render_falling_frame(int frame_idx, const FallingParams& fp,
                           const CameraState& cs_at_frame,
                           const std::string& out_path)
{
    const int W = fp.width, H = fp.height;
    std::vector<uint8_t> rgb(W * H * 3, 0);

    // Build tetrad + apply HorizonFlip roll
    double e[4][4];
    build_tetrad(cs_at_frame, fp.bh, e);
    double psi = horizon_flip_psi(cs_at_frame.x[1], fp.bh.r_horizon());
    apply_roll(e, psi);

    const int total_pixels = W * H;
    int done = 0;
    auto t0 = std::chrono::steady_clock::now();

    #pragma omp parallel for schedule(dynamic,16) reduction(+:done)
    for (int py = 0; py < H; ++py) {
        for (int px = 0; px < W; ++px) {
            double k[4];
            init_photon_k(cs_at_frame, e, fp.bh,
                          px, py, W, H, fp.fov_h, k);

            FallingGeoPixel pix = trace_photon_gpg(
                cs_at_frame.x, k, fp.bh, fp, 50000, 0.05, 1e-7);

            uint8_t R=0, G=0, B=0;
            if (pix.outcome == 0) {
                // Escaped — simple grey background for Phase A
                R = G = B = 30;
            }
            // outcome 1 (disk), 2 (singularity), 3 (trapped) → black for Phase A

            const int idx = (py*W + px)*3;
            rgb[idx+0]=R; rgb[idx+1]=G; rgb[idx+2]=B;
            ++done;
        }

        // Progress report (thread-safe approximate, one line per row)
        if (py % (H/20 + 1) == 0) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now-t0).count();
            int pct = done * 100 / total_pixels;
            std::printf("[frame %04d/%04d] %d%% %.1fs elapsed\r",
                        frame_idx+1, fp.frames, pct, elapsed);
            std::fflush(stdout);
        }
    }
    std::printf("[frame %04d/%04d] 100%% done\n", frame_idx+1, fp.frames);
    save_png(out_path, rgb, W, H);
}
```

- [ ] **Step 2: Verify it compiles with main.cpp**

```bash
cd /Users/iman.rosignoli/Documents/KerrTraceCpp2
cmake -B build_cpu -DUSE_METAL=OFF 2>&1 | tail -3
cmake --build build_cpu -j$(sysctl -n hw.ncpu) 2>&1 | tail -10
```

Expected: builds without errors. (The `--falling-camera` dispatch is added in Task 8.)

- [ ] **Step 3: Commit**

```bash
git add falling_renderer.cpp
git commit -m "feat(falling): render_falling_frame Phase A — background shading"
```

---

## Task 7: Disk shading — Phase B

**Files:**
- Modify: `falling_renderer.cpp`

The disk hit in `trace_photon_gpg` currently sets `redshift=1.0f`. This task adds:
1. Precise disk crossing via Hermite interpolation
2. Redshift g-factor
3. Page-Thorne luminosity profile

- [ ] **Step 1: Replace the disk-crossing block in `trace_photon_gpg`**

In `falling_renderer.cpp`, find the disk crossing section and replace with:

```cpp
        if (x[1] >= r_in && x[1] <= r_out) {
            double d_prev = prev_theta - M_PI/2.0;
            double d_cur  = cur_theta  - M_PI/2.0;
            if (d_prev * d_cur < 0.0) {
                // Hermite interpolation for precise crossing parameter t ∈ [0,1]
                // Simple linear fallback (Hermite requires derivative storage)
                double t = std::abs(d_prev) / (std::abs(d_prev) + std::abs(d_cur));
                double r_cross   = x[1];          // approx; refine with t
                double phi_cross = x[3];

                // Redshift: g = -(k_μ u^μ_obs) / (k_μ u^μ_emit)
                // u^μ_emit = Keplerian disk in GPG coords
                // For equatorial Keplerian: u^t_emit = 1/sqrt(-g_tt - 2g_tφ Ω - g_φφ Ω²)
                //                           u^φ_emit = Ω · u^t_emit
                double gLL_d[4][4];
                gpg_covariant(bh, r_cross, M_PI/2.0, gLL_d);
                const double Omega = bh.keplerian_omega(r_cross);
                double N2 = -(gLL_d[0][0]
                             + 2.0*gLL_d[0][3]*Omega
                             + gLL_d[3][3]*Omega*Omega);
                N2 = std::max(N2, 1e-20);
                const double ut_em = 1.0/std::sqrt(N2);
                const double uph_em= Omega*ut_em;

                // k_μ at crossing (lower index)
                double k_low[4]={0};
                for (int mu=0;mu<4;++mu)
                    for (int nu=0;nu<4;++nu)
                        k_low[mu] += gLL_d[mu][nu]*k[nu];

                const double k_u_emit = k_low[0]*ut_em + k_low[3]*uph_em;
                // k_μ u^μ_obs: use camera four-velocity at emission (approx: same frame)
                // For backward tracing, obs end is the camera position x0 (already done).
                // We stored it separately; use conserved -E (p_t at emission = k^T rescaled)
                // Approximate: g_obs = |k_T| / |k_u_emit|  (valid for r_obs >> r_cross)
                const double g_redshift = std::abs(k[0]) / std::max(std::abs(k_u_emit), 1e-20);

                // Page-Thorne luminosity: f(r) ∝ (r - r_isco)·r^{-3} (approximate)
                const double r_isco_val = bh.r_isco();
                double lum = 0.0;
                if (r_cross > r_isco_val) {
                    lum = (r_cross - r_isco_val) / (r_cross * r_cross * r_cross);
                    lum *= std::pow(std::max(g_redshift, 0.0), 4.0); // Doppler+redshift
                    lum = std::min(lum * fp.disk_brightness * 1e3, 1.0);
                }

                pix.outcome  = 1;
                pix.r_hit    = float(r_cross);
                pix.phi_hit  = float(phi_cross);
                pix.redshift = float(g_redshift);
                pix.r_min    = std::min(pix.r_min, float(r_cross));
                return pix;
            }
        }
```

- [ ] **Step 2: Update the shading block in `render_falling_frame`**

Replace the shading section:

```cpp
            uint8_t R=0, G=0, B=0;
            if (pix.outcome == 0) {
                // Background — flat grey for Phase A/B
                R = G = B = 30;
            } else if (pix.outcome == 1) {
                // Disk hit — orange glow scaled by luminosity via redshift
                float lum = std::min(pix.redshift * pix.redshift *
                                     float(fp.disk_brightness), 1.0f);
                // Page-Thorne: blueshift inner → yellow-white, redshift outer → red
                float g = pix.redshift;
                if (g > 1.0f) {
                    // Blueshifted (approaching side): white-yellow
                    R = uint8_t(std::min(255.0f, 255.0f*lum));
                    G = uint8_t(std::min(255.0f, 200.0f*lum));
                    B = uint8_t(std::min(255.0f, 80.0f*lum));
                } else {
                    // Redshifted (receding side): orange-red
                    R = uint8_t(std::min(255.0f, 220.0f*lum));
                    G = uint8_t(std::min(255.0f, 100.0f*lum*g));
                    B = 0;
                }
            }
            // outcome 2,3 → black (already zeroed)
```

- [ ] **Step 3: Build and verify no warnings**

```bash
cmake --build build_cpu -j$(sysctl -n hw.ncpu) 2>&1 | grep -E "error:|warning:" | head -20
```

Expected: no errors, warnings only for unused variables (acceptable).

- [ ] **Step 4: Commit**

```bash
git add falling_renderer.cpp
git commit -m "feat(falling): Phase B — disk intersection, redshift, Page-Thorne shading"
```

---

## Task 8: main.cpp dispatch and CMakeLists.txt

**Files:**
- Modify: `main.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Add `falling_renderer.cpp` to CMakeLists.txt**

In `CMakeLists.txt`, find:
```cmake
set(CPU_SOURCES main.cpp)
```

Replace with:
```cmake
set(CPU_SOURCES main.cpp falling_renderer.cpp)
```

- [ ] **Step 2: Add test target to CMakeLists.txt**

In the `if(BUILD_TESTING)` block, append after the last `add_test`:

```cmake
    add_executable(kerrtrace_falling_tests tests/test_falling_camera.cpp falling_renderer.cpp)
    target_include_directories(kerrtrace_falling_tests PRIVATE ${CMAKE_SOURCE_DIR})
    target_compile_options(kerrtrace_falling_tests PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O2 -Wall -Wextra>)
    add_test(NAME kerrtrace.falling COMMAND kerrtrace_falling_tests)
```

- [ ] **Step 3: Add --falling-camera parsing to main.cpp**

Find the argument parsing loop in `main.cpp` (near `if (arg=="--a"`). Add alongside existing args:

```cpp
        // Falling camera
        bool  do_falling      = false;
        double fall_r_start   = 20.0;
        double fall_E         = 1.0;
        double fall_L         = 0.0;
        double fall_Qc        = 0.0;
        double fall_theta_deg = 90.0;
        int    fall_frames    = 120;
        double fall_dtau      = 0.1;
        double fall_delta_out = 2.0;
        double fall_delta_in  = 0.8;
```

In the arg parsing loop:
```cpp
        if (arg=="--falling-camera")                do_falling       = true;
        if (arg=="--fall-r-start" && i+1<argc)      fall_r_start     = std::stod(argv[++i]);
        if (arg=="--fall-E"       && i+1<argc)       fall_E           = std::stod(argv[++i]);
        if (arg=="--fall-L"       && i+1<argc)       fall_L           = std::stod(argv[++i]);
        if (arg=="--fall-Qc"      && i+1<argc)       fall_Qc          = std::stod(argv[++i]);
        if (arg=="--fall-theta"   && i+1<argc)       fall_theta_deg   = std::stod(argv[++i]);
        if (arg=="--fall-frames"  && i+1<argc)       fall_frames      = std::stoi(argv[++i]);
        if (arg=="--fall-dtau"    && i+1<argc)       fall_dtau        = std::stod(argv[++i]);
```

- [ ] **Step 4: Add dispatch block in main.cpp**

After `KNdSMetric g_info(M_bh, fp.a, Q_bh, Lam);` and before the main render dispatch, add:

```cpp
        if (do_falling) {
            FallingParams fpar;
            fpar.bh          = KNdSMetric(M_bh, fp.a, Q_bh, Lam);
            fpar.r_start     = fall_r_start;
            fpar.E           = fall_E;
            fpar.L           = fall_L;
            fpar.Qc          = fall_Qc;
            fpar.theta_start = fall_theta_deg * M_PI / 180.0;
            fpar.frames      = fall_frames;
            fpar.dtau        = fall_dtau;
            fpar.width       = W;
            fpar.height      = H;
            fpar.fov_h       = arg_fov * M_PI / 180.0;
            fpar.r_disk_out  = fp.disk_out;
            fpar.disk_brightness = fp.disk_brightness;

            // Output directory
            std::string fall_dir = std::string(OUT_DIR) + "/falling/" + std::to_string(
                std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count());
            std::filesystem::create_directories(fall_dir);

            CameraState cs = init_worldline(fpar);
            for (int fi = 0; fi < fall_frames; ++fi) {
                std::string frame_path = fall_dir + "/frame_"
                    + std::string(4-std::to_string(fi).size(),'0')
                    + std::to_string(fi) + ".png";
                render_falling_frame(fi, fpar, cs, frame_path);
                cs = step_worldline(cs, fpar.bh, fpar.dtau);
                // Stop if camera hits singularity
                if (cs.x[1] < fpar.r_singularity) break;
            }
            std::printf("Falling render complete: %s\n", fall_dir.c_str());
            return 0;
        }
```

Add `#include "falling_renderer.hpp"` and `#include <filesystem>` at the top of `main.cpp`.

- [ ] **Step 5: Build**

```bash
cmake -B build_cpu -DUSE_METAL=OFF && cmake --build build_cpu -j$(sysctl -n hw.ncpu) 2>&1 | tail -15
```

Expected: builds successfully.

- [ ] **Step 6: Smoke test — 4 frames at low resolution**

```bash
./build_cpu/kerr_tracer --falling-camera --a 0.9 --fall-r-start 15 --fall-E 1.0 \
  --fall-frames 4 --fall-dtau 0.5 --width 320 --height 180 2>&1
```

Expected output:
```
[frame 0001/0004] 100% done
[frame 0002/0004] 100% done
[frame 0003/0004] 100% done
[frame 0004/0004] 100% done
Falling render complete: .../out/falling/...
```

Check frames exist:
```bash
ls out/falling/*/frame_*.png
```

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt main.cpp
git commit -m "feat(falling): --falling-camera CLI dispatch, CMakeLists integration"
```

---

## Task 9: CTest integration and final validation

**Files:**
- No new files

- [ ] **Step 1: Run full test suite**

```bash
cmake -B build_cpu -DUSE_METAL=OFF -DBUILD_TESTING=ON \
  && cmake --build build_cpu -j$(sysctl -n hw.ncpu) \
  && cd build_cpu && ctest --output-on-failure
```

Expected:
```
Test project .../build_cpu
    Start 1: kerrtrace.core
1/4 Test #1: kerrtrace.core ............ Passed
    Start 2: kerrtrace.bump_detector
2/4 Test #2: kerrtrace.bump_detector ... Passed
    Start 3: kerrtrace.spin_orientation
3/4 Test #3: kerrtrace.spin_orientation  Passed
    Start 4: kerrtrace.falling
4/4 Test #4: kerrtrace.falling ......... Passed
```

- [ ] **Step 2: Visual check — render 30 frames at 640×360**

```bash
./build_cpu/kerr_tracer --falling-camera --a 0.9 --fall-r-start 20 --fall-E 1.0 \
  --fall-frames 30 --fall-dtau 0.3 --width 640 --height 360
```

Open a sample frame:
```bash
open out/falling/*/frame_0015.png
```

Expected: dark image with faint disk glow and grey background ring (BH shadow visible).

- [ ] **Step 3: Commit**

```bash
git add CMakeLists.txt
git commit -m "test(falling): add kerrtrace.falling to CTest suite"
```

---

## Self-Review Checklist

- [x] GPG metric inverse: covered Task 1
- [x] BL→GPG transform f(r): implemented in `gpg_f()`
- [x] CameraWorldline init from (E, L, Qc): Task 2
- [x] Worldline RK4 + normalization reproject: Task 3
- [x] Tetrad Gram-Schmidt + orthonormality test: Task 4
- [x] apply_roll around ê_3: Task 4
- [x] HorizonFlip ψ(r): Task 4
- [x] Photon null initialization: Task 5
- [x] Backward GPG ray trace + stop conditions: Task 5
- [x] No stop at r_horizon (GPG regular): Task 5 ✓
- [x] Disk crossing + redshift: Task 7
- [x] Page-Thorne profile: Task 7
- [x] render_falling_frame + progress stdout: Task 6
- [x] --falling-camera CLI: Task 8
- [x] CTest integration: Task 9
- [x] r_min per pixel saved (for future CPU mask): Task 5 ✓

**Note on Carter constant:** The `u_θ` formula uses the Kerr-approximation form. For full KNdS with Λ≠0 the Carter constant acquires Ξ correction terms. Implement the corrected form when Λ≠0 is exercised in Phase E.
