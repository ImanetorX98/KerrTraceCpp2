# KerrTraceCpp2 — Claude Instructions & Development Plan

## Project Goal
Relativistic ray-tracer for Kerr-Newman-de Sitter black holes.
Physically correct null-geodesic integration, multi-backend GPU/CPU,
ray-bundle antialiasing.  Target: publication-quality images like DNGR/Interstellar.

---

## Codebase Map

| File | Role |
|------|------|
| `knds_metric.hpp` | KNdS metric, BL + KS Cartesian charts, transformations |
| `geodesic.hpp` | Hamiltonian RHS, adaptive RK4 (step-doubling) |
| `camera.hpp` | BL camera, pixel→(α,β)→initial state, `angle_ray()` |
| `ray_bundle.hpp` | Jacobi-field bundle: Hessian(H), variational equations, magnification |
| `main.cpp` | Render loop, backend dispatch, PPM output |
| `gpu/metal/tracer.metal` | MSL compute shader (one thread = one pixel) |
| `gpu/metal/metal_renderer.hpp/.mm` | Objective-C++ Metal bridge |
| `gpu/cuda/tracer.cu/.cuh` | CUDA kernel + host launcher |
| `CMakeLists.txt` | Build: `-DUSE_METAL=ON`, `-DUSE_CUDA=ON`, default=CPU |

---

## Build Commands

```bash
# CPU (default)
cmake -B build && cmake --build build -j$(nproc)
./build/kerr_tracer           # single-ray
./build/kerr_tracer --bundles # ray-bundle mode

# Metal GPU (macOS)
cmake -B build -DUSE_METAL=ON && cmake --build build -j$(nproc)

# CUDA GPU (Linux/Windows, requires nvcc)
cmake -B build -DUSE_CUDA=ON && cmake --build build -j$(nproc)
```

---

## Physics Summary

### Metric: Kerr-Newman-de Sitter
Parameters `(M, a, Q, Λ)` in `KNdSMetric`.  Set to zero to reduce:
- `Q=0, Λ=0` → Kerr
- `a=0, Λ=0` → Reissner-Nordström
- `a=0, Q=0` → Schwarzschild-de Sitter
- `a=0, Q=0, Λ=0` → Schwarzschild

Key functions in BL coordinates:
- `Σ = r² + a²cos²θ`
- `Δ_r = (r²+a²)(1−Λr²/3) − 2Mr + Q²`
- `Δ_θ = 1 + (Λa²/3)cos²θ`
- `Ξ = 1 + Λa²/3`

### KS Cartesian chart (Λ=0)
`g_ab = η_ab + H·l_a·l_b` where:
- `H = (2Mr−Q²)/ρ²`
- `l_a^{in} = (1, (rX+aY)/(r²+a²), (rY−aX)/(r²+a²), Z/r)`  [ingoing]
- `l_a^{out} = (−1, (rX−aY)/(r²+a²), (rY+aX)/(r²+a²), Z/r)` [outgoing]
- `r` implicit from `r⁴ − (X²+Y²+Z²−a²)r² − a²Z² = 0`

### Geodesic integration
Hamiltonian `H = ½g^μν p_μ p_ν = 0` (null).
Conserved: `p_t = −E`, `p_φ = L` (Killing symmetries).
Reduced state: `(r, θ, p_r, p_θ)`.
RHS via numerical `∂H/∂r`, `∂H/∂θ`.
Two adaptive integrators (`enum class Integrator` in `geodesic.hpp`):

| Flag | Enum | Method | RHS evals/step | Notes |
|------|------|--------|----------------|-------|
| *(default)* | `RK4_DOUBLING` | RK4 + Richardson step-doubling | 12 | Simple, robust |
| `--dopri5`  | `DOPRI5` | Dormand-Prince RK45 | 6 (FSAL→5 net) | `ode45` standard |

DOPRI5 Butcher tableau: 6 stages, embedded 4th/5th order pair, error = `‖y5−y4‖`.
FSAL: `k7 = f(y5)` reused as `k1` of the next step → 5 net evaluations per accepted step.
RK4-doubling: 1 full step + 2 half steps, Richardson factor = 2⁴−1 = 15.

### Ray bundles
Variational equations `d(δz)/dλ = M(z)·δz` alongside main geodesic.
`M = J_s · Hess(H)`, Hessian by central finite differences.
Jacobi matrix `W[:,0]` = deviation in α, `W[:,1]` = deviation in β.
`|det J|` at disk crossing → magnification → flux correction.
Reference: James et al. (2015) CQG 32 065001.

### Disk redshift
Full formula (Bardeen 1972):
`g = ν_obs/ν_emit = √(−g_tt−2g_tφΩ−g_φφΩ²) / (1−Ω·b)`
where `b = p_φ/(−p_t)` and `Ω_K = √M/(r^{3/2}+a√M)`.

---

## Development Roadmap

### Phase 1 — Core (done)
- [x] KNdS metric (BL + KS Cartesian)
- [x] Adaptive RK4 Hamiltonian integrator
- [x] Camera, image plane, null-condition enforcement
- [x] Thin-disk renderer with relativistic redshift
- [x] CPU/OpenMP backend

### Phase 2 — GPU & quality (in progress)
- [x] Metal compute shader (macOS)
- [x] CUDA kernel (Linux/Windows)
- [x] Ray-bundle Jacobi-field renderer

### Phase 3 — Accuracy & features (TODO)
- [ ] Chart switching BL↔KS when `Δ_r < ε` (Arcmancer-style)
- [ ] Carter constant `Q_c` for exact orbit classification
- [ ] Analytical sub-steps via elliptic integrals (Gralla & Lupsasca 2019)
- [ ] Thick disk / GRRMHD texture maps
- [ ] Photon ring detection and sub-ring counting
- [ ] Polarisation via parallel-transported polarisation vector
- [ ] Spectral rendering (multi-band, synchrotron emission model)

### Phase 4 — Performance (TODO)
- [ ] CUDA ray-bundle kernel (Jacobi field on GPU)
- [ ] BVH-like adaptive sampling density
- [ ] Semi-analytic fast path for pure Kerr (elliptic integrals)
- [ ] Tile-based rendering for very large images

---

## Key References (in `sources/`)
- James et al. (2015) — DNGR / Interstellar technique, ray bundles
- Gralla & Lupsasca (2019) — Analytical Kerr null geodesics
- Pihajoki et al. (2018) — Arcmancer: multi-chart library
- Luminet (2019) — History of BH imaging
- Chan et al. (2013) — GRay: GPU ray-tracing

---

## Coding Conventions
- C++17, geometric units `G=c=1`.
- All angles in radians internally.
- BL coordinate order: `(t=0, r=1, θ=2, φ=3)`.
- KS Cartesian order: `(T=0, X=1, Y=2, Z=3)`.
- Metric signature: `(−+++)`.
- `pt < 0` always (future-directed, `E = −p_t > 0`).
- Headers are self-contained; no `.cpp` files for metric/geodesic.
- GPU code mirrors CPU math exactly (same formulas, different types).
- Never use `std::clamp` for C++14 compatibility in GPU ports.

---

## Known Issues & Ongoing Work

### elliptic-closed solver — Region III bug (2026-04-26)

**Status**: Root cause fully identified. Fallback guard in place; ~53% fallback rate at
default a=0.998 (near-extremal). Correct images produced; performance is suboptimal.

**Symptom (original)**: enlarged black shadow, secondary image missing.
**Fix applied (2026-04-24)**: guard `if (r_now < mp.r1) return fallback` rejects hits in
the inner inaccessible region. This fixed the visual corruption.

**Remaining efficiency problem**: ~23 000 rays per 1920×1080 frame hit
`r_now < rh_cut` (rh_cut = 1.03·r₊) and fall back to numerical.

**Root cause of the 23k fallback rays (diagnosed 2026-04-26)**:
These are **Region III** photons (2 real + 2 complex roots, both real roots inside r₊).
The GL B75 radial formula for Region III computes r on the **hypothetical post-bounce
leg**: in BL/GL coordinates the inner turning point r_hi is inside r₊, so the formula
treats the photon as bouncing at r_hi and returning outward.  In reality these photons
cross θ=π/2 **before reaching the horizon**, giving a genuine disk hit at r ≈ 2–2.4M.
The GL formula evaluates r at τ_first on the (unphysical) outgoing leg, returning
r ≈ 1.09M (sub-horizon) — completely wrong.

**Confirmed**: diagnostic run showed 100% (5001/5001) of sub-horizon Region III rays
give DISK_HIT (not HORIZON) under both BL and KS numerical tracers.

**Guard in place**: `if (r_now < rh_cut) return fallback_trace(DIRECT_RADIAL_INVALID)`
(lines ~1263 in `trace_single_elliptic_closed`) with explanatory comment.  This is
correct and safe; do not replace it with a direct HORIZON return.

**Correct long-term fix**:
For Region III where r_hi < r₊, the GL B75 formula is only reliable for r > r_hi on
the *ingoing* leg.  Properly handling these rays requires either:
(a) Detecting when τ_first corresponds to the outgoing leg and computing τ_first for
    the actual ingoing disk crossing instead, or
(b) Switching to direct numerical integration for all Region III photons.
Option (b) is already in effect via the fallback guard.

**Standard renders confirmed correct** at both a=0.5 and a=0.998 (near-extremal).

### Build notes (2026-04-24)
- CPU binary: `cmake -B build_cpu -DUSE_METAL=OFF && cmake --build build_cpu`
- Metal binary: `cmake -B build -DUSE_METAL=ON && cmake --build build`
- tracer.metal is loaded at runtime from `build/tracer.metal` (copied from source).
  Changes to `gpu/metal/tracer.metal` require rebuild to copy to build dir.
- Metal elliptic-closed: `can_separable_kerr` in shader checks Q≈0, Λ≈0 (NOT chart),
  so elliptic solver runs even in KS chart mode on GPU.
