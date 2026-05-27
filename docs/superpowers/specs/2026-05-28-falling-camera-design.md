# Falling Camera Mode — Design Spec
*Date: 2026-05-28*

## Overview

Simulate a camera freely falling into a KNdS rotating black hole. The camera
follows a timelike geodesic in generalized Painlevé-Gullstrand (GPG) coordinates
(Lin & Soo 2009, arXiv:0905.3244), emitting backward null geodesics from a local
orthonormal tetrad. Output: MP4 animation with live per-frame preview in the
frontend.

---

## Section 1: Architecture and Files

### New files

| File | Role |
|------|------|
| `falling_camera.hpp` | GPG metric g_μν, numerical Christoffel symbols, `CameraWorldline`, `CameraTetrad` |
| `falling_renderer.cpp/.hpp` | `render_falling_frame()`, frame loop over τ, backward ray integration, disk hit, shading |
| `gpu/metal/tracer_falling.metal` | Metal compute kernel for falling camera (float, far-field pixels) |

### Modified files

| File | Change |
|------|--------|
| `main.cpp` | Add `--falling-camera` dispatch branch calling `render_falling_frame()` |
| `server/index.js` | Add `/api/falling-render`, `/api/falling-frames/:id/:frame`, `DELETE /api/falling-render/:id` |
| `frontend/src/app/` | Add falling camera tab component |

### Per-pixel CPU/GPU split (two-pass per frame)

```
Pass 1 — Metal (float, all pixels):
  Integrate photon geodesic.
  Save r_min reached during integration per pixel.

Pass 2 — CPU (double, pixels where r_min < r_switch):
  Re-integrate with double precision.
  Overwrite Metal result.

r_switch = r_horizon × k     (k ≈ 3–5, configurable, default 3)
```

`r_min` stored as an extra float in the `GeoPixel` buffer. CPU pass reads the
mask and re-traces only pixels that need it — typically the shadow edge and
photon ring.

---

## Section 2: Worldline Physics and Tetrad

### GPG Coordinates (Lin & Soo 2009)

Coordinate system: `(T, r, θ, φ_P)`. Transformation from Boyer-Lindquist:

```
dT   = dt_BL + (R²/Ξ) · √(f²−Δ) / (Δ·f) · dr
dφ_P = dφ_BL + (a/Ξ)  · √(f²−Δ) / (Δ·f) · dr
r, θ unchanged
```

Recommended choice for f(r):
```
f(r) = √(R² + Q² + Λa⁴/3)          for Λ ≥ 0
f(r) = √(R²(1 − Λr²/3) + Q²)       for Λ < 0
```

Guarantees `f² > max(Δ, a²Ξ)` for all `(M, a, Q, Λ)`.

Key metric components (regular at Δ=0):
```
g_rr  =  ρ²/f²
g_Tr  =  √(f²−Δ) / (f·Ξ)
g_rφ  = −a·sin²θ·√(f²−Δ) / (f·Ξ)
```

Standard KNdS functions:
```
R²  = r² + a²
ρ²  = r² + a²cos²θ
Ξ   = 1 + Λa²/3
Ξ_θ = 1 + Λa²cos²θ/3
Δ   = R²(1 − Λr²/3) − 2Mr + Q²
```

### CameraWorldline

State vector: `(T, r, θ, φ_P, u^T, u^r, u^θ, u^φ)`.

Initial conditions from `(r_start, θ_start, E, L, Q_c)`:
```
u_T = −E                          (conserved, Killing vector ∂_T)
u_φ = +L                          (conserved, Killing vector ∂_φ)
u_θ from Q_c: Q_c = u_θ² + cos²θ·(a²·u_T² − L²/sin²θ)
u_r from normalization: g^μν u_μ u_ν = −1  →  solve quadratic for u^r
```

Integration per proper time step Δτ (RK4, double precision):
```
dx^μ/dτ = u^μ
du^μ/dτ = −Γ^μ_{αβ} u^α u^β    (numerical Christoffel from GPG g_μν)
```

Normalization check every step: if `|g_μν u^μ u^ν + 1| > tol`, re-project u^r.

### CameraTetrad

Local orthonormal frame `{ê_0, ê_1, ê_2, ê_3}` via metric Gram-Schmidt:

```
ê_0 = u^μ                         (four-velocity, timelike)
ê_1 = ∂_r direction               (radial outward, before orthonormalization)
ê_2 = ∂_θ direction               (polar)
ê_3 = completes right-hand frame  (azimuthal)
```

Each step uses the metric inner product `g(v,w) = g_μν v^μ w^ν`, not Euclidean.
Verify: `g_μν ê_a^μ ê_b^ν = η_ab` after construction.

### Camera roll — rotation around ê_3 (azimuthal axis)

Rotates `ê_1` (radial) and `ê_2` (polar) in the radial-polar plane:
```
ê_1' =  cos(ψ)·ê_1 + sin(ψ)·ê_2
ê_2' = −sin(ψ)·ê_1 + cos(ψ)·ê_2
```

Effect: the camera's forward direction cycles through:
- ψ=0°   → looks radially outward (BH behind)
- ψ=90°  → looks toward pole (BH to the side)
- ψ=180° → looks radially inward (BH in front)
- ψ=270° → looks toward opposite pole

With `ψ(τ) = ω_roll · τ` the camera tumbles continuously. `ê_0` and `ê_3`
remain fixed; only the viewing plane rotates.

### LookMode: HorizonFlip

Automatic ψ(r) driven by camera position relative to the horizon, avoiding
the singularity. Camera looks toward BH when outside, flips to look outward
after crossing:

```
r_far   = r_horizon + δ_out    (default: 2.0 × r_horizon)
r_near  = r_horizon − δ_in     (default: 0.8 × r_horizon)

ψ(r) = π · smoothstep(r_near, r_far, r)   [cubic easing]

r ≥ r_far  → ψ = π   (looking inward toward BH)
r ≤ r_near → ψ = 0   (looking outward, back through horizon)
transition → smooth cubic interpolation
```

What the camera sees:
| Phase | View |
|-------|------|
| Far from BH | BH at center, disk around it, stars behind |
| Approaching | BH fills frame, flip begins |
| At horizon | Outside universe compressed into luminous cone |
| Inside | Entire universe visible in bright circle, darkness around |

HorizonFlip and `ω_roll` are composable — tumble + flip can be combined.
`δ_out` and `δ_in` configurable from frontend as "flip zone width".

---

## Section 3: Backward Ray Tracing and Shading

### Photon initialization

For each pixel `(px, py)`, local null direction `k̂^a = (1, n_x, n_y, n_z)`:

```
α = fov_h · (px − W/2) / (W−1)
β = fov_v · (H/2 − py) / (H−1)

n_x = sin(β)·cos(α)     (ê_1' component, radial)
n_y = sin(β)·sin(α)     (ê_2' component, polar)
n_z = cos(β)            (ê_3  component, azimuthal)

k^μ = ê_0^μ + n_x·ê_1'^μ + n_y·ê_2'^μ + n_z·ê_3^μ
```

Verify `g_μν k^μ k^ν = 0`; if violated, rescale k^T to enforce null condition.

### Backward integration (null geodesic)

```
dx^μ/dλ =  k^μ
dk^μ/dλ = −Γ^μ_{αβ} k^α k^β     (numerical Christoffel, GPG)
```

Integrator: adaptive RK4, same scheme as `geodesic.hpp`.

Stop conditions:
| Condition | Outcome |
|-----------|---------|
| `r ≥ r_escape` | skybox / background |
| `r < r_singularity` (e.g. `r < 0.05M`) | black (physical singularity) |
| θ crosses π/2 with `r_in ≤ r ≤ r_out` | disk hit |
| steps > max_steps | black (trapped photon) |

No stop at `r = r_horizon` — GPG metric is regular there. The BH shadow
emerges naturally from the photon capture geometry. From inside the horizon,
outward-going photons cross back through and reach `r_escape` → skybox.

`r_min` saved per pixel for the CPU refinement mask (Pass 2).

### Disk intersection

Sign-change detection of `(θ − π/2)` between steps. Precise `(r_hit, φ_hit)`
via cubic Hermite interpolation + bisection — same logic as `geodesic.hpp`.

### Redshift

```
g = −(k_μ u^μ_obs) / (k_μ u^μ_emit)

u^μ_obs  = camera four-velocity in GPG at emission event (backward ray)
u^μ_emit = disk matter four-velocity (Keplerian Ω_K, expressed in GPG)

I_obs = g⁴ · I_emit     (bolometric)
```

### Shading

| Outcome | Color |
|---------|-------|
| Disk hit | Page-Thorne profile + redshift g + existing palette |
| Background | HDRI skybox lookup at (θ_esc, φ_esc) |
| Black hole shadow | black (natural, no explicit stop at horizon) |
| Singularity / trapped | black |

Output per frame: PNG → same colorizer pipeline as existing renders (`.kgeo`
temporary buffer).

---

## Section 4: Server, Frontend Tab, MP4 Output

### Server endpoints

```
POST   /api/falling-render         start job
GET    /api/falling-render/:id/status   job state
GET    /api/falling-frames/:id/:frame   live frame preview PNG
DELETE /api/falling-render/:id     cancel + cleanup
```

POST body:
```json
{
  "r_start": 20.0, "E": 1.0, "L": 0.0, "Qc": 0.0, "theta_start": 90.0,
  "a": 0.998, "Q": 0.0, "Lambda": 0.0, "M": 1.0,
  "frames": 120, "fps": 24,
  "width": 1280, "height": 720,
  "roll_rate": 0.0, "flip_delta_out": 2.0, "flip_delta_in": 0.8,
  "disk_palette": "interstellar", "disk_brightness": 1.0,
  "background": "milkyway.hdr",
  "r_switch_factor": 3.0
}
```

### Server job pipeline

```
1. Spawn: ./kerr_tracer --falling-camera [params]
2. Binary writes frame_0000.png … frame_NNNN.png
   to out/falling/<jobId>/
3. Per-frame stdout: "[frame 0012/0120] 47% 2.3s elapsed · ETA 2.6s"
4. Server parses → WebSocket {
     type: 'falling_progress',
     frame: 12, totalFrames: 120,
     framePct: 47, elapsed: 2.3, eta: 2.6,
     previewUrl: '/api/falling-frames/<id>/frame_0011.png'
   }
5. All frames done → ffmpeg -r fps -i frame_%04d.png -c:v libx264 output.mp4
6. WebSocket { type: 'falling_done', mp4_url: '...' }
```

### Output files

```
out/falling/<jobId>/
  frame_0000.png … frame_NNNN.png   (removed after ffmpeg)
  falling_<jobId>.mp4               (final output)
  metadata.json                     { params, r_horizon, tau_total, ... }
```

### Frontend — Left panel

```
┌─ Worldline ──────────────────────┐
│ r_start  [  20.0 ] M             │
│ E        [  1.00 ]               │
│ L        [  0.00 ] M             │
│ Q_c      [  0.00 ] M²            │
│ θ_start  [  90   ] °             │
├─ Metrica ────────────────────────┤
│ a, Q, Λ  (sync dai params globali)│
├─ Cinematica ─────────────────────┤
│ Roll rate    [ 0.0  ] rad/frame τ │
│ Flip zone out[ 2.0  ] × r_h      │
│ Flip zone in [ 0.8  ] × r_h      │
├─ Render ─────────────────────────┤
│ Frames  [ 120 ]  FPS  [ 24 ]     │
│ Width   [ 1280 ] Height [ 720 ]  │
│ CPU/GPU threshold [ 3.0 ] × r_h  │
├──────────────────────────────────┤
│ [    Avvia render caduta    ]    │
│ [    Annulla               ]    │
└──────────────────────────────────┘
```

### Frontend — Right panel (viewer)

Two progress bars:
```
Frame 12 / 120  ████████████░░░░░░░░░░  47%   ETA 2.6s
Job             ██░░░░░░░░░░░░░░░░░░░░  10%   ~4m 20s rimanenti
```

- Live preview: last completed frame, updated on each `falling_progress` event
- On completion: inline MP4 player
- MP4 appears in gallery with "Falling" badge

---

## Implementation Phases

| Phase | Scope |
|-------|-------|
| **A** | GPG metric + worldline CPU only, background skybox, no disk |
| **B** | Thin disk intersection + Page-Thorne shading |
| **C** | Redshift + Doppler, HorizonFlip LookMode |
| **D** | Metal GPU pass + CPU refinement mask |
| **E** | Roll cinematografico, frontend tab completo, MP4 output |

---

## Key References

- Lin & Soo (2009) arXiv:0905.3244 — GPG coordinates for Kerr-Newman with Λ
- Natário (2008) arXiv:0805.0206 — PG coordinates for Kerr (Λ=0 reference)
- Zaslavskii (2018) arXiv:1802.07069 — Regular frames for rotating BHs
- James et al. (2015) CQG 32 065001 — Ray bundle technique (existing codebase)
