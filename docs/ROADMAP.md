# Roadmap e stato

Fonte unica dello stato del progetto: cosa funziona, cosa manca, in che ordine.
Le stime e le evidenze del backlog vengono da
[AUDIT-MIGLIORIE-2026-11-15.md](AUDIT-MIGLIORIE-2026-11-15.md) (S < ½ giorno,
M 1–2 giorni, L 3–5, XL oltre una settimana). I problemi aperti puntuali sono in
[OPEN-ISSUES.md](OPEN-ISSUES.md).

Modo di lavorare concordato: una voce alla volta, dalla più piccola, ciascuna con
un test che fallisce prima della correzione e una propria PR su `main`, unita
quando le due suite (`build/` Metal, `build_rcpu/` CPU) passano.

---

## Funziona

| area | stato | note |
|---|---|---|
| Metrica KNdS, carte BL e KS | ✅ | KS coincide con BL e con l'ombra analitica di Schwarzschild (`kerrtrace.chart_consistency`) |
| Integratori RK4 a passo doppio e DOPRI5 | ✅ | |
| Camera statica a distanza finita, redshift di Bardeen | ✅ | |
| Disco sottile: blackbody, Interstellar, NASA, stratificato | ✅ | NASA e stratificato solo CPU |
| CPU / OpenMP | ✅ | riferimento |
| Metal | ✅ | FP32; forze BL e KS analitiche; rumore su hash intero bit-identico alla CPU |
| CUDA | ⚠️ | test solo lato host; mai validato su GPU NVIDIA |
| Fasci di raggi (campo di Jacobi), impronte, copertura | ✅ | solo CPU |
| Costante di Carter | ✅ | usata da semi-analitico ed ellittico; manca la classificazione delle orbite (C7) |
| Solver semi-analitico (`--semi-analytic`) | ✅ | tempo di Mino, ~4× più veloce di BL (A1, PR #2) |
| Forma chiusa ellittica (`--elliptic-closed`) | ✅ | corretta in B2 (PR #1); ricade sul numerico per il cielo e i casi limite |
| Wormhole DNEG | ✅ | senza disco, solo CPU |
| Formato `.kgeo` v5 autodescrittivo con provenienza | ✅ | D2, PR #4 |
| Gate di verifica: manifest dei binari, confronto impronte, valori golden | ✅ | D1, D3, PR #3 |
| Navigazione web (server + frontend) | ✅ | |

---

## Backlog

Fatte: D1, D3, B2, A1, D2, A5 (PR #1–#5), D4 (questa riorganizzazione).

| ID | voce | effort | note |
|---|---|---|---|
| D5 | inventario e pulizia di build e rami vecchi | S | mostrare l'elenco prima di cancellare; tenere `falling-camera-wip` |
| C3 | significato di `magnif` (determinante grezzo della mappa angolare) | S–M | validare contro raggi vicini, non imporre mediana ≈ 1 |
| C1 | temperatura del disco r^-0,5 → r^-0,75 | M | opzione di profilo con legacy riproducibile |
| B4 | export NumPy integrato nella CLI | M | oggi c'è `kgeo_to_npy` |
| A3 | palette NASA e stratificata su Metal | M / L | |
| A2 | blackbody Metal "slavato" | M diagnosi | non riprodotto: prima un caso |
| A6 | tolleranza minima Metal, residui sul bordo del disco | M | 21 pixel a 4K, disco di taglio |
| C4 | anelli di fotoni | M CPU / L multi-backend | esiste su `feature/photon-ring-multipass@e40814b` con un GeoPixel da 56 byte: va portato al formato v5 come campo vero, non in `_pad` |
| C2 | verso di rotazione del disco indipendente dal segno di a | L | oggi il disco si ribalta attraversando a = 0 |
| A4 | export della geometria dalla GPU | L / XL | |
| B1 | Metal: secondo passaggio, riordino dei raggi, coda persistente | L / XL | dopo profiling aggiornato |
| B3 | CUDA: visibilità Interstellar, fasci | L / XL | serve hardware NVIDIA |
| D7 | schema unico dei parametri CLI / server / frontend | L / XL | |
| D8 | campionamento sferico del cielo, cuciture | M | |
| E2 | filtro spaziale e temporale alla DNGR | L / XL | |
| C5 | polarizzazione | XL | la premessa "campo B da Connors–Stark–Piran 1980" va corretta |
| C6 | wormhole con potenziale, disco, rotazione | L / XL | |
| C7 | cambio di carta, classificazione di Carter, disco spesso, spettri | XL | |
| E1 | modalità interattiva nativa | XL | piano sotto |

---

## Studio di fattibilità: modalità interattiva (E1)

Scritto prima dell'audit; resta valido come piano, con una correzione dell'audit:
la navigazione **web** esiste già (`server/index.js`, `frontend/`). Mancano la
finestra nativa, la camera libera e una pipeline persistente. Stime dell'audit:
5a 3–5 settimane, 5b 1–2, 5c 4–8, 5d 3–6.

### Feasibility Summary (2026, pre-audit)

**Overall verdict: Feasible. Medium-high complexity. Recommended phased approach.**

The core physics engine (geodesic integrator, metric, disk model) is already production-quality.
The main new work is: (1) a real-time preview pipeline at low res/quality, (2) a native window +
3D camera controller, (3) progressive refinement up to cinematic quality, (4) an optional LUT
pre-bake for "instant" preview. The web-based frontend (Express + WebSocket) already handles
parameter dispatch and image delivery, which simplifies integration.

### Module Feasibility Breakdown

| Module | Difficulty | Notes |
|--------|-----------|-------|
| **PreviewInteractive** (live Metal, 64–256 px, <100 ms) | Low | Metal tile dispatch already exists; just reduce resolution |
| **PhysicsPreview** (720p, full physics, 1–5 s) | Low | Already works via `metal_render()`; add async dispatch |
| **CinematicRender** (4K, bundles, full quality) | Low | Already works; just needs UI trigger |
| **CameraController** (orbit, pan, roll, FOV) | Medium | New: 3D arcball/quaternion math; maps to `Camera` params |
| **RenderSettings** (live param sliders) | Low | Frontend already does this via JSON POST |
| **MetricModule** (KNdS param switching) | Low | Already runtime-configurable |
| **GeodesicIntegrator** (RK4/DOPRI5 switch) | Low | Already a runtime flag |
| **DiskModule** (profile, palette, taper) | Low | Already runtime flags |
| **RedshiftDopplerModule** | Done | v0.2.12 exact g formula |
| **Background/Skybox** (HDR HDRI rotation) | Medium | Need equirect rotation by camera yaw; currently fixed |
| **LookupTable** (pre-baked r,θ → hit-record) | High | New: offline bake step; significant engineering; optional |
| **ProgressiveRefinement** (coarse→fine) | Medium | Needs multi-pass accumulation buffer |
| **TemporalAccumulation** (TAA-style) | High | Needs per-pixel history; complex for non-linear projection |
| **Upscaling** (bilinear / MetalFX SR) | Medium | Bilinear trivial; MetalFX requires AVFoundation integration |
| **Native Window / UI** | High | Biggest new component; recommend SDL2 + Dear ImGui |

### Risks & Constraints

- **LUT bake** is the highest-risk item: a full `(r_obs, θ_obs, α, β)` → hit-record table is
  O(GB) for 4K. Practical only for fixed-metric, fixed-r_obs slices. Mark as optional/stretch.
- **Temporal accumulation** with GR lensing is non-trivial: pixel history is in screen space but
  lensing maps non-linearly. Reprojection from previous frame requires storing geodesic endpoints
  per pixel. Defer to stretch goal.
- **MetalFX Super Resolution** requires macOS 13+ and a Metal Performance Shaders dependency.
  Bilinear 2× upscale is a free 1-day task; MetalFX is a separate sprint.
- **Native window** is the largest new dependency. SDL2 (MIT license, single `.framework` on
  macOS) is the lowest-friction choice. Dear ImGui adds ~5k LOC but gives free sliders/panels.
- **Camera controller**: arcball orbit is standard; the tricky part is mapping 3D screen-space
  mouse deltas to BL coordinate `(θ_obs, φ_obs)` increments. Gimbal-lock-free quaternion
  accumulation then re-projection to BL angles is the correct approach.

### Recommended Architecture

```
┌─────────────────────────────────────────┐
│  SDL2 window  +  Dear ImGui sidebar     │  ← new (Phase 5a)
├─────────────────────────────────────────┤
│  CameraController                       │  ← new (Phase 5a)
│   arcball quaternion → BL (r,θ,φ,fov)  │
├──────────────┬──────────────────────────┤
│ Preview path │ Full-quality path        │
│ 128×72 px    │ up to 4K                 │
│ Metal, 1 ms  │ Metal/CPU, async         │  ← mostly exists
├──────────────┴──────────────────────────┤
│  Shared render core (metal_render / CPU)│  ← unchanged
│  KNdSMetric, Geodesic, Camera, Disk     │
└─────────────────────────────────────────┘
```

### Implementation Roadmap

#### Phase 5a — Live Preview Window (2–3 weeks)
- [ ] Add SDL2 + Dear ImGui as optional CMake deps (`-DUSE_INTERACTIVE=ON`)
- [ ] `InteractiveWindow` class: SDL2 event loop, MTLTexture → SDL_Texture blit
- [ ] `CameraController`: arcball quaternion, map to BL `(θ_obs, φ_obs, fov_h)`; mouse drag = orbit, scroll = zoom, middle-drag = pan
- [ ] Async render thread: on param change, dispatch low-res Metal (128×72) immediately; dispatch 720p after 300 ms debounce; dispatch cinematic on explicit button press
- [ ] ImGui sidebar: sliders for `a`, `Q`, `Λ`, inclination, FOV, disk brightness, p exponent, toggle Doppler/bundles/chart
- [ ] Background rotation: add `phi_bg_offset` uniform to shader for skybox yaw sync with camera φ

#### Phase 5b — Progressive Refinement (1–2 weeks)
- [ ] Multi-pass accumulation: render at 1/8 res, 1/4, 1/2, full — blit each pass to screen
- [ ] Pixel-stable jitter: Halton (2,3) sub-pixel offset per pass for free MSAA convergence
- [ ] Cancel-on-move: abort in-flight render when camera moves (use `MTLCommandBuffer cancel` + a `std::atomic<bool>`)

#### Phase 5c — LUT Pre-bake (stretch, 3–4 weeks)
- [ ] Offline bake tool (`kerr_bake`): for fixed `(M, a, Q, Λ, r_obs)` sweep `(θ_obs, α, β)` grid, store `GeoPixel` hit-records in a flat binary `.kgeo` file
- [ ] Runtime LUT loader: on LUT hit, skip geodesic integration entirely; interpolate `(r_hit, theta_hit, phi_hit, redshift)` from 4 nearest samples
- [ ] LUT memory budget: 1920×1080×(1 byte type + 3×4 byte coords + 4 byte redshift) ≈ 24 MB per inclination slice — manageable for a small grid of inclinations

#### Phase 5d — Temporal Accumulation & Upscaling (stretch, 2–3 weeks)
- [ ] Per-pixel geodesic endpoint cache: store `(r_hit, phi_hit)` in a secondary Metal buffer
- [ ] TAA reprojection: reproject previous frame endpoint into current frame UV; blend with α=0.1
- [ ] Bilinear 2× upscale: render at 960×540, blit to 1920×1080 (Metal blit encoder, trivial)
- [ ] MetalFX SR (macOS 13+): optional CMake flag `-DUSE_METALFX=ON`; wrap `MTLFXSpatialScaler`

### File Changes Required

| File | Change |
|------|--------|
| `CMakeLists.txt` | Add `USE_INTERACTIVE` flag, find SDL2 + ImGui |
| `interactive/window.hpp/.mm` | New: SDL2 window, MTL texture blit |
| `interactive/camera_controller.hpp` | New: arcball quaternion → BL angles |
| `interactive/render_scheduler.hpp` | New: async multi-pass dispatch |
| `interactive/ui.hpp` | New: Dear ImGui sidebar layout |
| `gpu/metal/tracer.metal` | Add `phi_bg_offset` uniform to `CameraParams_C` |
| `renderer/cli.hpp` | Add `--interactive` flag dispatching to `InteractiveWindow::run()` |
