# Photon Ring Multi-Hit Compositing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Track photon ring order (n) per pixel and composite up to 3 disk hits per ray with physically correct opacity stacking and an artistic `ring_opacity_falloff` control.

**Architecture:** Extend `GeoPixel` to store primary + 2 extra disk hits; refactor `trace_single` to accumulate hits and continue past semi-transparent disk crossings; update `colorize_buffer` for multi-hit weighted blend. All changes confined to `main.cpp`.

**Tech Stack:** C++17, no new dependencies.

---

## Files

| File | Changes |
|------|---------|
| `main.cpp` | All changes: GeoPixel layout, KGEO_VERSION, ColorParams, TraceResult, trace_single, 3 GeoPixel population sites, colorize_buffer |
| `tests/core_tests.cpp` | New tests: GeoPixel layout, opacity weight formula |

---

## Weight formula reference

```
// α = disk_opacity, f = ring_opacity_falloff, n_hits = total disk hits (1–3)
// w_0   = α
// w_1   = (1−α) × α × f          [extra hit 1]
// w_2   = (1−α)² × α × f²        [extra hit 2]
// w_bg  = (1−α)^n_hits            [final background, no falloff]
// final = w_0·disk0 + w_1·disk1 + w_2·disk2 + w_bg·bg
// At f=1.0 + opaque disk (α=1): degenerates to current single-hit behaviour.
```

---

## Task 1: Update GeoPixel layout and KGEO_VERSION

**Files:**
- Modify: `main.cpp:103-113` (GeoPixel struct + static_assert)
- Modify: `main.cpp` near `KGEO_VERSION` constant

**Current struct (28 bytes):**
```cpp
struct GeoPixel {
    uint8_t outcome;
    uint8_t _pad[3];   // [0]=debug_tag, [1]=behind_code, [2]=free
    float r, redshift, magnif, phi_disk, theta_esc, phi_esc;
};
static_assert(sizeof(GeoPixel) == 28, "GeoPixel size mismatch");
```

- [ ] **Step 1: Replace GeoPixel struct**

```cpp
struct GeoPixel {
    // ── header (8 bytes) ─────────────────────────────────────
    uint8_t outcome;      ///< 0=escaped, 1=disk_hit, 2=horizon, 3=ESCAPED_B
    uint8_t n_ring;       ///< ring order of primary hit (0=direct, 1=photon ring…)
    uint8_t n_extra;      ///< extra disk hits stored: 0, 1, or 2
    uint8_t behind_code;  ///< background after ALL disk passes: 0=bg_A escaped,
                          ///<   2=horizon, 3=bg_B escaped
    uint8_t debug_tag;    ///< elliptic solver debug tag (was _pad[0])
    uint8_t _pad[3];      ///< alignment padding
    // ── primary disk hit (24 bytes) ──────────────────────────
    float r;              ///< BL radius at primary disk crossing
    float redshift;       ///< g = ν_obs/ν_emit at primary crossing
    float magnif;         ///< flux magnification (bundle mode; 1 in single-ray)
    float phi_disk;       ///< BL azimuthal angle at primary crossing
    float theta_esc;      ///< direction at final escape (background lookup)
    float phi_esc;
    // ── extra hits for opacity compositing (12 bytes each) ───
    float r1, redshift1, phi1;   ///< ring n+1 (valid if n_extra >= 1)
    float r2, redshift2, phi2;   ///< ring n+2 (valid if n_extra >= 2)
};
static_assert(sizeof(GeoPixel) == 56, "GeoPixel size mismatch");
```

- [ ] **Step 2: Bump KGEO_VERSION to 2**

Find `static const uint32_t KGEO_VERSION = 1;` and change to `= 2;`.

- [ ] **Step 3: Verify build still compiles (static_assert fires if layout is wrong)**

```bash
cmake --build build_cpu 2>&1 | grep -E "error:|GeoPixel"
```
Expected: no errors. If static_assert fires, fix padding.

---

## Task 2: Add `ring_opacity_falloff` to ColorParams

**Files:**
- Modify: `main.cpp` ColorParams struct (~line 69)

- [ ] **Step 1: Add field to ColorParams**

After `double disk_opacity = 1.0;` insert:
```cpp
double ring_opacity_falloff = 1.0; ///< brightness multiplier per ring order beyond n=0
                                   ///< 1.0 = pure physics; <1.0 = artistic darkening
```

- [ ] **Step 2: Parse `--ring-falloff` CLI argument**

In the CLI argument parsing section (search for `"--disk-opacity"` to find the pattern), add:
```cpp
} else if (arg == "--ring-falloff" && i+1 < argc) {
    cp.ring_opacity_falloff = clamp(std::stod(argv[++i]), 0.0, 1.0);
```

- [ ] **Step 3: Build and run `--help` to verify flag exists**

```bash
cmake --build build_cpu 2>&1 | tail -3 && ./build_cpu/kerr_tracer --help 2>&1 | grep ring
```

---

## Task 3: Extend TraceResult for multi-hit data

**Files:**
- Modify: `main.cpp` TraceResult struct (~line 206)

- [ ] **Step 1: Add fields to TraceResult**

Replace the current TraceResult:
```cpp
struct TraceResult {
    Outcome out; double r, redshift;
    double phi_disk=0.0;
    double theta_esc=0.0, phi_esc=0.0;
    Outcome behind_out=Outcome::HORIZON;
};
```
With:
```cpp
struct DiskHitRecord {
    double r=0.0, redshift=1.0, phi=0.0;
};

struct TraceResult {
    Outcome out            = Outcome::ESCAPED;
    double  r              = 0.0;
    double  redshift       = 1.0;
    double  phi_disk       = 0.0;
    double  theta_esc      = 0.0;
    double  phi_esc        = 0.0;
    Outcome behind_out     = Outcome::HORIZON;
    int     n_ring         = 0;   ///< ring order of primary hit
    int     n_extra        = 0;   ///< extra hits stored in extra[]
    DiskHitRecord extra[2] = {};  ///< extra[0]=n_ring+1 hit, extra[1]=n_ring+2 hit
};
```

- [ ] **Step 2: Compile**

```bash
cmake --build build_cpu 2>&1 | grep "error:" | head -20
```
Expected: no errors (new fields have defaults; existing callsites still compile).

---

## Task 4: Refactor `trace_single` for multi-hit

**Files:**
- Modify: `main.cpp:463–591` (function `trace_single`)

This is the core change. The current function returns on the first disk hit (optionally calling `trace_terminal_no_disk` for behind). The new version accumulates hits and continues.

- [ ] **Step 1: Add ring counters before the main loop**

After line `Vec4d fsal=Vec4d::nan_init();` insert:
```cpp
// ── Multi-hit ring tracking ───────────────────────────────
const double opacity    = cp_ptr ? clamp(cp_ptr->disk_opacity,  0.0, 1.0) : 1.0;
const bool   multi_hit  = (opacity < 1.0 - 1e-9);
int  n_eq_cross  = 0;  // total equatorial crossings so far (in + out of disk)
int  n_disk_hits = 0;  // disk hits recorded (max 3)
bool done_record = false; // true once we've recorded the max useful hits
DiskHitRecord hits_buf[3] = {};
int  hit_nring[3] = {};   // ring order of each hit
```

- [ ] **Step 2: Replace the `if (best_event == StepEvent::DISK)` block**

Find and replace lines 552–580 (the `StepEvent::DISK` handler):

```cpp
        if (best_event == StepEvent::DISK) {
            if (!done_record) {
                // Record this hit
                hits_buf[n_disk_hits] = {disk_r_hit, disk_redshift_hit, disk_phi_hit};
                hit_nring[n_disk_hits] = n_eq_cross;
                ++n_disk_hits;
                // Stop recording if: disk is opaque OR we've hit the max (3)
                if (!multi_hit || n_disk_hits >= 3) done_record = true;
            }
            n_eq_cross++;  // count the crossing regardless

            // Advance state past the equatorial plane crossing
            const double alpha = clamp(best_alpha, 0.0, 1.0);
            GeodesicState s_hit = s_prev;
            s_hit.r      = s_prev.r      + alpha * (s.r      - s_prev.r);
            s_hit.theta  = s_prev.theta  + alpha * (s.theta  - s_prev.theta);
            s_hit.phi    = s_prev.phi    + alpha * (s.phi    - s_prev.phi);
            s_hit.pr     = s_prev.pr     + alpha * (s.pr     - s_prev.pr);
            s_hit.ptheta = s_prev.ptheta + alpha * (s.ptheta - s_prev.ptheta);
            s_hit.pt     = s_prev.pt;
            s_hit.pphi   = s_prev.pphi;
            const double th_eps = 1e-6;
            const double q_dir  = (s.theta - s_prev.theta);
            if (std::abs(s_hit.theta - M_PI/2.0) < 1e-5)
                s_hit.theta += (q_dir >= 0.0 ? th_eps : -th_eps);
            s_hit.theta = clamp(s_hit.theta, th_eps, M_PI - th_eps);

            if (done_record && !multi_hit) {
                // Opaque disk: stop here, no behind needed
                break;
            }
            // Semi-transparent (or still recording): continue from just past crossing
            s = s_hit;
            fsal = Vec4d::nan_init();
            dlam = std::max(1e-10, step_used * std::max(1e-3, 1.0 - alpha));
            continue;
        }
```

- [ ] **Step 3: Count non-disk equatorial crossings**

Immediately AFTER the `maybe_equator` block (after line 528, where `best_event` may or may not have been set to DISK), add:
```cpp
        // Count equatorial crossings that were NOT recorded as disk hits.
        // The DISK branch above already increments n_eq_cross; here we handle
        // crossings that fell outside the disk radial range.
        if (best_event != StepEvent::DISK) {
            const double q0b = s_prev.theta - M_PI/2.0;
            const double q1b = s.theta      - M_PI/2.0;
            if (sign_change(q0b, q1b)) n_eq_cross++;
        }
```

- [ ] **Step 4: Replace the function return path after the loop**

After the main loop ends (the `return {Outcome::ESCAPED, s.r, 1.0, 0.0, s.theta, s.phi};` fallback), replace the entire exit logic. Currently lines 581–591 handle HORIZON and ESCAPE events. Keep those, but change how we return:

Replace this entire block at the end:
```cpp
        if (best_event == StepEvent::HORIZON) {
            const double r_h = s_prev.r + best_alpha * (s.r - s_prev.r);
            return {Outcome::HORIZON, r_h, 0.0};
        }
        if (best_event == StepEvent::ESCAPE) {
            const double th_esc = s_prev.theta + best_alpha * (s.theta - s_prev.theta);
            const double ph_esc = s_prev.phi   + best_alpha * (s.phi   - s_prev.phi);
            return {Outcome::ESCAPED, r_escape, 1.0, 0.0, th_esc, ph_esc};
        }
    }
    return {Outcome::ESCAPED, s.r, 1.0, 0.0, s.theta, s.phi};
```

With:
```cpp
        if (best_event == StepEvent::HORIZON || best_event == StepEvent::ESCAPE) {
            Outcome terminal_out;
            double th_esc = 0.0, ph_esc = 0.0;
            if (best_event == StepEvent::HORIZON) {
                terminal_out = Outcome::HORIZON;
            } else {
                terminal_out = Outcome::ESCAPED;
                th_esc = s_prev.theta + best_alpha * (s.theta - s_prev.theta);
                ph_esc = s_prev.phi   + best_alpha * (s.phi   - s_prev.phi);
            }
            if (n_disk_hits == 0) {
                return {terminal_out, s.r, 1.0, 0.0, th_esc, ph_esc};
            }
            // Build multi-hit result
            TraceResult out;
            out.out       = Outcome::DISK_HIT;
            out.r         = hits_buf[0].r;
            out.redshift  = hits_buf[0].redshift;
            out.phi_disk  = hits_buf[0].phi;
            out.n_ring    = hit_nring[0];
            out.behind_out = terminal_out;
            out.theta_esc  = th_esc;
            out.phi_esc    = ph_esc;
            out.n_extra    = n_disk_hits - 1;
            for (int k = 0; k < out.n_extra; ++k)
                out.extra[k] = hits_buf[k + 1];
            return out;
        }
    }
    // Max steps exhausted
    if (n_disk_hits == 0)
        return {Outcome::ESCAPED, s.r, 1.0, 0.0, s.theta, s.phi};
    TraceResult out;
    out.out       = Outcome::DISK_HIT;
    out.r         = hits_buf[0].r;
    out.redshift  = hits_buf[0].redshift;
    out.phi_disk  = hits_buf[0].phi;
    out.n_ring    = hit_nring[0];
    out.behind_out = Outcome::ESCAPED;
    out.theta_esc  = s.theta;
    out.phi_esc    = s.phi;
    out.n_extra    = n_disk_hits - 1;
    for (int k = 0; k < out.n_extra; ++k)
        out.extra[k] = hits_buf[k + 1];
    return out;
```

- [ ] **Step 5: Build and fix any compile errors**

```bash
cmake --build build_cpu 2>&1 | grep "error:" | head -30
```

---

## Task 5: Update GeoPixel population sites

Three sites write GeoPixel fields. Update all three.

**Files:**
- Modify: `main.cpp` around lines 2409, 2603, 2569

- [ ] **Step 1: Fix site 1 — KS chart tracer (line ~2409, escape path)**

This site writes escaped/horizon pixels. Change:
```cpp
pix._pad[0]   = pix._pad[1] = pix._pad[2] = 0;
```
To:
```cpp
pix.n_ring    = 0;
pix.n_extra   = 0;
pix.behind_code = 0;
pix.debug_tag = 0;
pix.r1 = pix.redshift1 = pix.phi1 = 0.0f;
pix.r2 = pix.redshift2 = pix.phi2 = 0.0f;
```

- [ ] **Step 2: Fix site 2 — bundle/ray_bundle tracer (line ~2569)**

Find the block starting with `pix.outcome = res.disk_hit ? 1 : 0;`. After `pix.phi_esc`, add:
```cpp
pix.n_ring      = 0;           // bundles don't track ring order yet
pix.n_extra     = 0;
pix.behind_code = 0;
pix.debug_tag   = 0;
pix.r1 = pix.redshift1 = pix.phi1 = 0.0f;
pix.r2 = pix.redshift2 = pix.phi2 = 0.0f;
```
Remove the old `pix._pad[0] = pix._pad[1] = 0` lines.

- [ ] **Step 3: Fix site 3 — main CPU tracer (line ~2603)**

This is the primary site. Replace all `pix._pad[*]` assignments with:
```cpp
pix.n_ring      = (uint8_t)clamp(res.n_ring, 0, 255);
pix.n_extra     = (uint8_t)clamp(res.n_extra, 0, 2);
pix.behind_code = (res.out == Outcome::DISK_HIT)
                    ? ((res.behind_out == Outcome::ESCAPED_B) ? 3
                      :(res.behind_out == Outcome::HORIZON)   ? 2 : 0)
                    : 0;
pix.debug_tag   = debug_elliptic ? static_cast<uint8_t>(fb_reason) : 0;
pix.r1          = (pix.n_extra >= 1) ? (float)res.extra[0].r        : 0.0f;
pix.redshift1   = (pix.n_extra >= 1) ? (float)res.extra[0].redshift : 1.0f;
pix.phi1        = (pix.n_extra >= 1) ? (float)res.extra[0].phi      : 0.0f;
pix.r2          = (pix.n_extra >= 2) ? (float)res.extra[1].r        : 0.0f;
pix.redshift2   = (pix.n_extra >= 2) ? (float)res.extra[1].redshift : 1.0f;
pix.phi2        = (pix.n_extra >= 2) ? (float)res.extra[1].phi      : 0.0f;
```

- [ ] **Step 4: Update debug_elliptic colorize path**

In `colorize_buffer` the debug path reads `geo[i]._pad[0]`. Replace with `geo[i].debug_tag`.

- [ ] **Step 5: Compile**

```bash
cmake --build build_cpu 2>&1 | grep "error:" | head -20
```

---

## Task 6: Update `colorize_buffer` for multi-hit compositing

**Files:**
- Modify: `main.cpp:2260–2306` (`colorize_buffer`, the `if (p.outcome == 1)` branch)

- [ ] **Step 1: Replace the single-hit compositing block**

Current code (lines 2260–2296):
```cpp
for (int i = 0; i < W*H; ++i) {
    const GeoPixel& p = geo[i];
    if (p.outcome == 1) {
        // ... compute disk_col ...
        const double alpha = clamp(cp.disk_opacity, 0.0, 1.0);
        if (alpha >= 1.0 - 1e-9) {
            image[i] = disk_col;
        } else {
            // single behind composite
            ...
        }
    }
    ...
}
```

Replace with:
```cpp
for (int i = 0; i < W*H; ++i) {
    const GeoPixel& p = geo[i];
    if (p.outcome == 1) {
        // ── weight formula ─────────────────────────────────────
        // w_k = α × (1−α)^k × f^k   (k = 0,1,2)
        // w_bg = (1−α)^n_hits  (no falloff on background)
        const double alpha   = clamp(cp.disk_opacity, 0.0, 1.0);
        const double f       = clamp(cp.ring_opacity_falloff, 0.0, 1.0);
        const int    n_hits  = 1 + (int)p.n_extra;
        const double ia      = 1.0 - alpha;

        // Disk color helper — calls the appropriate palette function
        auto disk_col_for = [&](float r, float g_factor, float mag,
                                float phi) -> RGB {
            if (cp.palette == DiskPalette::STRATIFIED)
                return disk_colour_stratified(r, phi, g_factor, mag,
                                              r_disk_in, r_disk_out,
                                              M_bh, r_isco, cp);
            if (cp.palette == DiskPalette::INTERSTELLAR)
                return disk_colour_interstellar(r, phi, g_factor, mag,
                                                r_disk_in, r_disk_out,
                                                M_bh, r_isco, cp);
            return disk_colour(r, g_factor, mag, M_bh, r_isco, cp);
        };

        // Background after all disk passes
        auto bg_for = [&]() -> RGB {
            if (p.behind_code == 0 && !bg.px.empty())
                return bg.sample(p.theta_esc, p.phi_esc);
            if (p.behind_code == 3) {
                const BackgroundImage& bgB = (bg_b && !bg_b->px.empty()) ? *bg_b : bg;
                if (!bgB.px.empty()) return bgB.sample(p.theta_esc, p.phi_esc);
            }
            return {0, 0, 0};  // HORIZON or no background
        };

        double R = 0.0, G = 0.0, B = 0.0;

        // k=0: primary hit
        {
            const double w = alpha;
            const RGB c = disk_col_for(p.r, p.redshift, p.magnif, p.phi_disk);
            R += w * c.r;  G += w * c.g;  B += w * c.b;
        }
        // k=1: first extra hit
        if (p.n_extra >= 1) {
            const double w = ia * alpha * f;
            const RGB c = disk_col_for(p.r1, p.redshift1, p.magnif, p.phi1);
            R += w * c.r;  G += w * c.g;  B += w * c.b;
        }
        // k=2: second extra hit
        if (p.n_extra >= 2) {
            const double w = ia * ia * alpha * f * f;
            const RGB c = disk_col_for(p.r2, p.redshift2, p.magnif, p.phi2);
            R += w * c.r;  G += w * c.g;  B += w * c.b;
        }
        // background
        {
            double w_bg = 1.0;
            for (int k = 0; k < n_hits; ++k) w_bg *= ia;
            const RGB bg_col = bg_for();
            R += w_bg * bg_col.r;  G += w_bg * bg_col.g;  B += w_bg * bg_col.b;
        }

        image[i] = {
            (uint8_t)clamp(R, 0.0, 255.0),
            (uint8_t)clamp(G, 0.0, 255.0),
            (uint8_t)clamp(B, 0.0, 255.0),
        };
    } else if (p.outcome == 3) {
        // unchanged
    } else if (p.outcome == 0 && !bg.px.empty())
        image[i] = bg.sample(p.theta_esc, p.phi_esc);
}
```

- [ ] **Step 2: Build and run a test render**

```bash
cmake --build build_cpu -j$(sysctl -n hw.logicalcpu) 2>&1 | tail -5
./build_cpu/kerr_tracer --720p --theta 80 --a 0.998 --disk-opacity 0.85 --ring-falloff 1.0
```
Expected: image renders without crash, looks physically plausible.

- [ ] **Step 3: Test falloff parameter**

```bash
./build_cpu/kerr_tracer --720p --theta 80 --a 0.998 --disk-opacity 0.85 --ring-falloff 0.5
```
Expected: photon ring (n=1 arc) noticeably dimmer than with `--ring-falloff 1.0`.

---

## Task 7: Add `--debug-ring` colorization mode

Add a visual debug mode that colors pixels by ring order (n=0 green, n=1 yellow, n=2 red).

**Files:**
- Modify: `main.cpp` (colorize_buffer + CLI arg)

- [ ] **Step 1: Add `debug_ring` parameter to colorize_buffer signature**

Add `bool debug_ring = false` after `bool debug_elliptic = false`.

- [ ] **Step 2: Add debug_ring branch at top of colorize_buffer**

After the `debug_elliptic` block:
```cpp
if (debug_ring) {
    static const RGB ring_colors[] = {
        { 60, 200,  60},   // n=0 direct  → green
        {220, 200,   0},   // n=1 ring    → yellow
        {220,  60,  60},   // n=2 ring    → red
        {180,  60, 220},   // n=3         → purple (edge case)
    };
    for (int i = 0; i < W*H; ++i) {
        const GeoPixel& p = geo[i];
        if (p.outcome == 1) {
            const int n = clamp((int)p.n_ring, 0, 3);
            image[i] = ring_colors[n];
        }
    }
    return image;
}
```

- [ ] **Step 3: Wire --debug-ring CLI flag**

Near `--debug-elliptic` CLI handling, add:
```cpp
} else if (arg == "--debug-ring") {
    debug_ring = true;
```
Pass `debug_ring` through to `colorize_buffer` call.

- [ ] **Step 4: Test debug mode**

```bash
cmake --build build_cpu && ./build_cpu/kerr_tracer --720p --theta 80 --a 0.998 --debug-ring
```
Expected: green pixels for direct image, yellow pixels tracing the photon ring arc.

---

## Task 8: Tests

**Files:**
- Modify: `tests/core_tests.cpp`

- [ ] **Step 1: Add GeoPixel layout test**

```cpp
bool test_geopixel_layout() {
    // Layout must match KGEO_VERSION=2 format
    static_assert(sizeof(GeoPixel) == 56, "GeoPixel layout broken");
    GeoPixel p{};
    p.n_ring    = 1;
    p.n_extra   = 2;
    p.behind_code = 3;
    p.debug_tag = 0;
    p.r         = 5.0f;
    p.r1        = 4.0f;
    p.r2        = 3.5f;
    return p.n_ring == 1 && p.n_extra == 2 &&
           approx(p.r,  5.0, 1e-6) &&
           approx(p.r1, 4.0, 1e-6) &&
           approx(p.r2, 3.5, 1e-6);
}
```

- [ ] **Step 2: Add opacity weight formula test**

```cpp
bool test_ring_compositing_weights_sum_to_one_opaque() {
    // With opacity=1.0 and n_hits=1: w_0=1, w_bg=0
    const double alpha = 1.0, f = 1.0, ia = 1.0 - alpha;
    const double w0 = alpha;
    const double w_bg = ia; // (1-alpha)^1
    return approx(w0 + w_bg, 1.0, 1e-12);
}

bool test_ring_compositing_weights_sum_to_one_semi() {
    // opacity=0.7, falloff=1.0, 2 hits: w0 + w1 + w_bg should = 1
    const double alpha = 0.7, f = 1.0, ia = 1.0 - alpha;
    const double w0   = alpha;
    const double w1   = ia * alpha * f;
    const double w_bg = ia * ia;  // (1-alpha)^2
    return approx(w0 + w1 + w_bg, 1.0, 1e-12);
}

bool test_ring_compositing_falloff_dims_ring() {
    // With falloff=0.5, n=1 ring should be dimmer than falloff=1.0
    const double alpha = 0.7, ia = 1.0 - alpha;
    const double w1_phys    = ia * alpha * 1.0;  // falloff=1
    const double w1_dimmed  = ia * alpha * 0.5;  // falloff=0.5
    return w1_dimmed < w1_phys && approx(w1_dimmed, w1_phys * 0.5, 1e-12);
}
```

- [ ] **Step 3: Register tests in main()**

```cpp
{"GeoPixel layout v2",               test_geopixel_layout},
{"ring compositing weights opaque",  test_ring_compositing_weights_sum_to_one_opaque},
{"ring compositing weights semi",    test_ring_compositing_weights_sum_to_one_semi},
{"ring falloff dims ring",           test_ring_compositing_falloff_dims_ring},
```

- [ ] **Step 4: Build and run tests**

```bash
cmake --build build_cpu && ./build_cpu/kerr_tests
```
Expected:
```
[PASS] GeoPixel layout v2
[PASS] ring compositing weights opaque
[PASS] ring compositing weights semi
[PASS] ring falloff dims ring
All tests passed (10).
```

---

## Task 9: Commit

- [ ] **Step 1: Verify full build and render**

```bash
cmake --build build_cpu -j$(sysctl -n hw.logicalcpu) && \
./build_cpu/kerr_tracer --720p --theta 80 --a 0.998 --disk-opacity 1.0 && \
./build_cpu/kerr_tracer --720p --theta 80 --a 0.998 --disk-opacity 0.8 --ring-falloff 0.7 && \
./build_cpu/kerr_tracer --720p --theta 80 --a 0.998 --debug-ring
```
Expected: three renders complete without crash or NaN.

- [ ] **Step 2: Commit**

```bash
git add main.cpp tests/core_tests.cpp docs/superpowers/plans/2026-05-17-photon-ring-multipass.md
git commit -m "feat(tracer): photon ring order tracking + multi-hit opacity compositing

- GeoPixel v2 (56 bytes): adds n_ring, n_extra, behind_code, debug_tag,
  plus r1/g1/phi1 and r2/g2/phi2 extra disk hit fields
- trace_single counts all equatorial crossings; accumulates up to 3 disk
  hits (primary + 2 extra) per ray before falling through to bg
- colorize_buffer composites hits with weights w_k = α(1-α)^k·f^k where
  f = ring_opacity_falloff (1.0 = pure physics, <1 = artistic darkening)
- --ring-falloff <0..1> CLI flag; --debug-ring colorizes by ring order
- KGEO_VERSION bumped to 2 (old .kgeo files require re-render)"
```

---

## Self-review

**Spec coverage:**
- ✅ Ring order counter (n_ring per pixel)
- ✅ Multi-hit opacity compositing (up to 3 hits)
- ✅ Artistic `ring_opacity_falloff` control
- ✅ CLI flag `--ring-falloff`
- ✅ Debug visualization `--debug-ring`
- ✅ KGEO version bump
- ✅ Tests for layout + weight formula

**Gaps / notes:**
- Bundle tracer (`ray_bundle`) sets `n_ring=0, n_extra=0` (no multi-hit in bundle mode — bundles already compute magnification correctly; ring order can be added in a follow-up)
- Semi-analytic and elliptic paths already fall back to `trace_single` when `disk_opacity < 1`, so they inherit multi-hit automatically
- Metal/CUDA GPU tracers store old GeoPixel layout — they will need updating separately to match KGEO_VERSION=2 when used with `--color-only`
