# KerrTraceCpp2

Relativistic ray tracer for Kerr / Kerr-Newman / Kerr-de Sitter black holes.

The project integrates null geodesics and renders thin accretion disks with
relativistic effects (Doppler boosting, gravitational redshift, lensing).
It includes CPU and GPU backends, plus a basic web UI.

## Features

- Boyer-Lindquist (BL) and Kerr-Schild (KS) chart support
- Adaptive geodesic integration (`RK4` step-doubling, optional `DOPRI5`)
- Thin accretion-disk shading with redshift and temperature mapping
- Ray-bundle mode for magnification-aware rendering
- GPU backends:
  - Metal (`gpu/metal`) on macOS
  - CUDA (`gpu/cuda`) on NVIDIA platforms
- CLI renderer + Node/Angular UI
- CI + tests (`core` + bump detector)

## Repository Layout

- `main.cpp`: CLI, render pipeline, backend dispatch
- `knds_metric.hpp`: metric and chart transforms
- `geodesic.hpp`: Hamiltonian RHS and integrators
- `camera.hpp`: camera model and initial ray setup
- `ray_bundle.hpp`: Jacobi/ray-bundle machinery
- `gpu/metal/*`: Metal backend
- `gpu/cuda/*`: CUDA backend
- `frontend/`, `server/`: web UI and API layer
- `tests/`: regression and geometry tests

## Build

Requirements:

- CMake >= 3.18
- C++17 compiler
- Optional:
  - Metal framework (macOS)
  - CUDA Toolkit (for `USE_CUDA=ON`)

### CPU (default)

```bash
cmake -S . -B build
cmake --build build --parallel
./build/kerr_tracer
```

### Metal (macOS)

Option A (recommended on macOS): default configure builds `kerr_tracer_metal`
alongside CPU.

```bash
cmake -S . -B build
cmake --build build --parallel
./build/kerr_tracer_metal
```

Option B: explicit Metal build for `kerr_tracer`.

```bash
cmake -S . -B build -DUSE_METAL=ON
cmake --build build --parallel
./build/kerr_tracer
```

### CUDA

```bash
cmake -S . -B build -DUSE_CUDA=ON
cmake --build build --parallel
./build/kerr_tracer
```

Windows/Linux notes for CUDA builds:

- By default the project builds a fat binary for common architectures:
  `60;61;70;75;80;86;89`.
- You can override architectures explicitly:

```bash
cmake -S . -B build -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
```

- For maximum numerical fidelity (recommended for FP64-heavy runs), keep
  fast-math disabled (default). For speed-focused experiments you can enable it:

```bash
cmake -S . -B build -DUSE_CUDA=ON -DCUDA_ENABLE_FAST_MATH=ON
```

## CLI Quick Start

### Single frame (KS chart)

```bash
./build/kerr_tracer --ks \
  --custom-res 1280 720 \
  --a 0.5 --charge 0 --lambda 0 \
  --r-obs 40 --disk-out 12 \
  --theta 80 --phi 0 --fov 45
```

On macOS Metal binary:

```bash
./build/kerr_tracer_metal --ks \
  --custom-res 1280 720 \
  --a 0.5 --charge 0 --lambda 0 \
  --r-obs 40 --disk-out 12 \
  --theta 80 --phi 0 --fov 45
```

Useful flags:

- `--ks` / `--bl` (or `--chart ks|bl`)
- `--bundles`
- `--dopri5`
- `--gpu-fp64` / `--cuda-fp64` (strict FP64 mode on GPU: on CUDA this enforces
  device FP64 capability checks; on Metal currently falls back to CPU)
- `--no-gpu-fp64` / `--no-cuda-fp64`
- `--solver-mode standard|semi-analytic|elliptic-closed`
- `--metal-kernel auto|unified|single|bundle` (Metal only)
- `--semi-analytic` / `--elliptic` (legacy alias for `--solver-mode semi-analytic`)
- `--elliptic-closed` (alias for `--solver-mode elliptic-closed`)
- `--bg <path>`
- `--4k`, `--2k`, `--720p`, `--custom-res W H`

Experimental note:

- `semi-analytic` and `elliptic-closed` modes currently target Kerr BL
  (`Q=0`, `Lambda=0`) and are optional, never default.
- `elliptic-closed` uses a closed-form elliptic CPU path (Carlson/Jacobi)
  with per-ray fallback to separable stepping when constraints are not met.
- Metal executes `elliptic-closed` with a dedicated elliptic GPU path and
  per-ray fallback to semi-analytic when needed for robustness.
- CUDA currently routes `elliptic-closed` to CPU fallback.
- CUDA single-ray kernel uses native `double` precision arithmetic.
- Metal dispatch is adaptive-tiled for high resolutions (2K/4K). You can
  override tile rows with `KERR_METAL_TILE_ROWS=<n>` when tuning stability
  vs throughput.
- `--bundles` on Metal is GPU-native for BL + `standard` solver (finite-
  difference bundle proxy). Other bundle configurations fall back to CPU.
- Kernel entrypoints are selectable:
  - `auto` (default): picks `single` or `bundle` by mode
  - `unified`: legacy all-in-one kernel (`trace_pixel`)
  - `single`: force single-ray kernel
  - `bundle`: force ray-bundle kernel

Rendered frames are written under `out/`.

## Recommended Usage Cases

| Scenario | Recommended backend/mode | Why |
|---|---|---|
| Fast local preview on macOS | Metal + `standard` + BL/KS | Lowest latency when kernel is stable |
| Highest numerical robustness on macOS | CPU (`kerr_tracer`) | Full double precision path, stable fallback behavior |
| Windows/Linux NVIDIA quality runs | CUDA + `--gpu-fp64` | Native CUDA double path + strict FP64 capability check |
| Windows/Linux NVIDIA speed runs | CUDA (without `--gpu-fp64`) + optional `-DCUDA_ENABLE_FAST_MATH=ON` | Maximum throughput when minor precision loss is acceptable |
| Non-standard solver (`elliptic-closed`) | CPU | Current CUDA path falls back for this mode |

## Tests

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## CI

GitHub Actions workflow: `.github/workflows/ci.yml`

- builds on `ubuntu-latest` and `macos-latest`
- runs the test suite on push and pull request

## License

This project is licensed under the MIT License.
See [LICENSE](LICENSE).
