# Fisica e convenzioni

Fonte unica per le formule, le convenzioni del codice e i riferimenti. `CLAUDE.md`
e `AGENTS.md` rimandano qui; la mappa del codice è in `ARCHITETTURA.md`, lo stato
delle funzionalità in `ROADMAP.md`. Se il codice contraddice questa pagina, ha
ragione il codice e la pagina va corretta.

## Sintesi fisica

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
| `--dopri5`  | `DOPRI5` | Dormand-Prince RK45 | 7 first, 6 with FSAL | `ode45` standard |

DOPRI5 Butcher tableau: 7 stages, embedded 4th/5th order pair, error = `‖y5−y4‖`.
FSAL: `k7 = f(y5)` reused as `k1` of the next step → 6 new evaluations per subsequent accepted step; rejected trials add evaluations.
RK4-doubling: 1 full step + 2 half steps, Richardson factor = 2⁴−1 = 15.

### Ray bundles
Variational equations `d(δz)/dλ = M(z)·δz` alongside main geodesic.
`M = J_s · Hess(H)`, Hessian by central finite differences.
Jacobi matrix `W[:,0]` = deviation in α, `W[:,1]` = deviation in β.
`|det J|` at disk crossing → magnification → flux correction.
Reference: James et al. (2015) CQG 32 065001.

### Disk colour temperature
Blackbody palette: `T = 6500 K · temp_scale · (6M/r)^q · g`, with `q = 3/4` by
default: for a thin disk `σT⁴ = F ∝ r^-3` away from the inner edge, so
`T ∝ r^-3/4`. Before 2026-11 the code used `q = 1/2` (`--disk-temp-exponent 0.5`).
The luminosity uses the radial flux profile, not this temperature. The
Interstellar palette has no temperature, but its colour ramp is placed by the
same law (`interstellar_ramp_position`, `q = 3/4`; `0` = linear in radius).

### Disk redshift
Full formula (Bardeen 1972):
`g = ν_obs/ν_emit = u_obs^t · √(−g_tt−2g_tφΩ−g_φφΩ²) / (1−Ω·b)`
where `b = p_φ/(−p_t)`, `u_obs^t = 1/√(−g_tt_obs)` for the static camera,
and corotating Kerr `Ω_K = s√M/(r^{3/2}+|a|√M)`, with `s = −1` for `a<0`, otherwise `+1`.
The KNdS implementation uses metric derivatives for the general circular orbit.
The observer-at-infinity approximation is recovered when `u_obs^t → 1`.

### Spacetime non-Kerr: wormhole DNEG
Metrica di James, von Tunzelmann, Franklin & Thorne (2015), "Visualizing
Interstellar's Wormhole" (arXiv:1502.03809): Morris–Thorne statico con Φ = 0,
generalizzazione del wormhole di Ellis. Formule, parametri (ρ, a, M_lens) e
Hamiltoniana in `wormhole_metric.hpp`. Nessun disco; solo CPU.

### Solver alternativi per Kerr puro (Q = 0, Λ = 0)
- `--semi-analytic`: potenziali separati R(r), Θ(θ) in tempo di Mino, al secondo
  ordine (r'' = R'/2, θ'' = Θ'/2), con proiezione sugli integrali primi.
- `--elliptic-closed`: forma chiusa di Gralla & Lupsasca (2019), Appendice B,
  con integrali di Carlson e funzioni di Jacobi; ricade sul numerico per raggio.
Dettagli e difetti corretti in `ROADMAP.md` e nei commit di A1 e B2.

## Convenzioni del codice
- C++17, geometric units `G=c=1`.
- All angles in radians internally.
- BL coordinate order: `(t=0, r=1, θ=2, φ=3)`.
- KS Cartesian order: `(T=0, X=1, Y=2, Z=3)`.
- Metric signature: `(−+++)`.
- `pt < 0` always (future-directed, `E = −p_t > 0`).
- Headers are self-contained; no `.cpp` files for metric/geodesic.
- GPU code mirrors CPU math exactly (same formulas, different types).
- Never use `std::clamp` for C++14 compatibility in GPU ports.

## Riferimenti (in `sources/`)
- James et al. (2015) — DNGR / Interstellar technique, ray bundles
- Gralla & Lupsasca (2019) — Analytical Kerr null geodesics
- Pihajoki et al. (2018) — Arcmancer: multi-chart library
- Luminet (2019) — History of BH imaging
- Chan et al. (2013) — GRay: GPU ray-tracing
- James, von Tunzelmann, Franklin & Thorne (2015) — wormhole DNEG (non in `sources/`: arXiv:1502.03809)
