# Problemi aperti

Solo ciò che è aperto oggi, con un rimando alla voce del backlog in
[ROADMAP.md](ROADMAP.md). Quando una voce si chiude, toglierla da qui e scriverne
le misure nel commit o nella PR. Il diario storico con le diagnosi è in
[STORICO-PROBLEMI-2026-09.md](STORICO-PROBLEMI-2026-09.md).

Ultimo allineamento: 28 novembre 2026.

## Fisica e resa

- **Verso di rotazione a `a = 0`** (C2). Il segno di Ω segue quello di a: animando
  lo spin attraverso lo zero il disco si ribalta.
- **Residui Metal sul bordo del disco visto di taglio** (A6): 21 pixel con errore
  > 48 livelli a 4K, da differenze di traiettoria FP32.

## Solver ellittico (`--elliptic-closed`)

Corretto in B2; restano, da togliere solo dopo validazione:
- la guardia `r_now < rh_cut`;
- il ritracciamento numerico di ogni colpo diretto per la precedenza
  dell'orizzonte: nei test non corregge più nessun colpo, quindi è puro costo;
- non calcola gli attraversamenti successivi (`tau_period` inutilizzato) né φ
  nel punto d'impatto: le palette diverse da blackbody ricadono sul numerico.

## Formato e dati

- **Wormhole nel render completo con `--geo-file`**: `renderer/cli.hpp` costruisce
  `KGeoMeta` con l'ISCO di Kerr anche per il wormhole. La provenienza v5 dichiara
  comunque `spacetime = dneg-wormhole`; `--geo-only` è corretto.

## Backend

- **CUDA** non è mai stato compilato ed eseguito su una GPU NVIDIA: i test
  coprono solo il codice host.

## Pulizia (D5)

- Build vecchie: `build_cpu/` (configurata con un cmake x86 che non gira più),
  `build/kerr_tracer_metal` (avanzo di un layout precedente), `build_audit/`,
  `build_edge/`, `build_fixes/`.
- Rami remoti già integrati in `main`; `falling-camera-wip` invece non lo è e va
  tenuto.
