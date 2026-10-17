# Metal KS: raggi bloccati dopo il bordo trasparente Interstellar — 17 ottobre 2026

## Sintomo

Con `--disk-interstellar` il backend Metal era 20× più lento della CPU
(640×360: 51 s di GPU contro 1,9 s CPU; con blackbody 2,3 s). La CPU non
mostrava differenze fra le palette.

## Indagine

- `KERR_METAL_STATS=1` (nuovo, vedi sotto): nessuna tessera fallita, 51 s di
  tempo GPU effettivo. Il kernel lavorava davvero di più.
- `--disk-interstellar-edge-transparency 0` riportava il tempo a 2,3 s: il costo
  nasceva dai raggi che attraversano il bordo sfumato del disco.
- Il tempo scalava con `--max-steps` (5 000 → 10 s, 50 000 → 46 s,
  500 000 → 51 s), e fra i due estremi cambiavano 677 pixel, tutti sul bordo
  esterno del disco.

## Causa

Nel ciclo KS di `tracer.metal`, dopo il pass-through, un `continue` saltava
l'aggiornamento `prevX/prevY/prevZ/prev_r_ks` di fine iterazione e i controlli
di orizzonte e fuga. Con `prevZ` fermo a prima dell'attraversamento, ogni passo
rilevava di nuovo lo stesso attraversamento del piano, l'interpolazione
ricadeva sul bordo trasparente e il raggio proseguiva fino a `max_steps`.
Il percorso BL dello stesso shader non usa `continue` ed era corretto.
Introdotto in e971388 (v0.2.10).

## Correzione e misure

Il pass-through ora cade nel normale flusso dell'iterazione.

| caso | prima | dopo |
|---|---:|---:|
| 640×360 Interstellar, tempo GPU | 51,1 s | 5,7 s |
| 1600×900 "Gargantua" (galleria) | 192 s | 16,4 s |
| pixel che dipendono da `--max-steps` | 677 | 3 |

Sul bordo esterno sparisce la frangia scura: il cielo arriva al bordo come nel
riferimento CPU. `tests/metal_passthrough_regression.py`
(`kerrtrace.metal_passthrough`) fallisce con lo shader vecchio (677 pixel) e
passa con quello nuovo (3). I 3 residui sono raggi vicini all'orbita fotonica
che richiedono davvero più di 5 000 passi.

## Aperto, non toccato qui

- **Tessere sincrone.** `metal_render` invia tessere piccole una alla volta e
  attende ciascuna (a 4K 512×2 px, griglia arrotondata a 16 righe: 7 thread su
  8 inattivi). Ogni tessera dura quanto il suo raggio più lento. Con tessere
  64×4096 (`KERR_METAL_TILE_ROWS/COLS`) il tempo scende del ~40 %. Il residuo
  Interstellar (3,0 s con cap 5 000 contro 5,7 s) è tutto in 2–3 raggi lunghi
  che bloccano la loro tessera.
- **Stesso `continue` nel percorso separabile** (`--semi-analytic`): salta
  l'aggiornamento di `prev_*_sep`. Non corretto perché non misurabile ora: il
  percorso separabile su GPU impiega 472 s a 320×180 anche col blackbody.
- **Forze KS su GPU per differenze finite FP32** (`geodesic_rhs_KS`, passo
  relativo 1e-5): la stessa cancellazione corretta in BL il 9 settembre.
  Candidata principale per gli scarti residui di 2 px in `metal_contour`.

## `KERR_METAL_STATS`

`KERR_METAL_STATS=1 ./build/kerr_tracer_metal …` stampa tessere iniziali,
dimensione, riuscite e fallite, tempo GPU sommato e tempo reale. Se il tempo
GPU è vicino al tempo reale il kernel è lento; se è molto minore la GPU sta
ferma fra una tessera e l'altra.
