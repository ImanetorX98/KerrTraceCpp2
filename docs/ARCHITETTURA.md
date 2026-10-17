# Architettura — mappa del progetto

Scopo: capire dov'è cosa senza rileggere 12 000 righe. Se una riga qui contraddice
il codice, ha ragione il codice — e va corretta questa.

Ultimo allineamento: v0.2.34 + refactor modulare (17 ottobre 2026). Numeri di riga
e conteggi storici sono indicativi; cercare i simboli nel sorgente.

Aggiornamento Metal BL (9 settembre 2026): `geodesic_rhs` nello shader usa
derivate analitiche della Hamiltoniana, mentre il riferimento CPU conserva
le differenze finite in double. Il tracciamento BL limita il passo vicino
all’asse e il single-ray riusa gli eventi del tracciatore condiviso. Vedere
[il resoconto del bump](METAL-BUMP-FIX-2026-09-09.md) e il test nativo
`metal_rhs_regression.mm`; non reintrodurre differenze finite FP32 nelle forze BL.

---

## Albero

```
KerrTraceCpp2/
├── main.cpp                      29  solo stb + main(): include renderer/cli.hpp
├── renderer/                         il renderer CPU, un header per argomento
│   ├── cli.hpp                  836  parsing dei flag, frame singolo, animazione
│   ├── render.hpp               350  fase 1 + fase 2 sul backend scelto
│   ├── backend_select.hpp       168  pianificatore unico CPU/Metal/CUDA
│   ├── phase1.hpp               280  trace_geodesics: setup del frame, resampling di bordo
│   ├── pixel_tracers.hpp        276  un tipo per tracciatore + std::variant + ciclo
│   ├── parallel.hpp              37  ciclo parallelo per righe (OpenMP o thread)
│   ├── tracer_bl.hpp            197  BL numerico, il riferimento
│   ├── tracer_ks.hpp            478  Kerr-Schild Cartesiano
│   ├── tracer_separable.hpp     296  Kerr separabile (Q=0, Λ=0)
│   ├── tracer_elliptic.hpp      613  forma chiusa, Carlson + Jacobi
│   ├── tracer_wormhole.hpp       47  integrazione DNEG di un raggio
│   ├── colorize.hpp             172  fase 2: GeoPixel → RGB
│   ├── palettes.hpp             649  palette del disco, profili radiali
│   ├── frame.hpp                 54  FrameParams, Spacetime, spacetime_traits
│   └── …                             trace_types, color_params, background,
│                                     disk_physics, util, common
├── knds_metric.hpp              444  metrica Kerr-Newman-de Sitter, carte BL e KS
├── geodesic.hpp                 444  RHS hamiltoniana, RK4 a passo doppio, DOPRI5
├── camera.hpp                   146  tetrad dell'osservatore statico, pixel → raggio
├── ray_bundle.hpp               687  fascio: deviazione geodetica, impronta, copertura
├── wormhole_metric.hpp          270  metrica DNEG del wormhole
│
├── gpu/metal/
│   ├── tracer.metal            3079  shader MSL: rispecchia renderer/, float32
│   ├── metal_renderer.mm        374  ponte Objective-C++
│   └── metal_renderer.hpp        81  struct dei parametri, gemella di quella MSL
├── gpu/cuda/
│   ├── tracer.cu                624  kernel CUDA + launcher
│   └── tracer.cuh                26
│
├── tests/                             5 test, tutti via ctest
│   ├── core_tests.cpp           124  unità pure
│   ├── bump_detector.cpp        324  self-test
│   ├── spin_orientation_regression.cpp   268
│   ├── ray_bundle_regression.cpp         329
│   └── chart_consistency_regression.cpp  293
│
├── server/index.js                    Express + WebSocket: traduce JSON → flag CLI
├── frontend/src/app/                  Angular: app.ts, app.html, render.service.ts
├── assets/backgrounds/                sfondo1..5.jpg, black.png
├── sources/                      25   i paper (DNGR, Gralla-Lupsasca, GRay, …)
├── docs/                              questi file
└── out/                               render e .kgeo — ignorato da git, rigenerabile
```

---

## Il flusso: due fasi, un file in mezzo

È la struttura portante, e spiega perché certe cose sono difficili.

```
   parametri CLI
        │
        ▼
  ┌───────────────┐   GeoPixel[]    ┌───────────────┐
  │   FASE 1      │ ──────────────► │   FASE 2      │ ──► PNG
  │  geometria    │   (.kgeo)       │   colore      │
  └───────────────┘                 └───────────────┘
   traccia le geodetiche             applica palette,
   e registra DOVE finiscono         redshift, filtri
```

**Fase 1** (`trace_geodesics`, `renderer/phase1.hpp`) non conosce i colori. Per ogni pixel
traccia un raggio e scrive un `GeoPixel`: esito, raggio d'impatto, redshift,
azimut, direzione di fuga, impronta.

**Fase 2** (`colorize_buffer`, `renderer/colorize.hpp`) non traccia niente: legge il
buffer e produce colori.

Si possono eseguire separatamente: `--geo-only` scrive il `.kgeo`, `--color-only
file.kgeo` lo rilegge. Utile per riprovare palette senza ritracciare.

**Conseguenza da tenere a mente**: la fase 2 vede un solo campione per pixel. Per
i pixel di bordo il colore corretto sarebbe la media dei colori dei
sotto-campioni, ma noi mediamo la *geometria* e ombreggiamo una volta.
È il limite residuo documentato in `PROGRESS-P2.md`.

---

## `GeoPixel`: il contratto fra le fasi (`render_data.hpp`)

64 byte, `KGEO_VERSION` = 4. **Il formato non è auto-descrittivo**: chi legge deve
conoscere il layout. Cambiarlo senza incrementare la versione è già successo
(v0.2.3) ed è passato inosservato per mesi.

| campo | significato |
|---|---|
| `outcome` | 0 = fuga, 1 = disco, 2 = orizzonte, 3 = universo B |
| `r`, `phi_disk` | punto d'impatto sul disco |
| `redshift` | rapporto di frequenza fra camera statica finita ed emettitore |
| `magnif` | `\|det J\|` del fascio (1 in single-ray) |
| `theta_esc`, `phi_esc` | direzione di fuga, per lo sfondo |
| `fp_dr_*`, `fp_dphi_*` | impronta del pixel **sul disco** |
| `sky_dth_*`, `sky_dph_*` | impronta del pixel **sulla sfera celeste** |
| `coverage` | frazione di pixel coperta dal disco |

I tre test di regressione replicano questo layout e hanno uno `static_assert`
sulla dimensione: se cambi `GeoPixel` e non i test, il build fallisce subito.
È voluto.

**Lettura da Python.** `kgeo_to_npy file.kgeo out` scrive `out.npy` (array NumPy
strutturato H×W, `numpy.load("out.npy")["redshift"]`) e `out.json` con
`KGeoMeta`. Il dtype non è scritto a mano: viene da `tools/kgeo_reflection.hpp`,
che descrive ogni campo con Sane C++ Reflection (`third_party/sane_cpp/`). I suoi
`static_assert` fermano il build se un campo aggiunto a `GeoPixel` non è descritto
lì o se compare padding. Se cambi `GeoPixel`, aggiorna anche quel file
(tag nuovi in coda, mai riusati).

---

## I tracciatori: quale corre quando

Selezionati da spaziotempo, `--solver-mode` e carta.

| tipo in `pixel_tracers.hpp` | chiama | file | quando |
|---|---|---|---|
| `BlPixelTracer` | `trace_single` | tracer_bl.hpp | BL, numerico, il riferimento |
| `KsPixelTracer` | `trace_single_ks` | tracer_ks.hpp | KS Cartesiana, default quando `Q=0, Λ=0` |
| `SeparablePixelTracer` | `trace_single_separable_kerr` | tracer_separable.hpp | Kerr puro, potenziali separati |
| `EllipticPixelTracer` | `trace_single_elliptic_closed` | tracer_elliptic.hpp | forma chiusa con integrali ellittici |
| `BundlePixelTracer` | `trace_bundle` | ray_bundle.hpp | modalità `--bundles` |
| `WormholePixelTracer` | `trace_single_wormhole` | tracer_wormhole.hpp | metrica DNEG |

### Come si sceglie, e perché non con `virtual`

Ogni tracciatore è un piccolo oggetto funzione con
`void operator()(int px, int py, GeoPixel&) const`. `trace_geodesics` ne
costruisce **uno per frame** e lo mette in `PixelTracer`, uno `std::variant`.
`trace_frame` fa un solo `std::visit`, che istanzia `trace_rows<T>` per quel
tipo: il ciclo sui pixel è monomorfo, senza if-chain né chiamata virtuale per
pixel. Costo misurato: nessuno (stessi tempi a 720p di prima del refactor).

Per aggiungere un tracciatore: un tipo nuovo in `pixel_tracers.hpp`, una voce nel
variant, un ramo nella scelta in `phase1.hpp`. Ciclo, thread e barra di
avanzamento restano condivisi.

### Spaziotempo e backend

`FrameParams::spacetime` (`KNDS` o `DNEG_WORMHOLE`) sostituisce il vecchio
`bool wormhole`. Quel che cambia fra i due fuori dal tracciatore (kernel GPU
disponibili? `r_obs` può essere negativo?) si chiede a `spacetime_traits`, non
controllando "è il wormhole?" in ogni punto.

"Questo frame può girare su Metal/CUDA?" ha **una sola** risposta:
`plan_metal` / `plan_cuda` in `backend_select.hpp`. `render_image` la esegue, la
CLI la usa (`planned_backend`) per la riga `Mode:` e i nomi dei file. Prima la
stessa logica stava in tre copie sincronizzate a mano.

**BL e KS devono dare la stessa immagine.** Non era così fino a v0.2.18: KS
sbagliava il raggio d'ombra del 5%. Ora `kerrtrace.chart_consistency` lo verifica
sia contro l'altra carta sia contro il valore analitico di Schwarzschild.

Il solver ellittico ricade sul numerico in molti casi (~53% a `a=0.998`): vedi
`CLAUDE.md`, sezione Region III.

---

## Il fascio (`ray_bundle.hpp`)

Integra la deviazione geodetica accanto al raggio centrale:

```
z = (r, θ, φ, p_r, p_θ)      parametri: (p_t, p_φ)
d(δz)/dλ = (∂f/∂z)·δz + (∂f/∂p_t)·δp_t + (∂f/∂p_φ)·δp_φ
```

Tre cose non ovvie, tutte imparate rompendosi la testa (vedi `PROGRESS-P1.md`):

1. `p_t` e `p_φ` sono conservati *lungo* un raggio ma **differiscono fra i raggi**
   del fascio: sono parametri, non costanti, e il termine di forzamento serve.
2. `δφ` va tracciato: l'impronta vive nel piano `(δr, r·δφ)`, non `(δr, δθ)`.
3. La deviazione va letta sulla **superficie di attraversamento**, non a λ uguale:
   i raggi vicini attraversano l'equatore a λ diversi.

`H = ½g^{μν}p_μp_ν` è esattamente quadratico nei momenti, quindi il blocco
momento-momento **è** la metrica inversa: nessuna Hessiana numerica.

Costo: **1.13×** il single-ray per il fascio nudo. Il resto è il resampling di
bordo (2.11× totale a 720p).

---

## Le palette (fase 2)

Quattro, tutte funzioni pure di `(r, φ)` più scalari per-pixel — per questo il
filtro d'impronta sta **fuori** da esse e nessuna ha dovuto cambiare.

| palette | flag | note |
|---|---|---|
| blackbody | `--disk-blackbody` | l'unica con temperatura di colore vera |
| stratified | `--disk-stratified` | piastrelle con `cell_hash` |
| interstellar | `--disk-interstellar` | bande + turbolenza, default |
| NASA | `--disk-nasa` | turbolenza propria, `--disk-nasa-gain` |

**Trappola dell'azimut**: le texture sono funzioni di φ, che ha un taglio di ramo.
Il rumore a reticolo va **piastrellato** (`fbm2d_tiled`) o si vede una barra
verticale. Tre siti: la maschera condivisa, la turbolenza Interstellar, quella
NASA. La stratified ha un `cell_hash` proprio **non ancora piastrellato**.

---

## GPU

`tracer.metal` rispecchia i tracciatori di `renderer/` in float32. Regole:

- **Si carica dal sorgente**, `exeDir/../gpu/metal/tracer.metal`
  (`metal_renderer.mm:49`), non dalla copia in build. Modificarlo ha effetto al
  run successivo senza ricompilare; patchare `build*/tracer.metal` non fa nulla.
- Le due struct dei parametri (`metal_renderer.hpp` e `tracer.metal`) sono
  **gemelle**: campi nuovi vanno aggiunti **in coda a entrambe**, nello stesso
  ordine.
- NASA e stratified, esportazione KGEO e bundle con impronte sono instradati
  esplicitamente alla CPU. `Backend used:` dichiara il backend effettivo.
- CUDA restituisce `GeoPixel[]` e condivide la fase colore CPU; la funzione
  realmente chiamata dal kernel è confrontata con la CPU anche nei test host.
- I test Metal compilano lo shader e confrontano frame BL/KS sulla GPU reale;
  senza dispositivo accessibile vengono saltati esplicitamente.
- Ogni correzione di fisica va portata di là a mano. Il bug del covettore BL→KS
  c'era identico e l'ho scoperto solo perché mi è stato chiesto se fosse a posto.

---

## Web

`server/index.js` traduce il JSON del frontend in flag CLI e lancia il binario.
Non parsa i `.kgeo`. Quindi **una nuova opzione va aggiunta in tre punti**:
`renderer/cli.hpp` (parsing), `server/index.js` (inoltro), `frontend/src/app/` (controllo
e tipo in `render.service.ts`).

---

## Build e test

```bash
cmake -B build_cpu -DUSE_METAL=OFF && cmake --build build_cpu -j8
cmake -B build -DUSE_METAL=ON     && cmake --build build -j8
ctest --test-dir build_cpu
```

**Refactor senza cambi di comportamento**: `tests/render_fingerprint.py` rende
una ventina di configurazioni piccole (carte, solutori, bundle, palette, Q, Λ,
wormhole, round trip `.kgeo`) e ne salva gli hash. Baseline col binario prima
della modifica, poi `--compare`: ogni hash deve restare uguale. Gli hash valgono
solo sulla stessa macchina e con gli stessi flag (`-march=native`, driver Metal).

Su Apple Silicon il build è vincolato ad arm64: `CMakeLists.txt` interroga
`sysctl hw.optional.arm64` **prima** di `project()` e fallisce se il toolchain
punta a x86_64, perché cmake stesso può girare sotto Rosetta e mentire.

---

## Dove sta scritto cosa

| file | contenuto |
|---|---|
| `CLAUDE.md` | fisica, convenzioni, roadmap, note sul solver ellittico |
| `PLAN-2026-09-05.md` | piano P0–P8 con criteri di verifica |
| `PROGRESS-P1.md` | diario del campo di Jacobi e dei filtri |
| `PROGRESS-P2.md` | diario dello sfondo e dei pixel di bordo |
| `RESOLVED-2026-09-05.md` | difetti chiusi, con le misure |
| `OPEN-ISSUES.md` | quel che resta |

---

## Trappole, in ordine di quanto tempo mi hanno fatto perdere

1. **zsh non fa word-splitting.** `ARGS="--a --b"; cmd $ARGS` passa **un solo**
   argomento. Mi ha ingannato quattro volte. Usare array.
2. **Misurare sulla maschera giusta.** Le statistiche sul fotogramma intero sono
   dominate dallo sfondo: a 720p sono 776 000 pixel contro 145 000 di disco. Due
   diagnosi sbagliate sono nate da qui.
3. **Lo shader si carica dal sorgente**, non dalla build dir.
4. **`grep "fallback"` matcha la riga `Mode:`**, che contiene
   `elliptic-fallback-black=off`.
5. **Guardare il frame prima di committare.** Le statistiche possono essere vere
   e indicare il colpevole sbagliato.
6. **Il `.kgeo` non è auto-descrittivo**: cambi `GeoPixel` → incrementa
   `KGEO_VERSION` → aggiorna i tre test.
