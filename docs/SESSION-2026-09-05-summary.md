# Sessione 2026-09-04/05 — riepilogo

Da **v0.2.5** a **v0.2.14**. Tutto su `origin/main` (`eb957dc`).

---

## 1. Wormhole DNEG — due bug fisici

**`dr/dℓ` con un fattore `1/M_lens` di troppo.** La derivata corretta è
`(2/π)·arctan(2(|ℓ|−a)/(πM))·sgn(ℓ)` — il `1/M` di `dx/d|ℓ|` si cancella col
prefattore `M` di `r(ℓ)`. Invisibile al default `M_lens=1`, ma al valore
Interstellar `M_lens=0.035` gonfiava la forza di lensing ~28× e **nessun raggio
attraversava la gola**. Dopo il fix la transizione universo A/B cade esattamente
su `|L| = ρ`.

**Sfere celesti al 5% dalla camera.** `escape_radius = max(|r_obs|·1.05, 10)`
invece che a `ℓ = ±∞` (Fig. 5 del paper). Produceva un fisheye senza lensing.
Ora `max(|r_obs|·20, wh_sky_dist)` con default `1e4`, che riproduce gli angoli
d'uscita di `1e5` entro 0.03° a un settimo del costo. Nuovo flag `--wh-sky-dist`.

**Come mostrare l'interno**: allungare il tunnel con `--wh-tunnel` (gli anelli
concentrici *sono* la parete interna), o mettere la camera dentro con `--r-obs`
tra `−a` e `+a`. Misurato: nel tubo `r=ρ` costante ⇒ `r'=0` ⇒ esiste un
**cilindro di fotoni** degenere lungo tutto il tunnel. A `B/ρ = 1` esatto il
fotone resta intrappolato (20 420 giri con a=5). Solo i raggi con `B < ρ`
sondano l'interno: il lato `B > ρ` è insensibile alla lunghezza del tunnel.

Commit: `4ca61fa`

---

## 2. arm64 senza Rosetta

**I sorgenti erano già puliti.** Audit su ogni file tracciato: nessun binario
committato, nessun assembly, nessun intrinsic x86. Le SSE2 in `stb_image.h`
sono dietro `#if defined(__x86_64__)` con percorso NEON alternativo; le 36 voci
x64 nel lockfile sono tutte `optional` con `cpu:["x64"]`.

**Il difetto era nel toolchain.** `cmake` può essere un binario x86_64 tradotto
(l'Homebrew Intel in `/usr/local`): riporta un host x86_64 e punta il build su
`-arch x86_64`. Non arrivava neanche a compilare, perché `-march=native`
risolveva a `apple-m2`, nome ARM che il backend x86_64 rifiuta.

Fatto: `sysctl hw.optional.arm64` prima di `project()` (dice il vero anche sotto
traduzione), `FATAL_ERROR` se il target non è arm64, `-march=native` sondato
invece che assunto, gate `lipo` post-build su ogni artefatto, assert in CI.
Escape hatch: `-DKERRTRACE_ALLOW_X86=ON`, `-DKERRTRACE_NATIVE_ARCH=OFF`.

**Fuori dal repo**: aggiunto `eval "$(/opt/homebrew/bin/brew shellenv)"` in
`~/.zshrc` (backup `~/.zshrc.bak-pre-arm64-20260904`). Ora `cmake`, `ctest`,
`git` risolvono agli arm64 nativi. Rosetta è supportata fino a macOS 27 e
rimossa in macOS 28.

Commit: `a2893cb`, `b80be48`

---

## 3. Backend Metal — da inutilizzabile a più veloce della CPU

Produceva un frame **tutto nero** in ogni scena e ripiegava sempre sulla CPU.
Tre difetti nel controllo di passo adattivo:

1. **Norma d'errore assoluta in float32.** Il rumore di cancellazione di
   `y_full − y_half` a r~60 è `~|r|·eps ~ 4e-6`, quindi la norma non scendeva
   mai sotto `tol=1e-7`: ogni passo rifiutato, `h` collassato al pavimento, il
   raggio avanzava 1e-6 per passo — 0.2M percorsi in 20000 iterazioni contro i
   ~55M necessari. La CPU sopravvive alla stessa norma solo perché in doppia
   precisione quel rumore sta a 1e-15. Ora norma scalata, `err ≤ 1` = accettato.
2. **`clamp` con estremi invertiti.** `clamp(x, ADAPT_H_MIN, h*0.5f)` è
   `min(max(x,lo),hi)`: appena `h*0.5 < ADAPT_H_MIN` gli estremi si invertono e
   `h` cade *attraverso* il proprio pavimento.
3. **Tolleranza irraggiungibile in float32.** Corretti 1 e 2, un frame 854×480
   costava 103 s: tasso di accettazione 82.5% ma ~99 500 passi accettati per
   raggio. `tol=1e-7` e `tol=1e-5` danno **la stessa accuratezza** (~17/255
   contro la CPU in float64), perché il residuo è il float32 stesso. Pavimento
   `ADAPT_TOL_MIN_F32 = 1e-5`.

Risultato: nero → **1.22 s** a HD, contro 3.27 s della CPU. Differenza media
10.1/255, concentrata su anello di fotoni e bordi disco.

Commit: `471daea`

---

## 4. `kerrtrace.spin_orientation` — era il test, non il renderer

Rosso da **v0.2.3** (`1587f19`, trovato col bisect). Il test porta una copia
hardcoded del record `.kgeo`: `1587f19` ha inserito `phi_disk` in `GeoPixel`
portandolo da 24 a 28 byte **senza incrementare `KGEO_VERSION`**, quindi il
controllo di versione non poteva accorgersene. Il test camminava con passo 24
su record da 28 e leggeva spazzatura.

Verificato sui byte: un frame 320×180 pesa 1 612 896 = 96 di header + 320·180·**28**.

Aggiunta una guardia che valida la dimensione del payload e fallisce nominando
il problema. Il renderer era corretto da sempre: la simmetria di specchio regge
al pixel — `cx(−a) + cx(+a) = 319 = W−1` su tutti gli spin testati, e le due
immagini a `a=±0.5` su sfondo nero sono identiche **bit per bit** dopo
riflessione orizzontale (diff 0.000, max 0).

Controllo negativo: sabotando il segno di Ω il test fallisce; rimettendo il
record a 24 byte la guardia scatta.

Commit: `948cc7c`

---

## 5. CI verde per la prima volta

**38 run, 38 fallimenti** nella storia del repo. Due cause indipendenti:

- **macOS**: falliva solo `spin_orientation` (§4).
- **Linux**: non compilava proprio. `geodesic.hpp` passava un coefficiente di
  Butcher DOPRI5 come `(const double[]){a21}`, compound literal C99 che GCC
  rifiuta in C++ ("taking address of temporary array"). Presente dal commit
  iniziale del 19 aprile.

Render DOPRI5 byte-identico dopo il fix (`sha256 7099e886…`).
Run `33915680314`: **entrambi i job SUCCESS**.

Commit: `1aade79`

---

## 6. Ray bundle — terminazione

Costavano **~18 400×** un raggio singolo; un frame HD non finiva mai. Solo ~9×
è l'Hessiano legittimo, il resto erano raggi che non terminavano.

`trace_single_bundle` protegge l'analisi del disco con `maybe_equator`, vero
ogni volta che il raggio sta entro 0.35 rad dal piano equatoriale — con camera
quasi equatoriale, quasi ogni passo. Dentro quel blocco un `continue` saltava
**anche i test di orizzonte e fuga**, che stanno sotto.

Prove raccolte prima di toccare il codice: il tempo scalava linearmente con
`--max-steps` (1k → 0.075 s, 500k → 28.3 s) mentre il raggio singolo restava
piatto (0.0021 s a entrambi); e il costo crollava esattamente alla soglia dei
0.35 rad (θ=80 → 28.4 s, θ=60 → 5.9 s, θ=40 → 0.011 s).

Risultato: 16×9 a θ=80 da **28.43 s a 0.018 s**. HD con bundle da mai-finito a
**21.9 s** contro 3.29 s single-ray, cioè 6.6×. Non-regressione: a θ=40, fuori
dalla banda, frame byte-identico.

Commit: `88e214c`

---

## 7. Palette interstellar — fisica di default

**Decadimento esponenziale.** `exp(-(r−r_in)/(0.7·r_in))` non ha corrispettivo
in nessun modello di disco sottile. Con `a=0.998` l'ISCO è ≈1.24M, quindi la
scala vale 0.87M mentre il disco arriva a 14M: attenuazione **9.75e6** dove
Novikov-Thorne chiede **51.8**. Risultato: 56.2% dei pixel di disco a zero
assoluto, mediana 0/255 (blackbody: 0.2% e 48). Alzare il guadagno non
recuperava nulla perché moltiplicava zeri.

La differenza è di **famiglia**, non di ripidezza. Pendenza logaritmica del
profilo fisico: +1.94 a 1.5M, −2.10 a 3M, −2.79 a 14M, −2.96 a 200M — limitata,
tende a −3. Quella di un esponenziale è `−(r−r_in)/L` e **cresce senza limite**:
−2.0 a 3M ma −14.7 a 14M.

**Profilo radiale.** `disk_colour_interstellar` non chiamava mai `disk_flux_raw`,
a differenza di blackbody e stratified. Ora usa lo stesso flusso fisico
`F ∝ r⁻³(1−√(r_isco/r))`, normalizzato su `disk_flux_reference`.

Misurato (a=0.998, θ=82, disk_out=14, soli pixel di disco):

| configurazione | a zero | mediana |
|---|---|---|
| **fisico (nuovo default)** | **1.8%** | **52** |
| artistico, legge di potenza | 2.1% | 19 |
| artistico + inner glow (vecchio default) | 56.2% | 0 |
| blackbody, riferimento | 0.2% | 48 |

Il default fisico centra la luminosità di blackbody **senza taratura di
guadagno**. Entrambi gli artifici restano, da CLI e da UI:
`--disk-interstellar-artistic-profile`, `--disk-interstellar-inner-glow`.

Commit: `18fb4e1`, `b1c863e`

---

## 8. Magnificazione dei ray bundle — committata ma INCOMPLETA

`GeoPixel::magnif` porta `|det J|`, il determinante di `d(r,θ)/d(α,β)`: un
Jacobiano **dimensionale** la cui scala è fissata dalla geometria della camera,
non una magnificazione. Mediana misurata 27.66, quindi `1/magnif` aveva mediana
0.0362 e il **70% dei pixel di disco era appiattito sul fondo** del
`clamp(1/magnif, 0.05, 5.0)`, tutti allo stesso valore. Da qui il disco
collassato in una linea con l'anello luminoso attorno.

`|det J|` si appiattisce lontano dal lensing forte — 24.59, 23.80, 23.64, 23.70,
23.66 per `r_hit = 10..14` — confermando che è una costante geometrica. La
mediana sui pixel di disco la stima in modo robusto.

Normalizzato **in un punto solo**, dove il pixel viene letto, così tutte e
quattro le palette la ereditano. Dopo: mediana 1.000, fondo mai raggiunto
(0.00%). Single-ray **byte-identico** (`sha256 16373061613a6810`), perché in
quella modalità ogni `magnif` è 1 e la correzione è inerte per costruzione.

> **ATTENZIONE — vedi `OPEN-ISSUES`.** Il commit peggiora l'immagine. Il tetto
> del clamp a 5.0 è ora il vincolo attivo (7.20% dei pixel lo tocca, contro
> 0.2% prima) e li appiattisce insieme: la caustica esce come chiazza bianca
> satura larga invece che anello con gradiente. Le due modifiche sono
> accoppiate e ne è stata spedita solo metà.

Commit: `1ded000`

---

## Cronologia commit

```
eb957dc  Merge fix/interstellar-physical-default        v0.2.14
35cde47  Merge fix/ray-bundle-termination               v0.2.11
1ded000  fix(bundles): normalise |det J|                v0.2.14
b1c863e  feat(disk): Novikov-Thorne flux in interstellar v0.2.13
18fb4e1  feat(disk): exponential decay opt-in           v0.2.12
88e214c  fix(bundles): termination near the equator     v0.2.11
602e9be  Merge fix/spin-orientation-kgeo-layout
1aade79  fix(build): drop the C99 compound literal      v0.2.10
948cc7c  fix(test): sync spin_orientation kgeo layout   v0.2.9
335e3ca  Merge fix/metal-adaptive-step
471daea  fix(metal): unstall the GPU integrator         v0.2.8
f23fe01  Merge chore/arm64-native-build
b80be48  build(arm64): refuse Rosetta-dependent binary  v0.2.7
a2893cb  build(arm64): native arm64 on Apple Silicon
4ca61fa  fix(wormhole): dr/dl and celestial spheres
```

Branch integrati su origin, cancellabili: `chore/arm64-native-build`,
`fix/metal-adaptive-step`, `fix/spin-orientation-kgeo-layout`,
`fix/ray-bundle-termination`, `fix/interstellar-physical-default`.
