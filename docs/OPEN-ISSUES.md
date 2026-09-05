# Aperto al 2026-09-05

Ordinato per urgenza. Il primo punto è una regressione visiva già su `main`.

---

## 1. ~~La magnificazione dei bundle è a metà strada~~ — RISOLTO in v0.2.15 (`dfde570`)

La diagnosi qui sotto era **sbagliata**. La chiazza bianca non veniva dalla
magnificazione: tolta la modulazione, l'anello restava identico. La causa vera
era in `ray_bundle.hpp`, segno di Ω:

```cpp
const double Omega = g.keplerian_omega(r_hit);   // manca il meno
```

`keplerian_omega()` restituisce **−Ω_K** per convenzione. Senza la negazione,
`d2 = −(g_tt + 2·g_tφ·Ω + g_φφ·Ω²)` finiva sul ramo retrogrado e diventava
negativo sotto r ≈ 1.5M → fallback `red = 1.0` → disco interno non spostato,
con g⁴ un fattore 10⁴ di troppo. Un secondo errore di segno su `b` mascherava
il primo annullandosi nel denominatore.

Dopo il fix, su 326 845 pixel comuni: `|Δg|` mediano 0.0285 → 0.0153, p99
0.924 → 0.069, e i 4864 pixel bloccati a g esatto 1.000 sono spariti.

Normalizzazione **rimossa del tutto** (`disk_magnif_reference()` cancellata).
La modulazione da magnificazione è ora opzione spenta di default
(`--bundle-magnification`), perché per un disco risolto Liouville dà brillanza
superficiale `g⁴ ×` emessa, indipendente dal footprint del fascio.

**Nota di processo, ancora valida**: `1ded000` era stato spedito senza guardare
l'immagine, sulle sole statistiche. Le statistiche erano vere ma non
catturavano il difetto — e per giunta indicavano il colpevole sbagliato.
**Guardare sempre il frame, e confrontare i `.kgeo` prima di attribuire una
causa.**

---

## 1-bis. ~~La geodetica del bundle diverge da quella single-ray~~ — DIAGNOSI SBAGLIATA

Confrontavo bundle **in BL** con single-ray **in KS** (la carta di default). A
parità di carta il fascio segue la geodetica centrale in modo esatto:

| confronto | `\|Δr\|` mediano | p99 |
|---|---|---|
| single KS vs bundle BL | 0.5313 | 2.531 |
| single **BL** vs bundle **BL** | **0.0000** | 0.001 (max) |

Il difetto è reale ma sta altrove → punto **1-ter**.

---

## 1-ter. APERTO — BL e KS non danno la stessa immagine

E la differenza **non cala stringendo la tolleranza**, quindi non è errore
d'integrazione: BL vs KS dà `|Δr|` mediano 0.5313 a `tol=1e-7` e 0.5312 a
`tol=1e-11`, mentre ciascuna carta confrontata con se stessa a tolleranza
diversa dà 0.0000.

- KS trova 23 541 hit sul disco, BL 21 208: **2699 pixel** di differenza, situati
  a **r ≈ 11.4**, cioè sul bordo **esterno** del disco, non vicino all'orizzonte.
- `|Δr|` **cresce con r** (0.27 dentro 2M, 0.68 fra 4M e 8M): il contrario di
  quello che darebbe un problema di carta vicino all'orizzonte.
- Un riallineamento rigido porta la mediana solo da 0.53 a 0.42 → c'è anche un
  piccolo disallineamento della camera fra le carte, ma non è la causa principale.
- Il solver ellittico trova esattamente lo stesso insieme di hit di KS (23 541),
  indizio a favore di KS ma debole, perché i suoi fallback girano in KS.

Piano in `PLAN-2026-09-05.md`, voce **P0**. È bloccante: il bundle esiste solo in
BL, quindi eredita la carta eventualmente sbagliata.

---

## 2. Ricerca DNGR — appena iniziata

Il paper primario è in locale: `sources/DNGR_James_Thorne_2015_1502.03808.pdf`,
47 pagine. Conteggi grezzi: "caustic" 84 occorrenze, "bundle" 42,
**"magnification" zero**. Suggerisce che DNGR tratti le caustiche come struttura
geometrica da risolvere infittendo i fasci, non come valori da tagliare.

Dalla scheda del paper: DNGR usa **filtraggio spaziale** per raccordare le
interfacce fra fasci adiacenti e **filtraggio temporale** fra fotogrammi. Non
un clamp.

Da fare: leggere le sezioni pertinenti del PDF. È la fonte primaria, meglio dei
forum. Link: <https://iopscience.iop.org/article/10.1088/0264-9381/32/6/065001>

---

## 3. ~~Metal non coperto dalla normalizzazione~~ — non si applica più

Con la normalizzazione rimossa non c'è nessuna statistica frame-wide da
replicare sulla GPU. Lo shader ha già il segno di Ω corretto
(`tracer.metal:1678`), quindi il bug del punto 1 era solo lato CPU.

---

## 4. Mappatura temperatura→colore (diagnosticato, non toccato)

Due difetti indipendenti in `disk_colour` (blackbody):

**Clamp che morde.** `red_phys = clamp(g, 0.2, 5.0)`. Il tetto è inerte (0.00%),
ma il **fondo a 0.2 blocca il 3.7%** dei pixel: le regioni fortemente
redshiftate non diventano rosse quanto dovrebbero. Il minimo misurato di `g` è
0.031, sei volte sotto il clamp.

**Esponente sbagliato.** Il codice usa `T ∝ √(6M/r)` = `r^-0.50`. La fisica dà
`T = (F/σ)^(1/4)` con `F ∝ r^-3`, quindi **`r^-0.75`**. Il disco esterno esce
troppo caldo (giallo-bianco invece di rosso profondo): a 14M il codice dà 0.378
contro 0.232 fisico, normalizzato a 2M.

Asimmetria da notare: la **luminosità** usa già il flusso corretto via
`disk_flux_raw`; è solo la **temperatura di colore** ad avere l'esponente
diverso.

Il fattore Doppler in sé è stato verificato ed **è corretto**: `disk_redshift`
implementa il fattore di Bardeen `g = E/[u^t(E−ΩL)]`, equivalente alla forma
`√d2/(1−Ωb)`; `doppler_exp = 4` è giusto per l'intensità bolometrica; su 106 681
pixel di disco l'intervallo è 0.031–1.460 con mediana 0.763 e **zero** pixel al
clamp `kGMax=6`.

---

## 5. `KGEO_VERSION` è ancora 1

Il layout del record `.kgeo` è cambiato in v0.2.3 senza bump. La guardia aggiunta
protegge il test, non altri eventuali lettori. Rimedio: incrementare la versione,
o meglio scrivere il passo del record nell'header — entrambi invalidano la cache
in `out/` (oltre 1200 file). Il server non parsa i record, usa i `.kgeo` solo
come nomi di file, quindi non è esposto.

---

## 6. Discontinuità del verso di rotazione del disco a `a = 0`

`s = (a < 0) ? 1 : −1` fa cadere `a = 0` nel ramo positivo. Misurato: offset del
lato brillante −89.2 a `a=0` contro +87.2 a `a=−0.2`. Legittimo — in
Schwarzschild non esiste un verso preferito e bisogna sceglierne uno — ma se si
anima lo spin attraverso lo zero il disco si ribalta di colpo.

---

## 7. ~~Modalità bundle senza copertura di test~~ — RISOLTO in v0.2.16 (`5e16819`)

`tests/ray_bundle_regression.cpp`, test ctest `kerrtrace.ray_bundle`, 5.9 s.
Tre render 320×180 (single-ray, bundle, bundle con `--max-steps 200000`) e otto
controlli sui `.kgeo`. Soglie calibrate su binari costruiti da entrambi i lati
di `dfde570`: contro un build pre-fix fallisce esattamente i tre controlli sul
redshift e passa gli altri cinque.

Terminazione coperta due volte: tetto di 300 s per render e **identità byte a
byte** dell'output a `--max-steps` 60000 e 200000 — un raggio che si ferma su un
evento vero non risente di un cap più alto, uno che esaurisce i passi sì.

---

## 8. Pulizia

- `frontend/package.json`: modifica `ng serve --host 0.0.0.0` mai committata,
  mai decisa. Espone il dev server alla LAN.
- `.gitignore`: `build*/`, `out/`, `*.tmp` sono untracked ma **non ignorati** —
  un `git add -A` distratto se li porta dentro.
- `CMakeLists.txt.tmp` e sei directory `build_*/` da rimuovere.
- Cinque branch integrati su origin, cancellabili.
- `CLAUDE.md` afferma che `tracer.metal` è caricato da `build/tracer.metal`:
  **è falso**, il bridge carica `exeDir/../gpu/metal/tracer.metal`, cioè il
  sorgente.

---

## Trappole incontrate, da non ripetere

**zsh non fa word-splitting.** `ARGS="--a --b"` poi `cmd $ARGS` passa **un solo
argomento**. Mi ha ingannato tre volte, producendo render coi default e facendomi
annunciare due regressioni inesistenti. Usare array o flag espliciti; verificare
con `set -- $VAR; echo $#`.

**Misurare sul disco, non sul frame.** Le statistiche sull'intera immagine sono
dominate dallo sfondo stellato. Correlare sempre col `.kgeo` filtrando
`outcome == 1`.

**`grep "fallback"` matcha la riga `Mode:`**, che contiene
`elliptic-fallback-black=off`. Ha prodotto falsi "nero" su render riusciti.
Verificare la luminanza reale del PNG.

**Lo shader Metal si carica dal sorgente**, non dalla copia in build. Patchare
`build_cpu/tracer.metal` non ha alcun effetto.

**Il bisect va fatto per davvero.** Testare due estremi e chiamarlo bisect ha
prodotto un'attribuzione sbagliata (`d3f28da` invece di `1587f19`), con 16
commit non esaminati in mezzo.
