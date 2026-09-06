# P1 — Dare al fascio un lavoro vero

Diario di lavoro. Aggiornato man mano; l'ultima sezione è sempre lo stato attuale.

Piano di riferimento: `PLAN-2026-09-05.md`, voce P1.

---

## Stato: P1 sostanzialmente chiuso sul disco, aperto sullo sfondo

| passo | stato |
|---|---|
| 1. Misurare quanto vale il fascio oggi | fatto |
| 2. Capire perché `\|det J\|` è sbagliato | fatto |
| 3. Ricostruire il campo di Jacobi | **fatto, v0.2.21** |
| 4. Validare contro differenze finite | **fatto, corr +0.9996** |
| 5. Esportare l'impronta (semiassi, orientamento) | **fatto, v0.2.22** |
| 6. Mediare l'emissione sull'impronta | **fatto, ma sul bersaglio sbagliato** |
| 7. Verificare contro il riferimento 16 spp | **fatto — vedi passo 7** |
| 8. Copertura analitica al bordo | **fatto, v0.2.23** |
| 9. Resampling di bordo (A.3.1) | **fatto, v0.2.24** |
| 10. Trasparenza del disco nel fascio | **fatto, v0.2.25** |

**Dove siamo**, RMSE contro il riferimento a 16 spp: single-ray 11.59, bundle
**11.96** (era 14.61 all'inizio di P1). Dei 3546 pixel ancora più scuri di 25
livelli, **3181 sono sfondo** — il single-ray ne ha 3689, quindi sul disco il
bundle è ormai davanti e il residuo è quasi tutto **P2**.

---

## Passo 1 — Il fascio non filtrava niente

480×270, a=0.9, θ=82°, riferimento a 16 spp:

| | RMSE vs riferimento | energia HF | eccesso |
|---|---|---|---|
| riferimento 16 spp | — | 29.31 | — |
| single-ray | 11.59 | 58.22 | +98.7% |
| bundle | **14.61** | 58.29 | **+98.9%** |

Stesso aliasing del single-ray, RMSE peggiore, 6× il costo. Sono i numeri da
battere al passo 7.

---

## Passo 2 — `|det J|` era anticorrelato con il vero

Referenza: Jacobiana per differenze finite `∂(r,φ)/∂(α,β)` dalle mappe
single-ray, su 16 279 pixel con tutti e quattro i vicini sul disco, regione
liscia `6 < r < 11`.

Rapporto bundle/vero: mediana 0.00745, spread p90/p10 **37×**, correlazione
log-log **−0.416**. Un campo corretto darebbe rapporto costante, spread ~1×,
correlazione +1. Non era una questione di scala.

---

## Passo 3 — Tre difetti, non separabili

**a. L'equazione variazionale era omogenea.** Stato `z = (r, θ, p_r, p_θ)` con
`p_t`, `p_φ` trattati come fissi. Sono conservati *lungo* un raggio ma
**differiscono fra i raggi** del fascio: sono parametri del flusso, e `δp_t`,
`δp_φ` sono costanti non nulle della famiglia. La deviazione vera è
inomogenea:

```
d(δz)/dλ = (∂f/∂z)·δz + (∂f/∂p_t)·δp_t + (∂f/∂p_φ)·δp_φ
```

Il forzamento mancava dal primo passo in poi.

**b. `δφ` non era tracciato.** L'impronta vive nel piano `(δr, r·δφ)`; il codice
sostituiva `δθ`, che a `θ=π/2` è normale al disco e non descrive area in esso.

**c. La deviazione era letta a parametro affine uguale.** I raggi vicini **non
attraversano l'equatore allo stesso λ**, quindi `W(λ)` contiene moto lungo il
raggio. L'impronta è la deviazione sulla **superficie di attraversamento**:
imporre che anche il vicino stia su `θ=π/2` fissa il suo scarto di λ,

```
δθ + θ'·δλ = 0   ⟹   δλ = −δθ/θ'
δr_sup   = δr   + r'·δλ
δφ_sup   = δφ   + φ'·δλ
```

È la stessa proiezione trasversa che DNGR ottiene per costruzione lavorando nel
piano ortogonale al raggio (A.2.3: *«the equation of geodesic deviation for the
separation vector whose spatial part in a FIDO reference frame is the vector
Y»*). Per un disco sottile la superficie giusta è il disco, che è quel che A.6
campiona.

**Semplificazione ottenuta**: `H = ½g^{μν}p_μp_ν` è esattamente quadratico nei
momenti, quindi il blocco momento-momento **è** la metrica inversa e i blocchi
misti sono sue derivate prime contratte con `p`. Nessuna Hessiana di `H`:
`hessian_H` e le sue 9 valutazioni sono state eliminate.

---

## Passo 4 — Validazione in due stadi

Una statistica sull'immagine intera non localizza il guasto. Prima l'integrazione
variazionale **isolata**, contro differenze finite di due raggi vicini sulla
stessa griglia di λ, senza logica del disco (`scratchpad/p1/devcheck.cpp`):

```
λ=40:   dr/dα  = +36.89092   contro  +36.89092
        dφ/dα  =  -6.55077   contro   -6.55077
```

Poi l'impronta completa:

| | rapporto med | spread p90/p10 | corr log-log |
|---|---|---|---|
| originale | 0.00745 | 37.06× | −0.416 |
| +forzamento, +δφ | 0.11239 | 13.73× | −0.625 |
| +proiezione | **1.00303** | **1.008×** | **+0.9996** |

La riga di mezzo conta: sistemare il forzamento **peggiorava** la correlazione.
Le tre correzioni non sono separabili, e lo stato intermedio non è un posto
sensato dove fermarsi.

### Residuo dello 0.3%: non è quello che avevo detto

Nel messaggio di `dafbad0` avevo scritto che era troncamento della referenza.
**Falso.** Raffinando la referenza il fattore atteso per un errore O(h²) sarebbe
4×; misurato 0.95×:

| risoluzione | passo | scarto da 1 |
|---|---|---|
| 480×270 | 1.64e-3 | 0.303% |
| 960×540 | 8.19e-4 | 0.318% |

Dipende dalla tolleranza ma **satura**: 1.235% a 1e-7, 0.303% a 1e-11, 0.303% a
1e-13. Fondo sistematico non spiegato, indipendente da passo e risoluzione.
Candidati non esclusi: i passi `hr`/`ht` delle derivate della metrica in
`bundle_ops`; l'interpolazione lineare di `W` all'attraversamento contro quella
di Hermite di `r`; il passo `eps` in `init_bundle`.

Irrilevante per scegliere la larghezza di un filtro. **Da spiegare prima di
P2**, dove la magnificazione entra direttamente nella luminosità delle stelle.

---

## Passo 5–6 — Cosa serve fare, dal paper

`§2.2 (iii)`: *«add up the spectrum and intensity of all the light emitted from
within that ellipse»*. L'ellisse è un **dominio d'integrazione**.

`A.3.1`, per le sorgenti estese: *«we minimise moiré artefacts by adapting the
resampling filter according to the shape of the beam»*. Più i dettagli
operativi: raggio iniziale del fascio pari a **due volte** il passo fra pixel,
modulazione con **gaussiana troncata**, tarata per uno sfarfallio ≤ 2%.

`A.3.1` avverte anche: *«This result assumes the final size of the ellipse is
small and the shape of the beam does not change significantly between adjacent
pixels. In extreme cases these assumptions can break down»* — con rimedio
tracciare più fasci per pixel. Da tenere presente vicino all'anello di fotoni.

### Conseguenze concrete

1. `trace_bundle` deve restituire i **due vettori di bordo** dell'impronta nel
   piano del disco, in unità di pixel: `e_α = (δr_α, δφ_α)·Δα` e
   `e_β = (δr_β, δφ_β)·Δβ`, non solo il determinante.
2. `GeoPixel` deve trasportarli → il record cresce → **`KGEO_VERSION` va portato
   a 2** (che è anche il punto 5 di `OPEN-ISSUES.md`, finora mai fatto).
3. La media va fatta con pesi gaussiani troncati sull'impronta, campionando la
   palette in K punti. Le palette sono già funzioni di `(r, φ)`, quindi si può
   mediare **fuori** da esse senza toccarne il codice.
4. La media avviene sul valore restituito dalla palette, cioè dopo il
   tonemapping. È lo stesso dominio in cui opera il riferimento `--spp`, quindi
   il confronto del passo 7 resta coerente.

---

## Passo 7 — La misura dice che stavo filtrando il posto sbagliato

Filtro implementato: impronta esportata in `GeoPixel` (record da 28 a 44 byte,
`KGEO_VERSION` finalmente portato a **2**), pattern di campioni a anelli
concentrici con pesi gaussiani troncati, media fuori dalle palette così nessuna
di esse è stata toccata. Flag `--bundle-filter` / `--no-bundle-filter`,
`--bundle-filter-rings`, `--bundle-filter-sigma`.

Effetto globale: quasi nullo. HF da 58.26 a 57.58 (eccesso +98.8% → +96.5%),
RMSE da 14.601 a 14.521.

**Perché**: scomponendo l'energia HF per regione si vede che l'aliasing non è
nella texture del disco.

| regione | riferimento 16 spp | bundle senza filtro | bundle con filtro |
|---|---|---|---|
| interno disco (39 188 px) | 19.93 | 19.23 | 18.24 |
| **bordi disco** (1 536 px) | 80.24 | **159.92** | 145.49 |
| **sfondo** (88 876 px) | 31.14 | **66.16** | 66.01 |

L'interno del disco era **già** al livello del riferimento (19.23 contro 19.93):
la texture non aliasava. Il mio filtro l'ha portato a 18.24, cioè leggermente
**sotto** il riferimento — sovra-liscia dell'8%.

L'aliasing vero sta in due posti, entrambi **doppi** rispetto al riferimento:

1. **I bordi del disco** — è aliasing di *copertura*: se il pixel cade dentro o
   fuori il disco è una decisione binaria. Nessun filtro sulla texture può
   risolverlo; serve la **frazione dell'impronta** che effettivamente cade sul
   disco.
2. **Lo sfondo stellato** — è il campionamento del background, cioè esattamente
   ciò di cui parla A.3.1 per le sorgenti non risolte. È territorio di **P2**.

Dimensioni dell'impronta misurate, per contesto: `|δr|` mediano 0.076 M e
0.152 M sui due assi, arco `r·δφ` mediano 0.031 M e 0.065 M. Sono piccole
rispetto alla scala su cui varia la texture, il che conferma che lì non c'era
niente da filtrare.

### Conseguenza sul piano

Il macchinario è corretto e validato, ma applicato al bersaglio sbagliato. Il
seguito di P1 non è «mediare meglio la texture», è:

- **P1b — antialiasing di copertura al bordo del disco**: usare l'impronta per
  stimare la frazione di pixel coperta dal disco e comporre con ciò che sta
  dietro, invece della decisione binaria attuale. → **fatto, v0.2.23**
- **P2** assorbe il filtraggio dello sfondo.

---

## Passo 8 — Copertura al bordo (v0.2.23)

Misura preliminare, per non sbagliare bersaglio due volte: dei 1536 pixel di
bordo, **tutti** confinano con lo sfondo e **nessuno** con l'orizzonte; 987 sono
contro `r_out`, 459 contro `r_isco`, solo 90 sono silhouette. Quindi il 94% è
risolvibile analiticamente contro i **bordi radiali**, senza tracciare un solo
raggio in più. E l'impronta scavalca il bordo per costruzione: estensione
radiale mediana 0.243 M contro una distanza mediana da `r_out` di 0.118 M.

Implementato: la copertura è la frazione dell'impronta con
`r_in ≤ r ≤ r_out`, campionata sull'ellisse; entra come `alpha` di
composizione. E, poiché la parte scoperta del pixel vede quel che sta **dietro**,
il raggio viene proseguito oltre il disco — altrimenti il bordo comporrebbe su
nero e si scurirebbe invece di sfumare. Lo pagano solo i pixel di bordo, l'1.2%
del fotogramma. 834 pixel hanno copertura < 0.999, mediana 0.796.

Record da 44 a 48 byte, `KGEO_VERSION` a **3**.

### Risultato

| | RMSE | interno | **bordi** | sfondo |
|---|---|---|---|---|
| riferimento 16 spp | — | 19.93 | **80.24** | 31.14 |
| single-ray | 11.59 | 27.46 | 152.22 | 65.10 |
| bundle + filtro texture | 14.52 | 18.24 | 145.49 | 66.01 |
| **bundle + copertura** | **14.06** | 18.10 | **122.61** | 65.52 |

Sui bordi: **−16%**, da 145.49 a 122.61. Reale e misurato, ma il riferimento sta
a 80.24: **non è chiuso**.

### Perché resta il divario, e cosa manca

Il trattamento è **unilaterale**. Un pixel il cui centro cade appena dentro il
disco ottiene copertura < 1 e sfuma; uno il cui centro cade appena fuori non ha
impronta — in modalità bundle un raggio che non colpisce il disco non registra
niente — quindi resta un taglio netto. Il bordo si ammorbidisce solo dal lato
interno ed è polarizzato verso l'interno di mezza impronta. Il riferimento a 16
spp ha invece un bordo simmetrico.

Rimedi possibili, in ordine di costo: registrare un'impronta anche al passaggio
più vicino all'equatore per i raggi che mancano il disco; oppure
supercampionare i soli pixel di bordo (~3% del fotogramma), che è il rimedio che
A.3.1 stesso indica per i casi estremi (*«we can trace multiple beams per pixel
and resample»*). → **fatto, v0.2.24, passo 9**

---

## Passo 9 — Il rimedio di A.3.1 (v0.2.24)

> *«In extreme cases these assumptions can break down, leading to a distortion in
> the shape of a star's image, flickering, and aliasing artefacts. In these cases
> we can trace multiple beams per pixel and resample.»* — A.3.1

Seconda passata dopo la fase 1: un pixel viene ricampionato **solo** se il suo
vicinato a quattro disagrees sul fatto che il disco ci sia. Su 480×270 sono
**3048 pixel, il 2.35% del fotogramma**, con una griglia 3×3 di fasci ciascuno
(`--bundle-edge-grid`, default 3).

Il risultato di ogni pixel di bordo viene ricostruito dai sotto-fasci:
copertura = frazione di fasci che colpiscono il disco pesata per la copertura
parziale di ciascuno; `r` e `redshift` mediati sui fasci che colpiscono;
`theta_esc`/`phi_esc` sui fasci che sfuggono. **Gli azimut sono mediati come
vettori unitari**, non aritmeticamente: un pixel a cavallo del taglio a `φ=π`
darebbe altrimenti una media priva di senso.

Questo rende la copertura **bilaterale**: la misura da entrambi i lati invece di
inferirla da uno solo, che era il limite del passo 8. 2239 pixel escono con
copertura parziale (mediana 0.612, minimo 0.002) contro gli 834 di prima.

### Risultato

Maschere ricalcolate sul buffer finale, quindi confrontabili riga per riga:

| | RMSE | interno | **bordi** | sfondo |
|---|---|---|---|---|
| riferimento 16 spp | — | 23.74 | **63.32** | 30.69 |
| single-ray | 11.59 | 35.17 | 127.83 | 64.59 |
| copertura analitica (passo 8) | 14.06 | 26.48 | 122.18 | 64.50 |
| **+ resampling A.3.1** | **13.80** | **24.13** | **79.51** | 64.47 |

Bordi: eccesso sul riferimento da **+93% a +26%**. Interno: 24.13 contro 23.74,
cioè **allineato** (+1.6%). Lo sfondo non si muove, ed è atteso: è P2.

### Costo

Trace da 2.07 s a 3.82 s sullo stesso fotogramma, cioè **+85%** di tempo per
+21% di raggi. Lo scarto è perché i pixel di bordo sono i più cari — stanno
vicino all'anello di fotoni, dove il paper stesso avverte che i render sono
molto più lenti — e perché la seconda passata parallelizza peggio (282% di CPU
contro 586% della prima). Riducibile abbassando `--bundle-edge-grid` a 2.

### Nota sull'RMSE globale

Resta ~14 perché è **dominato dallo sfondo**: 88 876 pixel di sfondo contro
40 724 di disco, e lo sfondo sta a 65.5 contro 31.1 del riferimento. Finché non
si filtra il campionamento del background (P2) l'RMSE globale non si muove, e
non è la metrica giusta per giudicare il lavoro sul disco.

---

## Cronologia

| versione | commit | contenuto |
|---|---|---|
| v0.2.21 | `dafbad0` → `2dce7e0` | ricostruzione del campo di Jacobi (passi 3–4) |
| — | — | correzione dell'attribuzione del residuo 0.3% |
| v0.2.22 | `861be72` → `deccc8a` | impronta esportata, `KGEO_VERSION`=2, filtro sull'emissione |
| v0.2.23 | `ab36883` → `f92aaf3` | copertura al bordo, continuazione dietro il disco, `KGEO_VERSION`=3 |
| v0.2.24 | `7dec149` → `e5ebc73` | resampling dei pixel di bordo (A.3.1), copertura bilaterale |
| v0.2.25 | `a25960b` → `6dd00f0` | trasparenza del disco onorata anche dal fascio |


---

## Passo 10 — Il bordo nero: difetto preesistente, non introdotto (v0.2.25)

Segnalato guardando il render. **Non l'avevo introdotto io**: il primo render
bundle della sessione (`20260905-232400`, v0.2.20, prima di qualunque modifica a
P1) ha già luminanza **0.0** nel punto incriminato.

**Causa**: `trace_bundle` non applicava la trasparenza parziale del disco che i
tracciatori single-ray onorano da sempre. Le palette Interstellar sfumano il
disco verso il bordo e trattano tutto ciò che sta sotto
`--disk-interstellar-edge-transparency` come un buco da attraversare; la palette
stratified ha vuoti veri fra le piastrelle. Il fascio si fermava alla prima
intersezione con l'equatore comunque, e rendeva la maschera sbiadita come nero.

**Prova**: rendendo il single-ray con `--disk-interstellar-edge-transparency 0`
si ottengono **esattamente** i numeri del bundle:

| x | single normale | single trasp=0 | bundle |
|---|---|---|---|
| 398 | 170.3 | 3.7 | 3.7 |
| 401 | 195.0 | 0.0 | 0.0 |
| 405 | 192.0 | 0.0 | 0.0 |

**Fix**: il predicato di trasparenza viaggia verso `trace_bundle` come callback
(`DiskTransparency`), così le due strade concordano su cosa conti come impatto.
Vive in `main.cpp` con le palette, senza trascinare `ColorParams` dentro
`ray_bundle.hpp`.

| x | 16 spp | single | bundle prima | bundle dopo |
|---|---|---|---|---|
| 398 | 148.0 | 170.3 | 3.7 | **119.7** |
| 401 | 195.3 | 195.0 | 0.0 | **194.3** |
| 405 | 191.3 | 192.0 | 0.0 | **192.0** |

RMSE contro il riferimento da **14.60 a 11.96**, contro l'11.59 del single-ray.
Pixel più scuri di 25 livelli: da 4256 a 3546, e di questi **3181 sono sfondo**
(aliasing dello starfield, cioè P2) — il single-ray ne ha 3689, quindi su quel
fronte il bundle è ora migliore. Restano 365 pixel di disco a `r≈11.6` con
copertura 0.662: bordo esterno parzialmente coperto.

**Nota**: la linea scura dove il disco incontra l'ombra c'è **anche nel
riferimento a 16 spp**. Quella è fisica — bordo interno fortemente spostato
verso il rosso — e non va tolta.
