# P1 — Dare al fascio un lavoro vero

Diario di lavoro. Aggiornato man mano; l'ultima sezione è sempre lo stato attuale.

Piano di riferimento: `PLAN-2026-09-05.md`, voce P1.

---

## Stato: metà fatta

| passo | stato |
|---|---|
| 1. Misurare quanto vale il fascio oggi | fatto |
| 2. Capire perché `\|det J\|` è sbagliato | fatto |
| 3. Ricostruire il campo di Jacobi | **fatto, v0.2.21** |
| 4. Validare contro differenze finite | **fatto, corr +0.9996** |
| 5. Esportare l'impronta (semiassi, orientamento) | **fatto, v0.2.22** |
| 6. Mediare l'emissione sull'impronta | **fatto, ma sul bersaglio sbagliato** |
| 7. Verificare contro il riferimento 16 spp | **fatto — vedi passo 7** |

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
  dietro, invece della decisione binaria attuale.
- **P2** assorbe il filtraggio dello sfondo.

---

## Cronologia

| versione | commit | contenuto |
|---|---|---|
| v0.2.21 | `dafbad0` | ricostruzione del campo di Jacobi (passi 3–4) |
| — | — | correzione dell'attribuzione del residuo 0.3% |
| v0.2.22 | | impronta esportata, `KGEO_VERSION`=2, filtro sull'emissione |
