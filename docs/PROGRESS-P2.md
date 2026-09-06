# P2 — Il fascio sullo sfondo

Diario di lavoro. Piano di riferimento: `PLAN-2026-09-05.md`, voce P2.

---

## Correzione al piano, prima di eseguirlo

P2 era scritto così:

> 1. Spostare la modulazione [della magnificazione] sul campionamento dello sfondo.
> 2. Raggio iniziale del fascio a 2× il passo fra pixel e gaussiana troncata.
> 3. Criterio: una stella che attraversa la griglia dei pixel varia di luminosità ≤ 2%.

**Il punto 1 non si applica al nostro sfondo.** A.3.1 tratta le stelle come
sorgenti **puntiformi**: *«If we were to treat these rays as infinitely thin,
there would be zero probability of any ray intersecting a star»*. È da lì che
nasce il fattore di magnificazione — serve perché un raggio infinitamente sottile
mancherebbe sempre una stella puntiforme.

Il nostro sfondo non è un catalogo di stelle: `BackgroundImage::sample()` è una
**lookup bilineare su una bitmap equirettangolare** (`main.cpp:231`). Ogni raggio
colpisce sempre qualcosa. È il caso della **sorgente estesa**, per cui la stessa
A.3.1 prescrive tutt'altro: *«we minimise moiré artefacts by adapting the
resampling filter according to the shape of the beam»*.

Quindi P2 si divide:

- **P2a — filtrare la lookup dello sfondo sull'impronta del fascio.** Applicabile
  ora, ed è dove sta l'aliasing misurato.
- **P2b — la magnificazione come moltiplicatore.** Ha senso **solo** se in futuro
  si aggiungono stelle puntiformi vere da catalogo. Anche il criterio del punto 3
  non è misurabile finché le stelle sono cotte in una bitmap: non si può
  osservare «una stella che attraversa la griglia» se la stella *è* la griglia.

Nota collegata: il residuo dello 0.3% sull'impronta (vedi `PROGRESS-P1.md`,
passo 4) era stato segnato come «da spiegare prima di P2». Riguarda **P2b**, dove
l'impronta entrerebbe direttamente nella luminosità. Per P2a, che la usa come
larghezza di filtro, resta irrilevante.

---

## P2a — Fatto (v0.2.26)

### Impronta sulla sfera celeste

Stessa costruzione del disco, con `r = r_escape` come superficie di
attraversamento invece di `θ = π/2`. I raggi vicini raggiungono la superficie a
parametro affine diverso, quindi il moto lungo il raggio va proiettato via:

```
δr + r'·δλ = 0   ⟹   δλ = −δr/r'
δθ_sky = δθ + θ'·δλ
δφ_sky = δφ + φ'·δλ
```

I quattro campi `fp_*` del record sono riusati: i due casi sono **esclusivi** —
o il raggio colpisce il disco, e portano `(δr, δφ)` nel piano equatoriale, o
sfugge, e portano `(δθ, δφ)` sulla sfera celeste. Nessun allargamento del record,
`KGEO_VERSION` resta 3.

### Risultato

480×270, a=0.9, θ=82°, contro il riferimento a 16 spp:

| | RMSE | interno | bordi | **sfondo** |
|---|---|---|---|---|
| riferimento 16 spp | — | 20.85 | 76.88 | **31.03** |
| single-ray | 11.59 | 27.48 | 156.98 | 64.92 |
| bundle dopo P1 | 11.96 | 23.70 | 82.41 | 64.50 |
| **bundle + filtro sfondo** | **6.77** | 23.70 | 81.32 | **33.88** |

Sfondo: eccesso sul riferimento da **+108% a +9%**. RMSE globale **quasi
dimezzato**, da 11.96 a 6.77, contro l'11.59 del single-ray — per la prima volta
il fascio vale il suo costo.

Interno e bordi non si muovono, come atteso: il filtro tocca solo i raggi che
sfuggono.

### Cosa restava

- Il caso misto: un pixel di bordo con copertura parziale portava nei campi
  `fp_*` l'impronta **del disco**, quindi lo sfondo composto dietro restava
  campionato senza filtro. → affrontato in v0.2.27, sotto.
- **P2b** resta non applicabile finché non esistono stelle puntiformi.
- Il costo: 3.92 s contro 3.75 s, cioè +4%. Il filtro dello sfondo è quasi
  gratis perché non traccia raggi, campiona solo la texture più volte.

---

## P2b-bis — I pixel misti del bordo (v0.2.27)

### Quanto pesavano

Non 365 come avevo detto — quello era il conteggio dei pixel *molto* più scuri.
I pixel di disco a copertura parziale che compongono sfondo dietro sono **1630**,
l'1.26% del fotogramma, ma valevano il **16.2%** dell'errore quadratico totale
(RMSE locale 24.32 contro 6.77 globale). Da qui la decisione di allargare il
record invece di lasciar perdere.

### Fatto

Campi `sky_*` separati da `fp_*`: un pixel di bordo ne ha bisogno di **entrambi**
— l'impronta sul disco per ombreggiare e misurare la copertura, quella sul cielo
per filtrare lo sfondo che compone dietro di sé. Record da 48 a 64 byte,
`KGEO_VERSION` a **4**. La continuazione oltre il disco ora registra anche la sua
impronta di cielo, e il resampling di bordo la media sui sotto-fasci che sfuggono.

| | RMSE totale | RMSE sui misti | contributo |
|---|---|---|---|
| prima | 6.769 | 24.32 | 16.2% |
| dopo | **6.728** | **22.95** | 14.6% |

### Il guadagno è reale ma piccolo, e so perché

Avevo stimato che azzerare l'errore su quei pixel avrebbe portato l'RMSE da 6.769
a 6.195. Ne abbiamo recuperato **6.728**, cioè circa il **7%** del recuperabile.
Il campionamento dello sfondo non era la causa principale del loro errore.

Infittire la griglia di sotto-fasci **non aiuta** — peggiora:

| griglia | RMSE totale | RMSE sui misti |
|---|---|---|
| 3×3 | **6.728** | **22.95** |
| 5×5 | 6.763 | 23.82 |
| 7×7 | 6.764 | 23.84 |

Non è rumore di campionamento: è **sistematico**. La causa è strutturale nel
modo in cui compongo. Per un pixel di bordo faccio

```
colore = copertura · disco(r̄, φ̄) + (1−copertura) · sfondo(θ̄, φ̄)
```

cioè **medio la geometria e poi ombreggio una volta sola**. Il riferimento
ombreggia ogni sotto-campione e poi media i colori. Le palette non sono lineari,
quindi le due cose non coincidono, e infittire la griglia rende la geometria
media più precisa senza avvicinare il risultato.

Per chiudere davvero servirebbe ombreggiare in fase 1, cioè rompere la
separazione in due fasi su cui è costruito il renderer (e il formato `.kgeo`).
Non lo faccio adesso: è una scelta d'architettura, non un ritocco. La griglia
resta a 3, che è la migliore delle tre misurate.

---

## Cronologia

| versione | commit | contenuto |
|---|---|---|
| v0.2.26 | `b816dc5` → `523c8e8` | impronta sulla sfera celeste, filtro della lookup di sfondo |
| v0.2.27 | | campi `sky_*` separati, `KGEO_VERSION`=4, sfondo filtrato anche dietro il bordo |
