# Qualità GPU vs CPU a 4K, e l'ibrido "bordi in FP64" — 17 ottobre 2026

Scena: Kerr a = 0,998, θ = 84°, r_obs = 60, FOV 50°, disco esterno 20,
`--disk-interstellar`, 3840×2160, un raggio per pixel, cielo panoramico
senza cucitura. Riferimento: CPU, tolleranza 1e-9. Macchina: M2 Max.

## Riferimento affidabile

CPU 1e-7 contro CPU 1e-9: 0,002 % di pixel con errore > 16. Il riferimento
è convergente; quello che segue è errore della GPU, non rumore d'integrazione.

## Tempi

| backend | tempo |
|---|---:|
| CPU, 12 core | 70 s |
| Metal, tessere di default (512×2 a 4K) | 398 s |
| Metal, `KERR_METAL_TILE_ROWS=64 KERR_METAL_TILE_COLS=4096` | **40 s** |

Le tessere grandi danno un'immagine identica bit per bit e nessuna tessera
fallita. Da allora sono il default (tessera ≈ 256 000 pixel, righe multiple di 16):

| risoluzione | CPU | Metal, default nuovo |
|---|---:|---:|
| 640×360 | 1,9 s | 3,4 s |
| 1600×900 | 12,1 s | 9,9 s |
| 3840×2160 | 69 s | **38 s** |

Provati e scartati perché senza guadagno misurabile: tre command buffer in volo
sulla stessa coda (Metal li esegue in ordine) e tre code separate (nessuna
sovrapposizione: una tessera così grande occupa già tutta la GPU). A 640×360
la CPU resta più veloce: il frame è troppo piccolo per riempire la GPU, e pochi
raggi lunghi vicini all'orbita fotonica pesano molto su un singolo thread GPU.

Perché le vecchie tessere erano così lente: invio sincrono di 8 640 tessere da
512×2 pixel, con la griglia arrotondata a 16 righe, quindi 7 thread su 8 inattivi.

## Dove sbaglia la GPU

Metriche: errore massimo per canale contro il riferimento, e "grana" = energia
del laplaciano rispetto al riferimento (1,00 = come la CPU).

| frame | ritracciati su CPU | err > 48 | grana |
|---|---:|---:|---:|
| GPU, carta KS (default) | 0 % | 3,04 % | 2,52× |
| GPU KS + ibrido, bordi T=40 | 23 % | 0,17 % | 1,57× |
| GPU KS + ibrido, bordi T=24 R=2 | 48 % | 0,01 % | 1,27× |
| GPU, carta BL | 0 % | 3,46 % | 1,07× |
| GPU BL + ibrido, bordi T=24 R=2 | 22 % | 0,83 % | 0,95× |

"Ibrido" è simulato sostituendo nell'immagine GPU i pixel selezionati con
quelli CPU: è esattamente ciò che produrrebbe il ritracciamento in FP64.
Selezione: pixel con salto di colore > T verso un vicino, dilatati di R.

1. **Il cielo GPU in KS è sgranato**: le scie delle stelle stirate dalla lente
   diventano puntini, perché pixel vicini ricevono direzioni di fuga
   rumorose. In BL la GPU dà scie lisce come la CPU. La differenza fra le due
   carte sulla GPU è il calcolo delle forze: BL usa derivate analitiche (dal
   9 settembre), KS differenze finite FP32 con passo relativo 1e-5.
2. **L'errore non sta sui bordi.** In KS l'ibrido raggiunge la qualità CPU
   solo ritracciando quasi metà dei pixel: costa più della CPU da sola. In BL
   l'errore residuo è uno spostamento liscio delle scie (tolleranza minima
   FP32 1e-5 contro 1e-9) e non si concentra sui bordi.
3. **Le chiazze sul disco** sono identiche in KS e in BL: differenza di
   ombreggiatura Interstellar fra CPU e GPU, non di traiettoria.
4. La convenzione di campionamento della texture era diversa fra CPU e GPU
   (corretta in c793610), ma il suo peso su queste metriche è trascurabile.

## Conseguenze

- L'ibrido "bordi in FP64" non è la strada per la parità: non è stato
  implementato.
- Le leve sono: derivate analitiche anche per le forze KS sulla GPU (come per
  BL), tessere più grandi nel dispatch Metal, e allineare l'ombreggiatura
  Interstellar della GPU a quella CPU.

## Aggiornamento 19 ottobre: forze KS analitiche sulla GPU

`geodesic_rhs_KS` calcola ora −∂H/∂x in forma chiusa (H = ½(η pp − F S²),
∂r/∂x dalla definizione implicita di r) invece che con differenze centrali di
due Hamiltoniane FP32. `kerrtrace.metal_rhs` confronta la funzione GPU con
differenze a cinque punti in double su 720 stati KS (a da −0,9 a 0,95, r da
1,5 a 40, fuori dal cono nullo inclusi).

| | prima | dopo |
|---|---:|---:|
| errore forze KS (massimo, scalato) | 9,2·10⁻² | 4,1·10⁻⁷ |
| grana del cielo 4K vs CPU | 2,52× | 1,01× |
| pixel 4K con errore > 48 | 3,04 % | 0,32 % |
| pixel 4K con errore > 16 | 12,4 % | 2,4 % |
| tempo 4K Interstellar | 38 s | 16,9 s (CPU 69 s) |
| `metal_contour` ks-custom/upper | max 2 px, 11 oltre 1 px | max 1 px, 0 oltre |

`metal_contour` passa per la prima volta da settembre: tutta la suite è verde.
L'errore residuo sul disco (chiazze uguali in KS e BL) è ombreggiatura
Interstellar, non traiettoria. Gli hash Metal di `render_fingerprint` nei casi
KS cambiano: vanno registrati di nuovo.
