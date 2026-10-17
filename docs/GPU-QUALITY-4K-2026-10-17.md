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
fallita. Le tessere di default a 4K tengono la GPU quasi ferma (invio
sincrono, griglia arrotondata a 16 righe su tessere da 2).

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
