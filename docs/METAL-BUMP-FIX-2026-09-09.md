# Correzione del gradino Metal BL — 9 settembre 2026

Il punto cerchiato sul bordo esterno sinistro è corretto. Il riferimento
prima della modifica è `004bab8`, già pubblicato sul branch
`codex/audit-physics-backend-fixes`; il renderer di quel commit coincide con
`16677fd`. I frame precedenti sono conservati in `out/contour-bump/`.

## Causa e modifica

La forza hamiltoniana BL veniva calcolata sottraendo due Hamiltoniane FP32,
con incrementi di `1e-6` radianti in θ e circa `1e-5 r` in r. La cancellazione
numerica produceva derivate imprecise e deformava la traiettoria: il bordo
Metal restava nella stessa colonna per cinque righe, dove il riferimento
CPU seguiva una curva regolare.

`geodesic_rhs` ora usa le derivate analitiche della stessa Hamiltoniana KNdS.
Non sono cambiati il modello del disco, il Doppler o la tolleranza richiesta.
La forma utilizzata, fuori dalla guardia dell'asse BL, è:

```
A = r² + a²
B = A p_t + a p_phi
C = a sin(theta) p_t + p_phi / sin(theta)
N = Delta_r p_r² + Delta_theta p_theta²
    + Xi² (C²/Delta_theta - B²/Delta_r)
H = N / (2 Sigma)
```

Le derivate includono il termine `N dSigma`: non si assume `H=0` negli
stadi intermedi di Runge–Kutta. È conservata la guardia già presente in
`gUU` sull'asse BL, differenziando coerentemente anche quel ramo.

Le forze più accurate hanno fatto emergere un secondo problema: un passo
RK troppo grande poteva saltare la stretta regione di inversione vicino
all'asse, producendo una colonna di pixel mancanti. Per momento azimutale
non trascurabile il passo viene ora limitato durante l'avvicinamento al polo:
`h <= 0,25 |sin(theta)| / |dtheta/dlambda|`, rispettando il minimo esistente.
Il limite è applicato prima di salvare il passo usato dall'interpolazione.
È un limite sulla traiettoria, indipendente da pixel o risoluzione.

È stato inoltre eliminato il ciclo BL duplicato del renderer single-ray:
ignorava la selezione Hermite e interpolava sempre linearmente in cos(θ).
Single-ray e fallback BL riusano ora il tracciatore condiviso, che rispetta
la modalità di intersezione richiesta e l'ordine degli eventi disco,
orizzonte e fuga.

## Verifica causale

Il nuovo `kerrtrace.metal_rhs` chiama la funzione dello shader sulla GPU,
con 288 stati anche fuori dalla superficie nulla. Comprende Schwarzschild,
Kerr e parametri Q/Λ non nulli, spin negativo, raggi 2,5…40M e angoli
0,03…3,1 radianti. Il riferimento usa la metrica CPU in doppia precisione
e differenze a cinque punti; non ricopia le nuove formule analitiche.

- Shader precedente: **455 componenti su 1440** oltre soglia.
- Shader corretto: **zero componenti** oltre soglia.
- Massimo errore normalizzato `|GPU-ref|/(1+|ref|)`:
  **0,109073 → 0,0000137557**, contro un limite di 0,00003.

Le prove intermedie sono conservate: correggere soltanto l'intersezione
lasciava lo scarto del gradino a 5 pixel. Le derivate analitiche portavano
il gradino entro un pixel, ma introducevano la colonna mancante vicino
all'asse. Il limite del passo elimina anche quella colonna.

## Frame e risultati

La distanza è la metrica di contorno descritta nel
[test del punto cerchiato](CONTOUR-BUMP-2026-09-09.md), senza allargare la
soglia di **un pixel**. Tutti i massimi elencati sono uguali alle soglie RGB
1, 3 e 8. M=1, Q=Λ=0, r_obs=40M, θ=80°, φ=0°, FOV=45°, disco esterno 12M.

| Caso | Punto cerchiato prima → dopo | Arco superiore prima → dopo |
|---|---:|---:|
| BL RK4, 640×360, T×0,65 / Doppler 2 | **5 → 1 px** | 3 → 1 px |
| BL RK4, 640×360, controlli standard | **5 → 1 px** | 3 → 1 px |
| BL DOPRI5, 640×360 | **5 → 1 px** | 3 → 0 px |
| BL RK4, jitter (+0,25; −0,25) px | **5 → 1 px** | 3 → 1 px |
| BL RK4, 1280×720 | **9 → 0 px** | 5 → 2 px |
| KS RK4, 640×360 | 1 → 1 px | 2 → 2 px |

Altri tre frame BL a 640×360 con spin 0, −0,5 e +0,9 passano entrambe le
regioni. Non sono presentati come coppie prima/dopo: verificano che la
correzione funzioni anche fuori dallo spin +0,5 della segnalazione.

Gli **11 test rimanenti passano**, inclusi colori Metal, compilazione delle
entrypoint e nuove derivate GPU. La matrice estesa del contorno esegue
108 controlli: **102 PASS, 6 FAIL**, senza SKIP. Tutti i controlli sul punto
cerchiato passano. I fallimenti sono le tre soglie su `ks-custom/upper` e
le tre su `bl-1280/upper`.

Questi residui restano aperti: in BL a 1280×720 c'è un punto candidato a
(590,173) distante due pixel dal contorno CPU dell'anello sottile; in KS
sono undici confronti diretti a distanza due. Non si dichiara verde l'intera
suite. Aumentare le bisezioni dell'intersezione da 7 a 14 non li ha risolti:
la prova è conservata e la modifica non è stata adottata.

## Riproduzione

```sh
cmake -S . -B build_fixes -DBUILD_TESTING=ON
cmake --build build_fixes --parallel
ctest --test-dir build_fixes -E 'kerrtrace.metal_contour' --output-on-failure
./build_fixes/kerrtrace_metal_contour_test --extended --output-dir out/contour-fix/after

# Controllo indipendente sulla funzione GPU (richiede un dispositivo Metal):
./build_fixes/kerrtrace_metal_rhs_test gpu/metal/tracer.metal
```

Il renderer Metal carica lo shader dal sorgente a runtime. I test richiedono
esecuzione GPU reale; un fallback CPU non può certificare la correzione.

`out/contour-fix/` contiene PNG originali, CSV, log e prove intermedie.
`bump-prima-dopo.zip` raccoglie le coppie prima/dopo e i riferimenti CPU,
insieme al resoconto e alle misure. Le evidenze sono ignorate da Git.
L'ingrandimento interattivo visualizza i PNG originali e non modifica i pixel.
