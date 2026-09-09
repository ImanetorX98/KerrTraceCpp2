# Verifica del contorno Metal — 9 settembre 2026

**Stato aggiornato:** il gradino BL è ora corretto. Vedere
[correzione e verifica prima/dopo](METAL-BUMP-FIX-2026-09-09.md).
Questo documento conserva la diagnosi e le misure precedenti alla correzione.

**Precisazione successiva alla foto cerchiata:** il gradino sul bordo esterno
sinistro era fuori dalla regione misurata qui. Il [test aggiornato sul punto
indicato](CONTOUR-BUMP-2026-09-09.md) aggiunge quella regione: BL raggiunge
5 pixel a 640×360 e 9 a 1280×720; KS resta entro un pixel su quel tratto.
I risultati sotto restano riferiti alla prima regione e alla prima esecuzione.

Le correzioni precedenti sono nel commit `16677fd2ed25f35b53cdbfa25905a9449609abe4`,
pubblicato su `origin/codex/audit-physics-backend-fixes`. `main` è rimasto a
`ce1dd33b34f9bfc9710550239d9f714e01a9662f` alla verifica del remoto.
Questa aggiunta introduce test e diagnostica; il renderer e lo shader sono
quelli di `16677fd`.

## Esito

Il nuovo test **fallisce sul contorno del renderer attuale**. Il riferimento
CPU rimane stabile e il vecchio test sui colori interni continua a passare.
La suite non viene marcata verde ignorando il fallimento del contorno.

| Caso | Risoluzione | Max CPU normale vs CPU stretta | Max Metal vs CPU stretta | Coppie di contorno oltre 1 px, soglia 3 |
|---|---|---:|---:|---:|
| BL, RK4, temperatura ×0,65 / Doppler 2 | 640×360 | 0 px | **3 px** | 6 / 2035 |
| BL, RK4, temperatura ×1 / Doppler 4 | 640×360 | 0 px | **3 px** | 6 / 2035 |
| KS, RK4, controlli personalizzati | 640×360 | 0 px | **2 px** | 11 / 2086 |
| BL, DOPRI5, controlli personalizzati | 640×360 | 0 px | **3 px** | 8 / 2029 |
| BL, RK4, offset (+0,25; −0,25) px | 640×360 | 0 px | **3 px** | 8 / 2021 |
| BL, RK4, controlli personalizzati | 1280×720 | 1 px | **5 px** | 240 / 4412 |

I massimi e gli esiti sono uguali alle tre soglie RGB provate: 1, 3 e 8 su 255.
La variazione CPU a 1280×720 riguarda pochissimi campioni; il suo p99 è 0 px.
I massimi Metal a 640×360 sono oltre soglia anche quando **p99 è soltanto
1 px**: una statistica media o il solo p99 nasconderebbero il difetto locale.
I conteggi sono confronti diretti in entrambe le direzioni, non pixel unici.

CTest mirato: `contour_metrics` PASS, `metal_render` PASS,
`metal_contour` FAIL, senza test saltati. Quest'ultimo esegue quattro scene
e passa i 12 controlli di convergenza CPU, fallendo i 12 confronti GPU.
I due casi aggiuntivi passano altri 6 controlli CPU e falliscono altri 6
confronti GPU. Le sette verifiche sintetiche della metrica passano.

## Metodo riproducibile

- Kerr: M=1, a=0,5, Q=Λ=0; camera r=40M, θ=80°, φ=0°, FOV=45°;
  disco esterno a 12M, bordo interno automatico; palette blackbody,
  esposizione 1, gamma 2,2. Un raggio per pixel, sfondo nero uniforme.
- CPU a tolleranza `1e-7` e `1e-10`, stessa carta e stesso integratore del
  caso Metal; massimo 500.000 passi. Metal riceve `1e-7`, ma lo shader
  applica il proprio limite FP32 `1e-5`. Questo test conserva il comportamento
  di produzione e non forza una precisione che il backend non usa.
- La GPU esegue `render_image` senza esportazione geometrica, passando
  attraverso il dispatch di produzione. Il test controlla che il backend
  effettivo sia `gpu-metal`: un fallback CPU fa fallire il test. Un dispositivo
  Metal non accessibile restituisce 77 e CTest dichiara SKIP.
- Maschera dei pixel con `max(R,G,B) > soglia`. Bordo = pixel acceso con
  almeno un vicino cardinale spento. Regione inclusiva:
  x=⌊0,30W⌋…⌊0,70W⌋, y=1…⌊0,45H⌋. Include arco superiore, bordo interno
  e immagine sottile interna. I vicini vengono letti nel frame originale,
  quindi il perimetro della regione non crea bordi artificiali.
- Distanza di Hausdorff simmetrica con norma Chebyshev:
  `max(|Δx|, |Δy|)`, in pixel nativi. Soglia di accettazione **≤1 pixel**
  e almeno 500 punti nel riferimento. Nessuna erosione o rimozione dei
  campioni peggiori. Una maschera vuota non può passare.
- Il test sintetico verifica identità, traslazione diagonale di un pixel,
  bump locale di tre pixel, buco isolato, candidato nero, due maschere vuote
  e assenza di falsi bordi introdotti dalla regione di misura.

## Interpretazione e limiti

Il contorno Metal differisce dal riferimento numerico CPU convergente.
La discrepanza sopravvive al cambio dei controlli di colore, delle soglie
di luminosità, dell'integratore e al piccolo spostamento del campionamento.
A risoluzione doppia cresce in pixel; non basta attribuirla alla sola
quantizzazione di un pixel.

Questi dati sostengono la diagnosi di artefatto del backend, ma **non isolano
ancora il termine numerico responsabile**. Le derivate per differenze finite
in FP32 e il controllo del passo sono candidati da investigare. Le precedenti
prove con incrementi più grandi nelle derivate spostavano il difetto e
introducevano altri artefatti: non sono state adottate come correzione.

Il massimo misura tutta la regione selezionata, compreso l'anello sottile;
non è la misura della sola gobba al vertice nella foto dell'utente. A 640×360
il massimo BL è presso (443,157), a cinque pixel dai tagli della regione. Le immagini
CPU e Metal devono quindi essere guardate insieme alla posizione dei residui.
L'accordo fra backend non è una prova analitica della soluzione fisica.
La copertura riguarda questa camera e questo spin, non ogni scena possibile.

## Esecuzione

```sh
cmake -S . -B build_fixes -DBUILD_TESTING=ON
cmake --build build_fixes --parallel --target \
  kerrtrace_contour_metrics_tests kerrtrace_metal_contour_test kerrtrace_metal_render_test
ctest --test-dir build_fixes \
  -R 'kerrtrace\.(contour_metrics|metal_contour|metal_render)$' --output-on-failure

# Matrice aggiuntiva, ciascun comando restituisce attualmente 1:
./build_fixes/kerrtrace_metal_contour_test --extended --case bl-jitter \
  --output-dir build_fixes/contour-jitter
./build_fixes/kerrtrace_metal_contour_test --extended --case bl-1280 \
  --output-dir build_fixes/contour-1280

# Grafico opzionale, richiede matplotlib:
python3 tests/plot_contour_diagnostics.py \
  build_fixes/contour-output/bl-custom-metal-vs-cpu-t3.csv \
  out/contour-tests/contorno-bl.png
```

Non è impostato `WILL_FAIL`: il test diventerà verde quando il renderer
rispetterà il criterio, senza modificare il criterio stesso.

## Evidenze locali

In `out/contour-tests/`: 18 PNG originali, CSV dei punti e dei vicini più
prossimi, `summary-all.csv`, log di build/CTest, grafici delle coordinate e
`contour-evidence.zip`. I grafici rappresentano i dati misurati; i frame non
sono ritoccati. Sono confronti CPU/Metal attuali, non un prima/dopo di una
correzione del bump. Le evidenze sono ignorate da Git e si rigenerano con i
comandi sopra. Il codice dei test e questo resoconto sono versionati.
