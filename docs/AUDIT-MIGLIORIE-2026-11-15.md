# Audit di fattibilità dell’handoff — 15 novembre 2026

## Esito e perimetro

Audit di tutte le 27 voci di `HANDOFF-CODEX-MIGLIORIE.md`, sul ramo `refactor/modular-architecture`, commit `5da122a5d0e7892db58362334096b66918bada74`. Nessuna correzione al codice di produzione, nessun merge, pulizia o push. La modifica preesistente a `.gitignore` è stata preservata. I programmi diagnostici sono solo in `out/audit-migliorie-2026-11-15/`.

Le migliorie sono in larga parte fattibili, ma alcune premesse cambiano sensibilmente il lavoro:

- **B2: bug matematico riproducibile nell’integrale ellittico incompleto**, prima ancora di discutere il ramo entrante. La mappa Region III non riproduce la posizione iniziale nei campioni verificati.
- **D1: problema operativo anche nel server**, che preferisce vecchi binari per semplice esistenza del file. Registrare il test da solo non basta.
- **A2: aspetto slavato non riprodotto** nei quattro casi blackbody della suite; non c’è ancora una causa da correggere.
- **C4: implementazione già presente su un altro ramo locale**, ma con un layout incompatibile con quello corrente: va recuperata e verificata, non duplicata né importata alla cieca.
- **E1: navigazione web già esistente**; finestra nativa, camera libera e pipeline persistente sono invece lavoro aggiuntivo.

Le stime comprendono implementazione, test mirati e documentazione, per una persona che conosce il progetto. Non sono benchmark né promesse di velocità. S = meno di mezza giornata; M = 1–2 giorni; L = 3–5 giorni; XL = oltre una settimana. Le voci di ricerca hanno incertezza maggiore; le dipendenze non sono comprese nella stima. Dove una voce contiene più progetti, sono separati.

## Verifiche eseguite

1. Letti AGENTS.md, CLAUDE.md, architettura, storico problemi, rapporti GPU e pass-through; confrontati con i sorgenti attuali.
2. Ricompilata `build/` con CMake nativo. `ctest --test-dir build --output-on-failure`: **14/14 PASS, nessuno SKIP, 92,84 s**. Questo risultato riguarda i casi registrati, non tutta la roadmap.
3. `ctest -N`: 14 test in `build/`, 15 in `build_rcpu/`; confermata la differenza di registrazione di `metal_passthrough`.
4. Test pass-through eseguito manualmente contro **build/kerr_tracer**: **PASS, 0 pixel cambiati** aumentando il cap da 5 000 a 500 000, limite corrente 10.
5. Test blackbody nativo: 992 pixel interni del disco per configurazione; MAE RGB 0,133/0,540 in BL e 0,130/0,543 in KS, scala 0–255, controlli standard/custom. Nessun pixel con errore medio >20. Non è una validazione della scena della galleria, non fornita come preset riproducibile.
6. Diagnostici CPU che chiamano le funzioni di produzione: identità dell’integrale F, posizione iniziale e velocità della mappa ellittica; piccolo esperimento sullo stallo del separabile. Dettagli sotto.
7. GitHub interrogato in sola lettura: HEAD remoto uguale al locale; main fermo a `ce1dd33`; **17 commit avanti, 0 indietro**, non 13. Nessuna PR aperta con questo head. Diff confermato: 64 file, +9 623/−5 147.
8. Consultati i PDF locali Gralla–Lupsasca (1910.12881, Appendice B) e DNGR (1502.03808, Appendice A.3), e documentazione primaria Apple / Connors–Piran–Stark citata nelle note.

Non ripetuti: benchmark 4K, conteggio dei 21 pixel, mediana 11 400, quota totale 53%, profiling GPU per raggio, test su hardware CUDA. Questi numeri restano misure storiche dell’handoff/documentazione, non risultati di questo audit. La suite CPU separata non è stata rieseguita per intero; la suite Metal esegue anche i test CPU previsti, ma non sostituisce la matrice di build proposta in D6.

## Tabella di valutazione

| ID | Ancora valida? Evidenza | Fattibilità / effort | Rischio e test | Valore | Dipendenze | Come misurare il successo |
|---|---|---|---|---|---|---|
| A1 | Parziale: `continue` presente, ma non replica esattamente il bug KS: [gpu/metal/tracer.metal:2840](/Users/iman.rosignoli/Documents/KerrTraceCpp2/gpu/metal/tracer.metal:2840), [gpu/metal/tracer.metal:2911](/Users/iman.rosignoli/Documents/KerrTraceCpp2/gpu/metal/tracer.metal:2911). CPU con potenziali clampati: [renderer/tracer_separable.hpp:71](/Users/iman.rosignoli/Documents/KerrTraceCpp2/renderer/tracer_separable.hpp:71). | **Media, L** per stabilizzare il solver; **S** solo aggiornamento degli stati al pass-through. | Turning point, eventi fuori ordine, falsi escape al cap. Test per raggio contro BL, cap e tolleranze variabili. | Alto, se si mantiene il solver. | D1; D3 per regressioni. | Istogrammi passi/rifiuti/stalli; stesso esito e hit del riferimento convergente; tempi blackbody e Interstellar separati. |
| A2 | **Non dimostrata** come difetto Metal generalizzato: [renderer/palettes.hpp:612](/Users/iman.rosignoli/Documents/KerrTraceCpp2/renderer/palettes.hpp:612), [gpu/metal/tracer.metal:2066](/Users/iman.rosignoli/Documents/KerrTraceCpp2/gpu/metal/tracer.metal:2066); test attuale passa. | **Alta per diagnosi, M**; correzione non stimabile finché manca il caso. | Cambiare esposizione per nascondere errori geometrici o cambiare entrambi i backend senza motivo. `metal_render` e test colore isolato. | Medio, alto se riprodotto. | D1; C1 solo dopo diagnosi. | Preset esatto; RGB/luminanza/saturazione su intersezione e unione delle maschere, più shader-colore su geometria identica. |
| A3 | Sì: fallback esplicito in [renderer/backend_select.hpp:13](/Users/iman.rosignoli/Documents/KerrTraceCpp2/renderer/backend_select.hpp:13); NASA [renderer/palettes.hpp:524](/Users/iman.rosignoli/Documents/KerrTraceCpp2/renderer/palettes.hpp:524); hash stratificato [renderer/palettes.hpp:151](/Users/iman.rosignoli/Documents/KerrTraceCpp2/renderer/palettes.hpp:151). | **Alta, M NASA; L stratificata**, inclusa parità e migrazione legacy. | Trasparenza, cucitura φ, rumore e parametri non supportati. Test noise/render/pass-through estesi. | Alto per chi usa queste palette. | D1/D3; alternativa A4 con shading CPU. | Backend effettivo Metal, shader e visibilità confrontati separatamente; nessuna cucitura; benchmark sulle stesse scene. |
| A4 | Sì: GPU scrive uint32 RGB, [gpu/metal/metal_renderer.mm:151](/Users/iman.rosignoli/Documents/KerrTraceCpp2/gpu/metal/metal_renderer.mm:151); export forza CPU, [renderer/backend_select.hpp:12](/Users/iman.rosignoli/Documents/KerrTraceCpp2/renderer/backend_select.hpp:12). | **Alta, L MVP single-ray; XL parità completa** con impronte/bundle/compositing. | ABI, campi non inizializzati, semantica outcome/padding/coverage. Test KGEO, confronti campo per campo, fallback espliciti. | Alto, architetturale. | D2 consigliata; D1. | Round trip e ricolorazione equivalenti; errore r/g/φ per esito; memoria e tempo geometry-only a 4K. |
| A5 | Sì, ramo stocastico irraggiungibile nel normale piano: [gpu/metal/tracer.metal:1968](/Users/iman.rosignoli/Documents/KerrTraceCpp2/gpu/metal/tracer.metal:1968), [renderer/backend_select.hpp:17](/Users/iman.rosignoli/Documents/KerrTraceCpp2/renderer/backend_select.hpp:17). | **Alta, S** per rimozione/documentazione del ramo morto. | Abilitare per errore una trasparenza diversa dal compositing CPU. Test del planner e opacità 0/0,5/1. | Medio, pulizia semantica. | Nessuna; se si abilita trasparenza serve redesign con A4. | Frame supportati invariati; configurazioni trasparenti continuano ad avere fallback dichiarato. |
| A6 | Tolleranza minima confermata in [gpu/metal/tracer.metal:23](/Users/iman.rosignoli/Documents/KerrTraceCpp2/gpu/metal/tracer.metal:23); **21 pixel storici non rimisurati**. | **Media, M diagnosi; L intervento se giustificato**. | Rallentare tutti i raggi, perdere anelli sottili; `metal_contour` più scena 4K specifica. | Basso/medio per anteprime, alto solo con requisito scientifico preciso. | D3; A4 utile per separare geometria/colore. | Hit mismatch, distanza del contorno e r/g su CPU convergente; costo del passo locale. Non soglia “zero pixel” universale. |
| B1 | Struttura monolitica e attesa sincrona presenti: [gpu/metal/metal_renderer.mm:295](/Users/iman.rosignoli/Documents/KerrTraceCpp2/gpu/metal/metal_renderer.mm:295), [gpu/metal/metal_renderer.mm:324](/Users/iman.rosignoli/Documents/KerrTraceCpp2/gpu/metal/metal_renderer.mm:324). Tempi 3,4/1,9 s precedono gli ultimi fix. | **Media: L secondo passaggio, L riordino, XL coda persistente**. | Stato incompleto, FSAL/segni/eventi persi, pressione registri, determinismo. Confronto per raggio e stress GPU. | Medio, condizionato al profiling aggiornato. | A1 solo per quel solver; A4 non obbligatoria. | p50/p95/p99 dei passi e della latenza, distribuzione per tessera, warm/cold; guadagno netto incluse compattazione e memoria. |
| B2 | Sì, ma diagnosi storica insufficiente: [renderer/tracer_elliptic.hpp:49](/Users/iman.rosignoli/Documents/KerrTraceCpp2/renderer/tracer_elliptic.hpp:49), [renderer/tracer_elliptic.hpp:381](/Users/iman.rosignoli/Documents/KerrTraceCpp2/renderer/tracer_elliptic.hpp:381), [renderer/tracer_elliptic.hpp:598](/Users/iman.rosignoli/Documents/KerrTraceCpp2/renderer/tracer_elliptic.hpp:598). | **Media, L–XL** per solver validato; **M** primo lotto quadranti/normalizzazione con test. | Hit falsi, ordine orizzonte/disco, degenerazioni a≈0 e radici multiple. Test identità, IVP e confronto numerico indipendente. | Alto per correttezza; velocità da misurare. | D3; non togliere guardie prima della validazione. | r(0)=r_obs, dr/dτ coerente, identità F/cn, hit per regione, fallback per causa e numero reale di trace numerici. |
| B3 | Parziale: blackbody è una restrizione della **visibilità**, non una fase colore CUDA mancante: [renderer/backend_select.hpp:75](/Users/iman.rosignoli/Documents/KerrTraceCpp2/renderer/backend_select.hpp:75), [renderer/render.hpp:306](/Users/iman.rosignoli/Documents/KerrTraceCpp2/renderer/render.hpp:306). | **Alta, L** visibilità Interstellar; **Media, XL** bundle. Consegna condizionata a GPU NVIDIA. | Differenze double/float, divergenza GPU, impronte incomplete. Test host attuali più nvcc e GPU reale. | Medio, alto se NVIDIA è piattaforma obiettivo. | A3 condivisione specifiche/hash; D2 per campi nuovi. | Test nativi CUDA, stessa maschera e GeoPixel del riferimento; tempi su hardware dichiarato. |
| B4 | Sì, conversione già in [tools/kgeo_to_npy.cpp:58](/Users/iman.rosignoli/Documents/KerrTraceCpp2/tools/kgeo_to_npy.cpp:58). | **Alta, M** per estrarre writer condiviso e integrare CLI; wrapper esterno S. | Dtype/endian/header divergenti, doppi percorsi I/O. `kgeo_npy` e round trip. | Basso per velocità; medio per uso Python frequente. | D2 da coordinare, non bloccante. | Array identici al convertitore, I/O/peak RAM e tempo totale; non promettere guadagno sul ray tracing. |
| C1 | Sì, r^-0,5 in [renderer/palettes.hpp:626](/Users/iman.rosignoli/Documents/KerrTraceCpp2/renderer/palettes.hpp:626) e [gpu/metal/tracer.metal:2069](/Users/iman.rosignoli/Documents/KerrTraceCpp2/gpu/metal/tracer.metal:2069); il vecchio clamp g≥0,2 è già off per default. | **Alta, M** opzione di profilo; **L** default fisico documentato e migrazione. | Aspetto di tutti i render, proxy NT vs vero spettro, Planck extra. Test colore radiale/legacy e CPU/Metal. | Alto per coerenza fisica del colore. | A2 baseline; D3. | T^4 proporzionale al flusso scelto, limite r^-3/4 lontano dal bordo per flusso r^-3, legacy riproducibile. |
| C2 | Sì, segno fissato da a: [knds_metric.hpp:376](/Users/iman.rosignoli/Documents/KerrTraceCpp2/knds_metric.hpp:376); ISCO corotante [knds_metric.hpp:326](/Users/iman.rosignoli/Documents/KerrTraceCpp2/knds_metric.hpp:326). | **Alta, L** cross-backend/CLI/web. | Ω, ISCO, redshift e profilo devono usare lo stesso ramo; test spin attuale più verso indipendente. | Medio/alto per animazioni. | D7 utile non necessaria; D2 provenienza. | Animazione attraverso a=0 a verso fissato continua; simmetria (a,s,φ)→(−a,−s,−φ); limiti ISCO pro/retro. |
| C3 | **Interpretazione errata, bug non provato**: determinante grezzo per angolo in [ray_bundle.hpp:534](/Users/iman.rosignoli/Documents/KerrTraceCpp2/ray_bundle.hpp:534). Mediana non riprodotta. | **Alta, M** validazione e semantica; S documentazione. | Normalizzazione arbitraria falsifica footprint; test differenze finite dei raggi vicini. | Alto per affidabilità scientifica, non necessariamente visivo. | Nessuna; D2 se cambia il nome/contratto. | J vs raggi perturbati, dipendenza da r_obs/FOV e unità; non richiedere mediana≈1. |
| C4 | Assente nel ramo corrente: [render_data.hpp:15](/Users/iman.rosignoli/Documents/KerrTraceCpp2/render_data.hpp:15), [renderer/tracer_bl.hpp:98](/Users/iman.rosignoli/Documents/KerrTraceCpp2/renderer/tracer_bl.hpp:98). **Presente in feature/photon-ring-multipass@e40814b**. | **Alta, M CPU base; L integrazione multi-backend**. | Conteggio doppio eventi, underflow/overflow, debug/behind nel padding, formato del ramo divergente. | Alto diagnostico e visuale. | D2; coordinamento con ramo esistente; A4 per export GPU. | Ordine su geodetiche note, stabilità al passo, convenzione esplicita crossing vs turning; vecchi campi preservati. |
| C5 | Non implementata nel contratto corrente, [render_data.hpp:13](/Users/iman.rosignoli/Documents/KerrTraceCpp2/render_data.hpp:13). Premessa “campo B da Connors 1980” da correggere. | **Media, XL**: prototipo Kerr 2–4 settimane; trasferimento magnetizzato generale oltre. | Tetrad, gauge, segno EVPA, basi Stokes, depolarizzazione. Test trasporto e conservazione κ, k·f=0. | Alto per ricerca, basso per RGB ordinario. | D2; momenti al punto d’emissione; C7 spettrale se Faraday. | Confronto PW/trasporto numerico, limiti analitici, Q²+U²≤I², media Stokes e non angoli. |
| C6 | DNEG attuale ultrastatico e senza disco: [wormhole_metric.hpp:6](/Users/iman.rosignoli/Documents/KerrTraceCpp2/wormhole_metric.hpp:6), [renderer/tracer_wormhole.hpp:12](/Users/iman.rosignoli/Documents/KerrTraceCpp2/renderer/tracer_wormhole.hpp:12); GPU esclusa [renderer/frame.hpp:40](/Users/iman.rosignoli/Documents/KerrTraceCpp2/renderer/frame.hpp:40). | **Media, L potenziale; L disco; XL rotazione; Alta, L GPU modello attuale**. | Nuova metrica, orbite alla gola, E/camera/redshift, due universi; test H e limiti Φ,J→0. | Medio/alto come nuova funzionalità. | D2 per identità metrica; disco dipende da potenziale validato. | Orbite effettivamente accessibili, conservati, attraversamento gola e CPU/GPU; non imporre r=m se fuori dominio. |
| C7 | Parziale: carte selezionate per frame [renderer/trace_types.hpp:71](/Users/iman.rosignoli/Documents/KerrTraceCpp2/renderer/trace_types.hpp:71); Carter già [renderer/tracer_separable.hpp:38](/Users/iman.rosignoli/Documents/KerrTraceCpp2/renderer/tracer_separable.hpp:38); disco sottile/colore scalare. | **Media, XL switching; Alta, M diagnostica Carter/L classificazione Kerr; Media, XL disco spesso/GRMHD; Alta, L multibanda/XL trasferimento spettrale**. | Trasformazione covettore/W, Λ≠0, classificazione vicino separatrice, unità e dati plasma. | Medio switching; alto diagnostica; alto ricerca GRMHD/spettri. | D2 per dati nuovi; C5 se polarizzato. | Invarianti tra carte; classificazione vs traccia; limite disco sottile; soluzione analitica trasferimento in slab. |
| D1 | **Confermata**, [CMakeLists.txt:311](/Users/iman.rosignoli/Documents/KerrTraceCpp2/CMakeLists.txt:311); binari obsoleti preferiti da [server/index.js:40](/Users/iman.rosignoli/Documents/KerrTraceCpp2/server/index.js:40). | **Alta, S registrazione; M naming/manifest/server**. | Test saltato come PASS, backend falso, client incompatibili. Matrice USE_METAL ON/OFF e server. | Alto, prerequisito delle misure. | Nessuna. | Test presente in entrambe le build, skip 77 reale; backend/build-id verificati, nessun binario obsoleto scelto per sola esistenza. |
| D2 | Parziale: header già esiste, manca auto-descrizione/provenienza: [render_data.hpp:40](/Users/iman.rosignoli/Documents/KerrTraceCpp2/render_data.hpp:40), [render_data.hpp:51](/Users/iman.rosignoli/Documents/KerrTraceCpp2/render_data.hpp:51). | **Alta, L** schema v5 e lettore v4 compatibile. | Rottura cache, ABI host, campi semantici persi; audit I/O, kgeo_npy, fixture v4 e file malformati. | Alto. | Coordinare A4/C2/C4/C5/C6; D7 utile. | record stride/endian/metrica/provenienza verificati; rifiuto file incoerenti, migrazione controllata. |
| D3 | Sì, nessuna baseline versionata; confronto permissivo in [tests/render_fingerprint.py:95](/Users/iman.rosignoli/Documents/KerrTraceCpp2/tests/render_fingerprint.py:95). | **Alta, M** primi golden mirati; L matrice completa. | Baseline difettose cristallizzate, falsi positivi tra driver; test del test con mutanti. | Alto. | D1. | Fallimento su caso mancante, render fallito e backend errato; golden numerici più hash solo a toolchain identico. |
| D4 | Sì: [AGENTS.md:14](/Users/iman.rosignoli/Documents/KerrTraceCpp2/AGENTS.md:14), [CLAUDE.md:306](/Users/iman.rosignoli/Documents/KerrTraceCpp2/CLAUDE.md:306); mappa/roadmap e stato problemi non sincronizzati. | **Alta, S** punto d’ingresso comune; M se generazione/check automatico. | Perdere convenzioni specifiche agenti; controllo link e build documentata. | Medio/alto. | D1 per comandi definitivi. | Una sola fonte per mappa/stato e due entrypoint brevi, nessuna istruzione contraddittoria. |
| D5 | Cinque rami remoti elencati **integrati** in main; falling-camera-wip **non integrato**, 32 commit propri. Quattro build vecchie circa 181 MB. | **Alta, S** inventario/pulizia selettiva; port falling separato XL possibile. | Cancellare artefatti diagnostici o un ramo utile; controllo ancestry/worktree/server. | Medio. | D1 prima di rimuovere build_cpu. | Manifest di ciò che si conserva/elimina, ancestry verificata, server ancora funzionante. Nessuna cancellazione fatta. |
| D6 | Sì; main antenato di HEAD, 17 commit e diff confermato. | **Alta, M** preparazione/review; rischio fisico medio nonostante merge meccanico semplice. | Cumulativo include fix di fisica e CUDA, non solo refactor. Test, fingerprint pre/post per commit e build separate. | Alto. | D1/D3 raccomandate come gate. | Merge-tree/fast-forward pulito, build CPU/Metal, test GPU non saltati; CUDA nativa oppure limite di validazione esplicito. |
| D7 | Sì: parametri duplicati CLI/server/frontend; reflection oggi solo record: [tools/kgeo_reflection.hpp:15](/Users/iman.rosignoli/Documents/KerrTraceCpp2/tools/kgeo_reflection.hpp:15). | **Alta, L MVP schema; XL migrazione completa**. | Alias, default, unità, vincoli e compatibilità API; test JSON→CLI→parametri. | Alto manutenzione, basso velocità. | D1; D2 coordina provenienza. | Schema generato riproducibile, stessi argomenti/valori e errori prima/dopo; eliminazione copie manuali. |
| D8 | Parziale: wrap longitudine già corretto, polo clampato: [renderer/background.hpp:63](/Users/iman.rosignoli/Documents/KerrTraceCpp2/renderer/background.hpp:63), [gpu/metal/tracer.metal:1797](/Users/iman.rosignoli/Documents/KerrTraceCpp2/gpu/metal/tracer.metal:1797). Asset della galleria non disponibile. | **Alta, M** sampling sferico/test; **bassa** eliminare ogni cucitura di qualunque asset con semplice mirror. | Cielo deformato, parità CPU/GPU, footprint al polo. Test panorama sintetico periodico e polo. | Medio/alto. | E2 per filtro footprint robusto. | Continuità su funzione analitica della direzione, φ e φ+2π uguali; asset validato separatamente dal sampler. |
| E1 | **Parzialmente esistente** web, [server/index.js:887](/Users/iman.rosignoli/Documents/KerrTraceCpp2/server/index.js:887); manca pipeline nativa persistente e camera libera. | **Media, XL**: 5a 3–5 settimane; 5b 1–2; 5c 4–8; 5d 3–6 per TAA affidabile. | Cache/globali, stale frame, camera/tetrad, caustiche; stress cancel/move e confronto offline. | Alto se prodotto interattivo è obiettivo. | D1/D7; B1 misurare, non requisito assoluto; A4 utile. | p95 evento→preview e frame finali uguali all’offline; target <100 ms da dimostrare. |
| E2 | **Parzialmente già presente**: filtro impronta disco [renderer/colorize.hpp:114](/Users/iman.rosignoli/Documents/KerrTraceCpp2/renderer/colorize.hpp:114) e cielo [renderer/background.hpp:37](/Users/iman.rosignoli/Documents/KerrTraceCpp2/renderer/background.hpp:37); temporale DNGR assente. | **Media, L spaziale corretto; XL temporale analitico** (3–6 settimane indicative). | Caustiche, overlap/occlusioni, media su RGB tonemappati; test stelle e shutter convergente. | Alto per animazioni e qualità finale. | Contratto radiance lineare; C3; D2 se nuovi dati; E1 solo per UI. | Stabilità flusso stellare subpixel, errore vs supersampling spazio/tempo e assenza ghosting. |

## Note per voce

### A1 — due problemi distinti, non una patch risolutiva già identificata

Nel separabile GPU `s_prev = s` viene creato **all’inizio di ogni iterazione** e l’evento equatoriale usa proprio `s_prev.theta` ([gpu/metal/tracer.metal:2840](/Users/iman.rosignoli/Documents/KerrTraceCpp2/gpu/metal/tracer.metal:2840), [gpu/metal/tracer.metal:2889](/Users/iman.rosignoli/Documents/KerrTraceCpp2/gpu/metal/tracer.metal:2889)). Orizzonte e fuga sono controllati **prima** del `continue`. Quindi non è corretto trasferire automaticamente la diagnosi KS “stesso attraversamento rilevato all’infinito”. Il `continue` salta comunque `prev_Rpot/prev_Thpot` e gli stati usati per interpolare la fuga: difetto reale da eliminare.

La CPU non contiene quel `continue`. Usa `sqrt(max(R,0))` e cambia segno solo quando il potenziale passa sotto soglia. Se il passo lascia lo stato in una regione vietata, la velocità può restare nulla e il cambio di segno da solo non lo riporta nel dominio. Esperimento diagnostico ridotto: 144 raggi campionati, 132 raggi arrivano al cap 5 000 con r e θ ripetuti. Il loop usa eventi semplificati: **è evidenza di un meccanismo di stallo, non una misura del tasso di errore del renderer di produzione**. Occorre strumentare passi accettati/rifiutati, potenziali, branch e ragioni di terminazione del tracer vero.

Non deprecare subito: marcare sperimentale, fissare invarianti e confrontare con BL. Se non offre vantaggio a qualità equivalente, ritirarlo dal percorso consigliato mantenendo un fallback corretto. Il cap CPU oggi può diventare `ESCAPED` ([renderer/tracer_separable.hpp:295](/Users/iman.rosignoli/Documents/KerrTraceCpp2/renderer/tracer_separable.hpp:295)): non è una prova di fuga fisica. La correzione locale S non giustifica annunciare risolti i 110/472 secondi storici.

### A2 — prima riprodurre, poi correggere

Il flusso di riferimento è calcolato dalla CPU e passato al kernel ([renderer/render.hpp:169](/Users/iman.rosignoli/Documents/KerrTraceCpp2/renderer/render.hpp:169)); entrambe le implementazioni usano Reinhard e la stessa legge T. C’è una differenza piccola e reale: CPU quantizza la rampa blackbody in RGB8 prima del tonemapping ([renderer/palettes.hpp:11](/Users/iman.rosignoli/Documents/KerrTraceCpp2/renderer/palettes.hpp:11)), Metal usa float fino al packing ([gpu/metal/tracer.metal:1781](/Users/iman.rosignoli/Documents/KerrTraceCpp2/gpu/metal/tracer.metal:1781)). Anche i floor del flusso differiscono (CPU 1e-12, Metal 1e-8). Sono ipotesi da isolare, non cause dimostrate dell’aspetto “slavato”.

La suite attuale smentisce un grande scarto nei quattro casi provati, ma non copre esposizioni estreme, tutti i taper e il preset originale. Conservare quel preset, usare cielo nero e confrontare anche percentili/saturazione: la sola media RGB potrebbe nascondere un errore localizzato. Un eventuale nuovo test deve fallire sulla riproduzione difettosa prima della patch; non serve aggiungere un test verde che replica quelli esistenti.

### A3 — port diretto oppure geometria GPU + colore condiviso

NASA non è solo rumore: servono tutti i parametri, la maschera di visibilità e il guadagno. Il port della stratificata deve risolvere anche la periodicità delle celle, non solo passare da seno a interi. Conservare una modalità legacy per i frame già prodotti.

A4 offre un’alternativa: geodetiche GPU e colorizzazione CPU già completa. Per palette con buchi/maschera non basta cambiare lo shader colore: il tracer deve sapere se continuare il raggio. Confrontare questa architettura con il port completo prima di mantenere altre copie delle quattro palette.

### A4 — MVP praticabile senza promettere bundle GPU

GeoPixel attuale è 64 byte. Un buffer completo vale **126,6 MiB a 1920×1080** e **506,25 MiB a 3840×2160**, contro 31,64 MiB per il buffer ABGR 4K attuale. Apple Silicon usa memoria condivisa, ma allocazione, scrittura, lettura CPU e serializzazione hanno comunque costo; non è corretto stimarlo dalla sola banda teorica.

Definire ABI con offset e static_assert, tipi scalari senza padding implicito MSL, outcome e campi validi per ogni esito. Per single-ray: magnif=1, coverage coerente, impronte assenti dichiarate. Non produrre un record apparentemente bundle con campi fittizi. `_pad[0]` contiene debug e `_pad[1]` è già usato per il risultato dietro il disco ([renderer/phase1.hpp:254](/Users/iman.rosignoli/Documents/KerrTraceCpp2/renderer/phase1.hpp:254)). Il formato disco può rimanere separato dal layout ottimale GPU. Per ridurre RAM, scrivere per tessere senza cambiare la semantica.

### A5 — non basta cambiare hash per abilitare l’opacità

La CPU miscela disco e sfondo, la funzione Metal decide stocasticamente se saltare il disco. Allineare l’hash non rende equivalenti questi due modelli. Preferisco rimuovere il ramo morto o renderlo esplicitamente sperimentale, lasciando la trasparenza nel fallback. Se diventa requisito GPU, progettare continuazione e compositing coerenti.

### A6 — accettazione provvisoria con un caso di regressione

Ventuno pixel su 4K non sono una metrica universale: dipendono da scena, soglia RGB, scala e risoluzione. Il test contorno attuale è verde; il caso 4K è diverso. Conservare coordinate e crop a risoluzione nativa, confrontare gli outcome e poi valutare un passo locale/dense output all’evento. Non riproporre il ritracciamento FP64 dei bordi già scartato; non abbassare indiscriminatamente il floor FP32.

### B1 — nuova baseline prima di una nuova architettura

I tempi 3,4/1,9 s sono precedenti alle derivate KS analitiche e all’hash intero. Il rapporto documenta poi 16,9 s GPU a 4K: la bassa risoluzione va misurata di nuovo. Prima scelta sperimentale: passi a budget limitato, stato dei raggi incompleti in buffer e secondo passaggio compattato. Riordino e thread persistenti sono alternative da valutare dopo aver misurato quanta divergenza rimane, non tre implementazioni da fare insieme.

Il buffer di continuazione deve salvare stato, passo adattivo, cache FSAL quando pertinente, segni dei turning point e metadati dell’ultimo evento. Serve una prova che spezzare in due il lavoro non cambi il risultato. Nessuna proposta di più code/command buffer concorrenti: già misurata e scartata.

### B2 — la diagnosi del ramo post-bounce non regge come unica causa

Gralla–Lupsasca, Appendice B, caso (3), eq. B74–B75, presenta la soluzione con condizioni iniziali corrette e senza turning point esterno all’orizzonte. Non c’è un ostacolo matematico intrinseco a usare B75 sul ramo entrante.

**Difetto confermato:** `ellint_F_incomplete` restituisce `sin(phi)*RF(cos²(phi),1−m sin²(phi),1)` senza estendere l’ampiezza oltre il quadrante principale. Perciò F(π−φ)=F(φ) nel codice, mentre deve essere 2K−F(φ). Region III chiama questa funzione con `acos(x3)`, che può essere >π/2. La GPU ha la stessa implementazione ([gpu/metal/tracer.metal:1171](/Users/iman.rosignoli/Documents/KerrTraceCpp2/gpu/metal/tracer.metal:1171)).

Il probe sulle funzioni reali, con camera r=60, a=0,998, θ=84°, trova quattro mappe Region III con errore iniziale r(0)−60 tra **−58,95 e −58,89**. Il problema esiste **a τ=0**, dunque non può essere spiegato soltanto da una scelta tardiva dell’intersezione. Il test elementare a m=0,5 restituisce F(π/3)=F(2π/3)=1,142429, mentre 2K−F(π/3)=2,565720: l’identità fallisce di 1,423291 già sul codice attuale. Primo lotto: test di quadranti/periodicità di F, inversione cn e identità r(0), poi correzione CPU/Metal.

**Seconda incoerenza da correggere e validare:** il quartico radiale viene reso monico dividendo per E² ([renderer/tracer_elliptic.hpp:269](/Users/iman.rosignoli/Documents/KerrTraceCpp2/renderer/tracer_elliptic.hpp:269)), le frequenze radiali non ripristinano E, mentre la frequenza polare lo contiene ([renderer/tracer_elliptic.hpp:484](/Users/iman.rosignoli/Documents/KerrTraceCpp2/renderer/tracer_elliptic.hpp:484)). Nei quattro campioni Region I/II, con E=0,983192, il rapporto fra derivata radiale ottenuta e quella attesa è 1,01710 ≈ 1/E. Va usato un unico tempo di Mino: o costanti tutte normalizzate a E=1, o fattore E coerente. Non confondere questo con il difetto di quadrante della Region III.

Inoltre:

- Il solver valuta solo `tau_first`, ignora `tau_period` ([renderer/tracer_elliptic.hpp:561](/Users/iman.rosignoli/Documents/KerrTraceCpp2/renderer/tracer_elliptic.hpp:561)); intersezioni successive richiedono altro lavoro.
- Il ramo diretto traccia comunque numericamente per controllare l’orizzonte (riga 598).
- Con palette non blackbody può poi tracciare di nuovo nel fallback (riga 606).
- Non calcola φ_hit robusto sul ramo diretto; restituisce 0.
- A=0 fa fallire il solver polare: non è ancora un fast path generale Kerr.

Il 53% non può essere attribuito interamente ai circa 23 000 raggi Region III: sono denominatori/gruppi diversi e manca il censimento. Misurare per motivo e regione, includendo il costo dei trace di verifica che non risultano nel solo contatore fallback. Conservare le guardie finché i nuovi test indipendenti passano. Stimare il guadagno solo dopo questo censimento.

### B3 — condividere matematica e contratti, non incollare MSL

CUDA usa double e produce GeoPixel; il colore passa già per `colorize_buffer`. Occorre portare il test di visibilità Interstellar e il relativo hash per sapere quando attraversare il bordo procedurale. Non serve riscrivere tutte le palette sul device per questo obiettivo. L’hash intero può avere una specifica e vettori di test comuni; dispatch, texture, qualificatori e precisione restano diversi.

I test host chiamano la vera funzione CUDA con qualificatori rimossi ([tests/audit_regression.cpp:1](/Users/iman.rosignoli/Documents/KerrTraceCpp2/tests/audit_regression.cpp:1)): utili, ma non provano compilazione nvcc, limiti device o prestazioni NVIDIA. Il port bundle richiede stato variazionale, impronte e controllo errore: XL, non un’estensione breve del kernel single-ray.

### B4 — comodità più che ottimizzazione

Estrarre il writer dal tool, riusando la descrizione Sane, invece di mantenere due implementazioni NPY. Il vantaggio è eliminare un file intermedio e una seconda lettura completa. Per ray tracing costoso è presumibilmente secondario, ma il rapporto va misurato su geometria già pronta e su frame completo. Una semplice scorciatoia di comando può essere sufficiente.

### C1 — r^-3/4 è un limite, non tutto il disco relativistico

Per flusso F∝r^-3, T_eff∝F^(1/4) dà r^-3/4. Vicino all’ISCO, con zero torque, taper e proxy NT attivi, non basta sostituire sqrt con pow(...,0.75): va definito quale flusso determina la temperatura, distinguendo parametri fisici e guadagni artistici. Anche l’attuale `blackbody` è una rampa RGB approssimata, non un’integrazione Planck/CIE.

Suggerisco `legacy` e `flux_temperature` espliciti, inizialmente opt-in per il nuovo profilo; promozione a default dopo confronto e migrazione preset. Rimuovere il vecchio floor del redshift non è lavoro ancora da fare: è già disattivato. Una nuova legge termica deve evitare di contare due volte il peso Planck opzionale.

### C2 — verso del disco distinto dallo spin del buco nero

Aggiungere un verso firmato nel sistema di coordinate, più modalità legacy corotante. Attraversare a=0 mantenendo s fisso cambia da progrado a retrogrado senza invertire il disco. Questo richiede ISCO, Ω, emissività e redshift coerenti; un semplice segno aggiunto alla sola Ω sarebbe fisicamente inconsistente. Salvare la scelta nelle cache.

### C3 — area per angolo, non magnificazione adimensionale

J mappa deviazioni angolari della camera in `(δr, r δφ)` sulla superficie d’intersezione. Il determinante salvato è quindi un’area coordinata per prodotto di angoli; non include automaticamente area propria del disco, Jacobiano esatto del solido angolo della proiezione o un riferimento non lensato. I commenti che lo chiamano rapporto solido-angolo/area sono fuorvianti rispetto al codice.

Un valore 11 400 non dimostra un errore. Confrontare J con raggi perturbati e verificare il footprint prima di parlare di normalizzazione. La modulazione della luminosità è off di default ([renderer/color_params.hpp:68](/Users/iman.rosignoli/Documents/KerrTraceCpp2/renderer/color_params.hpp:68)); non forzare la mediana a uno per alterare la brillanza di una sorgente estesa. La mediana specifica non è stata ricalcolata perché manca il preset/cache d’origine.

### C4 — recuperare il lavoro esistente con un nuovo contratto

Il worktree `.worktrees/feature/photon-ring-multipass` è attivo sul commit `e40814b`, con contatore e compositing multi-hit. Il suo `geo_pixel.hpp` ha **56 byte**, campi n_ring/n_extra/debug_tag e hit extra, mentre il ramo corrente ha **64 byte** con impronte. Quindi un cherry-pick diretto non è una migrazione sicura.

Definire prima che cosa significa ordine: conteggio di attraversamenti del piano, numero di turning point polari e numero di hit emissivi non sono sinonimi. Per disco opaco ci si ferma al primo hit fisico; gli attraversamenti precedenti possono essere nel foro o fuori dal disco. Per mostrare contributi multipli attraverso un disco trasparente serve un contratto multi-hit. Contare eventi una sola volta, gestire partenza sul piano ed eventuale saturazione. `_pad` non è tutto libero: debug e sfondo dietro il disco ne occupano già parte.

### C5 — trasporto e modello emissivo sono problemi diversi

Penrose–Walker rende fattibile un primo modulo Kerr, ma non determina da sola il campo magnetico né la polarizzazione emessa. [Connors, Piran e Stark 1980](https://adsabs.harvard.edu/pdf/1980ApJ...235..224C) tratta polarizzazione da scattering e trasporto relativistico: non è una prescrizione generale del campo B per emissione di sincrotrone. Scegliere esplicitamente disco termico con scattering oppure plasma magnetizzato.

Servono momenti/direzione nel frame dell’emettitore e una base sullo schermo dell’osservatore finito; GeoPixel non li contiene. In prima fase Kerr, senza Faraday, con confronto contro trasporto parallelo numerico. Per KNdS e plasma magnetizzato non assumere che le formule Kerr bastino. Conservare/filtrare Stokes I,Q,U (eventualmente V), non mediare direttamente pol_angle. Un solo angolo e grado possono bastare a un’uscita diagnostica, non a una pipeline polarimetrica completa.

### C6 — dominio della gola e modello rotante da specificare

Per la metrica statica `ds²=−exp(2Φ)dt²+dℓ²+r(ℓ)²dΩ²`, lontano dai punti dove r′=0, una deduzione locale dà: con Φ=−m/r, l’estremo del potenziale nullo è r=m e la marginale stabilità delle orbite massive è r=2m (L²=m r²/(r−m)). **Sono risultati condizionati**, non garanzie per qualsiasi DNEG: il dominio è r≥ρ e la gola/tunnel, dove r′=0, richiede un’analisi separata. Distinguere m dal parametro `M_lens` del profilo.

Il disco geodetico sostenuto dalla gravità richiede un potenziale appropriato; una superficie emissiva artificiale è invece renderizzabile anche nel modello ultrastatico, senza chiamarla disco di accrescimento fisico.

`ω′=−6J/r⁴` è incompleto senza coordinata della derivata, condizioni al contorno, regime di rotazione e metrica. Se primo indica d/dr, l’integrazione dà ω=2J/r³+costante, ma d/dℓ introduce r′ e le due asintotiche della gola vanno trattate esplicitamente. Non certifico come “piano fisico già derivato” una specifica priva di questi elementi. Il port GPU del DNEG attuale è separabile e più semplice del nuovo modello rotante.

### C7 — stime separate

- **Switch BL↔KS:** selezione attuale a livello frame, non cambio durante il raggio. Nel Kerr singolo raggio i trasformatori esistono; per bundle occorre trasformare anche W e invalidare FSAL. La carta KS qui non copre automaticamente Λ≠0. XL per un percorso robusto, non solo soglia Δr.
- **Carter:** già calcolata per Kerr separabile. M per diagnostica di conservazione nel tracer numerico; L per classificatore con radici/segni e soglie vicino alla separatrice. Carter da sola non identifica quale evento emissivo viene prima.
- **Disco spesso:** XL (2–4 settimane indicative) per superficie analitica/volume semplice; **GRMHD XL, più settimane ulteriori**, con formati dati, interpolazione, unità, frame del fluido e trasferimento. Il tracer oggi interseca l’equatore, non integra un volume emissivo.
- **Spettro:** L per multibanda di un disco termico semplice dopo C1; XL per jν/αν, plasma, integrazione lungo il raggio e possibile polarizzazione. `planck_emission` a 550 nm non è un renderer spettrale.

### D1 — gate prioritario

Registrare il test usando il target Metal effettivo in entrambe le configurazioni; definire nomi/manifest stabili per CPU/Metal/CUDA e aggiornare il server. Un alias CMake da solo non crea un secondo eseguibile né elimina quello vecchio sul disco.

C’è un secondo difetto del test: in assenza di backend Metal stampa SKIP ma esce **0** ([tests/metal_passthrough_regression.py:77](/Users/iman.rosignoli/Documents/KerrTraceCpp2/tests/metal_passthrough_regression.py:77)). CTest lo conterebbe come PASS. Deve usare un codice skip esplicito o fallire quando Metal è obbligatoria. Il caso eseguito oggi ha usato davvero la GPU.

### D2 — header v5 e semantica riproducibile

Magic/versione/dimensioni e vari parametri ci sono già; non è corretto dire “nessun header”. Mancano record size/schema, endian esplicito, identificatore metrica (inclusi parametri DNEG), FOV, solver/tolleranze/carta, versione codice/shader e impostazioni che cambiano la visibilità. Palette, soglia del bordo procedurale, opacità e tempo possono influenzare la geometria: una cache non è universalmente ricolorabile con qualsiasi impostazione.

Separare formato su disco e struct in memoria; v4 rimane leggibile attraverso un convertitore con provenienza dichiarata incompleta. Non scrivere come verificato ciò che il vecchio file non contiene. Incrementare versione/schema anche quando il significato cambia senza cambiare sizeof.

### D3 — prima rendere affidabile il comparatore

Il comparatore considera solo chiavi presenti in entrambe le esecuzioni e solo hash PNG. Casi mancanti, backend differente e fallimenti con `png=null` possono sfuggire. Richiedere insieme di casi identico, return code valido e backend atteso.

Usare hash esatti per refactor puro su stessa toolchain; fixture numeriche/scientifiche e tolleranze motivate per CPU/GPU o driver differenti. Non aggiornare golden automaticamente quando falliscono. Per i casi intenzionalmente cambiati, riferimento indipendente e approvazione visiva dei crop nativi.

### D4 — fonte comune con piccoli entrypoint

AGENTS e CLAUDE possono rimandare a un documento comune di convenzioni/architettura senza duplicare roadmap e comandi. Restano specifiche le istruzioni del singolo agente. Lo storico OPEN-ISSUES contiene sia bug ormai risolti sia descrizioni obsolete: distinguere stato corrente e diario, senza cancellare prove utili.

### D5 — pulizia sicura, nessuna cancellazione in questo audit

I cinque head remoti indicati coincidono con gli oggetti locali verificati antenati di main. Sono candidati alla rimozione. `falling-camera-wip@d9565c9` diverge da main (73 commit solo main, 32 solo ramo): **non è spazzatura**. Contiene renderer/camera cadente Metal e test; l’eventuale port nell’architettura modulare richiede audit a parte.

Le quattro build indicate occupano circa 181 MB, quindi il recupero di spazio è modesto. `build_cpu` è ancora cercata dal server: prima D1. Esistono anche worktree attivi e registrazioni prunable in /tmp; inventariare senza rimuovere quelli attivi. Non riparare toolchain storiche per ottenere nuove misure: usare build pulite native.

### D6 — integrare il ramo con review per lotti

Main è antenato, quindi il merge corrente è meccanicamente semplice; questo non prova correttezza numerica. I 17 commit comprendono fix scientifici/backend, estrazione modulare, campionamento, performance, tool e hash. Preferisco PR impilate per sequenza di commit, se costruibili autonomamente: (1) audit/fix iniziali, (2) refactor/planner, (3) fix Metal e rumore, (4) strumenti/docs dove separabili. Non dividere artificialmente per cartella duplicando modifiche dipendenti.

Prima dell’integrazione: D1, comparator D3, build pulite CPU/Metal e backend effettivo, shader/hash identificati, regressioni sui default e legacy, controlli su KGEO. CUDA host passa qui ma validazione nvcc/device manca: dichiararlo, o eseguirla prima di certificare quella piattaforma. Non esigere identità bitwise con main per fix che cambiano intenzionalmente la fisica.

### D7 — reflection non genera la semantica delle opzioni

La libreria vendorizzata descrive campi, ma alias CLI, unità, range, help, default, conflitti e disponibilità per backend richiedono uno schema aggiuntivo. Generare JSON Schema/TypeScript e parser/validazione dal descrittore; mantenere adattatori per vecchi nomi e preset. Pilotare un piccolo gruppo di opzioni, poi migrare progressivamente. Non serve cambiare lo standard C++17 per questa operazione.

### D8 — separare cucitura dell’asset e singolarità della parametrizzazione

CPU e Metal fanno già repeat in longitudine e clamp in latitudine. Un asset con bordi longitudinali incompatibili non diventa un cielo fisicamente continuo con un filtro diverso.

Per campioni che oltrepassano un polo bisogna ricondurre la direzione sulla sfera: θ→−θ e φ→φ+π al polo nord; θ→2π−θ e φ→φ+π al sud, con normalizzazione ripetuta se necessario. Il semplice mirrored-repeat verticale senza spostamento di π in longitudine non basta. Meglio campionare l’impronta nel piano tangente delle direzioni 3D vicino al polo. Verificare separatamente raggio centrale, interpolazione degli angoli e campioni del bundle. Non attribuire tutti gli artefatti all’immagine o tutti al sampler senza il caso originale.

### E1 — rivalutazione delle quattro fasi

Esistono già `buildNavArgs`, messaggi WebSocket navigate, anteprima 320×180 e bake di immagini. Questo riduce il lavoro di UX web ma non equivale a una LUT di GeoPixel né a un renderer persistente senza avvio di processi.

- **5a, 3–5 settimane:** finestra nativa opzionale, camera orientabile, render context non globale, scheduler a richieste versionate, presentazione texture. Le 2–3 settimane originali sono plausibili solo per demo orbitale ristretta. Pan/roll non si ottengono convertendo semplicemente un quaternion in θ/φ: la camera corrente punta verso il centro e serve orientare la tetrad. Rivedere anche `g_cache`/`last_render_backend` e altre globali prima di esecuzioni concorrenti.
- **5b, 1–2 settimane dopo 5a:** coarse-to-fine e accumulo a camera ferma. Esiste già jitter camera-spp, ma l’accumulo attuale è su immagini RGB, non un buffer radiance persistente. Reset obbligatorio al cambiamento dei parametri pertinenti.
- **5c, 4–8 settimane di prototipo/validazione:** LUT opzionale a metrica e camera ristrette. 1920×1080×17 byte = **35,25 MB**, non 24; GeoPixel attuale = **132,71 MB** per slice (126,6 MiB). Interpolare θ,α,β richiede otto vicini in un reticolo 3D, non quattro; gli esiti discreti e le caustiche non si interpolano linearmente. Servono maschere, error estimator e fallback.
- **5d:** bilineare M con pipeline texture pronta; MetalFX spaziale M–L; TAA con movimento **3–6 settimane** e fattibilità media/bassa vicino alle caustiche. Un endpoint non determina univocamente il pixel a causa delle immagini multiple; occorrono branch/history validity e disocclusioni.

Correzioni tecniche al piano CLAUDE: [MTLCommandBuffer](https://developer.apple.com/documentation/metal/mtlcommandbuffer) non espone un metodo `cancel`. Fermare i nuovi invii, scartare risultati obsoleti e limitare il lavoro già inviato. [MTLFXSpatialScaler](https://developer.apple.com/documentation/metalfx/mtlfxspatialscaler) appartiene a MetalFX, con texture e command buffer: AVFoundation non è il requisito d’integrazione. L’upscale bilineare richiede campionamento/render o compute, non una semplice copia blit che ridimensioni magicamente.

Non promettere 1 ms a 128×72 o <100 ms senza benchmark cold/warm e p95; separare compilazione shader, integrazione e presentazione. La rotazione artistica del cielo è un parametro distinto dalla rotazione della camera: aggiungere automaticamente lo stesso yaw rischia di ruotare due volte una sfera già campionata in coordinate globali.

### E2 — DNGR non coincide con blur RGB o TAA

Il paper distingue sorgenti puntiformi e strutture estese. Il progetto già filtra le impronte, ma non ha tutta la ricostruzione DNGR e il modello di stelle puntiformi. Per ridurre errori di compositing bisogna anche superare la media di geometrie seguita da un unico shading ([renderer/phase1.hpp:240](/Users/iman.rosignoli/Documents/KerrTraceCpp2/renderer/phase1.hpp:240)) e la media di colori già tonemappati ([renderer/colorize.hpp:134](/Users/iman.rosignoli/Documents/KerrTraceCpp2/renderer/colorize.hpp:134)).

DNGR A.3.2 integra il movimento durante l’esposizione, con derivate temporali dei fasci; non miscela semplicemente due frame finali. Primo prototipo: riferimento supersampled spazio/tempo su esposizione dichiarata, poi filtro spaziale in radiance lineare e infine swept footprint temporale. Il 2% citato da DNGR riguarda la variazione di luminosità stellare nel proprio setup e con ipotesi precise: è un utile obiettivo di confronto, non una soglia automaticamente trasferibile a ogni scena del progetto.

## Ordine consigliato: primi cinque lotti

1. **D1 — binari e test:** costo basso, evita misure e render con eseguibili vecchi, rende affidabile tutto ciò che segue.
2. **D3 — comparatore e regressioni mirate:** prima di altri fix fisici, chiudere i falsi PASS e aggiungere invarianti indipendenti.
3. **B2 — quadranti e normalizzazione ellittica:** difetti riprodotti; primo lotto ristretto con guardie mantenute, senza promettere subito un solver completamente analitico.
4. **A1 — terminazione separabile:** aggiornamenti pass-through e trattamento dei turning point; decidere se mantenere il solver sulla base di qualità/tempo misurati.
5. **D2 — contratto KGEO v5:** prepara A4 e il recupero C4, evitando più migrazioni incompatibili.

D4 può procedere come breve attività di supporto a D1. D6 può integrare i fix già validati dopo i gate D1/D3, senza attendere le grandi funzionalità. Dopo questi lotti: A4 se interessa velocizzare il workflow .kgeo; A3 se interessa soprattutto il rendering GPU diretto; C1 se la priorità è la fedeltà termica.

## Vittorie rapide e attività da rimandare

**S con valore medio/alto:** registrazione e skip corretto del test D1; entrypoint documentali D4; chiarimento semantico di magnif C3; rimozione/documentazione del ramo opacità morto A5. Il `continue` A1 è una piccola patch, ma non una soluzione completa certificabile senza test dedicato.

**Rimandare:** LUT e TAA fino alla misura della preview; persistent threads fino al profiling aggiornato; bundle CUDA senza hardware di validazione; polarizzazione/GRMHD/wormhole rotante fino a specifica emissiva e metrica. B4 è comodità, non priorità prestazionale. A6 può essere accettato provvisoriamente con caso riproducibile se il requisito è cinematografico.

**Scartare nella forma proposta:** stesso bug KS come spiegazione completa di A1; “solo scegliere un altro τ” come diagnosi completa di B2; normalizzare magnif a uno perché è grande; mirror verticale come cura universale dei cieli; cancellazione di `falling-camera-wip`; `MTLCommandBuffer cancel`; LUT interpolata indistintamente tra esiti diversi. Non riproporre gli esperimenti già scartati dall’handoff.

## Correzioni all’handoff e allo stato documentato

1. 17 commit avanti, non 13; HEAD e assenza PR confermati sul remoto.
2. 14/14 vero nella build Metal corrente, ma manca il pass-through e il suo skip può essere contato come PASS.
3. Il `continue` separabile esiste, ma l’equatore usa stato fresco e i controlli radiali precedono quel punto: meccanismo diverso dal vecchio KS.
4. L’aspetto blackbody Metal slavato resta un’osservazione non riprodotta; il floor g=0,2 non è più il default.
5. Region III: F perde il quadrante, r(0) è già sbagliato; la diagnosi post-bounce non basta. Anche scala del tempo e trace numerici nascosti richiedono intervento.
6. CUDA condivide il colore CPU: il blocco Interstellar è principalmente nella visibilità del disco.
7. Magnif è un determinante grezzo della mappa angolare, non una magnificazione adimensionale già normalizzata.
8. Carter è già presente; C4 è implementata in un ramo non integrato, con contratto incompatibile.
9. Il file KGEO ha già header e parametri KNdS; mancano schema estensibile e provenienza sufficiente.
10. La navigazione web e parte dei filtri spaziali esistono già; non tutta E1/E2 parte da zero.
11. La scelta dei vecchi binari è ancora codificata nel server, non soltanto in documentazione.
12. Formule wormhole condizionate al dominio e al modello; Connors 1980 non specifica un campo magnetico di sincrotrone.
13. Le stime LUT e le API Metal del piano interattivo vanno corrette come indicato sopra.

## Evidenze riproducibili

Tutti i log e i piccoli programmi diagnostici sono in `out/audit-migliorie-2026-11-15/`: `build.log`, `ctest.log`, `ctest-detail.log`, `passthrough.log`, `probe.cpp/.log`, `probe_region3.cpp/.log`, `probe_F.cpp/.log`. Gli ultimi due chiamano direttamente le funzioni del solver senza modificarle. Il probe separabile usa eventi semplificati ed è esplicitamente esplorativo.

Le soglie dei futuri test non sono state inventate qui: prima si salvano i casi difettosi e si verifica che il test fallisca sul commit attuale, poi si applica il fix e si confronta con un riferimento indipendente. Per funzionalità nuove si usa un mutante controllato (segno, campo, branch o callback rimosso) o un caso analitico, non il solo confronto con il vecchio comportamento. Per prestazioni: mediana e p95 su esecuzioni ripetute e senza competizione GPU; nessuna soglia assoluta derivata da una singola misura storica.

Fonti locali consultate: [Gralla–Lupsasca, Appendice B](/Users/iman.rosignoli/Documents/KerrTraceCpp2/sources/Gralla_Lupsasca_NullGeodesics_Kerr_2019_1910.12881.pdf), [DNGR, Appendice A.3](/Users/iman.rosignoli/Documents/KerrTraceCpp2/sources/DNGR_James_Thorne_2015_1502.03808.pdf).
