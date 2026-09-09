# Punto cerchiato: gradino sul bordo esterno sinistro

La seconda foto dell'utente localizza il difetto: un breve tratto quasi
verticale interrompe il bordo obliquo esterno sinistro del disco. Nei frame
di riferimento a 640×360 si trova circa in **(178,95)**, coordinate da zero.
Il bordo Metal BL resta a x=178 per le cinque righe y=93…97; il bordo CPU
stretto passa invece da x=174 a 170, senza quel tratto verticale.

**Correzione della copertura precedente:** la regione `upper` iniziava a
x=192 a questa risoluzione. Il punto cerchiato era fuori dal test. Gli scarti
di 2–3 pixel riportati nel primo resoconto riguardavano altre parti del
contorno e non misuravano questo gradino.

## Test aggiornato

Il test conserva `upper` e aggiunge `left-bump`, centrata sul tratto indicato:
x=⌊0,22W⌋…⌊0,30W⌋, y=⌊0,24H⌋…⌊0,32H⌋. A 640×360 sono x=140…192,
y=86…115. Si esige un minimo di 20 punti del riferimento (31 osservati),
con distanza massima consentita di **un pixel**. Per `upper` resta il minimo
di 500 punti. Camera, scene, tolleranze e soglie RGB sono quelle del
[primo resoconto](CONTOUR-TESTS-2026-09-09.md).

La distanza confronta i punti presenti nella regione con il bordo completo
dell'altra immagine. Il vicino più prossimo può essere appena fuori dalla
regione: tagliarlo introdurrebbe un errore artificiale alle estremità del
tratto. Una nuova verifica sintetica protegge questo caso. Le otto verifiche
della metrica passano.

## Risultati sulla regione left-bump

| Caso | Max CPU normale vs stretta | Max Metal vs CPU stretta | Esito Metal |
|---|---:|---:|---|
| BL, RK4, personalizzato | 1 px | **5 px** | FAIL |
| BL, RK4, standard | 1 px | **5 px** | FAIL |
| KS, RK4, personalizzato | 0 px | **1 px** | PASS |
| BL, DOPRI5 | 0 px | **5 px** | FAIL |
| BL, RK4, jitter | 0 px | **5 px** | FAIL |
| BL, RK4, 1280×720 | 0 px | **9 px** | FAIL |

Salvo l’ultima riga, tutti i frame sono 640×360. Massimi ed esiti Metal
identici alle soglie RGB 1, 3 e 8. Matrice completa: **39/72 PASS, 33/72 FAIL**;
36 controlli CPU passati, tre confronti Metal KS sul bordo sinistro passati,
33 confronti Metal oltre soglia. Nessun test saltato.

Il gradino in BL resta anche con temperatura/Doppler standard, DOPRI5 e
jitter subpixel. **KS invece rispetta la soglia in questa regione**, pur
avendo altri residui nella regione `upper`. La differenza restringe
l'indagine al percorso numerico BL di Metal. Non dimostra quale operazione
sia responsabile: derivate, adattamento del passo ed eventi di intersezione
restano da isolare. Non sono state modificate fisica, shader o tolleranze
di produzione.

Gli scarti qui riportati sono distanze Chebyshev fra contorni. Una differenza
orizzontale a parità di riga è una misura diversa: nel tratto del gradino
arriva a circa 9 pixel a 640×360. Non va confusa con il massimo di 5 pixel
del test, che cerca il punto più vicino anche nelle righe adiacenti.

## Riproduzione ed evidenze

```sh
cmake --build build_fixes --parallel --target \
  kerrtrace_contour_metrics_tests kerrtrace_metal_contour_test
./build_fixes/kerrtrace_contour_metrics_tests
./build_fixes/kerrtrace_metal_contour_test --extended --output-dir out/contour-bump

# Grafico opzionale dei dati misurati, richiede matplotlib:
python3 tests/plot_contour_diagnostics.py \
  out/contour-bump/bl-custom-left-bump-metal-vs-cpu-t3.csv \
  out/contour-bump/punto-cerchiato.png
```

La matrice estesa esegue due regioni × sei scene × tre soglie × due confronti.
Un fallback CPU produce errore; l'assenza di un dispositivo Metal dà SKIP.
Il test attualmente fallisce: la sua copertura è stata corretta, il bump
del renderer è ancora aperto.

Frame originali, CSV, profilo del bordo sinistro e log sono in
`out/contour-bump/`, ignorati da Git. Il grafico mostra coordinate misurate,
senza ritoccare i frame. `punto-cerchiato.zip` raccoglie queste evidenze.
