# KerrTraceCpp2 — istruzioni per Codex

Ray tracer relativistico per buchi neri di Kerr-Newman-de Sitter (e wormhole DNEG):
geodetiche nulle fisicamente corrette, CPU/Metal/CUDA, fasci di raggi. Obiettivo:
immagini di qualità da pubblicazione, come DNGR/Interstellar.

Questo file è solo il punto d'ingresso. Il contenuto comune a tutti gli agenti sta
in `docs/`, scritto una volta sola:

| leggi | per |
|---|---|
| [`docs/ARCHITETTURA.md`](docs/ARCHITETTURA.md) | **per primo**: albero dei file, flusso a due fasi, contratto `GeoPixel`, quale tracciatore corre quando, build e test, regole di lavoro, trappole |
| [`docs/FISICA.md`](docs/FISICA.md) | formule, convenzioni del codice (unità, segni, ordine delle coordinate), riferimenti |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | cosa funziona, backlog con priorità ed effort |
| [`docs/OPEN-ISSUES.md`](docs/OPEN-ISSUES.md) | problemi aperti oggi |

Se un documento contraddice il codice, ha ragione il codice: correggi il documento
nella stessa PR.

## Note specifiche per Codex

- Gli audit e le valutazioni richieste con un handoff vanno in
  `docs/AUDIT-<argomento>-<data>.md`, con evidenze `file:riga`, comandi eseguiti e
  ciò che **non** è stato verificato. Gli handoff (`HANDOFF-*.md` nella radice)
  sono locali e non versionati.
- Una misura non ripetuta in questo audit va dichiarata come storica, non come
  risultato.
- Lavori su un ramo proprio (`codex/<argomento>`), con le stesse regole di test e
  PR di `docs/ARCHITETTURA.md`.
