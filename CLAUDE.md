# KerrTraceCpp2 — istruzioni per Claude

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

## Note specifiche per Claude

- File temporanei, probe e render di confronto nella scratchpad della sessione, non
  nel repository né in `out/` se non servono all'utente.
- Mai `git stash` dentro un ciclo di prove: per confrontare con il codice vecchio
  usare un worktree (`git worktree add --detach <dir> <commit>`) o `git show`.
- I `HANDOFF-*.md` nella radice sono note locali fra agenti, non versionate.
- Commit e PR: messaggi in inglese, documentazione in italiano, una PR per voce del
  backlog (vedi le regole in `docs/ARCHITETTURA.md`).
