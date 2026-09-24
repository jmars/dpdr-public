# Self-application as a common source of benefit and failure

Artifact for the paper `paper.md`:

> *Self-Application as the Common Source of Benefit and Failure: A Control-Theoretic
> Model of Self-Regulation, Its Escapes, and a Standing-Cost Budget*

A five-state control-theoretic model whose failure mode is collapse of a
self-content generator into a self-maintaining state with no outward pull, together
with its measured thresholds, its escape taxonomy, and three non-dynamical
substrate experiments.

## Layout

| path | contents |
|---|---|
| `paper.md` | the paper (markdown — the source of truth) |
| `paper.pdf` | the paper rendered to PDF, generated from `paper.md`: `pandoc paper.md --pdf-engine=typst -o paper.pdf` |
| `figures/` | the paper's twelve figures (renumbered copies; originals in `dpdr/figs/`, `csd/figs/`) |
| `dpdr/` | the frozen model and its opt-in variants (permissive, regulator, window, generational), experiment drivers with their pre-registrations and design scans, cached results, tests, figures |
| `csd/` | the critical-slowing-down / border-collision audit |
| `sref2.py` | SLD-resolution engine (substrate leg 1: tabling) |
| `cutred2.py` | cut-elimination (substrate leg 2) |
| `datalog-leg.py` | Datalog provenance (substrate leg 3) |
| `claim-audit.md` | every claim sorted into dynamical fact / marked interpretation / demoted projection |
| `what-held.md` | the measured results that survived all revisions |
| `dual-control.md` | the literature read that positions the contribution |
| `dpdr/source-library-passages.md` | the tradition-attestation source record — quoted Source Library translations (CC-BY-SA-4.0; see its licence note) |

## License

Dual-licensed, by content type:

- **`paper.md` and `figures/`** — [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
  (see `LICENSE-paper`).
- **Everything else** — the code (`dpdr/`, `csd/`, and the standalone substrate scripts)
  and the companion documents (`claim-audit.md`, `what-held.md`, `dual-control.md`) —
  MIT (see `LICENSE`). The copyright holder is Jaye Timothy Marshall; the paper's
  author line carries the same name and ORCID iD
  ([0009-0002-5209-8161](https://orcid.org/0009-0002-5209-8161)), as does the
  Zenodo metadata (`.zenodo.json`).
- **One exception** — the quoted passages in `dpdr/source-library-passages.md` are
  Source Library's AI-generated translations, licensed by their source under
  [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/) with attribution
  to Source Library, as that file's licence note states; the underlying historical
  texts are public domain. The share-alike term attaches to reuse of those quoted
  translations, not to the rest of this repository.

## Reproducing

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # pinned numpy/scipy/matplotlib/pytest
cd dpdr && ../.venv/bin/python -m pytest tests/ -q     # 53 tests
../.venv/bin/python experiments/exp10_regimes.py        # regime map
```

Every number in the paper is traceable to a cached `.npz` under `dpdr/cache/` or
`csd/cache/`, regenerable by the committed drivers.

Scope: the artifact covers the paper's three contributions — the frozen model and its
escape structure, the opt-in regulatory/window variants with their measured thresholds,
and the generational line (`dpdr/dpdr/generational.py` with experiments exp12, exp15,
exp16, their pre-registrations, caches, and figures). The consolidation and values
modules (`exp8_consolidation.py`, `exp9_values.py`) and the composed-regulator work
(`exp13_compose.py`, `exp14_composed.py`) are architecture and follow-up-paper material
cited nowhere in `paper.md`; they are deliberately excluded, as are the project's
agent-side documents. The paper's twelve embedded figures are renumbered copies under
`figures/`; three further experiment figures that the paper cites as artifacts but does
not embed (`f16_generational.png`, `f19_blindlever.png`, `f20_traintest.png`) ship
under `dpdr/figs/`.

The standalone substrate scripts need only the standard library:
`python3 sref2.py`, `python3 cutred2.py`, `python3 datalog-leg.py`.

## Status

Prepared for public release as a preprint plus its computational artifact. Dual license
chosen (see above: CC-BY-4.0 for the paper and figures, MIT for everything else),
copyright holder Jaye Timothy Marshall. `.zenodo.json` carries the deposit metadata,
including the author's ORCID iD. A DOI is minted by the deposit itself.
