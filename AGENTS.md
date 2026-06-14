# AGENTS.md

Guidance for AI coding agents (and new human contributors) working in this repo.
It is **agent-agnostic** — load it however your tool ingests repo context. For
deeper, task-specific guides see [`skills/`](skills/).

## What ambit is

ambit measures and visualizes how a dataset **occupies** an embedding space —
density hotspots, coverage voids, and the resolution / isotropy of the space — as
a terminal scan or a self-contained, theme-adaptive HTML report. One in-memory
type (`EmbeddingSet`) flows through a streaming **scan** → a shared render context
(`Ctx`) → a registry of **figures**.

## Environment & setup

**Python ≥ 3.10. Always use a virtualenv — never install into system Python.**

```sh
python -m venv .venv && . .venv/bin/activate
pip install -e '.[all]'        # or core only: pip install -e .
```

Core install is just **numpy**. Heavier capabilities are **opt-in extras**:
`io` (parquet/csv/jsonl), `reduce` (PCA · kNN · scikit), `umap`, `ann`
(pynndescent), `gpu` (torch CUDA/MPS), `faiss`, `model` (sentence-transformers),
`all`. If a venv already exists at `.venv/`, use it (`.venv/bin/python`).

## Commands

ambit installs an `ambit` console script (`ambit.cli:main`); equivalently
`python -m ambit`.

- `ambit info <embeddings>` — streaming scan → resolution / occupancy diagnostics.
- `ambit report <embeddings> --out report.html` — the self-contained HTML report.
- `ambit embed <dataset> --out … --model …` — embed raw items via an
  OpenAI-compatible endpoint (or a local sentence-transformers model, `model` extra).

`<embeddings>` is `.npy` / `.npz` / `.parquet` / `.jsonl`, or a directory / glob of
parquet shards (streamed). Common flags: `--embedding-col`, `--label-col`,
`--device cpu|auto|cuda|mps`, `--approx N` (cap the scan for 1M+ rows),
`--knn-backend auto|pynndescent|sklearn|brute|faiss`, `--config <json>`. See
`skills/ambit-cli` for every flag.

## Conventions (hard rules)

- **Configuration is a `Config` object + CLI flags. Never environment variables.**
- **numpy-first.** Heavy deps (torch, scikit-learn, pyarrow, umap, pynndescent)
  are imported **lazily**, inside the function that needs them, so the core stays
  light and importable with numpy alone.
- **Figures are pure functions:** `@figure def fig_<slug>(ctx) -> dict` returning
  `num / order / name / tech / why / svg / legend / reveal / cls [/ script]`. They
  emit **token-colored static SVG**, so a theme swap re-skins with no re-render.
  Compute every number at **generation time** and bake it into the SVG — do not
  compute figure values in the frontend. See `skills/ambit-figures`.
- Stay in the project's visual voice: Tufte-style data-ink, a single accent color,
  good/bad communicated by **direction** (not a fixed hue), theme-adaptive across
  all themes. The design language lives in the study under `docs/design/`.

## Layout

- `src/ambit/` — the engine: `scan.py` (streaming scan), `pipeline.py`
  (`build_ctx` / `Ctx`), `project.py`, `knn.py`, `cluster.py`, `metrics.py`,
  `render.py` (+ `figures/`, `assets/`), `source.py` (input streaming),
  `ingest.py`, `embed.py` (embedding client), `accel.py` (torch backend),
  `cli.py`, `config.py`, `types.py`.
- `docs/concepts/` — the science (anisotropy & resolution), with citations.
- `docs/guide/` — how to interpret a generated report.
- `docs/design/` — the visualization design study (the design language).
- `skills/` — focused guides for using and developing ambit.

## Deeper guides

| want to… | read |
|---|---|
| understand ambit | `skills/ambit-overview` |
| run it | `skills/ambit-cli` |
| read the results | `skills/ambit-concepts` · `docs/guide/` |
| understand how it's built | `skills/ambit-architecture` |
| work on report figures | `skills/ambit-figures` |
| contribute | `skills/ambit-development` |
