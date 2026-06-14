---
name: ambit-development
description: >
  How to DEVELOP and contribute to ambit — dev environment setup, the
  optional-dependency tier system (io/reduce/umap/ann/gpu/faiss/model/all) and how
  to add one, the codebase conventions (numpy-first, lazy optional imports with
  helpful errors, dataclass contracts, Config-not-env-vars, graceful degradation,
  streaming, module docstrings that explain the "why"), and the concrete recipes for
  extending it: a new input format, a new diagnostic metric, a new kNN/projection/
  device backend, or a new figure. Plus how to smoke-test changes. Use this when
  modifying ambit's code or reviewing a change. For the structure see
  ambit-architecture; for figures see ambit-figures.
version: 1.0.0
---

# Developing ambit

A small numpy-first package with opt-in heavier backends. Read **ambit-architecture**
for the structure first; this skill is about *working in the code*.

## Dev setup

```bash
cd /path/to/ambit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"          # editable install + analysis extras (io, reduce, umap, ann)
# add what you're testing:  pip install -e ".[all,gpu]"   (torch)   or  ".[all,model]"  (sentence-transformers)
ambit info <some.npy>            # smoke test the console entry point
```

Build backend is **hatchling**; the package is `src/ambit` (src-layout), Python
**≥ 3.10**, console script `ambit = ambit.cli:main`. There is currently **no test
suite or CI** in the repo — verify changes by running the CLI / library directly
(see *Smoke-testing* below). Adding `tests/` (pytest) is a welcome contribution.

## The dependency tiers

Core install is **numpy only** — it ingests `.npy`/`.npz` and runs every
native-space diagnostic. Everything else is an opt-in extra
(`pyproject.toml [project.optional-dependencies]`):

| extra | brings | unlocks |
|---|---|---|
| `io` | pandas, pyarrow | parquet / csv / jsonl with a vector column |
| `reduce` | scikit-learn, scipy | PCA (randomized), exact kNN, projections |
| `umap` | umap-learn | non-linear UMAP projection |
| `ann` | pynndescent | approximate kNN for a large reservoir |
| `gpu` | torch | CUDA/MPS device backend (`accel.py`) |
| `faiss` | faiss-gpu | GPU kNN for very large reservoirs |
| `model` | sentence-transformers | embed raw items on the fly (tier 3) |
| `all` | the analysis set | io + reduce + umap + ann |

**Rule:** the core must stay importable and functional with numpy alone. Any heavier
capability is lazily imported *inside the function that needs it* and fails with a
message pointing at the extra.

## Conventions (match these)

- **`from __future__ import annotations`** at the top of every module.
- **Lazy optional imports with a helpful error.** Never import an optional dep at
  module top level. Inside the function:
  ```python
  try:
      import pyarrow.parquet as pq
  except ImportError as e:
      raise ImportError("reading parquet needs pyarrow: pip install 'ambit[io]'") from e
  ```
  For *optional enhancements* (not hard requirements), fall back silently to a numpy
  path instead (see `project.pca`'s randomized-SVD fallback, `knn.knn`'s backend
  cascade).
- **numpy-first, float32 vectors.** Vectors are `(n, d)` float32 contiguous;
  accumulate reductions in float64 (see `_NumpyCov`: float32 grams → float64 totals).
- **Dataclasses are the contracts.** `EmbeddingSet`, `Chunk`, `Scan`, `Ctx`, `Config`
  are dataclasses; validate invariants in `__post_init__` (fail loudly at the
  boundary so downstream code is clean). Don't smuggle state outside them.
- **No environment variables.** Every setting is a `Config` field set by a CLI flag
  or `--config` JSON. If you add a setting, add it to `Config` *and* the CLI.
- **Graceful degradation.** Optional capabilities return `None`/a fallback rather
  than crashing: `build_ctx` wraps kNN/clustering in try/except; figures return a
  terse fallback dict when a `Ctx` field is `None`; `render._load_figures` skips a
  broken figure with a stderr note. Preserve this — one missing dep or broken figure
  must never take down a run.
- **Streaming by default.** New ingestion stays an iterator of `Chunk`s so large
  corpora never fully load; scalar diagnostics run over the full corpus, visuals over
  the reservoir sample.
- **Module docstrings explain the *why*.** Every module opens with a paragraph on its
  role and the reasoning behind its design (see `types.py`, `scan.py`, `accel.py`).
  Match that voice; keep inline comments about *intent*, not mechanics.
- **Private helpers are `_`-prefixed** (`_box`, `_pick`, `_NumpyCov`, `_from_table`).
- **Figures** follow the zicato design language (token-only SVG, no hex) — see
  ambit-figures.

## Extension recipes

### Add an input format
1. `ingest.py` — handle the extension in `load()` (return one `EmbeddingSet`); reuse
   `_pick` + `_EMB_KEYS`/`_ID_KEYS`/`_LABEL_KEYS` for column detection, or add keys
   there.
2. `source.py` — add a streaming `_<fmt>_chunks` generator and wire it into
   `iter_chunks()` so the scan can consume it; add the extension to `expand()` if it
   should be discoverable in a sharded directory.
3. Keep the heavy parser lazily imported with an extra-pointing error.

### Add a diagnostic metric
1. `metrics.py` — a pure-numpy function over the high-dimensional vectors (or the
   precomputed `cov`/`eigs`). Keep it dependency-free.
2. Surface it: print it in `cli.cmd_info`, and/or read it from a new RES-family figure
   (ambit-figures). If it needs the eigenspectrum, take `eigs` so it works over the
   full corpus via the streaming scan.

### Add a kNN backend
`knn.knn()` dispatches on `backend`; add a branch returning `(idx, dist)` with self
excluded and cosine `dist = 1 - cos`. Add the name to the CLI `--knn-backend`
choices and `Config.knn_backend`. For a GPU variant, mirror it in
`accel.torch_knn`. Downstream is backend-agnostic — just honor the `(idx, dist)`
contract.

### Add a projector
`project.project()` switches on `method`; add a branch returning the `(m, n_components)`
array. Add the name to the `report --projector` choices and `Config.projector`.
Projection runs on the reservoir only.

### Add a device kernel
`accel.py` mirrors a CPU kernel on a torch device and **returns numpy at the
boundary** so the rest of ambit is unchanged. Add the kernel, route to it from
`scan`/`build_ctx` when `device != "cpu"`, and keep a numpy path for `device="cpu"`.

### Add a dependency tier
Add the group to `pyproject.toml [project.optional-dependencies]` (and to `all` if
it's part of core analysis). Then lazily import it with the
`pip install 'ambit[<tier>]'` error message. Never make it a hard dependency.

### Add a figure
See **ambit-figures** — drop an `@figure` module in `figures/`, return the contract
dict, register its default in `config.DEFAULT_FIGURES`. Discovery is automatic.

## Smoke-testing a change

No suite yet, so exercise the real paths on tiny data:

```bash
python - <<'PY'
import numpy as np
np.save("/tmp/toy.npy", np.random.default_rng(0).standard_normal((2000, 64)).astype("float32"))
PY
ambit info   /tmp/toy.npy                 # diagnostics path (scan + metrics)
ambit report /tmp/toy.npy --out /tmp/r.html   # full pipeline + every enabled figure
```

Library-level (what `report` does under the hood):

```python
from ambit.scan import scan
from ambit.pipeline import build_ctx
from ambit.render import build_report
ctx = build_ctx(scan("/tmp/toy.npy"))
open("/tmp/r.html", "w").write(build_report(ctx))
```

Check: `ambit info` numbers are sane; the report opens, every enabled figure renders,
and the theme picker recolors all of them (proves figures use tokens, not hex). When
touching scale paths, also try a sharded dir and `--approx`/`--device`/`--knn-backend`.

## Repo / workflow

- Remote: `git@github.com:pedapudi/ambit.git`.
- Commit subjects are short and descriptive (e.g. *"Add sharded directory / glob
  source support"*, *"Remove env-var config: everything is a CLI flag in a Config
  object"*). Keep one logical change per commit.
- Touch `README.md` / `docs/` when you change user-facing behavior; keep the
  `docs/concepts` and `docs/design` notes in sync with the diagnostics/figures they
  describe.
