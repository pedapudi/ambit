---
name: ambit-architecture
description: >
  How ambit is BUILT — the end-to-end data flow (source → scan → build_ctx →
  render), the core contracts (EmbeddingSet, Chunk, Scan, Ctx, Config), a
  module-by-module map of src/ambit/, and the key design decisions (streaming
  one-pass scan with a reservoir sample, single canonical type so the three input
  tiers degrade gracefully, optional-dependency backends for kNN/projection/GPU, no
  environment variables). Use this to navigate or modify the codebase, trace where a
  value comes from, or understand why something is structured the way it is. To work
  specifically on report figures see ambit-figures; for dev setup/conventions see
  ambit-development.
version: 1.0.0
---

# ambit architecture

ambit is a small Python package (`src/ambit/`) built around **one in-memory type**
that flows through a **streaming scan**, a **shared render context**, and a
**figure registry**. numpy is the only hard dependency; everything heavier is an
optional backend selected at runtime.

## The data flow

```
                          ┌──────────── tier 2/3: raw items ────────────┐
                          │ embedding.py: EmbeddingClient + embed_dataset│
                          │  (OpenAI-compatible /v1/embeddings, stdlib)  │
                          └───────────────────┬──────────────────────────┘
                                              │ writes .jsonl/.parquet
                                              ▼
  source.py  ──iter_chunks()──►  Chunk(X, ids, labels)   (streaming, fixed-size)
   (file / glob / sharded dir / ndarray; parquet via zero-copy Arrow)
                                              │
                                              ▼
  scan.py    ──scan()──►  Scan(n, dim, cov, mean, norm_mean/std, sample, scanned)
   one pass: _NumpyCov (or accel.TorchCov) accumulates covariance + norms;
   a bounded **reservoir sample** (Algorithm R) is kept for the visual layers;
   --approx N stops early; _source_rows() recovers the true corpus size cheaply.
                                              │
                                              ▼
  pipeline.py ──build_ctx()──►  Ctx(xy, xyz, eigs, cos, knn_idx/dist, labels, …)
   project.py (PCA/UMAP on the reservoir) · metrics.random_pair_cosine ·
   knn.py (kNN graph) · cluster.py (auto-labels if no label column) ·
   accel.py mirrors the hot kernels on a torch device when device != "cpu"
                                              │
                          ┌───────────────────┴───────────────────┐
                          ▼                                         ▼
  cli.cmd_info: metrics → terminal        render.build_report(ctx) → self-contained HTML
   (info path stops at Scan + metrics)     iterate FIGURES registry; figures/*.py; theme assets
```

`ingest.py` is the non-streaming sibling of `source.py`: it loads a whole file into
one `EmbeddingSet` (used where the full set fits in memory / by library callers),
sharing the same column-name detection.

## The core contracts

### `EmbeddingSet` (`types.py`) — the canonical type

Everything normalizes to this so downstream code never branches on provenance.

```python
EmbeddingSet(
  X: np.ndarray,            # (n, d) float32 — the vectors (validated finite, 2-D, non-empty)
  ids=None, labels=None,    # (n,) optional; ids default to arange(n)
  meta=None,                # optional pandas DataFrame of per-item metadata
  metric="cosine",          # "cosine" | "euclidean"
  normalized=False,         # True once rows are L2-normalized
  source=None,              # provenance string for reporting
  ids_provided=True,        # False when ids fell back to arange (no stable id column)
)
# .n, .dim, .normalize(eps=1e-12) -> copy, .subsample(k, seed) -> deterministic copy
```

`ids_provided` is `False` when no id column was supplied and `ids` defaulted to row
indices. It matters for `--compare`: aligning two independently sampled sets needs a
real, stable id, so `compare.build_cmp` refuses when `ids_provided` is `False` rather
than pairing unrelated rows.

`__post_init__` enforces the invariants (float32 contiguous, 2-D, finite, matching
id/label lengths) — invalid data fails loudly at construction, so no figure has to
defend against NaN/shape bugs.

### `Chunk` (`source.py`) — the streaming unit

`Chunk(X, ids, labels)` — a fixed-size block of rows yielded by `iter_chunks`, so a
1M+ corpus is never fully resident.

### `Scan` (`scan.py`) — the streaming-scan result

```python
Scan(n, dim, cov, mean, norm_mean, norm_std, sample: EmbeddingSet, source,
     metric="cosine", scanned=0)
#  .eigs       -> eigenvalues of cov (full-corpus spectrum)
#  .approximate-> True if scanned < n (i.e. --approx stopped early)
```

`cov` is the streamed covariance over the scanned rows (exact full-corpus spectrum
unless `--approx`); `sample` is the bounded reservoir the visuals work from; `n` is
the true corpus size when cheaply known, else `scanned`.

### `Ctx` (`pipeline.py`) — the shared render context

Computed once from a `Scan`, passed to every figure (read-only):

```python
Ctx(scan, xy, xyz, eigs, cos, knn_idx, knn_dist, labels, labels_source,
    hub_skew, mutual_knn, cmp)
#  xy (m,2), xyz (m,3)  projected reservoir
#  eigs                 full-corpus covariance eigenvalues
#  cos                  random-pair cosine sample
#  knn_idx/knn_dist     (m,k) kNN over the reservoir (None if no backend)
#  labels/labels_source provided metadata column, else unsupervised clusters
#  hub_skew             k-occurrence skewness
#  mutual_knn           bool — the kNN graph is filtered to reciprocal (mutual) edges
#  cmp                  CmpCtx (compare.py) when --compare was given, else None
#  .es  == scan.sample  (the reservoir EmbeddingSet, L2-normalized in build_ctx)
```

Optional fields (`knn_*`, `labels`, `hub_skew`, `xyz`, `cmp`) may be `None`; figures
must degrade gracefully. `mutual_knn` is applied once in `build_ctx` (every downstream
reader sees the same reciprocal graph); a figure that recomputes its own neighbors
reads `ctx.mutual_knn` and honors it. `cmp` is filled by `api.report` when
`config.compare` is set — it carries the two-embedding comparison the CMP figures draw.

### `Config` (`config.py`) — the single run description

A dataclass holding every ingestion / scan / projection / device / kNN / embedding
setting **plus** the `figures` enable map. The CLI builds it from flags; a
`--config` JSON is merged on top via `Config.merge()` (known keys set fields; a
`"figures"` object or a bare slug toggles a figure). `DEFAULT_FIGURES` is the
default enable map; `enabled(figures, key)` resolves a figure's visibility. There
are **no environment variables** — a run is fully described by its `Config`.

## Module map (`src/ambit/`)

| module | role |
|---|---|
| `cli.py` | argparse front end; `cmd_info` / `cmd_embed` / `cmd_report`; builds `Config` |
| `__main__.py` | `python -m ambit` → `cli.main` |
| `config.py` | `Config` dataclass, `DEFAULT_FIGURES`, `enabled()`, `merge()` |
| `types.py` | `EmbeddingSet` — the canonical in-memory contract |
| `source.py` | streaming `iter_chunks()` + `expand()` (glob/dir → shards); zero-copy Arrow vector reads |
| `ingest.py` | non-streaming `load()` → one `EmbeddingSet`; column-name detection (`_EMB_KEYS`/`_ID_KEYS`/`_LABEL_KEYS`, `_pick`) |
| `scan.py` | one-pass `scan()` → `Scan`; `_NumpyCov` accumulator; reservoir sampling; `_source_rows()` |
| `metrics.py` | native-space diagnostics: random-pair cosine, eigs, effective rank, participation ratio, dims-for-variance, isotropy ref, IsoScore, uniformity (Wang–Isola), hubness skew |
| `project.py` | `project()` → PCA (numpy SVD or sklearn randomized) / UMAP, on the reservoir; `pca_fit`/`pca_transform` (the shared PCA frame the drift figures reuse) |
| `cluster.py` | unsupervised labeling: HDBSCAN if present, else MiniBatchKMeans with silhouette-picked *k* |
| `knn.py` | `knn()` with backends auto/pynndescent/sklearn/brute/faiss; cosine dist = 1−cos; `topk_cosine`; `reciprocal_mask`/`reciprocal_filter` (mutual-kNN) |
| `separability.py` | the label-aware separability panel (RES 05b): centroid-cosine matrix, kNN purity, silhouette, Fisher ratio, and — for discovered clusters — stability + Laplacian eigengap |
| `local_anisotropy.py` | the local density field (RES 06/07): per-item k-NN density against a density-matched uniform null |
| `compare.py` | the two-embedding comparison + `CmpCtx`/`build_cmp`: id-aligned, linear & RBF CKA, MMD², energy, Procrustes, neighbor overlap |
| `accel.py` | optional torch backend mirroring the hot kernels (`TorchCov`, `torch_pca`, `torch_random_pair_cosine`, `torch_knn`); `resolve_device` |
| `pipeline.py` | `build_ctx()` — assembles the `Ctx` from a `Scan` |
| `api.py` | the library verbs (`report` / `diagnose` / `embed` / `build_context`); `report` wires `build_cmp` and the CMP auto-on figures when `config.compare` is set |
| `render.py` | `build_report()`, the `@figure` registry (`FIGURES`), `_DISPLAY_ORDER`, SVG helpers (`_box`/`_svg`/`_local_density`), theme assembly |
| `figures/*.py` | one module per figure; see **ambit-figures** |
| `embedding.py` | tier-2 `EmbeddingClient` (stdlib OpenAI-compatible) + `embed_dataset` streaming writer |
| `assets/theme.css`, `assets/picker.js` | inlined into every report (16-theme tokens + live picker) |

## Key design decisions (the "why")

- **One canonical type, three tiers.** Because everything becomes an `EmbeddingSet`,
  the three input tiers (have-embeddings / +dataset / +model) *degrade gracefully* —
  no downstream branch depends on where the data came from (`types.py` docstring).
- **Streaming, one pass.** `scan()` computes covariance + norms in a single pass with
  a `d×d` scatter (scales in *n*, not *n²*), keeping only a bounded reservoir for the
  visuals. Scalar diagnostics are over the full corpus; visuals over the sample.
  `--approx N` trades exactness for speed (rank/variance converge fast in *n*).
- **Covariance via the d×d scatter.** `_NumpyCov` accumulates `XᵀX` in float64 from
  float32 grams (BLAS sgemm), so the eigenspectrum is exact and memory is `O(d²)`.
- **Reservoir sample for visuals.** Projections, kNN, clusters, and the cosine
  histogram all run on the reservoir (`--sample`, default 20k), never the full set —
  the scatter/3-D views never draw more than a few thousand points anyway.
- **Optional backends, runtime-selected.** kNN (`pynndescent`/`sklearn`/`brute`/
  `faiss`), projection (`umap`), tabular IO (`pandas`/`pyarrow`), GPU (`torch`),
  on-the-fly embedding (`sentence-transformers`) are all optional; a missing one
  disables a capability rather than crashing. Lazy `import` inside the function that
  needs it is the standard pattern.
- **GPU mirrors, numpy boundaries.** `accel.py` reimplements the GEMM-heavy kernels
  on a torch device and returns numpy at the boundaries, so figures/metrics/render
  are unchanged whether you run on CPU or GPU.
- **Config-as-description, no env vars.** The whole run is one `Config`; reproducing
  a run means reproducing its `Config`/`--config` JSON.
- **Self-contained reports.** `render.build_report` inlines CSS + JS + SVG into one
  HTML file with no external requests; figures are theme-token-only SVG (see
  ambit-figures).
- **Resilience.** `build_ctx` wraps kNN/clustering in try/except (a missing backend
  → `None`, not a crash); `render._load_figures` skips a broken figure module with a
  stderr note rather than failing the whole report.

## Tracing a value (worked example)

"Where does the random-pair cosine histogram in the report come from?"
`cli.cmd_report` → `scan()` builds the reservoir → `pipeline.build_ctx` calls
`metrics.random_pair_cosine(es.X, …)` (or `accel.torch_random_pair_cosine` on GPU)
and stores it as `Ctx.cos` → the `cos_hist` figure reads `ctx.cos` and draws it
against the isotropic reference `1/√d` (`metrics.isotropy_ref`).
