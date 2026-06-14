---
name: ambit-overview
description: >
  Orientation and mental model for ambit — a tool that visualizes and measures how
  a dataset OCCUPIES an embedding space (density hotspots, coverage voids, and
  resolution/isotropy). Read this first whenever you are about to use, explain, or
  modify ambit and want the big picture: what problem it solves, the three facets
  of occupancy, the three input tiers (embeddings / +dataset / +model), the three
  CLI commands (info / embed / report), and how the pieces fit. Points to the other
  ambit-* skills for depth. Activates on questions like "what is ambit", "what does
  ambit do", "should I use ambit for this", or any first contact with the project.
version: 1.0.0
---

# ambit — what it is

> **ambit visualizes how a dataset occupies an embedding space** — surfacing
> density hotspots and sparse regions, and measuring how much of the space's
> *resolution* the data actually uses.

A scatter plot of projected embeddings only half-answers the real question. A
corpus can pile into one megacluster while leaving most of the reachable space
empty; it can wrap a ring of coverage around an interior hole; it can pack
unrelated items so tightly that cosine similarity can no longer tell them apart.
ambit makes that **occupancy structure** legible.

## The one idea behind everything

Crowding in an embedding space is a **loss of resolution**. When items pack into a
narrow cone (anisotropy), every pair — related or not — scores high on cosine, so
the signal you care about barely rises above the floor. Retrieval, clustering,
dedup, and RAG all inherit that ambiguity. ambit's job is to replace *"the
embeddings look fine"* with *"here is exactly how much resolution this dataset has,
and where it is spent."*

- The **density / coverage** views are the *picture* of crowding.
- The **resolution / isotropy** diagnostics are its *measurement* (computed on the
  original high-dimensional vectors, not a 2-D projection).
- The **3-D** views let you *rotate and see* the anisotropy directly.

For the science and how to read the numbers, see **`ambit-concepts`**.

## The three facets of occupancy

| facet | question it answers | example views |
|---|---|---|
| **Density / hotspots** | where do items pile up? | density-peak prominence, hexbin, KDE contours |
| **Coverage / voids** | what does the data fill vs. leave empty? | NN-sparsity field, alpha-hull reach, void discs |
| **Resolution / isotropy** | is the space well-used or collapsed into a cone? | random-pair cosine, scree/effective-rank, IsoScore, NN-margin, hubness |

Plus **topology** (kNN graph, bridge chokepoints), **comparison** (vs a reference
space), and **3-D volume** (occupancy you can orbit). See **`ambit-figures`** for
the full catalog.

## The three input tiers (graceful degradation)

ambit meets you wherever your data is. Everything normalizes into one
`EmbeddingSet`, so downstream code never branches on provenance:

1. **Embeddings you already have** — `.npy` / `.npz` / `.parquet` / `.csv` /
   `.jsonl`, a single file, a sharded directory, or a glob. Needs only numpy
   (tabular formats pull in the `io` extra). This is the common path.
2. **A raw dataset** — text rows you haven't embedded yet. `ambit embed` turns them
   into vectors via any OpenAI-compatible endpoint (no provider SDK).
3. **A model** — embed raw items on the fly with the `model` extra
   (sentence-transformers).

## The three commands

```bash
ambit info   <embeddings>   # one-pass streaming scan -> resolution diagnostics in the terminal
ambit embed  <dataset>      # raw text items -> vectors via an OpenAI-compatible endpoint
ambit report <embeddings>   # -> a self-contained, theme-adaptive HTML occupancy report
```

`info` is the quick read; `report` is the full visual story; `embed` is the bridge
from tier-2 to tier-1. See **`ambit-cli`** for every flag and recipes.

## How the pieces fit (one line)

`source` (streaming chunks) → `scan` (one pass: covariance + norms + a reservoir
sample) → `pipeline.build_ctx` (projections, eigenspectrum, kNN, clusters) →
`render` (iterate the figure registry → self-contained HTML). The full map is in
**`ambit-architecture`**.

## Design stance worth knowing up front

- **numpy-first, optional everything-else.** A minimal install (just numpy) ingests
  `.npy/.npz` and runs every native-space diagnostic. Heavier capabilities
  (parquet, UMAP, ANN kNN, GPU, on-the-fly embedding) are opt-in extras that
  **degrade gracefully** — a missing dependency disables a feature, never crashes
  the core.
- **No environment variables.** Every setting is a CLI flag gathered into a single
  `Config` object (optionally merged with a `--config` JSON). A run is fully
  described by its `Config`.
- **Streaming by default.** The scan never holds a 1M+ corpus in RAM; scalar
  diagnostics run over the full corpus while the visual layers work off a bounded
  reservoir sample.
- **Self-contained, theme-adaptive reports.** One HTML file, no external assets;
  every figure is token-only SVG that re-skins live via a theme picker (the
  "zicato" design language). See **`ambit-figures`**.

## When to reach for ambit

- Auditing an embedding model or a corpus before building retrieval/RAG on it.
- Comparing two encoders *without task labels* (effective rank is a label-free
  quality signal).
- Diagnosing why retrieval is noisy (hubness, thin NN margins, a crowded cone).
- Understanding coverage: what regions a dataset over- or under-represents.

## Status

Early scaffolding (`0.0.1`). The core pipeline, CLI, diagnostics, and figure
registry are in place; expect rough edges and rapid change.
