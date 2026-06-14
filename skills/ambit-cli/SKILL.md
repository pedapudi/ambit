---
name: ambit-cli
description: >
  How to RUN ambit from the command line — the three subcommands (info, embed,
  report), every flag and its default, the accepted input formats and column
  auto-detection, sharded-directory / glob sources, the knobs that make it scale to
  millions of rows (--approx, --device, --batch-rows, --knn-backend), figure
  toggling via --config JSON, and copy-paste recipes. Use this whenever you need to
  invoke ambit, choose flags, point it at data, embed a raw dataset, or generate an
  occupancy report. For what the output MEANS see ambit-concepts; for the big
  picture see ambit-overview.
version: 1.0.0
---

# Using the ambit CLI

Install (editable, from the repo root). Core is just numpy; pull the extras you
need (see `ambit-development` for the full tier list):

```bash
pip install -e ".[all]"      # everything for analysis (io, reduce, umap, ann)
pip install -e ".[io]"       # minimal + parquet/csv/jsonl reading
pip install -e ".[io,gpu]"   # + torch device backend for large corpora
```

The console entry point is `ambit` (→ `ambit.cli:main`). Three subcommands:

```bash
ambit info   <embeddings>   # streaming scan -> resolution diagnostics in the terminal
ambit embed  <dataset>      # raw text -> vectors via an OpenAI-compatible endpoint
ambit report <embeddings>   # -> a self-contained HTML occupancy report
```

> **No environment variables.** Every setting is a flag. Flags are gathered into a
> single `Config`; a `--config <file.json>` object is merged on top (it can set any
> `Config` field, a nested `"figures"` map, or a bare `figure_slug: bool`).

## `ambit info` — quick diagnostics

One streaming pass over the embeddings, then prints: source, items × dims, mean L2
norm, mean random-pair cosine (with the isotropic reference ≈ `1/√d`), an
anisotropy verdict, effective rank, participation ratio, dims-for-90%-variance, and
an ASCII histogram of the random-pair cosine distribution.

```bash
ambit info embeddings.parquet
ambit info ~/embed-legal/out/            # a sharded directory (all shards as one corpus)
ambit info vecs.npy --approx 200000      # cap diagnostics to ~200k rows (fast on 1M+)
ambit info big.parquet --device auto     # route covariance onto GPU if available
```

Shared scan flags (also used by `report`):

| flag | default | meaning |
|---|---|---|
| `embeddings` | — | path: `.npy` / `.npz` / `.parquet` / `.jsonl` (or a dir / glob) |
| `--embedding-col` | auto | vector column name (auto-detected if omitted) |
| `--id-col` | none | stable id column |
| `--label-col` | none | class/label column (becomes the report's groups) |
| `--metric` | `cosine` | `cosine` or `euclidean` |
| `--sample` | `20000` | reservoir size kept for the visual layers |
| `--pairs` | `200000` | random pairs sampled for the cosine histogram |
| `--batch-rows` | `50000` | streaming chunk size (rows per read) |
| `--k` | `10` | neighbors for the kNN graph |
| `--device` | `cpu` | `cpu` (numpy) \| `auto` \| `cuda` \| `mps` \| `torch` |
| `--approx` | none | stop after ~N rows; estimate spectrum from that sample |
| `--knn-backend` | `auto` | `auto` \| `pynndescent` \| `sklearn` \| `brute` \| `faiss` |
| `--config` | none | JSON object overriding any `Config` field / figure toggle |

## `ambit embed` — raw items → vectors

Streams a raw dataset through any OpenAI-compatible `/v1/embeddings` endpoint
(OpenAI, vLLM, text-embeddings-inference, LM Studio, llama.cpp, …) using stdlib
only — no provider SDK. Writes `.jsonl` or `.parquet` you can then feed to
`info`/`report`.

```bash
ambit embed corpus.jsonl \
  --out vecs.parquet \
  --model Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0 \
  --base-url http://localhost:8081/v1 \
  --text-col text --id-col uuid --batch 128
```

| flag | default | meaning |
|---|---|---|
| `dataset` | — | `.jsonl` / `.csv` / `.parquet` / `.txt` of raw items |
| `--out` | **required** | output `.jsonl` or `.parquet` |
| `--model` | **required** | model id sent to the endpoint |
| `--text-col` | `text` | column/field holding the text (`.txt` = one item per line) |
| `--id-col` | none | id column to carry through |
| `--label-col` | none | label column to carry through |
| `--base-url` | OpenAI | endpoint base URL (omit → `https://api.openai.com/v1`) |
| `--api-key` | none | bearer token (omit for an unauthenticated local endpoint) |
| `--batch` | `256` | items per request (the client also retries 5× with backoff) |

Output schema: `id`, `embedding`, `text` (+ `label` if given). The embedding column
is named `embedding`, which `info`/`report` auto-detect.

> ambit's built-in `embed` is a convenience for modest jobs. For very large corpora
> with resumability/sharding, a dedicated streaming embed script is often better;
> the output (a parquet/jsonl with an `embedding` column, or a sharded dir) feeds
> `ambit report` directly.

## `ambit report` — the HTML occupancy report

Runs the scan, builds the shared render context (projection, eigenspectrum, kNN,
clusters), then renders a **single self-contained, theme-adaptive HTML file**.

```bash
ambit report embeddings.parquet --out report.html
ambit report vecs.npy --projector umap --title "MiniLM · 100k legal"
ambit report data/ --label-col subset            # color by a metadata column
ambit report vecs.parquet --clusters 8           # force 8 k-means groups
ambit report vecs.parquet --no-cluster           # skip unsupervised labeling
```

Report-only flags (plus all shared scan flags above):

| flag | default | meaning |
|---|---|---|
| `--out` | `ambit-report.html` | output HTML path |
| `--projector` | `pca` | `pca` (fast, deterministic) or `umap` (needs the `umap` extra) |
| `--title` | "ambit — embedding-space occupancy" | report title |
| `--clusters` | auto | force *k* k-means clusters for auto-labeling |
| `--no-cluster` | off | disable unsupervised labeling entirely |

**Labeling:** if `--label-col` is present, groups come from that column
("provided"); otherwise ambit clusters the geometry itself (HDBSCAN if installed,
else k-means with a silhouette-picked *k*). `--no-cluster` turns this off.

## Input formats & auto-detection

Auto-detected by extension; columns auto-detected by name when not specified:

- **`.npy`** — a single `(n, d)` array (memory-mapped / streamed off disk).
- **`.npz`** — embedding array by key (`embeddings`/`embedding`/`X`/`emb`/`vectors`/…)
  or the largest 2-D array; `ids`/`labels` picked up from matching keys.
- **`.parquet` / `.csv`** — either a **vector column** (a list/array per row) or a
  **wide numeric matrix** (every non-id/label numeric column is a dimension);
  id/label/other columns kept aside as metadata. (Needs the `io` extra.)
- **`.jsonl`** — one object per line with an embedding field + arbitrary metadata.
- **directory / glob** — a sharded dump scanned as one source. A directory resolves
  to its sorted `*.parquet` (then `*.pq`/`*.npz`/`*.npy`/`*.jsonl`); a glob like
  `out/train-*.parquet` is sorted and concatenated.

Recognized column names — embedding: `embeddings, embedding, X, emb, vectors, vecs,
vector`; id: `ids, id, keys, key`; label: `labels, label, y, cluster`. Override
with `--embedding-col` / `--id-col` / `--label-col`.

## Scaling to millions of rows

The scan is streaming and never holds the whole corpus in RAM. Reach for:

- **`--approx N`** — stop the covariance/spectrum pass after ~N rows (rank/variance
  converge fast in n; a few hundred-k sample is ≈ exact). The headline item count
  still reflects the true corpus size when it's cheaply known (parquet metadata /
  npy shape).
- **`--device auto|cuda|mps`** — route the covariance accumulation, PCA, random-pair
  cosine, and brute kNN onto a torch device (needs the `gpu` extra).
- **`--knn-backend`** — `auto` tries pynndescent → sklearn → brute; use
  `pynndescent` (`ann` extra) for a large reservoir, or `faiss` (`faiss` extra) for
  GPU kNN over very large reservoirs.
- **`--sample` / `--batch-rows`** — reservoir size for the visuals / streaming chunk
  size. Scalar diagnostics run over the full (or `--approx`-capped) corpus
  regardless of `--sample`.

## Toggling figures with `--config`

`DEFAULT_FIGURES` (in `config.py`) decides which figures render; a curated set is on
and the rest are implemented-but-hidden. Flip any of them via a JSON object:

```json
{ "figures": { "den_contour": true, "cov_void": true, "d3_voxel": true } }
```

```bash
ambit report vecs.parquet --config show_more.json
```

A bare slug also works (`{"den_hexbin": true}`), and the same file can set any
`Config` field (e.g. `{"projector": "umap", "k": 15}`). The figure slugs and
families are listed in `ambit-figures`.

## Recipes

```bash
# Audit a parquet of embeddings you already have
ambit info vecs.parquet

# Embed a local corpus against a llama.cpp server, then report
ambit embed corpus.jsonl --out vecs.parquet --model my-embed \
  --base-url http://localhost:8081/v1 --text-col text --id-col uuid
ambit report vecs.parquet --out report.html --label-col subset

# Large sharded dump on GPU, approximate spectrum, ANN kNN
ambit report ~/embed-legal/out/ --device auto --approx 300000 \
  --knn-backend pynndescent --out legal.html

# Color by a real label column instead of auto-clusters
ambit report vecs.parquet --label-col category
```
