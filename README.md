# ambit

Visualize the embeddings of a dataset to understand how much of an embedding
space it covers — surfacing density hotspots and sparse regions.

## Overview

`ambit` projects high-dimensional embeddings down to 2D/3D and renders a
density view over the projected points, making it easy to see where a dataset
concentrates in the space and where it leaves the space empty.

The pipeline:

1. **Project** — reduce embeddings to 2D/3D (UMAP / t-SNE / PCA).
2. **Densify** — estimate density over the projection (KDE / hexbin / contours).
3. **Surface** — detect and rank hotspots and coverage gaps.

## Usage

ambit is a library first; the `ambit` command is one front end over it. Every CLI
command builds a `Config` and calls the matching library verb, so anything the CLI
does is available programmatically.

### Library

```python
import ambit

# full report -> self-contained, theme-adaptive HTML
rep = ambit.report("embeddings.parquet")     # Report(.html, .ctx, .scan, .shown/.total)
rep.write("report.html")

# just the numbers — a streaming scan + resolution metrics, no rendering, no stdout
diag = ambit.diagnose("embeddings.parquet")  # Diagnostics
print(diag.mean_cos, diag.verdict, diag.effective_rank, diag.dims_for_90pct)

# embed raw items via an OpenAI-compatible endpoint
ambit.embed("items.jsonl", "vecs.parquet", model="text-embedding-3-small",
            base_url="https://api.openai.com/v1", api_key="...")
```

Every verb takes a `Config` or plain keyword overrides (nothing is read from the
environment):

```python
cfg = ambit.Config(sample=100_000, projector="umap")
ambit.report("embeddings.parquet", config=cfg).write("report.html")
ambit.report("embeddings.parquet", sample=100_000, title="my corpus")   # or inline

# compare the same items embedded two ways (aligned by id) — adds the CMP figures
ambit.report("encoder-a.parquet", compare="encoder-b.parquet", id_col="uuid").write("diff.html")
```

Or drive the pipeline stage by stage:

```python
sc   = ambit.scan("embeddings.parquet")      # streaming scan      -> Scan
ctx  = ambit.build_ctx(sc)                   # project, kNN, cluster -> Ctx
html = ambit.build_report(ctx, out="report.html")
la   = ambit.localized_anisotropy(ctx.es.X)  # the local-density measure
```

### Command line

```
ambit info   embeddings.parquet                 # scan -> resolution diagnostics
ambit report embeddings.parquet --out report.html
ambit embed  items.jsonl --out vecs.parquet --model text-embedding-3-small

# the same items embedded two ways — a local neighbor-overlap view + CKA, aligned by id
ambit report encoder-a.parquet --compare encoder-b.parquet --id-col uuid --out diff.html
# reciprocal (mutual) kNN everywhere — suppresses hubs in every neighbor view
ambit report embeddings.parquet --mutual-knn --out report.html
```

## Status

Early scaffolding.
