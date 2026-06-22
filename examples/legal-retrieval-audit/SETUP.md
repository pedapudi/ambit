# Setup — environment and reproduction

This documents how the example is run and what you need to reproduce it. The work
splits cleanly into three roles; they can run on three separate hosts or all on a
single machine.

## Roles

| role | what it does |
|---|---|
| **embedding endpoint (GPU)** | serves Qwen3-Embedding over an OpenAI-compatible `/v1/embeddings` API; does all the GPU work. A GPU with a few GB free is ample for a 0.6B embedding model |
| **data host** | holds the dataset; runs the embedding *clients* and eval scripts (CPU only — they POST text to the endpoint and write parquet) |
| **orchestrator** | runs `ambit`, holds a copy of the embeddings, drives the authoring agents |

Nothing requires three machines — the endpoint, the dataset, and `ambit` can all
live on one host. The split exists because the GPU box (endpoint) and the dataset
were on different machines; clients ran where the data was and embedded over HTTP.

## The embedding endpoint

An **OpenAI-compatible `/v1/embeddings` server** is the only hard dependency for
embedding. In the demonstration this was **vLLM serving `Qwen3-Embedding-0.6B`**:

```
vllm serve <path-or-hf-id> --task embed        # -> http://<host>:8081/v1
```

- Native output dim **1024**; `max_model_len` 32768. Embeddings are L2-normalized
  for cosine. `embed_corpus.py --dims 768` does Matryoshka truncation + renorm if
  you want the smaller vectors.
- llama.cpp's `llama-server` (GGUF, e.g. Q8_0) serves the same API and was used for
  an earlier 768-d pass — note that **quantization costs resolution** (Q8_0/768-d
  measured mean-pair-cosine +0.30 vs full-precision/1024-d +0.235; don't mix vector
  sets from different precisions in one analysis).

Every script takes `--base-url` (default `http://localhost:8081/v1`); set it to
your endpoint.

## The dataset

**NVIDIA `Nemotron-Pretraining-Legal-v1`** (HuggingFace, ~14 GB git-LFS): 13 legal
subdomains, 9.6M rows, schema `text · license · metadata · uuid`. Important: the
`text` field is a single *pretraining-formatted blob* per row (instruction/QA/
summary text), **not** pre-shaped `(query, passage)` pairs — so the eval ground
truth has to be *derived* (Set A from citations) or *authored* (Set B), not read
off a column. Subdomains used here:

- embedded into the 1M occupancy map: `California-Code-Of-Regulations`,
  `Case-Law-Summary`, `CaseHOLD`, `eCFR-QA`
- the retrieval corpus / gold source: `eCFR` (35,173 raw federal-regulation sections)

## What gets produced

| artifact | size | by |
|---|---|---|
| 1M occupancy-map embeddings (1024-d, 4 subdomains, stratified) | ~4 GB | `embed_corpus.py` |
| 35k eCFR section embeddings (retrieval corpus) | cached `.npy` | `run_eval.py` (first run) |
| cached query embeddings (both eval sets) | small | `run_eval.py` / `run_open_eval.py` |

Data flow: dataset (data host) → embed over HTTP to the endpoint (GPU) → parquet
shards (data host) → copy to the orchestrator for `ambit`. Query/corpus embeddings
are cached on disk keyed by `(model, ids)`, so re-evals never re-embed unchanged
text — only a new model triggers work.

## Prerequisites

- An OpenAI-compatible embeddings endpoint (above).
- The dataset available locally on the data host.
- **ambit** installed with extras: `pip install -e '.[io,reduce,ann]'` (parquet,
  PCA/kNN, approximate-kNN). PCA is the default projector; `umap` is optional.
- Python deps for the example scripts: `requests pyarrow numpy scikit-learn scipy`.
- **Set B authoring only:** a harness that can run LLM subagents to author
  questions from `AUTHORING_{SINGLE,MULTI}.md`. The questions were authored *by the
  model itself* (no separate generative endpoint) — if you don't have an agent
  harness, any instruction-following LLM can play the authoring role by following
  those same files. **Concurrency lesson:** authoring agents will recursively spawn
  their own sub-agents unless told not to (the `.md` files forbid it); drive them
  from a **Workflow with batched `parallel()`** (small batches) or you will trip API
  rate limits.

## Running it

`run.sh` is the end-to-end driver (each stage is independent). Point `ENDPOINT`,
`DATA`, and `PY` at your environment and run the stages you need.
