# ambit

**ambit tells you where an embedded dataset is too crowded to work — and which
items are in trouble.**

## What it is, in plain English

When you embed a dataset, every item becomes a direction in space, and
"similar" means "pointing the same way." Everything built on top — search,
retrieval-augmented generation, dedup, clustering, recommendations — relies on
one thing: **items you can tell apart must stay apart**. When distinct items
land too close together, they become interchangeable to a query. That's
crowding, and it is the failure mode behind impostors at the top of search
results, near-duplicates that can't be ranked, and clusters that blur into one.

The catch is that crowding hides from averages. A dataset can look healthy on
every global number while one pocket of it has quietly collapsed — a small
clump barely moves a mean. So ambit measures occupancy the other way around:
**continuously over every scale** (no histogram bins or grid cells whose size
and placement change the answer), **against a calibrated reference** (every
number is read against what a well-spread dataset of the same size and shape
would show — never a bare count), and **down to named items** (not "this region
is dense" but "these documents, by id, are each expected to collide with ~12
others").

The output is a single self-contained HTML report that reads as a story: what
your data looks like → how crowded it is and at what scale → which items are
affected → what structure the crowding forms → why the space behaves that way.
Plus a terminal mode for just the numbers, a compare mode for the same items
embedded two different ways, and a training module that turns the measurements
into gradient signal.

ambit is **unsupervised by design** — no labels, no gold pairs, no ground
truth. It reads geometry the embeddings carry on their own. Core install is
numpy-only; it streams millions of rows.

## Why it works

Three ideas carry the tool (developed fully in
[Continuous Occupancy](docs/concepts/continuous-occupancy.md)):

- **The pair-distance distribution is the master object.** Global crowding
  questions are functionals of one curve — the fraction of item pairs closer
  than each scale — read exactly (a sorted list, no bins) against an analytic
  null. In high dimension the null is razor-thin: a well-spread corpus has
  essentially *no* close pairs, so every close pair is a finding.
- **Per-item scores replace per-cell counts.** Each item is scored by the
  radius it needs to gather a fixed share of the corpus — continuous, provably
  stable under data perturbation, and attributable to ids. Cells can't name
  names; fields can.
- **Operational units.** The exact law Φ(−distance/2σ) converts geometry into
  expected retrieval collisions under query noise σ, giving the headline
  scalar σ\*: *how much query sloppiness the corpus tolerates before wrong
  items win*.

## Quickstart

```
pip install -e .                      # numpy-only core
ambit report embeddings.parquet --out report.html
ambit info   embeddings.parquet      # terminal scan: just the numbers
```

```python
import ambit
rep = ambit.report("embeddings.parquet")     # Report(.html, .ctx, .scan)
rep.write("report.html")

diag = ambit.diagnose("embeddings.parquet")  # no rendering, just diagnostics
print(diag.mean_cos, diag.verdict, diag.effective_rank)
```

Compare the same items embedded two ways (aligned by id), or suppress hubs
with reciprocal kNN:

```
ambit report encoder-a.parquet --compare encoder-b.parquet --id-col uuid --out diff.html
ambit report embeddings.parquet --mutual-knn --out report.html
```

Drive the pipeline stage by stage, or reach the training-time regularizers:

```python
sc   = ambit.scan("embeddings.parquet")      # streaming scan       -> Scan
ctx  = ambit.build_ctx(sc)                   # project, kNN, cluster -> Ctx
html = ambit.build_report(ctx, out="report.html")

from ambit import training                   # pip install -e '.[train]'
# training.confusion_loss / preservation_loss / resolution_weights / miners
```

Every verb takes a `Config` or keyword overrides; nothing is read from the
environment. The CLI is a thin front end over the library. Embedding raw items
via an OpenAI-compatible endpoint is built in
(`ambit embed items.jsonl --out vecs.parquet --model ... --base-url ...`).

## Learning path

| read | to learn |
|---|---|
| [Background & context](docs/ambit-context.md) | the primer: the problem, the vocabulary, the method — from scratch |
| [Interpreting the report](docs/guide/interpreting-the-report.md) | every figure and header fact: how it's computed, how to read it, caveats |
| [Anisotropy & resolution](docs/concepts/anisotropy-and-resolution.md) | the science: crowding as resolution loss, with the literature |
| [Continuous occupancy](docs/concepts/continuous-occupancy.md) | the measurement theory: why bins fail, the continuous foundation, σ\* |
| [skills/](skills/README.md) | task-focused guides for agents and contributors — including tuning a model with the measurements |

Two self-contained slide decks tell the story visually:
[background deck](docs/ambit-context-presentation.html) ·
[field-guide deck](docs/ambit-presentation.html).

## Status

Research preview. The measurement layer is tested end to end (unit tests plus
report-level integration on million-row corpora); the training layer is newer
and its worked example reports measured numbers, stated honestly.
