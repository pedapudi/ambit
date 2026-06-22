# Legal retrieval audit — using ambit to diagnose an embedding model, then proving it

A worked, end-to-end example: take a general-purpose embedding model
(**Qwen3-Embedding-0.6B**), point **ambit** at how it organizes a domain corpus
(the **Nemotron-Pretraining-Legal-v1** dataset), form a hypothesis about where it
will fail at retrieval, and then **confirm that hypothesis with two grounded
retrieval evals**. The arc is the point: ambit tells you *where the resolution is
spent*; the evals tell you *what that costs downstream*; together they define a
fine-tuning objective.

> Use case behind the example: a *question → governing statute* retriever (and
> RAG) over U.S. legal text.

---

## Problem statement

You have a capable *general-purpose* embedding model and a domain — legal — where
you want semantic retrieval and RAG, concretely **question → governing statute**.
Before spending effort on fine-tuning (data prep + GPU time), you want answers to:

1. Does the off-the-shelf model already retrieve well enough on this corpus?
2. If not, *where exactly* does it break — and is that the kind of failure
   fine-tuning can fix?

The trap is shipping on vibes ("the embeddings look fine"): a space can look
high-dimensional yet pack unrelated documents so tightly that cosine can no longer
rank the right one to the top. This example demonstrates the disciplined
alternative — **diagnose → quantify → decide**:

- **Diagnose, label-free.** Point `ambit` at how the model *occupies* the embedding
  space on your corpus (anisotropy, effective rank, per-subdomain separation). No
  labels, no eval needed — it shows where the model spends its resolution.
- **Quantify.** Turn the diagnosis into two *grounded* retrieval evals (one
  citation-derived, one synthetically authored; single- and multi-hop) and measure.
- **Decide.** The numbers pin the failure modes, which define the fine-tuning
  objective — and become the before/after harness.

**As demonstrated, the answer was:** the model cleanly separates legal *genres*
(regulation vs case law) but **cannot finely discriminate which specific statute**
among similar ones, and **multi-hop retrieval is the hard ceiling** (all required
documents in the top-10 only ~48% of the time). Coarse retrieval is fine; fine and
multi-document retrieval are not — a fixable, fine-tuning-shaped problem.

## Setup

The only hard dependency is an **OpenAI-compatible `/v1/embeddings` endpoint** (here
vLLM serving Qwen3-Embedding-0.6B), plus the dataset and `ambit`. In the
demonstration the work spanned three hosts — a GPU box serving the endpoint, a data
host running the embedding clients, and an orchestrator running `ambit` — but it can
all run on one machine. Full environment, hardware, data flow, and prerequisites are
in **[SETUP.md](SETUP.md)**.

## The pipeline

```
embed corpus ──▶ ambit report ──▶ diagnose (where does the model lose resolution?)
   (endpoint)     analyze_by_label        │
                                          ▼
                          build two grounded retrieval evals
                          ├─ Set A: citation-grounded (eCFR)         build_eval.py + run_eval.py
                          └─ Set B: open-corpus synthetic            build_open_* + authoring + merge + run_open_eval.py
                                          │
                                          ▼
                          measure → numbers that match the diagnosis → fine-tuning objective
```

Everything runs against an **OpenAI-compatible `/v1/embeddings` endpoint** (here
vLLM serving Qwen3-Embedding; llama.cpp works too). No provider SDK.

## 1. Diagnose with ambit

Embed a balanced sample across legal subdomains (`embed_corpus.py`), then:

```
ambit report ./emb --embedding-col embedding --label-col subset --out report.html
python analyze_by_label.py ./emb --label-col subset --strip-prefix Nemotron-Pretraining-Legal-
```

What ambit found on 1M docs (full-precision, 1024-d), 4 subdomains:

| metric | value | reading |
|---|---|---|
| mean random-pair cosine | **+0.235** | an anisotropic *cone* — every pair already alike |
| IsoScore | 0.090 | low space utilization |
| effective rank / participation | 728 / **93** of 1024 | ~93 dims carry real variance (front-loaded, not collapsed) |
| hub skew | 2.0 | a few hub docs absorb many queries |

Per-subdomain (kNN purity = do a doc's neighbors share its label?):

| subdomain | purity | nearest other (centroid cos) |
|---|---|---|
| California-Code-Of-Regulations | 0.98 | eCFR-QA **0.81** |
| Case-Law-Summary | 0.96 | CaseHOLD **0.84** |
| CaseHOLD | 0.83 | Case-Law-Summary 0.84 |
| eCFR-QA | **0.78** | CA-Regs 0.81 |

**Diagnosis:** the model separates *genre* well (regulation vs case law) but the
two regulatory subdomains collapse onto each other (centroid 0.81) and the two
case-law ones do too (0.84). It groups by *kind of legal text*, not by *which
specific law* — so it should struggle to pick the right statute among similar
ones, and the regulatory-question content (eCFR-QA) is the least resolved.

## 2. Prove it — two grounded retrieval evals

Both strip all citations / section numbers from the questions, so retrieval is
**semantic, not string-matching a locator**. See [`DATASHEET.md`](DATASHEET.md)
for full schemas, label vocabularies, and caveats.

### Set A — citation-grounded eCFR (ground truth from the data)

`build_eval.py` extracts each eCFR-QA item's cited CFR section as gold, mapped to
the eCFR document containing it. 5,000-question reservoir, 35,173-section corpus.
`run_eval.py` (query embeddings cached):

| | R@1 | R@5 | R@10 | MRR@10 | median rank |
|---|---|---|---|---|---|
| closed (35k eCFR) | 0.29 | 0.69 | 0.75 | 0.47 | 2 |
| open (+555k cross-domain distractors) | 0.28 | — | 0.73 | 0.45 | 2 |

Median rank 2 but **R@1 only 0.29**, and adding cross-domain distractors barely
moves top-k yet explodes the *miss tail* (mean rank 345 → 1,638). Coarse retrieval
is robust; picking the *exact* governing section among siblings is not — exactly
the within-family collapse ambit predicted.

### Set B — open-corpus synthetic (single + multi-hop)

Questions authored by LLM subagents grounded in real documents (no external
model): `build_open_candidates.py` → `build_open_inputs.py` → authoring agents
reading [`AUTHORING_SINGLE.md`](AUTHORING_SINGLE.md) /
[`AUTHORING_MULTI.md`](AUTHORING_MULTI.md) → `merge_open_eval.py` →
`run_open_eval.py`. **4,987 questions** (2,489 single-doc, 2,498 multi-doc),
searched against the **~1.035M-doc open corpus** (1M map + 35k eCFR):

| | R@1 | R@5 | R@10 | R@100 | MRR@10 |
|---|---|---|---|---|---|
| single-doc | 0.44 | 0.91 | 0.95 | 0.99 | 0.66 |

| multi-doc | @5 | @10 | @20 | @100 |
|---|---|---|---|---|
| set-Recall (frac of gold docs found) | 0.59 | 0.69 | 0.75 | 0.87 |
| all-gold (ALL required docs found) | 0.36 | 0.48 | 0.55 | 0.75 |

**Single-doc is strong** (top-5 0.91 even over a million docs — a distinctive
question finds its doc), but **multi-hop is the wall**: all required documents land
in the top-10 only **48%** of the time. A RAG pipeline that needs every supporting
passage misses pieces about half the time.

> Difficulty is about gold *distinctiveness*, not corpus size: Set A (35k corpus)
> is *harder* for single-doc (R@5 0.69) than Set B (1M corpus, R@5 0.91), because
> Set A's gold is one section among near-identical siblings.

## 3. Fine-tuning — objective, and what happened

The diagnosis and the numbers agreed: coarse retrieval is fine; **fine, within-family
and multi-hop discrimination is the ceiling.** So the objective was an asymmetric
fine-tune — question anchor (query instruction) against governing-passage positives,
**hard negatives mined from the within-family confusable neighbors** ambit exposed
(sibling sections, similar regs, other holdings), and Matryoshka so truncated dims
stay strong.

**It was done in three rounds, and the result was instructive:**

| eval | metric | base | v1 (eCFR-only) | v2 (+synth) | v3 (multi-positive) |
|---|---|---|---|---|---|
| Set A — eCFR (trained task) | Recall@1 | 0.288 | **0.355** (+23%) | 0.336 | **0.350** |
| Set B — open-corpus single | Recall@10 | 0.952 | 0.830 ↓ | 0.906 | **0.918** |
| Set B — multi-hop | set-Recall@10 | 0.687 | 0.366 ↓↓ | 0.481 | 0.487 |
| ambit geometry | kNN purity | 0.89 | 0.91 | 0.81 | **0.90** |

**v1** over-specialized — catastrophic forgetting from a multi-GPU workaround that
forced eCFR-only training. **v2** folded the synthetic pairs back in under one uniform
schema and recovered most of the regression. **v3** then tested whether the residual
multi-hop gap was a *loss-function* limit (a multi-positive objective) — and showed it
is not: v3 is the best-balanced model with the cleanest geometry, yet multi-hop@10
stayed flat, because only **3.7%** of training queries are multi-positive. The gap is
**data composition**, not the loss. The two-eval harness is what made the false win
(v1), the rebalance (v2), and the honest negative result (v3) all legible. The complete
base→v1→v2→v3 before/after is in **[FINETUNE.md](FINETUNE.md)**; scripts:
`build_train_pairs.py`, `mine_hard_negs.py`, `mine_synth_negs.py`, `train.py`
(+ `train_v3.py` for the multi-positive experiment). A fourth round — **v4**, a
*balanced* training set raising multi-positive from 3.7% to ~1/3 — is built and
documented in FINETUNE.md but **not yet trained** (`build_v4_inputs.py`,
`mine_v4_negs.py`, `build_train_v4.py`).

---

## Files

| file | role |
|---|---|
| `embed_corpus.py` | embed a corpus subset via the endpoint → ambit-ready parquet (Matryoshka `--dims`, resumable) |
| `analyze_by_label.py` | per-label cohesion / kNN purity / centroid matrix (complements `ambit report`) |
| `build_eval.py` | Set A — citation-grounded eCFR eval (questions + corpus) |
| `run_eval.py` | Set A runner — Recall@k/MRR, cached embeddings, optional open-corpus distractors |
| `build_open_candidates.py`, `build_open_inputs.py` | Set B — sample source docs + form single/sibling/semantic authoring inputs |
| `AUTHORING_SINGLE.md`, `AUTHORING_MULTI.md` | instructions the authoring subagents follow (single-stream; no sub-agents) |
| `merge_open_eval.py` | label + train/eval split + validate → `questions_open.jsonl` |
| `run_open_eval.py` | Set B runner — single-doc R@k/MRR + multi-doc set-Recall@k / all-gold@k |
| `build_train_pairs.py` | fine-tune — derive holdout-disjoint `(query, positive)` training pairs |
| `mine_hard_negs.py` | fine-tune — mine within-family sibling hard negatives for the eCFR pairs via the base endpoint |
| `mine_synth_negs.py` | fine-tune (v2) — mine hard negatives for the Set B pairs so both sources share one uniform 7-col schema (DDP-safe) |
| `train.py` | fine-tune — full FT, MNRL + Matryoshka, multi-GPU DDP, both sources as one uniform dataset |
| `build_train_v3.py` | fine-tune (v3) — regroup pairs into multi-positive rows (all golds kept), fixed-shape for DDP |
| `train_v3.py` | fine-tune (v3) — multi-positive InfoNCE (+ false-neg masking) + Matryoshka; the multi-hop experiment |
| `build_v4_inputs.py` | fine-tune (v4) — fresh eval-disjoint multi-doc clusters (excludes eval golds) |
| `mine_v4_negs.py` | fine-tune (v4) — hard negs + gold texts for the new multi-doc questions |
| `build_train_v4.py` | fine-tune (v4) — balanced multi-positive set (~1/3 multi-doc), eCFR capped |
| `DATASHEET.md` | both eval sets: schemas, label vocab, distributions, caveats |
| `SETUP.md` | environment + how to reproduce |
| `FINETUNE.md` | fine-tuning: training details, multi-GPU lessons, base→v1→v2 before/after results |
| `eval-data/` | the generated question sets (Set A + Set B) + a sample baseline result |
| `run.sh` | end-to-end driver |

## Caveats

- **Synthetic label noise (Set B):** a question may be answerable by a sibling doc;
  gold credits the source only — treat single-doc R@1 as a lower bound.
- **Citation-grounded noise (Set A):** some "misses" are sibling sections that also
  answer the question.
- **Multi-hop necessity** is enforced by the author's judgment (recorded in each
  question's `note`), not formally proven.
- **Authoring concurrency:** subagents will self-parallelize into sub-agents unless
  told not to (the `AUTHORING_*.md` files forbid it); drive them from a Workflow
  with batched `parallel()` to stay under API rate limits.
