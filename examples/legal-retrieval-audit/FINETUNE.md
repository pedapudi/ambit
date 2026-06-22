# Fine-tuning — what was done, and the before/after

The audit (see [README](README.md)) diagnosed the failure modes; this is the
fine-tune that acted on them, and the measured result. **Headline: three rounds. v1
(eCFR-only) became a sharp *specialist* — +23% Recall@1 on the trained task but a
catastrophic multi-hop regression. v2 (eCFR + synthetic, one uniform schema) recovered
most of it. v3 (a multi-positive loss) tested whether the residual multi-hop gap was a
loss-function limit — and the answer was no: it's a *data-composition* limit. v3 is the
best-balanced model overall, but multi-hop@10 stayed flat because only 3.7% of the
training queries are multi-positive.** The two-eval harness is what made the false
"win" (v1), the rebalance (v2), and the honest negative result (v3) all legible.

## Training data

The corpus isn't in `(query, passage)` form, so pairs were *derived* (`build_train_pairs.py`):

- **eCFR-QA → eCFR section** (the citation-grounded path from `build_eval.py`): the
  cleaned question as anchor, the governing CFR section text as positive. ~45k pairs.
- **Set B synthetic train split**: the authored `question → gold doc` pairs (CA-Regs /
  Case-Law-Summary / CaseHOLD / eCFR), multi-doc expanded to multi-positive. ~6k pairs.
- **Strict holdout disjointness**: every training query is excluded from *both* eval
  splits by **qid and exact text** (a held-out question can recur under a different
  uuid after citation-stripping — 6 such leaks were caught and dropped). Verified
  **0 / 0** query overlap with Set A and Set B eval.

**Hard negatives** (`mine_hard_negs.py`) — the within-family lever: for each eCFR
query, retrieve the nearest eCFR sections with the **base** model and take the top
non-gold hits (the confusable siblings — e.g. for a "security risk assessment for
registration" question, the gold is the Title 42 HHS section and the hard negative
is the *parallel Title 9 USDA* section). Output: `train_ecfr.jsonl` (anchor +
positive + 5 hard negatives) and `train_synth.jsonl` (anchor + positive).

## Training recipe (`train.py`)

| | |
|---|---|
| base model | `Qwen/Qwen3-Embedding-0.6B` (full fine-tune — no LoRA; 0.6B fits easily) |
| loss | `MultipleNegativesRankingLoss` wrapped in `MatryoshkaLoss` (1024/768/512/256) |
| asymmetry | query instruction baked into the anchor (matching the eval harness); documents raw; last-token pooling, left pad |
| batch | per-device 128 × 2 GPUs = **global 256** (in-batch negatives gathered across GPUs) + 5 mined hard negs/eCFR row |
| other | bf16, gradient checkpointing, lr 2e-5, warmup 0.05, **2 epochs** |
| hardware | 2 GPUs via `torchrun --nproc_per_node=2`, ~130 min wall-clock |

### Multi-GPU lessons (paid for in debugging)

Getting DDP working on this box took real effort; the failure modes are worth recording:

1. **`CachedMultipleNegativesRankingLoss` is incompatible with DDP.** GradCache's
   chunked, multi-pass custom backward desyncs DDP's gradient all-reduce → NCCL
   timeout → abort. Use plain `MultipleNegativesRankingLoss` + `gradient_checkpointing`
   for multi-GPU (CachedMNRL is for *single*-GPU large-batch).
2. **Mixed-shape multi-dataset training deadlocks DDP.** Training two datasets with
   different column counts (`train_ecfr` has 5 hard-neg cols, `train_synth` has 0)
   makes the rank sampler hand different-shaped batches to the two ranks; the
   cross-GPU collective then gets mismatched tensor sizes and **hangs at a fixed early
   step (~7)**. A *deterministic* step-N hang is a data/code rank-desync, **not** a
   transport bug. Fix: train a **single, uniform-shape dataset**.
3. **NCCL env vars were red herrings here.** `NCCL_P2P_DISABLE` (outdated on current
   drivers — P2P works) and `NCCL_CUMEM_ENABLE=0` (a real workaround vLLM applies, kept
   as hygiene) neither caused nor fixed this hang. The reproducible-at-the-same-step
   signature was the tell.

> Because the fix was "single uniform dataset", **this run trained on eCFR-only** (the
> 47,858-pair set with hard negatives), dropping the synthetic pairs. That choice is
> the direct cause of the Set B regression below — see v2.

## Results — before/after (held-out eval splits, identical questions)

The tuned model was served via vLLM under a **distinct id** so the eval cache
re-embedded rather than reusing the base vectors.

### Set A — eCFR question→section (the trained task) ✅

| metric | base | tuned | Δ |
|---|---|---|---|
| Recall@1 | 0.288 | **0.355** | +0.067 (**+23%**) |
| Recall@5 | 0.693 | **0.788** | +0.095 |
| Recall@10 | 0.750 | **0.840** | +0.090 |
| MRR@10 | 0.467 | **0.549** | +0.082 |
| mean rank | 345 | **210** | −135 (miss-tail shrank) |

### Set B — open-corpus synthetic (single + multi-hop, NOT trained) ❌

| metric | base | tuned | Δ |
|---|---|---|---|
| single-doc Recall@5 | 0.91 | **0.78** | −0.13 |
| single-doc Recall@10 | 0.95 | **0.83** | −0.12 |
| multi-doc set-Recall@10 | 0.69 | **0.37** | −0.32 |
| multi-doc all-gold@10 | 0.48 | **0.16** | −0.32 |

### ambit geometry (1M map, base→tuned)

| | base | tuned |
|---|---|---|
| mean random-pair cosine | +0.235 | **+0.182** (cone opened) |
| participation ratio | 93/1024 | **115/1024** |
| effective rank | 728 | 769 |
| 90% variance in | 361 dims | 414 dims |
| within-family centroid {CA-Regs, eCFR-QA} | 0.81 | **0.66** (separated) |
| cross-family {eCFR-QA, Case-Law} | 0.71 | **0.47** |

The contrastive objective did what it should — **less anisotropic, more dimensions
in use, regulatory within-family blur reduced** — and that shows up as the Set A
gains. But the geometry improvement was **eCFR-task-aligned** and did *not* transfer
to the broader corpus (Set B fell). A useful reminder: **better isotropy ≠ better
retrieval on every task.**

## v2 — fold the synthetic pairs back in (uniform schema)

v1 was a clean specialist: train a narrow slice, win it, regress the rest
(catastrophic forgetting). v2 acts on that — **same recipe, but the Set B synthetic
pairs are folded back in with hard negatives mined for them too**, so every row
shares the **same 7-column schema** (anchor + positive + 5 hard negs). That single
uniform-shape dataset is what keeps DDP happy (see lesson #2 above) *and* gives the
model multi-domain signal. Trained 2 epochs, 2-GPU DDP, same LR.

**How it was generated.** `mine_synth_negs.py` is the new step: for each Set B
question it embeds the anchor with the **base** model and retrieves the 5 nearest
non-gold docs from a *text-available* pool — the 35k eCFR sections **plus** the Set B
candidate docs (CA-Regs / Case-Law / CaseHOLD) — **excluding the question's FULL gold
set**, so a sibling gold of a multi-hop question is never mined as its own negative.
That turns `train_synth.jsonl` from 2 columns into the same `{anchor, positive,
negative_1..5}` shape as `train_ecfr.jsonl`; `train.py` then trains **both** as one
`{"ecfr", "synth"}` dataset (per-source query instructions, MNRL gathering in-batch
negatives across the two GPUs). 47,858 eCFR rows + 3,257 synth rows, all 7 columns.

### Results — base → v1 → v2 (held-out eval splits)

**Set A — eCFR question→section** (the trained task)
| metric | base | v1 | v2 |
|---|---|---|---|
| R@1 | 0.288 | 0.355 | 0.336 |
| R@10 | 0.750 | 0.840 | 0.832 |
| MRR@10 | 0.467 | 0.549 | 0.532 |

**Set B single-doc** (not trained)
| metric | base | v1 | v2 |
|---|---|---|---|
| R@1 | 0.438 | 0.347 | 0.394 |
| R@5 | 0.914 | 0.780 | 0.855 |
| R@10 | 0.952 | 0.830 | 0.906 |
| MRR@10 | 0.662 | 0.539 | 0.605 |

**Set B multi-doc** (not trained *as multi-positive*)
| metric | base | v1 | v2 |
|---|---|---|---|
| set-Recall@10 | 0.687 | 0.366 | 0.481 |
| all-gold@10 | 0.481 | 0.159 | 0.207 |

**ambit geometry** (1M map, identical script)
| | base | v1 | v2 |
|---|---|---|---|
| mean random-pair cosine | +0.236 | +0.181 | +0.147 |
| kNN hubness skew | +1.90 | +2.27 | +1.50 |
| kNN purity | 0.89 | 0.91 | 0.81 |
| within-family {CA-Regs↔eCFR-QA} | 0.81 | 0.66 | 0.78 |

### Verdict

v2 **strictly beats v1 on Set B and ties it on Set A** — the uniform-schema refold
worked. But vs *base* the picture is honest, not triumphant: Set A is a clear win
(+17% R@1), Set B single-doc nearly recovers (R@10 −0.046 vs base), but **multi-hop
still regresses** (set-Recall@10 0.687→0.481). The geometry says why: v1 tore the
regulatory family apart (within-family 0.81→0.66) and spiked hubness — wrecking
multi-hop; v2 kept the reg family together (0.78≈base), opened the cone the most, and
pushed hubness *below* base — but kNN purity fell to 0.81 (neighborhoods more mixed),
which is the residual cost that keeps multi-hop under base.

**Ship v2 for single-target legal retrieval; base still wins multi-hop RAG.** The
remaining fix is **not data — it's the loss**: the synth multi-doc rows were flattened
to single-positive for MNRL, so "spread attention across several golds" was never
actually trained. A multi-positive / listwise objective is the next lever.

> v3 below tests that "it's the loss" hypothesis directly — and overturns it.

## v3 — does a multi-positive loss close the multi-hop gap? (no — it's the data)

v3 builds the next lever v2 called for: a custom **multi-positive InfoNCE** (each query
scored against its FULL gold set in the numerator, all in-batch docs in the denominator)
wrapped in MatryoshkaLoss, with **false-negative masking** — any in-batch doc that is a
gold of the current query is removed from its negatives. That masking is not optional:
**82% of golds are shared by ≥2 queries**, so a naive in-batch impl would train the
model to push true siblings apart. Multi-doc questions are kept as multi-positive (not
flattened), on fixed-shape padded tensors so 2-GPU DDP stays in sync. 48,413 training
queries, holdout-disjoint. Trainer: `train_v3.py`.

### Results — base → v1 → v2 → v3

**Set A — eCFR question→section** (trained task)
| metric | base | v1 | v2 | v3 |
|---|---|---|---|---|
| R@1 | 0.288 | 0.355 | 0.336 | 0.350 |
| R@10 | 0.750 | 0.840 | 0.832 | 0.829 |
| MRR@10 | 0.467 | 0.549 | 0.532 | 0.542 |
| mean rank | 345 | 210 | 240 | **197** |

**Set B single-doc**
| metric | base | v1 | v2 | v3 |
|---|---|---|---|---|
| R@10 | 0.952 | 0.830 | 0.906 | **0.918** |
| MRR@10 | 0.662 | 0.539 | 0.605 | 0.605 |

**Set B multi-doc** (the target)
| metric | base | v1 | v2 | v3 |
|---|---|---|---|---|
| set-Recall@10 | 0.687 | 0.366 | 0.481 | **0.487** |
| set-Recall@20 | 0.749 | 0.450 | 0.588 | **0.614** |
| set-Recall@100 | 0.872 | 0.634 | 0.790 | **0.814** |
| all-gold@100 | 0.745 | 0.438 | 0.593 | **0.646** |

**ambit geometry**
| | base | v1 | v2 | v3 |
|---|---|---|---|---|
| mean random-pair cosine | +0.236 | +0.181 | +0.147 | +0.212 |
| kNN hubness skew | +1.90 | +2.27 | +1.50 | +1.74 |
| kNN purity | 0.89 | 0.91 | 0.81 | **0.90** |
| within-family {CA-Regs↔eCFR-QA} | 0.81 | 0.66 | 0.78 | 0.78 |

### Verdict

v3 is the **best-balanced model**: best mean rank on Set A (197), best single-doc
recovery (R@10 0.918, nearly base), best deep-k multi-hop (set-Recall@100 0.814), and
the **cleanest geometry of any tuned model** — kNN purity restored to base level (0.90).
And yet **multi-hop@10 did not move** (0.487 ≈ v2's 0.481; only deeper-k improved).

That is the finding. With purity back at base *and* the loss provably correct, the
multi-hop ceiling held — so multi-hop@10 does **not** hinge on aggregate isotropy/purity,
and the residual gap is **not** a loss-function limit (overturning v2's guess). The
cause is **data composition**: only **1,808 / 48,413 (3.7%)** training queries are
multi-positive, so the 96% single-target eCFR signal dominates and shapes the space for
single-doc retrieval. A τ/lr/epoch variant addresses overfitting, not composition, so it
can't move set-Recall@10 by the needed +0.11 — it wasn't run.

**Ship v3** for single-target legal retrieval (best all-rounder). For multi-hop@10 RAG,
**base still wins**. The real next lever is **data, take two**: generate/oversample far
more *multi-document* training questions, and/or add a **distillation** term pulling
v3's query-doc similarities toward the base model's to preserve its co-retrieval
geometry.

## v4 — a more balanced training set (data only; not yet trained)

v3's verdict named the lever: more *multi-document* training signal. v4 builds it. The
v3 set was only **3.7%** multi-positive (1,808 / 48,413); v4 rebalances to **~1/3** by
(a) authoring a fresh batch of multi-doc questions and (b) capping the single-target
eCFR pool.

**Pipeline** (`build_v4_inputs.py` → author → `mine_v4_negs.py` → `build_train_v4.py`):
1. Sample a fresh candidate pool (`build_open_candidates.py --seed 42`), then form ~5k
   multi-doc clusters **excluding any cluster that touches an eval gold doc** — so the
   new training questions can't be near-paraphrases of eval multi-hop questions (the
   same cluster-disjointness Set B's split had by construction). 3,389 eval golds
   excluded up front.
2. Author one multi-hop question per cluster (subagents, `AUTHORING_MULTI.md`): 5,000
   authored, all valid (0 forbidden tokens / bad gold-sets).
3. Mine 5 hard negatives each + dedup vs eval (text **and** gold-set): **0 eval-text,
   0 eval-gold** overlaps → 4,929 kept.
4. Combine + cap eCFR to hit the ratio.

**Result — `train_v4.jsonl`, 20,211 rows:**
| | v3 | v4 |
|---|---|---|
| total | 48,413 | 20,211 |
| multi-positive | 1,808 (3.7%) | **6,737 (33.3%)** |
| — newly authored multi | 0 | 4,929 |
| single (eCFR capped + Set B single) | 46,605 | 13,474 |

Same multi-positive schema as v3 (uniform shape, DDP-safe), holdout-clean. **Training
is the next step** — point `train_v3.py` at `train_v4.jsonl`, then run the same serve →
Set A/B eval → ambit harness to test whether the rebalanced data lifts multi-hop@10
toward base.

## Reproduce

```sh
python build_train_pairs.py        # derive (query, positive) pairs (eCFR-QA + Set B), holdout-disjoint
python mine_hard_negs.py           # eCFR pairs -> train_ecfr.jsonl (anchor + positive + 5 hard negs)
python mine_synth_negs.py          # v2: Set B pairs -> train_synth.jsonl, SAME 7-col schema (uniform = DDP-safe)
NCCL_CUMEM_ENABLE=0 torchrun --nproc_per_node=2 train.py   # v2: full FT, MNRL + Matryoshka, both sources, 2-GPU DDP
# v3 (multi-positive experiment): keep multi-doc as multi-positive, then
# NCCL_CUMEM_ENABLE=0 torchrun --nproc_per_node=2 train_v3.py   # multi-positive InfoNCE + false-neg masking
# then serve the tuned weights under a NEW served-model-name and re-run run_eval.py / run_open_eval.py
```
