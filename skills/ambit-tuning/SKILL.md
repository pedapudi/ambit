---
name: ambit-tuning
description: >
  How to set up a fine-tune driven by ambit's measurements — the full protocol
  from diagnosis to verified result. Read this when embeddings are too crowded
  (impostors in search results, near-duplicates that can't be ranked, subsets
  blurring together) and you are considering training the embedding model to
  fix it. Covers: deciding whether to tune at all (the cheapest-fix decision
  tree), mining batches and negatives from the measurements, the three-term
  objective (ambit.training), the held-out verification loop, stop criteria,
  and the failure modes that make naive tuning silently backfire. Activates on
  "fine-tune my embeddings", "improve resolution/separability", "use ambit for
  training", "my retrieval confuses documents".
version: 1.0.0
---

# Tuning an embedding model with ambit

## The idea, in plain English

An embedding model earns its keep by keeping distinct things distinct. When a
dataset lands too crowded — many items packed into nearly the same direction —
queries stop being able to tell those items apart, and no amount of prompt or
index engineering downstream can recover the lost distinction. Fine-tuning can
recover it, but naive fine-tuning has a classic failure mode: the pair loss
pulls related items together (good for recall) and quietly spends the very
angular room needed to rank items apart (bad for precision). Push the other way
with a blunt "spread everything out" regularizer and you dissolve the grouping
that makes retrieval work at all.

The way out is to **let measurements drive every decision**: measure first,
apply the cheapest fix the measurement licenses, aim gradient only at the
measured scale and the measured items, and grade the result on held-out data
the training never touched. ambit provides each piece: the diagnosis (the
report), the aim (mining utilities), the gradient (training losses whose
influence is local by construction), and the audit (the compare report).

## Intuition for the two key quantities

- **σ\* (resolution bandwidth)** — model a query as its target item plus noise
  of scale σ. A competitor at distance r wins the retrieval exactly when the
  noise crosses the halfway plane between the two items, which happens with
  probability Φ(−r/2σ) — near zero for far competitors, a coin-flip for
  near-duplicates *at any noise level*. Summing over competitors gives the
  expected number of wrong items outranking the right one; σ\* is the largest
  noise at which that stays ≤ 1. **It is the corpus's noise budget, and the
  tune's job is to raise it.**
- **The confusable window** — the band of similarities where pairs are close
  enough to collide under realistic noise but not so close they're duplicates.
  The report's crowding curve marks where it begins (the liftoff scale). This
  window is where all the useful gradient lives — and also where unlabeled
  *true* relatives live, which is why mining it needs a guard.

## Stage 0 — Diagnose, and decide whether to tune at all

```
ambit report corpus.parquet --id-col uuid --out report.html
```

Read four things: the **crowding curve** (does the data exceed the
anisotropy-matched reference, and from what cosine?), **σ\*** vs its uniform
crossing (how much budget is spent), the **per-entity crowding field** (which
ids, at what collision cost), and the **pockets** (how many tight groups, born
at what scale). Then the decision tree — *cheapest sufficient fix wins*:

| diagnosis | fix |
|---|---|
| Data hugs the anisotropy-matched reference (cone only) | **Don't tune.** Mean-centering / dominant-direction removal / light whitening; re-measure. Linear post-processing has been measured to recover the entire impostor floor on real corpora. |
| Pockets born near distance 0 (near-duplicates) | **Fix the data, not the model.** Dedup or deliberately accept the pocket as one entity. Training against true duplicates wastes gradient and hurts recall. |
| Genuine clumping beyond the cone at moderate scale | **This is the tuning case.** Record the liftoff cosine and σ\*; they parameterize everything below. |

## Stage 1 — Mine: batches and negatives from the measurements

```python
import numpy as np
from ambit import training as tr

# reservoir/corpus embeddings under the BASE model, L2-normalized, ids aligned
Xn = ...                                   # (n, d) float32

# 1) oversample the items in trouble (uniform floor keeps the bulk anchored)
weights = tr.resolution_weights(Xn, sigma=SIGMA, floor=0.25)
# feed to a weighted batch sampler

# 2) negatives from the confusable window, with the false-negative guard:
#    an anchor's top-m base-model neighbors are NEVER negatives — the window
#    is exactly where unlabeled true relatives live
anchors, negatives = tr.mine_confusable_negatives(
    Xn,
    cos_window=(LIFTOFF_COS, 0.98),        # lo = the curve's liftoff; hi = just
    guard_top_m=5,                         #      below your true-pair scale
    per_anchor=8,
)
```

`SIGMA` is the measured σ\* (or slightly below it); `LIFTOFF_COS` is the
"crowding begins" cosine from the report. Neither is a folklore hyperparameter
— both come from Stage 0.

## Stage 2 — The objective: three terms with distinct jobs

```python
import torch
from ambit import training as tr

# z      : (B, d) embeddings from the model being tuned
# z_base : same batch under the FROZEN base model (no grad)
# pos    : (B, B) bool — known positive pairs (and mined-guard pairs), excluded

loss = (
    task_loss                                        # your pair loss: buys recall
    + LAM_C * tr.confusion_loss(z, sigma=SIGMA, exclude=pos)
    + LAM_P * tr.preservation_loss(z, z_base)
)
```

Why each term is safe by construction:

- `confusion_loss` minimizes expected retrieval collisions at the measured σ.
  Its gradient dies exponentially for pairs farther than ~3σ, so it **cannot
  disturb global dissimilarity structure** — it only widens margins inside the
  confusable window. (This is a theorem about Φ, and a unit test.)
- `preservation_loss` is the neighbor-overlap comparison in differentiable
  form: it pins *who is similar to whom* to the frozen base model, which is the
  explicit statement of "don't hurt similarity."
- Keep model drift structurally small too: LoRA / adapters / low LR.

Set `LAM_C` by sweeping and plotting **task metric vs σ\*** on held-out data;
take the knee. If the corpus structure is precious, invert the framing: hold
neighbor-overlap-vs-base above a threshold and maximize σ\* under it.

## Stage 3 — Verify: held-out data, and never the metric you optimized

- Keep a **held-out reservoir** that neither mining nor batching ever saw.
  Every verdict comes from it.
- Each epoch, embed it and run the compare report against the base:

```
ambit report base_heldout.parquet --compare tuned_heldout.parquet \
      --id-col uuid --out compare.html
```

  Watch: **σ\* rising**, the crowding-curve **liftoff retreating** toward
  higher cosine, **collision counts falling on the entities Stage 0 flagged**,
  and **neighbor overlap vs base staying high** — that last one is the drift
  alarm; if it collapses you are re-skinning the space, not repairing it
  (raise `LAM_P`, lower LR).
- **Goodhart discipline:** you optimized C(σ) at one σ — grade with the full
  curve, the per-entity field, the pockets, separability, and above all your
  real retrieval evals. A diagnostic that becomes a loss stops being a
  diagnostic.
- Stop at σ\* plateau or the first task-metric regression. Ship the compare
  report as the run's audit trail.

## What silently backfires (the unsound list)

1. **Regularizing before diagnosing** — you may be training against a cone
   that centering fixes for free.
2. **Uniform spreading pressure at all scales** (a large uniformity weight) —
   spends the grouping–resolution budget backwards; recall dies.
3. **Training against duplicates** — they need dedup, not gradient.
4. **Mining the confusable window without the guard** — you will push true
   relatives apart; this is the most common silent failure.
5. **Validating on the optimized functional** — see Goodhart above.
6. **Reading σ\* as an external-query guarantee** — its scope is intra-corpus
   confusability; your eval queries are the arbiter beyond that.

## References

- Worked end-to-end example (synthetic, honest measured numbers):
  `examples/training-regularizers/`
- The mathematics of σ\* and the confusion kernel:
  `docs/concepts/continuous-occupancy.md` §13
- Reading the report that drives Stage 0: `docs/guide/interpreting-the-report.md`
- A complete real-corpus workflow (embedding service, eval harness, compare
  reports): `examples/legal-retrieval-audit/`
