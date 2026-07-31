# Training-time regularizers — a worked example

`ambit.training` turns the occupancy measurements into gradient signal and batch
composition for adapting an embedding model over a profiled corpus. This example
runs the whole sound loop on a synthetic corpus with a planted crowded pocket,
using a linear adapter as a stand-in for whatever model you actually train:

1. **Measure** — resolution bandwidth σ\* and per-entity collision counts on the
   base embeddings (the diagnosis that licenses the intervention and sets the
   one hyperparameter that matters, the target scale).
2. **Mine** — `resolution_weights` oversamples the crowded entities;
   `mine_confusable_negatives` drafts negatives from the confusable window with
   the false-negative guard (an anchor's top-m base-model neighbors are never
   negatives).
3. **Train** — `confusion_loss` at the measured σ widens margins only inside the
   confusable window, while `preservation_loss` anchors who-is-similar-to-whom
   to the frozen base embeddings.
4. **Verify on held-out entities the training never saw** — σ\* up, collision
   counts on the flagged entities down, neighbor overlap with the base intact —
   and never grade with the one functional you optimized.

Run it (needs the optional `train` extra):

```
pip install -e '.[train]'
python examples/training-regularizers/run_example.py
```

Expected output: the flagged entities' expected collisions roughly **halve**
(≈2.6 → ≈1.3 median) while mean neighbor overlap with the base model stays
≈0.92 — resolution recovered without re-skinning the space. That is the honest
ceiling of a *linear* adapter, which cannot unfold a tight pocket much further
without moving everything else — precisely the trade the preservation term
enforces. A real encoder (or a nonlinear adapter) has far more capacity; the
loop, the measured σ target, the guard, and the held-out verdicts carry over
unchanged. Swap in your encoder and your reservoir; the diagnosis numbers come
from the report's header facts.
