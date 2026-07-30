# Analysis: resolution loss and dense-pocket detection in ambit

> A critical review of how ambit detects crowded/dense pockets, what the measure
> is actually for, where it is sound and where it misleads, and how it could be
> improved. The conclusions are grounded in controlled experiments run against
> ambit's own code (`localized_anisotropy`); the reproduction recipes are in the
> appendix. This is a review document, not part of ambit's own concept notes
> (`docs/concepts/`).

---

## TL;DR

- **ambit's objective is resolution-loss detection**, not density-mode discovery:
  *can cosine similarity still tell distinct entities apart, and where does it
  fail?* The salient harm is the tension it puts on recall, ranking, and
  precision — impostors crowding the top of a search, near-duplicates that cannot
  be ranked apart, cluster members that cannot be resolved.
- Judged as a **density/mode detector**, the pocket detector has real flaws (it
  fires on homogeneous low-dimensional data that contains no separated mode).
  Judged against its **actual objective (resolution)**, most of those flaws soften
  or invert: the isotropic reference is the *right* yardstick, and `global_crowding`
  is a legitimate, graded resolution-loss signal that even catches loss the
  headline mean-pair-cosine test misses.
- The **sharpest surviving critique is conceptual**: *density is not resolution.*
  The detector triggers and ranks on local density, but the operational quantity is
  the **NN margin / unresolvability**. High density with a wide margin (well-separated
  near-duplicates) is fine; low density with a flat similarity profile is not.
  Localization should key on thin margins, with density as corroboration.
- Two orthogonal issues stand regardless of framing: **no test coverage** for the
  detector, and the **2-D projected density figures** (computed on a PCA projection)
  are unreliable for native-space crowding and over-claim ("real cluster cores").
- The largest missed opportunity: ambit reports *that* a space is crowded but not
  **how much of the loss is recoverable** by a cheap, label-free post-process. A
  demonstrated example below removes ~100% of the impostor floor and widens margins
  4× with mean-centering + all-but-the-top — no training, no labels.

---

## 1. What ambit is for

ambit measures how a dataset **occupies** an embedding space and, at the local
level, whether items are packed so tightly that cosine similarity can no longer
separate them. The framing in `docs/concepts/anisotropy-and-resolution.md` is
sound: crowding is a **loss of resolution**, read against an isotropic reference
rather than an absolute threshold (learned-embedding cosine is only interpretable
relative to a distribution). The operational stakes are retrieval-shaped:

- **Precision** — unrelated impostors sit high in cosine and cannot be ranked out.
- **Ranking** — the best of several relevant items cannot be told from the runners-up.
- **Recall / dedup / diversity** — near-duplicates and genuinely-distinct neighbors
  blur together.

The key point for everything below: the target is **unresolvability between
entities**, not the topological question "is there a separated bump in the density
distribution." The two are related but not the same, and conflating them is the
root of both the tool's confusions and the confusions in an initial reading of it.

## 2. How pocket detection computes today

There are **two independent implementations**, and they are different estimators.

### 2a. Native-space local density field — `local_anisotropy.py` (RES 06 / RES 07)

Operates on the original high-dimensional (e.g. 768-d) unit vectors.

1. For each item, the **mean cosine to its `k` nearest neighbors** at scales
   `k = 10, 50, 200` — treated as a k-NN density estimate. (With `--mutual-knn`,
   the mean is over reciprocal neighbors only, a hubness correction.)
2. A **synthetic isotropic reference**: `n_ref = 4000` points drawn iid Gaussian and
   L2-normalized (uniform on the ambient d-sphere), with each scale's `k` rescaled by
   `n_ref / m` to hold `k/N` fixed.
3. Robust z-scores (`(field − median)/MAD`) computed *within* the dataset and *within*
   the reference. The **headline scale `k*`** is the one whose field has the most
   extended upper tail (in MAD units).
4. A point is flagged **crowded** when its dataset z exceeds the *reference's* own
   upper-tail z quantile (α = 0.01, Bonferroni-split across scales, unioned).
5. Flagged points are **angular-agglomerated** (`AgglomerativeClustering`, cosine,
   average linkage, threshold tied to the density level); clusters below a minimum
   size are dropped. The survivors are **pockets**.
6. `global_crowding = (bulk − iso_bulk) / iso_MAD` reports the global story; each
   pocket carries a concentration, an NN margin, an `IsoScore*`, and its scale.

RES 06 draws the field distribution against the reference; RES 07 recolors the 3-D
PCA cloud by each item's crowding z.

### 2b. Projected density peaks — `den_prom.py` (DEN 04), `den_hexbin`, `den_contour`

Bin the **2-D projection** (`ctx.xy`, default `projector="pca"`) onto a grid, smooth,
and find topographic peaks by prominence (marching-squares contours). DEN 04 ranks
peaks and labels the tall ones as cluster cores.

## 3. Two lenses — and why the objective decides the verdict

The same behavior reads very differently under two framings.

### The density-mode lens (a natural first reading — and the wrong target)

Read as "find separated over-densities," the detector has real problems, cleanly
demonstrated by controlled inputs (§5): it flags **pockets on uniform, mode-free
data that merely lives on a low-dimensional manifold.** A uniform distribution on a
40-dimensional subsphere is homogeneous — every neighborhood is statistically
identical — yet it is reported as having a pocket. By the mode definition, that is a
false positive, and it traces to the null: a uniform-on-the-*ambient*-768-sphere
reference has a pathologically short field tail (distance concentration), so its
tail threshold is tiny, and any lower-dimensional dataset has a naturally heavier
tail that crosses it.

Under this lens the recommendation would be to swap in a proper density-peak method
(Rodriguez–Laio decision graph; the point-adaptive, significance-tested PAk/DPA of
d'Errico–Facco–Laio–Rodriguez; or HDBSCAN cluster stability). **But this lens is the
wrong target.** ambit is not trying to enumerate modes; it is trying to find where
resolution is lost.

### The resolution lens (the actual objective — and what flips)

Read as "where can cosine no longer separate entities," several judgments invert:

- **The isotropic reference is appropriate, not a miscalibration.** Resolution is
  inherently measured against the achievable best (isotropy is where unrelated pairs
  are orthogonal and a true match stands out). Refusing to "grade on a curve" is the
  point. A manifold-matched null would *hide* the loss the user wants surfaced — it
  would say "you are as resolved as a 40-d manifold allows" when the operational
  truth is "your entities pack at cosine 0.55 with a ±0.16 impostor spread; retrieval
  will struggle." (Note: the theory note's §08–09 argues for a *breadth*-calibrated
  reference. That argument is about **localization** — "is this sub-population
  anomalous *for this model*" — a different question from "how much resolution does
  this space have.")
- **`global_crowding` is a genuine, graded resolution signal — and it catches loss
  the headline anisotropy test misses.** Uniform blobs on 40-, 15-, 4-d subspheres
  all have **mean random-pair cosine ≈ 0**, so ambit's "anisotropic if mean > ~4/√d"
  check calls them healthy — yet their impostor floor widens (±0.036 → ±0.16 → ±0.26
  → ±0.50) and neighbors tighten. `global_crowding` reads the neighborhood *level*
  against isotropy and rises accordingly, surfacing exactly this loss.
- **The low-dimensional "false pocket" downgrades from a broken measurement to a
  labeling blemish.** The dominant, correct finding (globally reduced resolution) is
  delivered by `global_crowding` and the margin; the spurious small "pocket" is an
  error of calling a *global* condition *local*. That still matters — because global
  vs. local imply different remedies (§7) — but it is a communication bug, not a
  measurement failure.

## 4. Density is not resolution — the load-bearing point

What the objective actually cares about — "can't tell these entities apart" — is
**unresolvability**: a thin top-1-vs-top-2 margin and a high local impostor floor.
Density is only a proxy, and the two come apart in both directions:

- **High density, fine resolution:** a cluster of paired near-duplicates, each item's
  true match at cosine 0.95 and everything else at 0.5 — dense, but a 0.45 margin
  resolves it perfectly. Density flags it; resolution says it is fine.
- **Low density, lost resolution:** items spread out but with a flat similarity
  profile (top-1 and top-2 both ≈ 0.3) — not dense, but unresolvable.

So the salient signal is **density ∧ thin margin**, not density alone. ambit already
computes a per-pocket margin, but it *detects and ranks* pockets by the density-field
z-score and reports margin as a secondary attribute. Under the objective that is
backwards: localization should key on the **margin / confusability field**, with
density as corroboration. This is also consistent with ambit's own docs, which call
the NN margin "the within-neighborhood resolution."

A caveat the margin exposes and that no unsupervised method can fully close: a thin
margin can mean "impostors crowd a genuine best match" (a real precision failure) or
"the neighbors are all genuinely equivalent/unrelated" (nothing to resolve).
Distinguishing them needs positive pairs — the alignment half ambit deliberately does
not claim. The honest deliverable is therefore a **battery** read together, not a
single score.

## 5. Empirical findings

All runs use ambit's own `localized_anisotropy` on synthetic 768-d inputs; see the
appendix to reproduce.

### 5a. Intrinsic dimension alone drives the "pocket" false positive

Three uniform, mode-free distributions differing *only* in intrinsic dimension:

| distribution (uniform, no separated mode) | mean pair cosine | `global_crowding` | pockets found | RES 06 tag |
|---|---|---|---|---|
| uniform on full **768**-sphere (= the null) | +0.000 | 0.2 | **0** | roomy/uniform |
| uniform on **40**-dim subsphere | −0.001 | 169.4 | **1** | 1 dense pocket |
| uniform on **15**-dim subsphere | +0.000 | 116.8 | **2** | 2 dense pockets |

The honest answer for all three is *zero local pockets*. The detector is calibrated
only for data uniform on the full ambient sphere.

### 5b. The multi-cone "dandelion" behaves as the docs promise

A proper dandelion (many tight cones in random directions) yields **no pockets** and
reads as globally denser — exactly as `docs/concepts/` claims. (An earlier informal
test that appeared to contradict this used a *single* cone, not a dandelion; the
docs' dandelion claim holds.)

| dandelion | mean pair cosine | `global_crowding` | pockets | tag |
|---|---|---|---|---|
| 200 cones × 45 pts | +0.001 | 31.4 | 0 | globally denser |
| 600 cones × 15 pts | +0.000 | 10.0 | 0 | globally denser |
| 1200 cones × 7 pts | +0.000 | 4.4 | 0 | roomy |

### 5c. A flagged "pocket" on homogeneous data has no separated mode

For the uniform 40-d subsphere the field is a single smooth unimodal bell
(range 0.376–0.475, median 0.426). The flagged pocket members (0.458–0.468) are the
top sliver of the tail — and are *not even the densest points* (non-pocket items run
to 0.475). There is **no valley/gap**: it is the tail carved out by the angular
clusterer. Contrast a genuinely planted tight cluster, where the field is truly
bimodal — bulk at 0.075–0.133, an empty gap, then the pocket at 0.778–0.810.

### 5d. Resolution loss tracks intrinsic dimension (why the isotropic null is right)

| uniform on… | NN cos (top-1) | NN margin (t1−t2) | random-pair cosine (impostor floor) |
|---|---|---|---|
| 768-sphere | 0.13 | 0.0065 | 0.00 ± 0.036 |
| 40-dim | 0.55 | 0.023 | 0.00 ± 0.162 |
| 15-dim | 0.80 | 0.024 | 0.00 ± 0.261 |
| 4-dim | 1.00 | 0.0019 | 0.00 ± 0.501 |

The mean is ≈ 0 throughout (so a mean-based anisotropy check sees nothing), while the
impostor floor *spread* grows as 1/√(intrinsic dim). Lower intrinsic dimension ⇒
unrelated items sit at higher cosine ⇒ real resolution loss. `global_crowding`,
reading the neighborhood level against isotropy, is the signal that catches it.

### 5e. Much resolution loss is recoverable — and unmeasured today

An 8-cluster structure buried under a shared cone direction + 2 rogue dimensions,
before/after cheap label-free post-processing:

| transform | impostor floor | NN margin | top-1 flips under ε-noise |
|---|---|---|---|
| raw (coney + rogue) | +0.267 | 0.0017 | 87.0% |
| mean-centered | +0.003 | 0.0025 | 82.1% |
| all-but-the-top-3 | +0.001 | 0.0044 | 72.1% |
| all-but-the-top-6 | +0.005 | 0.0068 | 61.7% |

Almost the entire impostor floor is a removable shared direction; margins widen 4×
and retrieval fragility drops 25 points — no training, no labels. ambit reports the
crowding but not this **headroom**.

## 6. Ledger: what holds, what was retracted

**Holds (under the resolution objective):**

- Density ≠ resolution; the direct signal is the NN margin / confusability, and
  localization should be margin-led (§4).
- Reporting global resolution loss as a *local* "pocket" (§5a, §5c) is a real
  communication error with remedy consequences (§7).
- No test coverage for the detector; a planted-scenario harness would catch §5a/§5c.
- The 2-D projected density path (DEN 04 / hexbin / contour) is unreliable for
  native crowding and over-claims ("real cluster cores") on a PCA projection.
- `IsoScore*` is near-blind (shrinkage → isotropic target) for the small pockets it
  is meant to characterize.

**Retracted or softened:**

- "The isotropic null is miscalibrated / should be manifold-matched" — *retracted for
  this objective*; isotropy is the correct resolution ceiling (§3).
- "`global_crowding` is low-specificity / duplicative" — *retracted*; it is a graded
  resolution signal that catches loss mean-pair-cosine misses (§3, §5d).
- "The dandelion claim is contradicted" — *retracted*; the docs' claim holds (§5b).
- "Multiple-testing budget compounds across scales" — *incorrect*; the split is a
  correct Bonferroni union. The issue was never the union, only (for the mode lens)
  the null.
- "Use PAk/DPA/density-peaks" — *redirected*; those target modes, not resolution.

## 7. Recommendations

### The organizing change: make unresolvability first-class

Redefine a pocket as a coherent region where the **local NN margin is thin AND the
local impostor floor is high**; density only corroborates. This fixes both false
modes (well-separated near-duplicates stop being flagged; homogeneous low-d data
reads as *global* loss, not a local pocket) and aligns the trigger with the objective.

### New measures (all label-free)

1. **Recoverable resolution ("headroom") — top priority.** Re-measure floor, margin,
   and fragility after the cheap post-processes the docs already cite (mean-centering;
   all-but-the-top-`k`, Mu et al. 2018; rogue-dimension standardization, Timkey & van
   Schijndel 2021; whitening) and report the *delta*: "≈100% of your impostor floor
   is a removable shared direction; all-but-top-6 widens margins 4× and cuts
   fragility to 62%" (§5e). Turns ambit prescriptive. Show as a counterfactual over
   `k`, never applied silently.
2. **Retrieval-fragility stress test.** Perturb each item by ε (or quantize to int8 /
   binary, as production ANN does) and report **top-k churn** — the "87% of top-1
   neighbors flip" number is a visceral, operational statement of the recall/ranking
   fragility density causes, and it is the per-item complement to the global floor.
3. **Confusability count.** Per item, the number of neighbors within a small cosine
   band of the top-1 (near-ties). This *is* "can't tell these entities apart," as an
   integer; its distribution against the isotropic null can replace the density
   histogram as the RES 06 headline.
4. **Local intrinsic-dimension field** (TwoNN, Facco et al. 2017). Explains *why* a
   region is crowded (impostor floor ≈ 1/√(local ID), §5d) and is a better small-`n`
   shape descriptor than `IsoScore*`.

### Dual reference + uncertainty

Report against **two** nulls, labeled: the **isotropic** reference ("how much
resolution has this space lost vs. the achievable best") and a **manifold-matched /
phase-randomized surrogate** of the data's own geometry ("is this pocket anomalous
*for your* space, or just your uniform texture"). Together they cleanly separate
global degradation (whiten / change model) from a genuine local pocket (a confined
sub-population). Add **bootstrap error bars** over the reservoir so "pocket" vs.
"sampling noise" is stated and the global-vs-local share is quantified.

### Visualization

- **Lead with a resolution verdict panel** (impostor floor, median margin,
  confusability, headroom-after-whitening — all against isotropic, good/bad by
  direction). Demote density/pockets to drill-down.
- **Recolor the 3-D cloud by native margin/confusability** (position stays
  layout-only, the existing RES 07 pattern) so "where retrieval is fragile" is visible.
- **A confusion-neighborhood inspector**: pick a crowded item, show its actual top-k
  with cosines and margin (and text/labels if present), making the loss auditable.
- Retire the "real cluster cores" language on DEN 04; use the 2-D projection for
  layout only, not for density claims.
- State scope (global vs. local) in words, with the implied remedy.

### Rigor

- **A planted-scenario test harness**: uniform-manifold → 0 local pockets but global
  loss flagged; planted tight cluster → 1 pocket; dense-but-wide-margin near-duplicate
  cluster → 0 resolution pockets. Encodes ambit's own promises and catches §5a/§5c.
- **Per-item hubness as a precision-risk overlay** (a hub is everyone's nearest
  neighbor — a precision failure by definition), not only a global skew scalar.

### Tradeoffs / honest floor

The perturbation and headroom passes cost extra compute, but bounded (reservoir-scale,
numpy). Whitening / all-but-the-top change the geometry, so they must be shown as
tunable counterfactuals, never silently applied. Local-ID is noisy at small scale
(report with a CI). And the hard limit stands: unsupervised, ambit cannot separate
"thin margin because impostors crowd a true match" from "thin margin because the
neighbors are genuinely equivalent" — that needs positives. The improvement is not a
better single number; it is pointing every measure at **unresolvability** instead of
density, telling the user **how much of the loss is recoverable**, and labeling scope
**global vs. local**.

### Suggested build order

1. Flip pocket detection to margin-led + add the planted-scenario tests.
2. Add the recoverable-resolution ("headroom") measure.
3. Add the perturbation / quantization fragility test.

Each is a few hundred lines against the existing `metrics.py` / `local_anisotropy.py`
scaffolding.

---

## Appendix — reproduction

All experiments run on synthetic 768-d data with a fixed seed against ambit's
`localized_anisotropy` and small numpy helpers. The two load-bearing constructions:

**Intrinsic-dimension ladder (§5a, §5d)** — uniform on an `idim`-dimensional
subsphere of R^768:

```python
import numpy as np
from ambit.local_anisotropy import localized_anisotropy
rng = np.random.default_rng(0); D = 768
def manifold(n, idim):
    B = rng.standard_normal((idim, D)); B /= np.linalg.norm(B, axis=1, keepdims=True)
    return rng.standard_normal((n, idim)) @ B          # uniform on an idim-subsphere
for idim in (768, 40, 15, 4):
    X = manifold(6000, idim) if idim < 768 else rng.standard_normal((6000, D))
    la = localized_anisotropy(X.astype(np.float32))
    print(idim, la.global_crowding, len(la.pockets))   # → 0 pockets only at idim=768
```

**Recoverable resolution / fragility (§5e)** — genuine cluster structure hidden under
a cone + rogue dimensions, measured before/after cheap fixes:

```python
def unit(A): return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
def all_but_top(A, k):
    Ac = A - A.mean(0); _, _, Vt = np.linalg.svd(Ac, full_matrices=False)
    return Ac - (Ac @ Vt[:k].T @ Vt[:k])
# impostor floor  = mean random-pair cosine
# NN margin       = median(top1 - top2 cosine)
# fragility       = fraction of items whose top-1 neighbor changes under ε-noise
```

Findings are stable across seeds; the exact figures above are from `seed=0`.
