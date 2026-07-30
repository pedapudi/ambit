# Interpreting an ambit Report

> A field guide to every figure and number in an ambit report. The audience is an
> engineer who has a report open in a browser and wants to know exactly what each
> visualization and metric means, how it was computed, which direction is "good,"
> and how the figures corroborate one another.
>
> This is the practical companion to
> [Anisotropy, Resolution, and What ambit Measures](../concepts/anisotropy-and-resolution.md),
> which provides the conceptual background and the academic citations reused here.

---

## 1. What an ambit report is

An ambit report is a single, self-contained HTML page that visualizes **how a
dataset occupies its embedding space**. You point ambit at a corpus of vectors
(or text it will embed), it scans a reservoir of them, computes a battery of
native-space and projected diagnostics, and renders a stack of figure "cards"
plus a header row of scalar facts.

Every report answers the same three questions, stated in its own lede:

1. **Where does the data concentrate?** — the density / hotspot views: where items
   pile up, and where they leave the space empty.
2. **How much of the space does it use?** — the coverage and dimensional views:
   the volume the data fills versus the capacity it wastes.
3. **How distinct are its items?** — the resolution / isotropy views: whether
   cosine similarity can still tell two items apart, globally and locally.

These map onto the **facet families**. In the report each card is titled by its
**name**; the short family codes like `RES 01` or `CMP 12` are this guide's own
cross-reference shorthand — the report surfaces figures by name and does not print
the codes as pills. The prefix is its family:

| Family | Question it serves | Reads on |
|---|---|---|
| **MAP** | where it concentrates | the projected 2-D footprint |
| **DEN** | where it concentrates | density peaks, contours, hotspots |
| **COV** | how much space it uses | coverage, sparsity, reach, voids |
| **3D** | how much volume / its shape | the 3-D occupied volume and its anisotropy |
| **RES** | how distinct items are | native-space resolution / isotropy diagnostics, **global *and* localized** (the local-density field, RES 06/07) |
| **CMP** | how two embeddings differ | the same items embedded **two ways** (`--compare`, aligned by id): neighbor-overlap drift, CKA / MMD / Procrustes, the drift field, the distribution shift |

> A given report only renders the figures that are **enabled** in its run
> configuration. Enabling and disabling figures is done through the `Config`
> object (a figure-toggle map) or the equivalent CLI flags — never through
> environment variables. This guide documents the **default-enabled** set: the
> default figures plus the header facts row — the **separability panel** joins them
> whenever a partition exists, `uniformity` is a header scalar, and the **CMP** figures
> appear when you pass `--compare`. A report with a custom configuration may
> show fewer or more; figures not covered here are hidden by default.

The figures appear in a fixed reading order. It is set by the explicit
`render._DISPLAY_ORDER` list (any enabled figure not named there sorts after, by
its own `order` field), not by the order in this guide's table. With the default
figure set the report reads top to bottom as: **RES 07 → 3D 02 → RES 06 → RES 04 →
RES 05 → RES 05b → RES 01 → RES 02b → RES 02 → 3D 05 → DEN 04 → COV 09**. So the
interactive **local crowding cloud (RES 07)** opens the report, the **orthographic
triptych (3D 02)** follows, and the **local density field (RES 06)** sits mid-stack
with the rest of the RES block. The **CMP** figures, when a `--compare` set is
present, sort ahead of all of these and lead the report; **RES 08 (uniformity)**
sits among the RES block in `--compare` / series mode. This guide documents the
figures grouped by family for teaching, but notes each figure's actual position.

---

## 2. How to read any ambit figure

A handful of conventions hold across every card. Internalize these once and each
figure becomes self-describing.

### The single accent color = "this dataset's own signal"

Every figure reserves one **accent** color for the thing the figure is actually
about — the dataset's own measured curve, its mean, its dominant peak, its median
margin. Everything else is drawn in neutral "ink" tones (full ink, soft ink, faint
ink) or in semantic tints. When you see the accent, you are looking at *this
dataset's reading*, not a reference or a guide line. There is exactly one accent
subject per figure, deliberately, so your eye is pulled to the measurement.

### Good/bad is communicated by DIRECTION, not by a single hue

This is the most important convention and the easiest to get wrong. ambit does
**not** encode "good" as one fixed color and "bad" as another that you memorize.
Instead:

- **Position and direction carry the verdict.** In a cosine histogram, mass to the
  *right* (toward +1) is crowded/bad and mass near 0 is healthy — regardless of
  what color the curve is rendered in. In the margin histogram, bars to the *right*
  (large margin) are decisively resolved and bars at the *left edge* (margin → 0)
  are near-ties. In the cumulative-variance plot, a curve that *bows up steeply* is
  concentrated/bad and one that *hugs the diagonal* is well-spread/good.
- Some figures *do* use the semantic tint tokens — `--good`, `--bad`, `--caution`
  — as a reinforcement (e.g. near-tie histogram bars tinted with the "bad" token,
  decisive bars with the "good" token). But the tint is redundant with the
  geometry, never a substitute for it. The axis and the reference line tell you the
  direction; the tint just colors it in.

The reason for this discipline: ambit is **theme-adaptive**. The report ships
**16 color themes** (monokai, solarized-dark/light, google-light/dark,
dracula, zenburn, paper, …) selectable live from the picker in the header. When
you swap themes, every figure re-skins instantly — *without re-rendering*, because
the SVGs are drawn with CSS color tokens (`var(--accent)`, `var(--good)`,
`var(--ink-faint)`, …) that the theme redefines. A meaning that depended on a
specific RGB value would break on a theme swap. So meaning lives in **where a mark
sits relative to a reference**, and color is only there to group and to draw the
eye. The accent might be cyan in one theme and teal in another; it is always "the
dataset's signal" in both.

### Every card has a legend and a `Reveals:` line

Each figure card is structured the same way:

- A **header** with the figure's number (`RES 01`), name, a terse method tag
  (`cosine density`), and a one-paragraph `why` describing what it shows for *this
  run*, with the run's actual numbers filled in.
- The **figure** itself (an SVG, or a live canvas).
- A one-line **legend** decoding the marks — which glyph is the dataset, which is
  the reference, which is the accent.
- A bolded **`Reveals:`** line: the single takeaway, the thing this figure is
  *for*. When skimming a report, read the `Reveals:` lines first.

### Interactive figures

Static figures are SVG and respond to theme swaps only. Three things are
interactive:

- **COV 09** (sparsity field) has a **#-samples slider** that thins a crowded
  field of rings down to a legible subset.
- **RES 07** (local crowding cloud) is a full interactive canvas: **drag to
  rotate**, **scroll or pinch to zoom**, a **#-samples slider**, and a
  **kNN-edges checkbox** that overlays the neighbor graph. It auto-spins when idle.
  This is the report's one rotatable 3-D view, recoloring and reshaping the cloud by
  local density.
- The **IsoScore readout** (inside RES 02b, and shown as a header fact) is a
  **dotted-underline hovercard**. Hover it with the mouse, or focus it with the
  keyboard and activate, and a card unfolds showing the exact formula, the
  intermediate terms for this dataset (d, δ, n), and a plain-language meaning of
  the 0–1 scale.

### Graceful degradation

Several figures need inputs that may be absent (a kNN backend, cluster labels, a
3-D projection, a covariance spectrum). When an input is missing, the figure does
not vanish — it renders a faint placeholder cloud and a "needs kNN backend" /
"needs cluster labels" note. If you see such a note, the figure isn't broken; the
run simply didn't have that ingredient.

### Reciprocal (mutual) kNN

By default each point's k nearest neighbors are taken **as found** — a directed graph,
where *i* listing *j* says nothing about *j* listing *i*. The `--mutual-knn` flag filters
**every** kNN graph in the report to its **reciprocal** edges: a neighbor survives only
when both points list each other. This suppresses **hubs** by construction (a point that
appears in many lists but reciprocates few keeps only the few), so the hub-skew fact
drops sharply, the retrieval margin (RES 04) and sparsity field (COV 09) read off the
mutual graph, the local-density field (RES 06/07) becomes a hubness-corrected estimate,
and a point with *no* mutual neighbor reads as maximally isolated. It re-skins every
neighbor-based figure at once — use it to separate genuine local structure from hubness
artifacts.

---

## 3. The header metrics (the facts row)

Directly under the lede, the report prints a row of key/value facts — the scalar
summary of the whole scan. These are computed in `render._facts(ctx)`; the math
lives in `metrics.py`. Read them as a profile that the figures below then unpack.

### items × dims

**What:** corpus size — number of items (rows) by embedding dimension. If the scan
was approximate (sampled), it appends `(≈N sampled)` to show the reservoir size
actually analyzed.

**How read:** `dims` is the *nominal* capacity (e.g. 768, 1536). Almost every
resolution metric below asks the same question in a different way: *how much of
that nominal capacity is really used?*

### mean L2 norm

**What:** the mean Euclidean length of the raw vectors, `ctx.scan.norm_mean`.

**How read:** Most resolution diagnostics are computed on **unit-normalized**
vectors (cosine geometry), so the norm itself doesn't enter them. But the mean norm
is a sanity check. If embeddings are already L2-normalized at the source, this is
≈ 1.000. A value far from 1, or a large spread, tells you the source vectors are
unnormalized — worth knowing before you compare runs.

### mean pair cosine

**What:** the mean cosine similarity of random pairs, `ctx.cos.mean()`, printed
with a sign. This is **anisotropy** in the sense of Ethayarajh (2019): the average
angular agreement of two unrelated items.

**How computed:** `metrics.random_pair_cosine` unit-normalizes the reservoir, draws
`pairs` (default 200,000) random index pairs i≠j, and takes `⟨u_i, u_j⟩` for each.
`ctx.cos` is that sample; this fact is its mean. (The full distribution is RES 01.)

**How read — DIRECTION:** lower (toward 0) is better.

- **≈ 0** → isotropic: unrelated items are ~orthogonal, the healthy case.
- **mildly positive (≈ 0.05–0.15)** → a gentle cone; usually fine.
- **strongly positive (≳ 0.2)** → a pronounced cone: every random pair already
  looks somewhat alike, so the dynamic range left to distinguish *related* items is
  compressed.

**Reference:** for i.i.d. points on a unit d-sphere, random-pair cosine is
≈ N(0, 1/d), so the isotropic mean is 0 with a tiny spread. Judge the number
against *that*, never against an absolute threshold — cosine values are only
interpretable relative to the space's own reference distribution (Steck et al.
2024).

### isoscore

**What:** **IsoScore** (Rudman et al. 2022) in [0, 1] — *the uniformity of
embedding-space utilization*: how evenly the variance fills all the dimensions.
This is the header's dotted-underline hovercard; hover/focus for the formula.

**How computed:** `metrics.isoscore` / `isoscore_parts`, from the covariance
eigenvalues λ (the variance along each principal axis):

```
v        = √d · λ / ‖λ‖₂            (normalize so an isotropic spectrum ⇒ v = all-ones)
δ        = ‖v − 1‖₂ / √(2(d − √d))  (isotropy defect, 0 = isotropic … 1 = one axis)
n_iso    = d − δ²·(d − √d)          (dimensions isotropically used, between √d and d)
IsoScore = (n_iso² − d) / (d² − d)  (0 = degenerate line … 1 = perfect sphere)
```

**How read — DIRECTION:** higher (toward 1) is better.

- **1.0** → variance spread equally over every dimension; a perfect sphere.
- **0.0** → all variance on a single axis; a degenerate line.

Real text embeddings sit **low** — single digits to low tens of a percent is
common and not alarming in itself; it reflects how much structure lives on a few
axes. Read it next to effective rank and dims-for-90%: those say *how many* axes;
IsoScore says *how unevenly* the variance is split among them.

### uniformity

**What:** **uniformity** on the unit hypersphere (Wang & Isola 2020) —
`U = log E exp(−2·‖x−y‖²)` over random pairs. For unit vectors `‖x−y‖² = 2 − 2·cos`, so
it is a function of the same random-pair cosines as *mean pair cosine* above
(`metrics.uniformity_from_cos`), and costs nothing extra.

**How read — DIRECTION:** lower (more negative) is better — a more uniform spread, more
of the sphere occupied. It is the formal, pair-potential version of the cone reading: a
strongly anisotropic cone reads as a *less* negative (worse) uniformity. Read it against
the isotropic reference in **RES 08** (below). This is the *uniformity* half of the
alignment/uniformity pair; the *alignment* half needs positive pairs (supervision) and
is outside ambit's unsupervised scope.

### occupancy z

**What:** the **Stolarsky occupancy discrepancy** — the mean pairwise chord
distance between entities, as a z-score against the uniform-sphere null with a
matched pair-sample size. By Stolarsky's invariance principle (1973), the mean
pair distance *is* (up to constants) the spherical-cap L2 discrepancy integrated
over **every cap position and every cap radius** — "all bins at all scales" as a
single number, with no lattice to choose.

**How computed:** `occupancy.stolarsky_z` — mean chord from the header's cosine
sample, against 24 replicates of the analytic uniform null (pair cosines drawn
directly via a Beta transform; no vectors materialized).

**How read — DIRECTION:** near **0** is healthy; **strongly negative** is crowded.
The null is razor-thin in high dimension, so real crowding produces very large
negative values; treat the magnitude as detection strength, not as a graded scale.

### resolution bandwidth

**What:** **σ\*** — the largest query-noise scale at which the corpus stays under
one expected retrieval collision per entity, under the Gaussian query channel
q = x + σ·g. This is the occupancy story in operational units: *how much query
perturbation can this corpus absorb before entities become interchangeable?*

**How computed:** `occupancy.sigma_star` — the exact pairwise confusion law
Φ(−‖x−y‖/2σ) integrated over the pair-cosine sample (union-bound conservative
under any competitor correlation), inverted by bisection at tolerance 1.

**How read — DIRECTION:** higher is better. Compare across corpora or embeddings
of the same corpus; a drop in σ\* means the space now confuses entities at noise
levels it previously tolerated. Scope: intra-corpus confusability (dedup,
clustering, corpus-as-queries retrieval) — extrapolation to an external query
workload is an assumption, not a measurement.

### effective rank

**What:** the continuous effective dimensionality, printed as `erank / dim`
(Roy & Vetterli 2007).

**How computed:** `metrics.effective_rank` takes the singular values `s = √λ`,
normalizes them to a probability vector `p = s/Σs`, and returns the **exponential of
the Shannon entropy** of p: `exp(−Σ p log p)`. Intuitively, the number of
dimensions that meaningfully share the variance — if k dimensions held it equally,
this would be k.

**How read — DIRECTION:** higher (toward the nominal dim) is better.

- **near `dim`** → variance is genuinely spread; little collapse.
- **far below `dim`** → **dimensional collapse**: the space lives on a
  lower-dimensional subspace; the rest of the axes are wasted capacity.

Note: effective rank is **entropy-based**, so a long tail of tiny eigenvalues can
*inflate* it relative to the participation ratio (see RES 02b). When the two
disagree, trust the cumulative-variance curve, not either scalar alone.

### dims for 90% var

**What:** the number of leading principal axes needed to capture 90% of the total
variance, printed as `d90 / dim`.

**How computed:** `metrics.dims_for_variance(eigs, 0.9)` cumulatively sums the
sorted eigenvalues, normalizes to a fraction, and returns the first index at which
the cumulative fraction reaches 0.9 (one-based).

**How read — DIRECTION:** higher (a larger fraction of `dim`) is better.

- **a large fraction of `dim`** → variance is broadly distributed.
- **a tiny handful** → a few axes dominate; the operational dimensionality is far
  below the nominal one. (5 of 768 is severe collapse; 276 of 768 is front-loaded
  but far from collapsed.)

### groups

**What:** present only when the run has cluster labels. Shows the number of distinct
groups and the label source, e.g. `7 · provided` or `12 · clustered`. The source
tag distinguishes labels you **supplied** from labels ambit **derived** by
clustering (`ctx.labels_source`).

**How read:** This is the partition that RES 05 (within- vs between-cluster cosine)
and the RES 05b separability panel read. If the source is `clustered`, the groups are
ambit's own guess at structure, not ground truth — interpret separability accordingly.

### hub skew

**What:** present only when a kNN graph was built. The **skewness of the
k-occurrence distribution** — how lopsidedly often points appear in *other* points'
neighbor lists (Radovanović et al. 2010).

**How computed:** `metrics.hubness_skew` counts, for each point, how many times it
occurs across all neighbor lists (`np.bincount` over the flattened kNN index), then
returns the third standardized moment (skewness) of that count distribution.

**How read — DIRECTION:** lower (toward 0) is better.

- **≈ 0** → neighbor relationships are roughly reciprocal and even.
- **large positive** → **hubness**: a few points are the nearest neighbor of
  disproportionately many queries ("hubs") while others are nobody's neighbor
  ("anti-hubs"). Hubness directly degrades kNN retrieval and is a practical
  fingerprint of poor resolution in high dimensions.

---

## 4. The visualizations

Each figure is documented with the same six sub-headings: **What it answers**,
**How it's computed**, **How to read it**, **What to look for**, a **Worked
example**, and **Caveats**.

Two abstract running examples recur, to show how the numbers read *together*:

- **An anisotropic cone:** mean pair cosine ≈ **+0.29**, IsoScore ≈ **0.06**,
  effective rank ≈ **553 / 768**, 90% variance in ≈ **276 / 768** dims, median NN
  margin **small**. Globally crowded, yet high-ish entropy rank with a front-loaded
  spectrum — a real, useful, but cone-shaped space.
- **A collapsed space (contrast):** IsoScore ≈ **0.00**, effective rank ≈ **20**,
  90% variance in ≈ **5** dims. Almost everything lives on a few axes; resolution is
  largely gone.

Notice these two can have *similar* mean pair cosine yet very different spectra —
which is exactly why ambit shows the spectrum several ways.

---

### RES 09 — Crowding curve

*Leads the resolution block. Source: `figures/res_kcurve.py`. Theory:
`docs/concepts/continuous-occupancy.md`.*

#### What it answers

At **what scale** does crowding begin, and how much of it does the anisotropy cone
explain? This is the continuous replacement for every binned occupancy count: the
fraction of random pairs at least as similar as each cosine scale — an exact CDF
(sort + lookup; no bin width, no lattice origin), read against two nulls.

#### How it's computed

`occupancy.exceedance` over the header's pair-cosine sample, on a log scale.
Two references with matched sample size: the **uniform-sphere envelope** (19
replicates of the analytic null — dashed; its pointwise max is the graphical
bound) and the **anisotropy-matched reference** (angular central Gaussian built
from the corpus's own covariance spectrum — the corpus's cone without its
clustering). The footer carries the *occupancy z* and *resolution bandwidth σ\**
scalars (see the facts row).

#### How to read it

- **Data hugging the ACG reference** → the corpus is cone-shaped but not clustered
  beyond it; the usual anisotropy story, no local pathology.
- **Data lifting above the ACG curve** → clustering the cone cannot explain.
- **Data above the uniform envelope at high cosine** (the marked "crowding begins"
  scale) → excess close pairs; in high dimension the null is exactly zero there,
  so *any* mass is a detection.
- The **shaded region** is the excess close-pair mass — the pairs a retrieval
  neighborhood of that size will confuse.

#### What to look for

The **liftoff scale**: near cos 1.0 = near-duplicates; at moderate cosine = topic
clumping; no liftoff = healthy. Compare the gap between the two nulls (cone cost)
with the gap between data and ACG (clustering beyond the cone).

#### Caveats

The curve is a *mean* over pairs — it does not name entities (RES 10 does) or
count pockets (RES 11 does). The envelope is pointwise, a display device; treat
marginal grazing of it as inconclusive.

### RES 10 — Per-entity crowding field

*Source: `figures/res_dtm.py`.*

#### What it answers

**Which entities** are crowded, by name — the per-entity layer the binned figures
could never provide (a cell count can't say who is in the cell).

#### How it's computed

Each entity's **distance to a measure** (`crowding.dtm`): the root-mean-square
chord distance to its nearest 2% of the reservoir — "the radius of the ball this
entity needs to gather a fixed share of the corpus." The figure draws the exact
CDF of the field, with the **uniform reference band** (1st–99th percentile of the
same field for a uniform corpus of matched size and dimension) shaded, the most
crowded entities listed by id in the left panel, and the most isolated (voids) below
them. DTM is provably stable: a small perturbation of the corpus moves every score
by a provably small amount (Wasserstein-Lipschitz) — the guarantee bin counts lack.
Above ~6,000 reservoir points the field runs on a seeded subsample.

#### How to read it

- **Curve inside the band** → entities spread as a uniform corpus would.
- **Low-tail spike escaping left of the band** → crowded entities, individually
  named and ranked; these are the retrieval collisions in waiting.
- **Right tail far beyond the band** → isolated entities / coverage voids (the
  continuous replacement for the void-grid figure).

#### Caveats

The 2% mass fraction is the one knob: pockets much smaller than 2% of the
reservoir blur (lower it to hunt smaller clumps). Scores are radii in chord
units — compare within a report or between reports of the same dimension.

### RES 11 — Crowding pockets

*Source: `figures/res_pockets.py`.*

#### What it answers

**How many** over-tight pockets exist, how tight each is, which entities belong to
each, and at what scale each detaches from the bulk — with no flat clustering
threshold chosen anywhere.

#### How it's computed

The merge tree of the reservoir (`crowding.pockets`): single linkage on the
mutual-reachability metric — the construction inside the default clustering
backend, **surfaced instead of flattened** — condensed with a minimum pocket size
of 8 and selected by excess-of-mass stability. Each bar spans a pocket's **birth**
(the scale at which its first members hold together) to its **death** (the scale
at which it merges into the bulk); bar length is its **prominence**. Sample member
ids are printed at each bar.

#### How to read it

- **Long bars born near zero** → near-duplicate pockets: entities that are almost
  interchangeable among themselves yet far from everything else — the crowding
  that hurts retrieval most. The listed ids make the finding actionable.
- **Short bars born late** → loose associations barely distinct from the bulk;
  usually benign.
- **"No prominent tight pockets"** → the corpus merges as one bulk; whatever
  crowding exists is diffuse (read RES 09/10 for it).

#### Caveats

Runs on a ≤4,096-point seeded subsample of the reservoir, so pocket **sizes are
shares of the sample**, not absolute corpus counts (scale by corpus/sample). The
minimum pocket size of 8 is a floor on what can be *reported*, not a density
threshold — smaller duplicate groups appear in RES 10's low tail instead.

### RES 01 — Random-pair cosine distribution

*Order: sixth card (after RES 05, before RES 02b). Source: `fig_cos_hist` in
`render.py`.*

#### What it answers

How crowded is the space *globally*? Where does the cloud of "any two unrelated
items" land on the cosine axis, relative to a perfectly isotropic space? This is
the visual form of the **mean pair cosine** header fact — the whole distribution,
not just its mean.

#### How it's computed

Take `ctx.cos` — the sample of `n` random-pair cosines (default ~200k pairs,
i≠j, on unit-normalized vectors). Render its **smooth density** over the full
**[−1, +1]** cosine axis. The smoothing is a numpy-only **KDE**: a fine 400-bin
histogram (`density=True`) convolved with a Gaussian kernel whose bandwidth follows
Scott's rule (`bw = sd · n^(−1/5)`), floored to stay visually smooth. The accent
curve is that density.

Three reference elements are drawn alongside:

- The **isotropic d-sphere reference** — an analytic Gaussian spike at cosine 0,
  `N(0, 1/√dim)`, where `1/√dim` is `metrics.isotropy_ref(dim)`. This is exactly
  what random-pair cosine would look like if the data filled the sphere
  uniformly. It is drawn as a tall dashed razor spike at 0.
- The **cos = 0 axis** — a faint dashed vertical rule up through the spike.
- The **anisotropy-gap wedge** — the shaded area under the dataset curve between
  cosine 0 and the dataset mean. This wedge *is* the anisotropy: the displacement of
  the dataset's mass away from the isotropic reference.

An accent **mean tick** with a dot marks the dataset mean on the curve, annotated
with `mean cos`, `sd`, and the right-tail (99th percentile) value. A verdict in the
top-right reads `near-isotropic` / `mild anisotropy` / `anisotropic cone` based on
where the mean falls relative to the reference spread.

#### How to read it

- **x-axis:** cosine similarity, −1 to +1, gridlines every 0.2. **0 is dead center
  and is "isotropic."**
- **The accent hump** is the dataset. **Where its mass sits is the verdict — and
  the DIRECTION of good is leftward, toward 0.** Mass centered on 0 (overlapping the
  reference spike) = isotropic and healthy. Mass shifted toward +1 = a crowded cone.
- **The dashed spike at 0** is the isotropic ideal. The further the accent hump sits
  to its right, the worse.
- **The shaded wedge** between 0 and the mean tick is the **anisotropy gap** — a
  wider, taller wedge means more crowding.
- A light tint fills the whole area under the dataset curve as a crowding cue; the
  wedge is a darker tint of the same.

#### What to look for

- **Isotropic / healthy reading:** the accent hump is narrow and centered on 0,
  nearly coincident with the dashed reference spike; the wedge is almost nonexistent;
  the verdict reads *near-isotropic*. Unrelated pairs are orthogonal, so a genuinely
  related pair will stand out.
- **Crowded / cone reading:** the accent hump is shifted bodily to the right
  (mean at, say, +0.3), often broader, with a right tail reaching toward +0.6 or
  beyond; the wedge is a large shaded slab; the verdict reads *anisotropic cone*.
  Every random pair already scores high, so the headroom to distinguish related
  items is small.

#### Worked example

For the anisotropic cone: the accent hump centers at **mean cos ≈ +0.29**, well to
the right of the dashed `N(0, 1/√768)` reference spike (whose sd ≈ 0.036). The
anisotropy-gap wedge spans the whole stretch from 0 to +0.29 — a large shaded slab.
The verdict reads **anisotropic cone**. Interpretation: pick two random items and they
already agree at cosine ≈ 0.29; a *truly* related item has to beat that floor, so
retrieval is operating in a compressed band. This is the global picture; RES 04 will
show whether *local* neighborhoods still resolve.

For a near-isotropic space, the hump would sit on top of the spike at 0 and the
wedge would nearly disappear.

#### Caveats

- This is a **global** statistic. A space can be globally a cone yet locally
  isotropic *within* clusters (Cai et al. 2021) — "crowded globally, useful
  locally." RES 04 and RES 05 are the local counterweights; read them together.
- High cosine here means **crowded/bad**, the opposite of the usual "high similarity
  = good" intuition for a single retrieval. Don't misread the direction.
- Absolute cosine values are not meaningful on their own; that's why the figure
  always draws the isotropic reference for you to read against rather than printing
  a pass/fail threshold (Steck et al. 2024).
- A few high-magnitude "rogue dimensions" can dominate the cosine and shift the
  whole hump right (Timkey & van Schijndel 2021). The figure shows the symptom, not
  the cause; standardizing those axes before re-measuring often restores resolution.

---

### RES 02 — Covariance eigenvalue scree

*Order: eighth card — it renders **after** RES 02b in `_DISPLAY_ORDER` (the
cumulative curve leads, the scree follows). Source: `fig_scree` in `render.py`.*

#### What it answers

How is variance distributed across the principal axes? Does it decay gently (broad
use of the space) or fall off a cliff after a few axes (collapse)?

#### How it's computed

From the covariance eigenvalues `ctx.eigs` (descending, non-negative; computed over
**all** items via the d×d scatter, not just the reservoir). Take the top `k` (≤ 80),
normalize by the largest eigenvalue, and plot them on a **log y-axis** (1e0 down to
1e−6) against eigenvalue index. An accent vertical rule marks the **effective rank**
(`metrics.effective_rank(ctx.eigs)`, the `exp(entropy)` measure described above).

#### How to read it

- **x-axis:** eigenvalue index (0 = largest-variance axis), ticked every 10.
- **y-axis:** normalized eigenvalue on a **log scale** — so a straight-looking
  descent is actually exponential decay, and small differences low down are real.
- **The ink curve** is the spectrum.
- **The accent vertical rule** is the effective rank, labeled. Think of it as "how
  far to the right the meaningful variance extends."
- **DIRECTION of good:** a **gentle, slowly-decaying** curve and an accent rule far
  to the right = healthy. A **steep early drop** and an accent rule hugging the left
  = collapse.

#### What to look for

- **Healthy reading:** the curve slopes down gradually; even at index 50–80 the
  eigenvalues are within a couple of orders of magnitude of the top; the effective
  rank rule sits well to the right.
- **Collapsed reading:** the curve plunges several decades within the first handful
  of indices, then flattens into a noise floor; the effective rank rule sits at the
  far left. A few axes carry essentially all the variance.

#### Worked example

For the anisotropic cone, the spectrum is **front-loaded** — the first several axes
carry a visibly larger share than the rest — yet the tail does not crater to the noise
floor; it decays gradually enough that the entropy-based effective rank rule lands
far right at **≈ 553 of 768**. Reading: a few dominant directions (the cone's
backbone) plus a broad, still-populated tail. Contrast the collapsed space, whose
curve would fall off a cliff in the first ~5 indices and whose effective-rank rule
would sit near the left edge at **≈ 20**.

#### Caveats

- Only the top 80 eigenvalues are drawn; the long low-variance tail past index 80
  isn't shown here (RES 02b integrates the *whole* spectrum and is the figure to use
  when the tail matters).
- The y-axis is log and normalized to the top eigenvalue, so the figure shows
  *shape*, not absolute variance.
- Effective rank can be inflated by a long thin tail; see RES 02b for why the single
  number can mislead and what the cumulative curve shows instead.

---

### RES 02b — Cumulative variance & dimensional concentration (carries the IsoScore)

*Order: seventh card — it renders **before** RES 02 (the cumulative curve leads, the
scree follows). Source: `figures/res_cumvar.py`.*

#### What it answers

How many dimensions *actually carry* the variance? This is the **integrated**
companion to the scree: it reconciles the three single-number summaries (effective
rank, participation ratio, dims-for-90%) that can disagree wildly on a heavy-tailed
spectrum, by showing the full cumulative curve. It also **carries the IsoScore**
readout (with the formula hovercard).

#### How it's computed

From the positive covariance eigenvalues, sorted descending. Compute the
**cumulative fraction of variance** `cum = cumsum(eigs)/sum(eigs)` and plot it
against dimension index (1…dim). A rank-deficient tail is padded to 1.0. Drawn
against the **isotropic diagonal** — the straight line from (0, 0) to (dim, 1),
which is what a full-rank, equal-variance (isotropic) space would produce. The
shaded **concentration gap** is the area between the dataset curve and that
diagonal.

Marked on it:

- **50% and 90% variance crosshairs** — the dimension at which the curve crosses
  half and nine-tenths of the variance (`dims_for_variance` at 0.5 and 0.9), each
  with a dotted crosshair and a `…% · N dims` label. These are the *operational*
  dimensionality.
- **Effective-rank rule** (caution-toned, dashed vertical) and a
  **participation-ratio** annotation beside it — so you can see *where on the curve*
  each summary lands and why they differ. Participation ratio is
  `(Σλ)² / Σλ²` (`metrics.participation_ratio`), a variance-weighted count that, unlike
  effective rank, is *not* inflated by a long thin tail.
- A verdict (top-right) reading `well spread` / `moderately concentrated` /
  `concentrated`, based on what fraction of `dim` holds 90%.
- The **IsoScore** readout in the lower-right corner — a dotted-underline
  hovercard. Hover or focus it: a card unfolds with the full four-line formula
  (shown in §3 above) and this dataset's intermediate terms (d, δ, n). The IsoScore
  is, in effect, this very gap-to-the-diagonal expressed as a single 0–1 scalar.

#### How to read it

- **x-axis:** dimensions, sorted by variance, 0 to `dim`.
- **y-axis:** cumulative fraction of variance, 0% to 100%.
- **The dashed diagonal** is the isotropic ideal (full utilization). **The accent
  curve** is the dataset.
- **DIRECTION of good:** a curve that **hugs the diagonal** is well-spread and
  healthy; a curve that **bows up sharply** toward the top-left, leaving a large
  shaded gap, is concentrated. The bow *is* the concentration.
- The **50%/90% crosshairs** read off directly: "half the variance is in the first
  N dims, 90% in the first M."
- The **effective-rank rule** and **participation-ratio** number let you see the two
  scalars against the actual curve.

#### What to look for

- **Healthy reading:** the curve tracks close to the diagonal; the shaded gap is
  thin; 90% of variance takes a large fraction of `dim`; effective rank and
  participation ratio are close together and both large; verdict *well spread*.
- **Concentrated reading:** the curve jumps to near-100% within a small fraction of
  the axes, then crawls flat; the shaded gap is a big lens above the diagonal; 90%
  lands in a tiny number of dims; verdict *concentrated*.
- **The diagnostic the curve is *for*:** when effective rank says "lots of
  dimensions" but participation ratio says "few," the curve shows why — a few
  dominant axes (steep early rise) plus a long low-variance tail (the slow crawl to
  100%). The tail inflates the entropy rank without carrying real variance.

#### Worked example

For the anisotropic cone: **50% of variance by some early dimension, 90% by ≈ 276 of
768.** The curve rises faster than the diagonal but doesn't slam into the ceiling —
it's a moderate upward bow, leaving a real but not enormous gap. The verdict is in
the *moderately concentrated* range (≈ 276/768 ≈ 36% of dims hold 90%). The
effective-rank rule lands far right at **≈ 553**, but the participation ratio is
**lower** — and the curve explains the discrepancy: a front-loaded head plus a long,
still-populated tail. The **IsoScore hovercard reads ≈ 0.06**, the small scalar
restatement of that gap. So: not a healthy isotropic space, but not a collapsed one
either — a broad cone.

For the collapsed contrast: the curve would shoot to ~100% within the first ≈ 5
dims and flatline, the gap would be a huge lens, the verdict *concentrated*, and the
IsoScore ≈ **0.00**.

#### Caveats

- The cumulative curve is monotonic and always ends at 100% — that's not
  information; the *shape on the way up* is.
- IsoScore, effective rank, participation ratio, and dims-for-90% are four views of
  the same spectrum; expect them to *rhyme*, and use the curve to adjudicate when
  they don't. Don't quote one in isolation.
- Like all the spectral figures, this is **variance**, not semantics. A space can
  spread variance broadly and still be poorly aligned; spectral spread is necessary
  for resolution, not sufficient for usefulness.

---

### RES 08 — Uniformity on the hypersphere

*Order: renders after RES 02b. Source: `figures/res_uniformity.py`. The `uniformity`
header fact is always shown; the figure is on by default in `--compare` / series mode.*

#### What it answers

How evenly the data spreads over the unit sphere — the **uniformity** half of the
alignment/uniformity decomposition (Wang & Isola 2020). Alignment (the other half) needs
positive pairs — a supervised signal — so it is outside ambit's unsupervised scope and is
not drawn.

#### How it's computed

`U = log E exp(−2·‖x−y‖²)` over random pairs, from the same cosines as the header *mean
pair cosine* (`metrics.uniformity_from_cos`). The figure places the dataset's U on a number
line against the **isotropic-sphere reference** (`metrics.uniformity_ref(dim)`), the gap
shaded. In `--compare` mode it draws the **trajectory** from A to B (the same items embedded
two ways) along the axis.

#### How to read it

- **More negative = more uniform = better** occupancy of the whole sphere. The axis puts
  *more uniform* to the right; the isotropic reference is the rightmost (ideal) mark.
- **The gap** to the reference is how far the data sits from a perfectly even spread — the
  formal, pair-potential version of RES 01's cone reading.
- **Trajectory (compare mode):** a rightward step from A to B is the second embedding
  spreading the same items more uniformly. Uniformity improving while *local* structure
  does not is the classic trap — read it alongside the separability panel and the
  neighbor-overlap drift.

#### Caveats

- Uniformity is **necessary, not sufficient**: a perfectly uniform space can still put the
  wrong things together. It is the occupancy half of the story, not the whole — which is
  exactly why ambit does not claim the alignment half it cannot measure unsupervised.

---

### RES 04 — Nearest-neighbor cosine margin

*Order: fourth card (after RES 06, before RES 05). Source: `figures/res_margin.py`.*

#### What it answers

At the **retrieval boundary**, how decisively does each item's *best* neighbor beat
its *runner-up*? This is the **local** resolution counterpart to RES 01's global
view: even in a globally crowded cone, are individual neighborhoods still resolvable?

#### How it's computed

Needs a kNN backend (`ctx.knn_dist`; degrades to a placeholder if absent). For each
item, convert neighbor distances to cosine similarities (`sim = 1 − dist`), sort
descending, and take the **margin = cos(top-1) − cos(top-2)** (clipped at 0). That's
one number per item: how much the nearest neighbor wins by. Histogram those margins
(40 bins, axis from 0 to a "nice" upper bound covering the bulk).

Two reference markers:

- **Dataset median margin** — an accent vertical rule with a dot, labeled.
- **Isotropic reference median** — a dashed faint rule. Computed by simulating an
  isotropic cloud: draw random unit vectors in `dim` dimensions, compute each query's
  cosines to a small random panel, and take the median top-1−top-2 gap. This is the
  "no-structure" yardstick: the margin you'd get from pure isotropic geometry.

Bars are split at the median: bins **at or below** the median are tinted with the
**bad** token (near-tie crowding), bins **above** with the **good** token
(decisively resolved). Directional captions label the left region "near-tie
crowding / resolution floor (margin → 0)" and the right tail "decisively resolved …
high resolution."

#### How to read it

- **x-axis:** margin = cos(top-1) − cos(top-2), from 0 (a perfect tie) to the upper
  bound. **DIRECTION of good is rightward — larger margin = more decisively
  resolved.**
- **y-axis:** number of items per margin bin.
- **Left edge (margin → 0):** near-ties; the index can barely tell the best match
  from the runner-up. **Bad** tint.
- **Right tail:** items whose nearest neighbor is distinctly separated. **Good**
  tint.
- **Accent rule = dataset median.** Its x-position is the headline: a median pinned
  near the left edge means *most* items are near-ties.
- **Dashed rule = isotropic reference median.** Compare the accent to it: a dataset
  median *below* the isotropic reference is worse than no-structure geometry would
  give.

The margin is, precisely, **within-neighborhood resolution**: how well the geometry
separates the items that are *already nearest* — whether or not they are related.
That distinction matters. The obvious failure mode is unrelated impostors crowding
the top of a search: a small margin there means you can't rank them out and
precision collapses. But a small margin among *genuinely related* neighbors is a
failure too — a relevant cluster packed too tightly loses its *internal* resolution,
so you can no longer tell which of several relevant items is the best match,
separate near-duplicates from distinct-but-related neighbors, or rank a cluster's
own members. Many tasks need exactly that within-set resolution (fine-grained
retrieval, dedup, re-ranking, recommendation diversity). The margin reads both the
same way, because crowding costs resolution either way.

#### What to look for

- **Healthy reading:** mass spread out into the right tail; the accent median sits
  comfortably away from the left edge and at or above the isotropic reference;
  retrieval is decisive — nearest neighbors are decisively separated, so both the
  best-vs-impostor and the best-vs-runner-up-relative judgments are well resolved.
- **Crowded reading:** a tall spike piled against the left edge (margin ≈ 0), a thin
  right tail, the accent median pinned near 0 — top matches are near-ties and
  retrieval is brittle. The two reference rules may even collapse onto the left edge
  together when the corpus is heavily floored. A floor like this hurts equally
  whether the near-ties are impostors (precision gone) or members of one tight
  relevant cluster (within-cluster ranking gone).

#### Worked example

For the anisotropic cone, the median NN margin is **small** — the histogram piles up
toward the resolution floor on the left, with the accent median rule sitting close
to 0, and only a thin right tail of items that have a clearly-separated nearest
neighbor. Reading: despite the corpus being usable in aggregate, *at the retrieval
boundary* the top-1 often barely beats the top-2, so a kNN retriever's ranking is
fragile and thresholds are unreliable. This is the local symptom that corroborates
the global cone from RES 01 — and it's exactly the regime where hubness (see the
hub-skew fact) tends to bite.

#### Caveats

- This is computed over the reservoir's kNN, so it reflects **reservoir sampling**:
  the neighbors are within the scanned sample, not necessarily the global nearest
  neighbors of the full corpus. The *shape* is robust; exact margins shift with
  sample size.
- The isotropic reference median is itself a small simulation (a random panel), so
  treat it as an approximate yardstick, not an exact bound.
- A small margin is a property of the *local geometry*; it does not by itself tell
  you whether the top-1 neighbor is the *correct* one — only that it's not decisively
  separated from the next candidate.

---

### RES 05 — Within- vs between-cluster cosine

*Order: fifth card (after RES 04, before RES 01). Source: `figures/res_wb.py`.*

#### What it answers

Does cosine geometry actually **separate the clusters**? Are same-cluster pairs
reliably more similar than different-cluster pairs — and by how much?

#### How it's computed

Needs cluster labels (degrades to a placeholder cloud if fewer than two clusters).
On the unit-normalized reservoir, sample ~60,000 random pairs (i≠j), compute each
pair's cosine, and split by whether the two items share a label:

- **within-cluster** = same-label pairs,
- **between-cluster** = different-label pairs.

Each set is capped to ~30,000 for stability, then turned into a smooth density (a
tiny Gaussian KDE) over the shared **[−1, +1]** cosine axis. Both curves are drawn
on the same axis. The **overlap region** (the pointwise minimum of the two
densities) is shaded as the **confusable zone**. The **separation = mean(within) −
mean(between)** is drawn as an accent caliper between the two peaks. A dashed
reference marks the **ideal isotropic between-peak at cosine 0** — where the
between-cluster density *should* sit if unrelated items were orthogonal.

#### How to read it

- **x-axis:** cosine similarity, −1 to +1, with a fine 0.1-step ruler.
- **Two density humps:** the **within-cluster** curve (rendered with the `good`
  token) and the **between-cluster** curve (rendered with a `bad`-leaning ink mix).
  Each is labeled with its peak cosine and sd.
- **DIRECTION of good:** the within-cluster hump should sit **to the right of** the
  between-cluster hump, and the between hump should sit **near 0** (the dashed
  isotropic reference). A clean rightward gap = separable.
- **The accent caliper** at the top measures the **separation** between the two
  peaks, with a signed number. Positive = within sits higher (good).
- **The shaded confusable zone** where the two curves overlap = the cosine range in
  which same-cluster and different-cluster pairs are *unresolvable*. Smaller is
  better.

#### What to look for

- **Separable reading:** a within-cluster hump clearly to the right of a
  between-cluster hump that sits near 0; a large positive separation caliper; a thin
  confusable overlap. Same-cluster items are reliably more similar than cross-cluster
  ones — cosine resolves the partition.
- **Entangled reading:** the two humps sit nearly on top of each other (often both
  shifted right of 0, the cone again); a small or negative separation; a fat
  confusable zone. Cosine alone does not resolve the clusters.

#### Worked example

For a healthy run with provided labels, you'd see a between-cluster hump near
cosine 0 and a within-cluster hump out around, say, +0.5, with a separation caliper
reading `separation = +0.5` and a slim confusable zone — clusters cleanly separable.

For an anisotropic cone, expect both humps shifted **right of 0** (the global cone
pulls *everything* positive), with the between-cluster peak sitting above the ideal 0
reference rather than on it. If clusters are genuine, the within hump still leads the
between hump (positive separation) but the confusable zone is wider than in an
isotropic space — same-cluster and cross-cluster pairs overlap more, so cosine
resolves the partition only partially. If the separation came out near zero, the
"clusters" aren't expressed in cosine geometry at all.

#### Caveats

- Entirely dependent on the **label source**. If `groups` reads `clustered`, the
  partition is ambit's own guess; "separable" then means "ambit's clusters are
  self-consistent," not "your ground-truth categories separate."
- Built on cosine, so it inherits cosine's caveats — a global cone shifts both humps
  right; what matters is their *relative* position, not their absolute cosine.
- The densities are sampled pairs, not exhaustive; the peaks are stable but the tails
  are estimates.

---

### RES 05b — Separability panel

*Order: renders right after RES 05. Source: `figures/res_separability.py`, reading
`separability.py`. On by default whenever a partition exists.*

#### What it answers

Do the categories you care about occupy **distinct regions** of the space? This panel
reports the per-label geometry — the centroid-cosine matrix and kNN purity — that fixes
the separability of a partition. The partition is your `--label-col` if present,
otherwise ambit's own clusters.

#### How it's computed

On the reservoir, for the partition `y`:

- **Centroid-cosine matrix** — `cos(μ_a, μ_b)` between per-group mean unit vectors; the
  off-diagonal cells show which groups collapse together (share a direction).
- **kNN purity** — per item, the fraction of its neighbors that share its label (reads
  `ctx.knn_idx`), drawn as per-group bars against an overall rule.
- **Scorecard** — cosine **silhouette**, **Fisher ratio** (between/within scatter), and
  overall purity.
- **Unsupervised extras** (discovered clusters only) — **stability** (mean adjusted-Rand
  index across bootstrap re-clusterings) and **n-modes** (the graph-Laplacian eigengap),
  with a badge that the read is *geometric, not semantic*.

#### How to read it

- **Heatmap:** deeper accent off-diagonal = two groups entangled (bad); light = separated.
  The diagonal is blanked.
- **Bars:** longer = higher purity (good); compare each to the overall rule.
- **Scorecard direction:** higher silhouette / Fisher / purity = more separable.
- **Unsupervised badge:** trust the heatmap only when **stability** is high — a low ARI
  means the clusters are an artifact of one run, so the separability is not meaningful.

#### Caveats

- On **clustered** (not provided) labels this is the separability of ambit's *own* guess at
  structure — geometric, not your ground-truth categories. Stability and the eigengap exist
  precisely to tell you whether that guess is worth interpreting.
- Reservoir-based; exact values reflect the sample, the picture is robust.

---

### RES 06 — Local density field

*Order: renders mid-stack, after RES 04/RES 05 (third card in the default report).
Source: `figures/res_field.py`, reading `local_anisotropy.py`.*

#### What it answers

Is the crowding **global** (the whole space is dense) or **localized** (a few dense
*pockets* in an otherwise roomy space)? RES 01 measured global anisotropy with a
single mean; this figure refuses to collapse the local picture to one number. It
shows the **distribution** of every item's local density, so you read the verdict
off the *shape* — and that shape is the localized-anisotropy reading of the corpus.

#### How it's computed

All the numbers come from `local_anisotropy.localized_anisotropy` (memoized on the
Ctx via `for_ctx`, shared with RES 07); this figure only draws them.

For each reservoir item, take the **mean cosine to its k nearest neighbors** — a
k-NN density estimate of how concentrated its neighborhood is — computed at several
**scales** (k = 10, 50, 200 by default). The **headline scale** `k*` is the one
whose field has the most extended upper tail (in MAD units): the scale that
separates pockets most cleanly. The figure histograms the field at `k*`.

Drawn against it is a **density-matched uniform reference** (the *null*): a synthetic
isotropic cloud of unit vectors sampled at the same effective density, with k scaled
to hold k/N fixed. It is rendered as a faint dashed ghost, peak-scaled to its own
height. Two medians are marked — the **dataset bulk** (accent rule) and the
**uniform-reference median** (dashed faint rule) — and a bracket between them
labels the **global-crowding gap** as a multiple of the reference's own spread
(`(bulk − iso_median)/iso_MAD`). Any **reference-calibrated pockets** get dashed
`bad`-toned flags above their bumps, labeled with their size.

Detection is **reference-tail-calibrated**, not bulk-relative: per scale, an item is
flagged crowded when its robust z exceeds the *reference's own* upper-tail threshold
(a small per-scale false-positive budget, unioned across scales), then angular
clustering plus a minimum pocket size reject scattered false positives. The cutoff
comes from "denser than the uniform reference ever gets," **not** from a fraction of
the dataset's own bulk-peak height — so a small but cleanly separated pocket is not
swamped by a tall bulk, and a uniformly crowded "dandelion" yields *no* pockets
because its z-tail matches the reference's.

#### How to read it

- **x-axis:** local density = mean cosine to the `k*` nearest, low (roomy) on the
  left to high (dense) on the right.
- **y-axis:** items per bin.
- **The accent rule** is the dataset bulk; **the dashed ghost** is the uniform null.
- **DIRECTION:** the *shape* is the verdict, not a single direction —
  - **one mode sitting near the reference** → **roomy / uniform**;
  - **the whole mode shifted far right of the reference** → **globally denser than
    uniform** (the gap bracket is wide);
  - **a separated high mode** above the bulk → a **dense pocket** (flagged).
- **The gap bracket** between the two medians is the global-crowding story: how far
  the *typical* neighborhood sits above uniform.

#### What to look for

- **Roomy reading:** a single mode hugging the uniform ghost; a small gap bracket;
  no pocket flags. Neighborhoods are about as concentrated as a uniform cloud's.
- **Globally dense reading:** the whole histogram bodily shifted right of the ghost,
  a wide gap bracket, but still *one* mode and no separated pockets — every
  neighborhood is crowded, evenly. This is the local-field echo of RES 01's cone.
- **Pocketed reading:** a main bulk near or modestly above the reference *plus* one
  or more separated high modes, each flagged as a pocket — a roomy space with a few
  crammed sub-populations the global mean would hide.

#### Worked example

For the anisotropic cone, the bulk sits well to the right of the uniform ghost — a
wide gap bracket and a `globally denser than uniform` tag — consistent with the global
anisotropy RES 01 reads. If a sub-population were crammed far tighter than the rest,
you'd instead see the bulk near the ghost with a separated high mode flagged
`pocket · n=…`, and the report would say the crowding is *local*, not global. The
distinction is invisible to a single mean-cosine scalar; it is the whole reason this
figure draws the distribution.

#### Caveats

- **Magnitude is a RANK, not a calibrated density.** The cosine→density map is
  strongly nonlinear in high d, and k-NN density estimation in ~768-d suffers
  distance concentration, hubness, and fixed-k boundary bias. Read the field as an
  ordering of neighborhoods, never as an absolute density. The honest signals are
  the *shape* and the *gap to the reference*, both of which are reference-relative.
- Pocket detection is **unsupervised** — it counts whatever sits in each
  neighborhood (see the relatedness note in §5 and RES 04); a flagged pocket is a
  geometrically dense sub-population, related or not.
- Computed over the reservoir, so exact pocket counts reflect the sample; the
  shape is robust.

---

### RES 07 — Local crowding cloud

*Order: renders **first** in the default report. Source: `figures/res_cmap.py`,
reading `local_anisotropy.py`. Interactive (drag · zoom · #-samples slider · kNN
edges).*

#### What it answers

**Where** does the local density sit — is the crowding spatially coherent (one
region of dense pockets) or an even wash across the cloud? This is the *spatial
companion* to RES 06: RES 06 says how much and what kind of crowding; RES 07 says
where it lives. It is the report's one rotatable 3-D view, carrying the local-density
signal in its color and shape.

#### How it's computed

The projected reservoir (`ctx.xyz`, the top-3 PCA axes) is baked into a vanilla-JS
canvas (no dependencies; reads CSS tokens so it re-skins on theme swap). Up to ~8,000
points are kept, centered and scaled into a unit box, depth-sorted (painter's order)
and depth-cued. Each point is **recolored and reshaped** by its local-density score
(the signed robust z of its neighborhood concentration at the headline scale, from
the same `local_anisotropy.for_ctx` result RES 06 reads):

- **color** runs **green → amber → red** on a crowding rank (so adjacent levels stay
  distinct under any theme): the *least*-dense items are flat green dots, the
  *most*-dense are red;
- **shape** reinforces it: open/typical items are **flat dots**; crowded items rise
  into little **pyramids**, taller and brighter the more crowded — readable from any
  angle. Items in a detected pocket get a ringed pyramid.

An optional **kNN-edge overlay** (precomputed over a ~1,500-point prefix, k≈6 by
cosine) wires each visible point to its native nearest neighbors when toggled on:
edges between two crowded points are tinted, the rest neutral. A **#-samples
slider** thins the field and a lower-left **x/y/z gnomon** tracks orientation.

#### How to read it

- **Drag to rotate, scroll/pinch to zoom.** It auto-spins when idle.
- **Read crowding from COLOR and SHAPE, never from screen position.** This is the
  load-bearing instruction: the position is a PCA projection of the *global*
  variance, and only a few percent of each item's true 768-d neighbors survive it —
  so dense and open points look equally clumped on screen. Color and pyramid height
  carry the local density; the layout does not.
- **What the layout *does* show:** whether the red pyramids **cluster in one region**
  (a localized pocket) or **spread evenly** (global density). That coarse
  arrangement is trustworthy; fine position is not.
- **kNN-edges toggle:** the edges link true 768-d neighbors, so they **fan across**
  the projection rather than staying local — that scatter is the projection limit
  made visible. Tinted edges mark crowded-to-crowded links.
- **#-samples slider:** thin the field when it's too dense to read.

#### What to look for

- **Roomy reading:** mostly flat green dots, few or no pyramids — little local
  crowding structure.
- **Pocketed reading:** a knot of tall red (ringed) pyramids concentrated in one
  region of the cloud — a spatially coherent dense pocket, the same pocket RES 06
  flagged, now located.
- **Globally crowded reading:** pyramids spread evenly across the whole cloud rather
  than bunched — the crowding is global, the RES 06 "dandelion" seen in space.

#### Worked example

Spin an anisotropic cone and you'll see crowding spread broadly rather than knotted in
one place — pyramids scattered across the cloud, matching RES 06's `globally denser than
uniform` read. If instead one sub-population were crammed, you'd find a localized
cluster of tall red pyramids; toggling kNN edges, the tinted crowded-to-crowded links
would concentrate there. Either way, resist reading density off how tightly the dots
*appear* to sit — the green dots are the least-crowded items **relative to this
dataset**, not necessarily roomy in absolute terms; RES 06 carries the absolute
level.

#### Caveats

- **Position is a PCA shadow and does not encode crowding** — the single most
  important caveat, repeated on the card itself. The top-3 axes capture global
  variance, not local 768-d structure; read color and shape.
- Color is **relative rank**: in a globally dense space the green dots are still
  cramped, just less than the median. Pair it with RES 06 for the absolute level.
- Capped to ~8,000 points for the cloud and ~1,500 for the edge graph — a
  representative sketch, not the whole corpus. The depth cue is perceptual, not
  metric.

---

### DEN 04 — Density-peak prominence

*Order: tenth card (after 3D 05, before COV 09). Source: `figures/den_prom.py`.*

#### What it answers

Where does the dataset *pile up* in the projected space, and how many **genuine
density modes** does it have (versus sampling noise)? This is a DEN-family
(concentration) view.

#### How it's computed

A two-panel figure built from the 2-D projection (`ctx.xy`).

**Left panel — topographic relief.** Bin the projected reservoir into an 84×60
density grid, smooth it with three light box-blur passes, and draw it as a
**topographic map**: nested **isodensity contour lines** at six density quantiles,
extracted by a **marching-squares** algorithm and stitched into closed polylines.
Contours ramp from low-density (toward accent) to high-density (toward the bad
token) so the densest cores read hottest. Faint reservoir dots underlie it.

On top, **grid local maxima** (8-neighbor peaks above the mean occupancy) are found
and ranked by **topographic prominence** — each peak's height minus the highest
saddle on a straight-line path to any taller peak, relative to a noise floor. The
result `ρ` is expressed as a multiple of the noise-floor density (`×`). The single
**dominant peak (P1)** carries the accent (double-ring + dot); up to two more real
peaks get neutral markers; sub-cutoff bumps are faint hollow rings.

**Right panel — prominence ranking.** A horizontal bar per peak, length = `ρ` in
`×noise-floor` units, against a dashed **cutoff at 2.0×**. P1's bar is the accent;
peaks above cutoff are ink bars; bumps below cutoff are faint dashed "bump" rows.

#### How to read it

- **Left:** an elevation map. **Closed nested contours** = density "hills"; tightly
  nested rings = a sharp peak; broad spacing = a gentle rise. The **color ramp** runs
  low→high density (legend strip at the bottom).
- **Peak dots** sit at local maxima, **sized by prominence**. The **accent
  double-ring (P1)** is the dominant mode.
- **Right:** the same peaks ranked by prominence. Bars past the **dashed cutoff** are
  real modes; bars short of it are noise bumps. The accent P1 bar shows how dominant
  the top peak is (`ρ ×`).
- **DIRECTION:** more, well-separated, high-prominence peaks = genuinely multi-modal
  structure; a single towering peak with everything else below cutoff = one
  megacluster dominating.

#### What to look for

- **Multi-modal / structured reading:** several peaks clear the 2.0× cutoff, each
  with its own nested contour island — the data has distinct cores.
- **Single-mode reading:** one accent peak towers (high `ρ`), everything else is
  sub-cutoff bumps — one dense blob with diffuse surroundings.
- **Flat reading:** no peak clears the noise floor at all — the figure says "no
  resolvable density peaks"; the projected field is featureless.

#### Worked example

The aria text and `why` are generated for the run, e.g. *"N hotspots clear the
prominence cutoff · P1 dominates (ρ×)."* For a corpus with one giant cluster and a
few satellites, you'd see a tall accent P1 ring over the densest contour island,
two neutral secondary peaks, and a scatter of sub-cutoff bumps — and on the right, a
long accent bar plus a couple of bars past the 2.0× line. Reading: one dominant mode
with a little genuine secondary structure.

#### Caveats

- This is computed on the **2-D projection**, so it inherits projection distortion:
  PCA flattens the cloud onto two axes, and density peaks in the projection are not
  guaranteed to be peaks in the native space (and vice versa — distinct native
  clusters can overlap in 2-D). Read it with the 3-D figures, which recover a
  dimension.
- Prominence uses a **straight-line saddle proxy**, not a true watershed; it's a fast
  approximation, robust for ranking but not a topology-exact measure.
- The noise cutoff (2.0×) is a heuristic; bumps just under it aren't necessarily
  noise, just below the figure's confidence line.

---

### COV 09 — Nearest-neighbor sparsity field

*Order: eleventh and **last** card. Source: `figures/cov_sparsity.py`. Interactive
(#-samples slider).*

#### What it answers

Where is the embedding space **open** (well-separated points, good resolution)
versus **packed** (crowded points)? This is a COV-family (coverage) view, and the
one place in the report where **larger is unambiguously good**.

#### How it's computed

Needs a kNN backend (`ctx.knn_dist`; degrades otherwise). For each reservoir point,
take its **1-NN distance** `d0` (distance to its single nearest neighbor). Plot every
point at its 2-D projected position as a **ring whose radius scales with that 1-NN
distance** (clipped to a max radius; scaled so the field is legible). A dashed
reference circle at the center shows the **median spacing**.

The **most-isolated decile** (1-NN distance ≥ the 90th percentile = the points
sitting in the most open space) is drawn as **good-token rings**; everyone else is a
small neutral faint dot. Because at 100k+ points the field is unreadably dense, the
rings are split into 40 shuffled groups and a **#-samples slider** thins the visible
set (defaulting to ~6,000 points).

#### How to read it

- Each **ring = one point; ring size = how far its nearest neighbor is.** Big ring =
  lots of empty space around it.
- **DIRECTION of good:** **larger NN distance = higher resolution = good
  separation.** The header literally says so. This is the opposite reading from the
  cosine figures, where larger meant crowded — here the metric is *distance*, so big
  is open is good.
- **Good-token rings** mark the **isolated decile** — the best-separated, most
  distinct points.
- **Faint dots** are typical-spacing points (neutral).
- The **dashed center circle** is the median spacing, a size reference to compare any
  ring against.
- **Slider:** drag to thin the field when it's too crowded to read; the count of
  visible points updates.

#### What to look for

- **Well-covered reading:** rings of varied, generally healthy size; the isolated
  decile spread around the periphery and into open regions — the space has room.
- **Packed reading:** a sea of tiny rings/dots with the isolated decile confined to a
  thin rind — most points are jammed against a neighbor; little open space.

#### Worked example

For the anisotropic cone, expect a dense interior of small neutral dots (points packed
together inside the cone) with the good-token isolated-decile rings concentrated
toward the *edges* of the projected cloud — the few items that sit in open space.
Reading: the bulk of the corpus is crowded (small NN distances), and genuine
isolation is the exception, confined to the periphery. Thin the field with the slider
to confirm the isolated rings really are edge-dwellers and not just hidden by the
crowd.

#### Caveats

- 2-D projection again: positions are PCA coordinates, so a "big ring" means open in
  the *native* metric (the 1-NN distance is native), but *where* the ring is drawn is
  the projected position. Two points adjacent on screen may be far apart natively.
- The isolated decile is a **relative** cut (top 10% of *this* dataset's NN
  distances), not an absolute standard — every dataset has an isolated decile, even a
  crowded one. It marks *this corpus's* most-open points, not "objectively isolated."
- Ring radius is clipped and rescaled for legibility, so compare rings to the median
  reference circle rather than reading absolute distances off them.

---

### 3D 02 — Orthographic triptych

*Order: second card — it renders right after RES 07. Source: `figures/d3_trip.py`.*

#### What it answers

What is the **shape** of the occupied volume — and specifically, is it **flattened**
(anisotropic) along one axis? Three undistorted axis-aligned views, all at one shared
scale, let you read the flattening straight off.

#### How it's computed

From the 3-D PCA projection (`ctx.xyz`; degrades if absent). Draw **three
orthographic projections** of the same cloud — **XY (top-down), XZ (front), YZ
(side)** — side by side. The key trick: **all three panels share one identical
numeric scale** (symmetric, rounded up to a 0.2 major), so the data→pixel ratio is
the same in every panel. That makes the cloud's extent in each axis directly
comparable across panels.

Within each panel: a fine grid (0.2 majors, 0.1 minors), faint accumulation dots
(accent only on the densest core, with a single accent **core ring** at the centroid
of the densest cells), and **green "extent calipers"** bracketing the data span on
each axis. The per-axis standard deviations (σx, σy, σz) quantify the anisotropy in
the prose, with the broad/thin ratio.

#### How to read it

- **Three panels, one scale.** Compare how far the cloud fills the box in each.
- **The calipers** bracket the actual data extent on each axis. A **shorter caliper =
  less spread on that axis.** The caption frames more space as higher resolution.
- **DIRECTION:** a cloud that fills the box roughly equally in all three panels is
  **isotropic in 3-space**; a cloud that fills X and Y but is squeezed in Z (short Z
  calipers, a flat band in the XZ and YZ panels) is **thin-in-Z anisotropic**.
- The **accent core ring** marks the dense megacluster in each panel — the same core
  seen from three sides.

#### What to look for

- **Isotropic-ish reading:** comparable extents and caliper lengths across panels; a
  roughly round footprint in all three; broad/thin σ ratio near 1.
- **Flattened reading:** the XY panel shows a broad cloud, but XZ and YZ show a thin
  horizontal band — the Z caliper is markedly shorter. The σ ratio is well above 1
  (the cloud is several times broader on its widest axis than its thinnest). This is
  anisotropy you can *see*.

#### Worked example

The `why`/`reveal` text fills in `σ x/y/z` and the ratio for the run, e.g. "σ is
0.34/0.31/0.12 along x/y/z — the cloud is ~2.8× broader on its widest axis than its
thinnest," and the Z calipers in the XZ and YZ panels come out visibly shorter than
the X and Y calipers. Reading: the occupied volume is a flattened disk, not a ball —
the third PCA axis carries far less spread. A plain 2-D footprint (the XY panel
alone) would hide this; the matched-scale triptych makes it measurable.

#### Caveats

- This is the **top 3 PCA axes only** — a 3-D shadow of a high-dimensional cloud.
  "Thin in Z" means thin along the *third principal component*, which is the third
  most-variant direction, not a meaningful named axis. The spectral figures
  (RES 02/02b) are the quantitative version across *all* dimensions.
- Orthographic projection is undistorted (unlike perspective), but it's still a
  projection — overlapping points along the view axis stack up and read as denser
  than they are.

---

### 3D · live — Live 3-D cloud (drag · zoom · kNN edges) *(hidden by default)*

*Status: **hidden by default** (`d3_live: False`); flip it on to add a cluster-colored
turnable cloud alongside RES 07's density-colored one. This section documents it for
runs that enable it; it is **not** part of the default report. Source:
`figures/d3_live.py`. Interactive.*

#### What it answers

How do the **clusters sit in the occupied volume**, seen from any angle, and where
does the neighbor graph cohere or bridge across clusters? The same `ctx.xyz` cloud
the triptych shows, but turnable and colored by cluster. (RES 07 turns the same cloud
but colors it by local density instead — enable `d3_live` when the cluster-coloring
view is what you specifically want.)

#### How it's computed

The 3-D projection is baked into a vanilla-JS canvas (no dependencies; reads CSS
tokens so it re-skins on theme swap). Up to 8,000 points are kept (subsampled if
more), centered and scaled into a unit box, and **colored by cluster** (the partition
labels mapped to a stable palette). Points are **depth-sorted** (painter's order)
and depth-cued — nearer points larger and brighter, farther points smaller and
dimmer. A small **x/y/z origin gnomon** at lower-left tracks orientation as you
rotate.

The optional **kNN-edge overlay** is precomputed: over a capped prefix (~1,500
shuffled points), the k≈6 nearest neighbors (by cosine on the normalized reservoir)
form an edge list. When toggled on, **same-cluster edges** are drawn in the cluster
color at low alpha and **cross-cluster "bridge" edges** in a neutral ink at higher
alpha, but only between points currently visible under the slider.

#### How to read it

- **Drag to rotate, scroll/pinch to zoom.** It auto-spins when idle and resumes
  spinning a couple seconds after you let go.
- **Point color = cluster** (legend lists each group's color). **Depth grades the
  color** (near brighter, far dimmer) so you can perceive 3-D layering.
- **#-points slider** thins the field; the count updates.
- **kNN-edges checkbox** overlays the neighbor graph: colored same-cluster threads
  show where a cluster *coheres*; neutral bridge threads show where the graph
  *stitches across* clusters (potential chokepoints / mixing).
- **The gnomon** at lower-left shows which way x/y/z point at the current rotation.

#### What to look for

- **Well-separated clusters:** rotating the cloud, each colored group occupies its
  own region of the volume; toggling edges on, same-cluster threads stay within a
  color and few neutral bridges cross between groups.
- **Entangled clusters:** colors interpenetrate from every angle; turning on edges
  shows a thicket of neutral bridge threads tying clusters together — the partition
  isn't spatially clean.
- **Anisotropy:** rotate until you find the thin axis — a cloud that looks round from
  the top but flat from the side is the same thin-in-Z flattening the triptych
  quantifies, now seen interactively.

#### Worked example

Spin an anisotropic cone and you'll likely see one dominant colored mass (the bulk of
the cone) with smaller colored satellites; flatten your view to find the thin axis
confirming the triptych's anisotropy. Toggle kNN edges: if you see many neutral
bridge threads weaving between colors, the clusters share a lot of boundary (mixing,
consistent with the wide confusable zone you'd read in RES 05). If the colored
threads stay tightly inside each color, the clusters cohere despite the global cone.

#### Caveats

- Capped to ≤ 8,000 points for the cloud and ~1,500 for the edge graph — it's a
  representative sketch of the structure, not the whole corpus.
- Cluster colors come from the **label source**; if labels are `clustered`, you're
  looking at ambit's own grouping.
- Same projection caveat as all 3-D views: it's the top-3 PCA shadow. The interaction
  helps you *find* the anisotropy, but the spectral figures *measure* it.
- The depth cue is perceptual, not metric — don't read exact distances off point
  size.

---

### 3D 05 — Radial shell occupancy

*Order: ninth card (after RES 02, before DEN 04). Source: `figures/d3_shell.py`.*

#### What it answers

How is the cloud distributed **radially** — from a dense filled core out to a sparse
rind? Is there a hollow cavity? This reads occupancy as a function of distance from
the centroid, in native high-dimensional space, rendered in 3-D.

#### How it's computed

A two-panel figure.

**Primary panel — concentric shells in 3-D.** For each reservoir row, compute its
**high-dimensional distance from the centroid** of `ctx.es.X` and normalize to
[0, 1]. Sort rows into **6 radius-quantile shells** (each holding a comparable
count), from the filled core (shell 0) to the sparse rind (shell 5). Then place
every row on its **nested sphere**: direction taken from the unit-normalized 3-D
projection (`ctx.xyz`), radius taken from its normalized high-dim distance. Render
the whole cloud in one fixed **isometric** pose (azimuth 35°, elevation 22°),
painter's-ordered (far first), points colored by shell on a **core→rind** ramp
(accent at the core, fading toward the bad/faint token at the rind). The shell
boundaries are drawn as **isometric ellipses** (each shell sphere's equatorial circle
projected through the camera), a few labeled with the radius value, over a centroid
dot and small x/y/z axis hints.

**Secondary panel — occupancy vs radius.** Bin rows by radius (22 bins) and compute
**occupancy = count / shell volume** (using a 3-D spherical-shell volume measure),
on a **log** axis against radius. A dashed **uniform-density reference (1×)** marks
what occupancy would be if points filled the ball uniformly. **Surplus over uniform
reads up as the good direction** (good-soft fill); the **cavity dip** below uniform
is marked neutral; the **single fullest shell** carries the accent dot.

#### How to read it

- **Primary panel:** a 3-D onion. **Color = which radial shell** a point lives in
  (legend strip: core → rind). The **dashed ellipses** are the shell boundaries.
  Dense accent color in the middle = a filled core; the outer shells fading to faint
  = the rind running into empty space.
- **Secondary panel:** occupancy vs radius on a log scale.
  - **x-axis:** radius from the centroid.
  - **y-axis:** occupancy as a multiple of uniform (1× = the dashed reference).
  - **DIRECTION:** **surplus (curve above 1×, shaded good-soft) is the good
    direction** — that radius is *fuller* than uniform expectation. The **accent dot
    marks the fullest shell.**
  - A **cavity** (curve dipping below 1× between core and rind) is marked neutral —
    a radius where the cloud thins out.

#### What to look for

- **Core-concentrated reading:** occupancy high at small radius (big surplus near the
  core, accent dot on an inner shell) and tapering outward — a filled ball, most mass
  near the centroid.
- **Shell / hollow reading:** a cavity dip at small-to-mid radius and a peak farther
  out — mass arranged in a shell rather than a filled core (the kind of structure a
  normalized or contrastively-trained space can show).
- **Rind:** the outermost shell is always sparse (few points in a large-volume
  shell); that's expected. The figure frames the rind's emptiness as *information*
  (where the cloud ends), not a defect.

#### Worked example

For the anisotropic cone, the centroid sits inside a filled core, so expect the
occupancy curve to **peak near small radius** (accent "fullest" dot on an inner shell)
and fall off outward, with the primary panel showing a dense accent center fading
through the shells to a thin rind. If instead the corpus were normalized to a sphere,
you'd see a **cavity** near the center and the fullest shell pushed outward — a hollow
shell rather than a filled ball.

#### Caveats

- **Direction** comes from the 3-D projection, but **radius** comes from the
  *native* high-dimensional distance — so the primary panel is a hybrid: the angular
  placement is a PCA shadow, the radial placement is real. Read the radial structure
  (which shell, the occupancy profile) as the trustworthy part; the angular spread is
  illustrative.
- The occupancy uses a spherical-shell volume in 3-D, a modeling choice; the *shape*
  of the profile (core-full vs hollow) is the signal, not the absolute × values.
- Subsampled to ~3,500 points for a clean isometric cloud.

---

## 4b. Comparison figures (CMP) — diffing two embeddings

These render only when you pass a **second embedding of the same items** with `--compare`:
the same corpus run through **two different embedding models / encoders / configurations**,
aligned by **id** (never row order — `--id-col` is required; without stable ids the run
refuses rather than pair unrelated rows). They answer one question — *how much, and **where**,
do the two embeddings differ?* — at two scales: a **local** view (which items' neighborhoods
reshuffled) and a **global** view (whether the whole representation moved). Source:
`compare.py` (`CmpCtx`) + `figures/cmp_*.py`. When present they lead the report, with the
local neighbor-overlap view first. The two sets are labeled **A** (the primary) and **B**
(the compare set); they may even have **different dimensionality**, and the figures degrade
cleanly when a metric needs them equal.

### CMP 12a — Neighbor-overlap drift *(the local headline; leads the block)*

The **local, retrieval-relevant** comparison, and the first CMP card. Each set gets its own
kNN graph; for every item, this measures the **fraction of its top-k nearest neighbors that
are shared** between the two embeddings (`compare.neighbor_overlap`) — 1 = an identical
neighborhood, 0 = fully reshuffled. A left panel histograms that per-item retention (with the
mean rule and the retention at each scale); a right panel re-draws A's projection with each
point **shaded by how much its neighborhood changed**, so you can see *where* the reshuffling
lives. Because it compares neighbor **identities**, it is **dimension-agnostic** — it works
even when `d_A ≠ d_B` (where the drift field cannot be drawn). This is the view the global
CKA scorecard can miss: CKA is a global second-moment statistic that can read "barely changed"
while the neighborhoods that drive retrieval reshuffle underneath it, so **read CMP 12a and
CMP 12 against each other.**

### CMP 12 — Representational drift (CKA & distances) *(the global view)*

A scorecard of how similar A and B are *globally*, each on a 0–1 track:

- **linear CKA** — the exact, whole-reservoir headline; **1 = the same space re-skinned**,
  lower = a genuine rebuild. Invariant to rotation, isotropic scaling and neuron
  permutation, so it ignores the nuisances that make raw coordinate diffs meaningless. It
  is a **global second-moment statistic** dominated by the top variance directions, so it
  can read "similar" while local neighborhoods reshuffle — which is exactly why it is read
  against CMP 12a. It is defined even when the two **dimensions differ** (e.g. a
  Matryoshka-truncated set).
- **RBF CKA**, **1 − MMD²** — sampled kernel / distributional companions.
- **Procrustes disparity** — the best rigid-alignment residual; **higher = more change**
  (drawn in the *bad* token). Needs equal dimensions.

### CMP 13 — Drift field

The "*where*" map: a standardized projection grid with each cell shaded by the **mean
per-point drift** (‖B − A‖ in A's projection frame), with a sparse **quiver** — one arrow
per high-drift cell, from its A-centroid to its B-centroid. A knot of deep cells and long
arrows is a region where the two embeddings place items most differently; no single-set
figure localizes *change* like this. Needs equal dimensions (a placeholder explains when it
can't be drawn — reach for CMP 12a, which localizes change without them).

### CMP 14 — Distance-distribution shift

The two random-pair cosine densities (A faint, B accent) overlaid on one −1…+1 axis, the gap
between their means shaded, with a `Δmean cos` (and `1 − MMD²`) callout. Answers whether the
*whole* similarity distribution moved, not just individual items. Works at any dimensions.

**Degradation matrix.** When `d_A ≠ d_B`, the two CKA variants and the neighbor-overlap drift
stay defined; MMD, energy, Procrustes, the drift field and the per-item drift cosine all need
equal dims and are dropped with a note. CKA and **neighbor-overlap drift** are the
dimension-agnostic headlines — the global and the local one, respectively.

---

## 5. Reading the report as a whole

No single figure is the verdict. The report is designed so the figures
**corroborate** one another across scales and facets; a confident reading comes from
seeing the same fact told several ways.

### The resolution story, scale by scale

- **Global anisotropy (RES 01)** — the mean random-pair cosine and its wedge: how
  crowded the space is *on average*.
- **Dimensional collapse (RES 02 → RES 02b)** — the same crowding seen in the
  *spectrum*: a steep scree and an upward-bowing cumulative curve are the dimensional
  signature of a cone. RES 02b carries the IsoScore, the single-scalar restatement of
  that gap.
- **Local retrieval margin (RES 04)** — whether, *down at the level of individual
  neighborhoods*, the geometry still resolves a best match from a runner-up. A global
  cone can still have decisive local margins (locally isotropic), or not.
- **Localized density (RES 06 → RES 07)** — whether the crowding is *global* (the
  whole field shifted above the uniform reference) or *localized into pockets* (a
  separated dense mode in RES 06; a coherent knot of red pyramids in RES 07). This is
  the local counterpart to RES 01's global mean: a single mean cannot distinguish "the
  whole space is dense" from "a few sub-populations are crammed and the rest is roomy."
- **Categorical separability (RES 05)** — whether the crowding *erases your
  categories*: do same-cluster pairs still out-score different-cluster pairs?

These are one phenomenon at several zoom levels. The instructive cases are when
they *disagree*: a globally anisotropic space (RES 01 hot) that nonetheless has good
local margins (RES 04 healthy) and clean cluster separation (RES 05 separable) is
"crowded globally, useful locally" — common and often fine. The dangerous case is
the readings pointing the same way: high mean cosine, collapsed spectrum, near-tie
margins, entangled clusters — resolution is genuinely gone.

### Local vs global: the scalars miss what the localized views catch

A theme runs through the whole report: **the global scalars summarize, and the
localized views catch what they smooth over.** A single number — mean pair cosine,
IsoScore, uniformity, or (when comparing) CKA — is an average over the entire space,
and an average can read "fine" while specific regions are anything but.

- **Mean pair cosine** (one global number) cannot tell "the whole space is crowded"
  from "a few sub-populations are crammed and the rest is roomy." The **local density
  field (RES 06/07)** draws the *distribution* and *locates* the pockets, which is
  exactly the structure the mean hides.
- **A high uniformity or a healthy spectrum** describes how the variance is *spread*,
  not whether your categories occupy *distinct* regions. The **separability panel
  (RES 05b)** localizes that to the per-group geometry — which groups share a
  direction, whose neighborhoods are pure — so two spaces with identical global
  scalars can separate the partition very differently.
- **CKA** (the global representational-similarity scalar in `--compare`) can read
  "barely changed" while the neighborhoods that drive retrieval reshuffle. The
  **neighbor-overlap drift (CMP 12a)** localizes the change item by item, catching the
  fine-grained reshuffling a second-moment statistic averages away.

The discipline is the same in every case: read the scalar for the headline, then let
the localized view (RES 06/07, RES 05b, CMP 12a) tell you whether the headline holds
*everywhere* or only on average.

**Crowding costs resolution whether or not the crowded items are related.** This is
worth holding onto when reading RES 04, RES 06, and RES 07 together. Unrelated items
squeezed together is the obvious failure — impostors crowd the top of a search and
can't be ranked out, so precision collapses. But a genuinely related cluster packed
*too* tightly is a failure too: it loses its *internal* resolution — you can no
longer tell which of several relevant items is the best match, separate
near-duplicates from distinct-but-related neighbors, or rank a cluster's own members.
Many tasks need exactly that within-set resolution (fine-grained retrieval, dedup,
re-ranking, recommendation diversity). ambit's field is **relatedness-agnostic by
construction**: it is unsupervised, so the local density counts whatever sits in each
item's neighborhood, related or not — high local density is lost resolution either
way. The per-item NN margin (RES 04, cos top-1 minus cos top-2) is precisely that
within-neighborhood resolution; a small margin means even the nearest neighbors are
near-ties. So a dense pocket flagged in RES 06/07 is *not* automatically benign just
because the items in it look like they "belong together" — if its internal margins
are small, you have lost the ability to resolve *within* the set, which is what many
downstream tasks actually need.

### Density/coverage vs resolution: picture and measurement

The **DEN / COV / 3D** facets are the *picture* of crowding; the **RES** facet is its
*measurement*.

- **DEN 04** (where it piles up) and **COV 09** (where it's open vs packed) are the
  spatial face of the same anisotropy the RES figures quantify. A space with one
  towering density peak (DEN 04) and tiny NN distances everywhere (COV 09) *is* a
  crowded cone — you can read it off the maps before you read the scalars.
- The **3-D figures** let you *see* the anisotropy: the triptych (3D 02) makes a
  thin-in-Z flattening visible, the radial shells (3D 05) show whether the mass is a
  filled core or a hollow shell, and the rotatable **local crowding cloud (RES 07)**
  lets you turn the cloud while reading local density off its color and shape (the
  default report's interactive 3-D view). These recover the dimension that the 2-D
  MAP/DEN/COV views flatten away — though, as RES 07's card insists, position in the
  projection encodes the *gross* shape, not fine local crowding.

A hotspot in the density view and a mass of random-pair cosines piled up near +0.29
are the **same fact told two ways**. The report's job is to put both in front of you,
so "the embeddings look fine" can be replaced with "here is exactly how much
resolution this dataset has, and where it is spent."

### The header row as the index

The facts row at the top is the table of contents for everything below. Read it
first as a profile, then let each figure unpack one fact:

| Header fact | The figure that unpacks it |
|---|---|
| mean pair cosine | RES 01 (the full *global* distribution + wedge); RES 06/07 give the *localized* counterpart — whether that crowding is global or pocketed, and where |
| isoscore | RES 02b (the gap-to-diagonal it summarizes; formula in the hovercard) |
| effective rank | RES 02 (the scree rule) and RES 02b (why it can mislead) |
| dims for 90% var | RES 02b (the 90% crosshair) |
| uniformity | RES 08 (the gap to the isotropic reference; a trajectory in `--compare` / series mode) |
| groups | RES 05 (within/between) and RES 05b (the separability panel) |
| hub skew | RES 04 (the retrieval-margin regime where hubness bites) |

Note there is no single header scalar for the *local* density field — that is
precisely the point of RES 06/07: the localized picture does not collapse to one
number, so it is read off the **shape** of the field (RES 06) and **where** the
pockets sit (RES 07), not off the facts row.

---

## 6. Glossary & references

**Anisotropy.** Embeddings concentrate into a narrow **cone**, sharing a dominant
direction and looking similar regardless of meaning; the "crowding" you observe.
Operationalized as the mean cosine similarity of random pairs (Ethayarajh
[2019](https://arxiv.org/abs/1909.00512)). ambit reads it in the header (mean pair
cosine), RES 01, and the spectral figures.

**Isotropy.** The healthy opposite: variance spread evenly across directions, the
cloud fills the space, random pairs are ~orthogonal. The ideal every RES figure
draws a reference for.

**Cone effect / representation degeneration.** The named pathology where training
drives embeddings toward a shared direction into a narrow cone (Gao et al.
[2019](https://arxiv.org/abs/1907.12009); Biś et al.
[2021](https://aclanthology.org/2021.naacl-main.403/)). Anisotropy is also argued to
be inherent to self-attention (Godey et al.
[2024](https://arxiv.org/abs/2401.12143)).

**Local isotropy.** A space can be globally a cone yet approximately isotropic
*within* clusters and along low-dimensional manifolds (Cai et al.
[2021](https://openreview.net/forum?id=xYGNO86OWDH)). This is why ambit pairs global
RES 01 with local RES 04 / RES 05 — "crowded globally, useful locally" is a real and
common state.

**Effective rank.** Continuous effective dimensionality:
`exp(−Σ pᵢ log pᵢ)` over the normalized singular values (Roy & Vetterli
[2007](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf)).
A label-free predictor of downstream quality (RankMe, Garrido et al.
[2023](https://arxiv.org/abs/2210.02885)). Header fact; marked in RES 02 and RES 02b.
Entropy-based, so a long thin tail inflates it.

**Participation ratio.** `(Σλ)² / Σλ²` over covariance eigenvalues — a
variance-weighted dimension count that, unlike effective rank, is *not* inflated by a
low-variance tail. Shown in RES 02b next to effective rank precisely so their
disagreement is legible.

**Dimensional collapse.** The embeddings span only a lower-dimensional subspace than
the model affords; the remaining axes carry almost no variance (Jing et al.
[2022](https://arxiv.org/abs/2110.09348)). The steep scree (RES 02) and steep
cumulative curve (RES 02b) are its signature.

**IsoScore.** "Uniformity of embedding-space utilization" in [0, 1] — how evenly
variance fills all dimensions (Rudman et al.
[2022](https://arxiv.org/abs/2108.07344)). 1 = perfect sphere, 0 = degenerate line.
Computed from the covariance eigenvalues (formula in §3 and in the hovercard). Header
fact; carried by RES 02b.

**Hubness.** A curse-of-dimensionality effect in crowded high-dimensional spaces: a
few points become the nearest neighbor of disproportionately many queries ("hubs")
while others are nobody's neighbor ("anti-hubs") — degrading kNN retrieval
(Radovanović et al.
[2010](https://www.jmlr.org/papers/v11/radovanovic10a.html)). The header "hub skew"
is the skewness of the k-occurrence distribution.

**NN cosine margin.** Per item, `cos(top-1) − cos(top-2)` — how decisively the best
neighbor beats the runner-up. A local resolution measure (RES 04). Small margins =
brittle retrieval. It is precisely **within-neighborhood resolution**, and it bites
*regardless of relatedness*: a small margin among unrelated impostors collapses
precision (you can't rank them out), while a small margin among genuinely related
neighbors collapses *internal* resolution (you can't tell which of several relevant
items is the best match, separate near-duplicates from distinct-but-related
neighbors, or rank a cluster's own members). Crowding costs resolution either way, so
a tight pocket (RES 06/07) with small internal margins is lost resolution even if its
members "belong together."

**Local density field.** The unsupervised, multiscale local-crowding measure (RES 06,
`local_anisotropy.py`): per item, the **mean cosine to its k nearest neighbors** — a
k-NN density estimate of how concentrated its neighborhood is — at scales k = 10, 50,
200, with the headline scale `k*` chosen as the one whose field has the
most-extended upper tail. Read as the **distribution** against a density-matched
uniform reference (the null): one mode near the reference = roomy, the whole mode
shifted up = globally dense, a separated high mode = a pocket. The magnitude is a
**rank, not a calibrated density** (the cosine→density map is nonlinear in high d,
and k-NN density estimation there suffers distance concentration, hubness, and
fixed-k boundary bias).

**Local crowding / pocket.** A reference-calibrated dense sub-population surfaced by
RES 06/07 (`local_anisotropy.py`). An item is flagged crowded where its robust z
exceeds the *uniform reference's own* upper-tail threshold (per scale, unioned across
scales) — a cutoff anchored to "denser than the reference ever gets," not to a
fraction of the dataset's own bulk peak, so small but cleanly separated pockets are
not swamped by a tall bulk. Surviving the angular-clustering and minimum-size filter,
each **pocket** is characterized by its mean local **density** (concentration), its
mean **NN margin** (within-pocket resolution), an **IsoScore\*** (an `n < d`-robust
isotropy of the pocket — collapsed sliver vs round dense ball), and the **scale** k
at which it is most anomalous. RES 06 shows pockets as flags on the field; RES 07
locates them spatially as ringed red pyramids. A uniformly crowded "dandelion" yields
*no* pockets, because its z-tail matches the reference's.

**Alignment & uniformity.** A representation-quality decomposition on the unit
hypersphere (Wang & Isola [2020](https://arxiv.org/abs/2005.10242)): **alignment** =
similar items map close; **uniformity** = items spread evenly. ambit reports the
**uniformity** half (header fact and RES 08) because it needs no labels; **alignment**
requires positive pairs (a supervised signal) and is outside ambit's unsupervised
scope. Crowding/anisotropy is essentially **low uniformity**; good representations need
both halves, only one of which ambit can measure unsupervised.

**Rogue dimensions.** A few high-magnitude dimensions that dominate the dot product
and inflate cosine; standardizing them away often restores hidden resolution (Timkey
& van Schijndel [2021](https://arxiv.org/abs/2109.04404)). A reason RES 01's hump can
sit right of 0 even when residual structure is healthy.

**On cosine itself.** Cosine similarity of learned embeddings can be arbitrary and is
only interpretable *relative to the distribution* of similarities in the space, not
against a fixed threshold (Steck et al.
[2024](https://arxiv.org/abs/2403.05440)). This is why every cosine figure in ambit
draws an isotropic reference rather than printing a pass/fail number.

**CKA (representational similarity).** Centered Kernel Alignment (Kornblith et al.
[2019](https://arxiv.org/abs/1905.00414)) — a [0,1] similarity between two embeddings of
the same items, invariant to rotation, isotropic scaling and neuron permutation, and
defined even when their dimensions differ. ambit's `--compare` headline (CMP 12). It is a
**global** second-moment statistic dominated by the top variance directions, so it can
read "similar" while local neighborhoods reshuffle — hence it is read against the
neighbor-overlap drift.

**Neighbor-overlap drift.** Per item, the fraction of its top-k nearest neighbors **shared**
between two embeddings of the same items (CMP 12a, `compare.neighbor_overlap`). The
**local**, retrieval-relevant counterpart to CKA: it compares neighbor *identities*, so it
is dimension-agnostic and sees the fine-grained reshuffling a global similarity score
averages away. 1 = an identical neighborhood, 0 = fully reshuffled.

**Reservoir.** The sampled subset of items ambit actually scans (default 20,000) when
the corpus is large. Several figures (RES 04, COV 09, RES 05, the 3-D views) are
computed over the reservoir and its kNN, so their *exact* values reflect the sample;
the *shapes* are robust.

For the full reference list with arXiv IDs and DOIs — including the remedies
(all-but-the-top, whitening, BERT-flow, SimCSE) referenced throughout — see
[§10 of the concept document](../concepts/anisotropy-and-resolution.md#10-references).
