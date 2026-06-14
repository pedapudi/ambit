---
name: ambit-concepts
description: >
  The science behind ambit and how to READ its results — anisotropy vs isotropy,
  resolution/crowding, and what every diagnostic means (random-pair cosine vs the
  isotropic reference, effective rank, participation ratio, dims-for-variance,
  IsoScore, nearest-neighbor margin, within/between-cluster cosine, hubness). Use
  this to interpret an `ambit info` printout or report, decide whether a space is
  healthy, explain WHY crowding hurts retrieval/clustering/RAG, and recommend
  remedies (mean-centering, all-but-the-top, whitening, contrastive uniformity).
  The deep reference is docs/concepts/anisotropy-and-resolution.md. For how to
  produce these numbers see ambit-cli; for the big picture see ambit-overview.
version: 1.0.0
---

# Anisotropy, resolution, and reading the diagnostics

> Crowding in an embedding space is a loss of **resolution**: when items pack too
> tightly, cosine similarity can no longer tell them apart. ambit's density/coverage
> views are the *picture* of this; the resolution diagnostics are its *measurement*.

The canonical reference, with full academic citations, is
`docs/concepts/anisotropy-and-resolution.md`. This skill is the working distillation.

## The core idea

An embedding model is only useful if it keeps **distinct things distinct**. The tell
is simple: take many random pairs and look at their cosine similarity.

- **Healthy (isotropic):** unrelated pairs average ≈ 0 (nearly orthogonal); a
  genuinely related pair stands out sharply. The space has high *effective
  resolution*.
- **Crowded (anisotropic):** embeddings concentrate into a narrow **cone** sharing a
  dominant direction, so *every* pair scores high (0.3–0.9). The dynamic range cosine
  can express is compressed; the signal you care about barely clears the floor.

Two cautions that shape how ambit reports:

1. **Cosine isn't meaningful in absolute terms** — for learned embeddings it depends
   on training/regularization (Steck et al. 2024). So ambit always reads cosine
   *against a reference distribution*, never a fixed threshold.
2. **A few "rogue" dimensions can dominate** the cosine (Timkey & van Schijndel
   2021); standardizing them away often *restores* resolution. This is why ambit
   reports the spectrum alongside the cosine.

## Vocabulary (just enough)

- **Isotropy ↔ anisotropy** — variance spread evenly across directions (good) vs.
  collapsed into a cone (bad). Operationalized as the *mean random-pair cosine*
  (Ethayarajh 2019).
- **Alignment & uniformity** (Wang & Isola 2020) — similar items close (alignment)
  *and* items spread over the sphere (uniformity). "Low resolution" = low uniformity.
- **The cone effect / representation degeneration** — likelihood training drifts
  embeddings toward a shared direction.
- **Dimensional collapse** — the cloud lives in a lower-dimensional subspace than the
  model affords; the rest of the axes are wasted capacity.
- **Hubness** (Radovanović et al. 2010) — in crowded high-d spaces a few points become
  the nearest neighbor of disproportionately many queries ("hubs"); a retrieval-side
  fingerprint of poor resolution.
- **"Crowding" (t-SNE sense)** is a *different* term of art — the difficulty of
  placing mid-distance relations when squeezing to 2-D. It's a projection artifact,
  not a property of the embeddings. ambit pairs every 2-D view with native-space
  diagnostics precisely to keep these separate.

## The geometry in one line

Write each embedding as a shared part plus a residual: `x_i = μ + r_i`. When `μ` (or
a few dominant directions) is large relative to the residuals, the cosine numerator
is dominated by `μ·μ` for *every* pair, so all cosines inflate toward a common
positive value — the cone. The information lives in the residuals `r_i` but is
swamped. This is exactly why **mean-centering**, **all-but-the-top** (drop the top
few directions), and **standardizing rogue dimensions** re-expose structure.

## The diagnostics ambit computes (and how to read each)

All run on the **original high-dimensional vectors** (not the 2-D projection).
Source: `src/ambit/metrics.py` and the RES-family figures.

| diagnostic | code | healthy | crowded | read it as |
|---|---|---|---|---|
| **mean random-pair cosine** | `random_pair_cosine` | ≈ 0 (± `1/√d`) | shifted toward +1 (e.g. 0.3) | the single most legible crowding signature — how far off orthogonal unrelated pairs sit |
| **isotropic reference** | `isotropy_ref(d)` = `1/√d` | — | — | the std of random-pair cosine for iid points on the unit d-sphere; ambit's yardstick. `info` flags "anisotropic" when the mean exceeds ~4× this |
| **effective rank** | `effective_rank(eigs)` = `exp(entropy(pᵢ))` | → full dim | ≪ dim | continuous effective dimensionality; a label-free quality signal (RankMe) |
| **participation ratio** | `participation_ratio(eigs)` = `(Σλ)²/Σλ²` | → full dim | small | a second effective-dimensionality read, weighted toward dominant eigenvalues |
| **dims for 90% variance** | `dims_for_variance(eigs, .9)` | spread over many | a handful | how concentrated variance is across axes |
| **eigenvalue scree** | `eigs_from_cov` | gentle decay | steep cliff | the structural *cause* of cosine crowding (a few axes hold the variance) |
| **IsoScore / utilization** | `res_iso` gauge (erank / nominal dims) | → 1 | → 0 | fraction of available dimensions actually exercised |
| **nearest-neighbor margin** | `res_margin` (top-1 minus top-2 cosine) | wide | thin (~0.04) | how much the best match beats the runner-up; a thin margin means neighbors are unresolvable and retrieval is fragile |
| **within- vs between-cluster cosine** | `res_wb` | clear rightward gap | overlapping | whether real cluster structure survives the crowding (alignment vs uniformity) |
| **hubness (k-occurrence skew)** | `hubness_skew(knn_idx)` | low | high positive | a few hubs absorb most retrievals |

Important nuance: **a global anisotropy number can mask local structure** — a space
can be a cone globally yet approximately isotropic *within* clusters (Cai et al.
2021). That's why ambit reports both global (cosine histogram, scree) and local
(NN-margin, within/between, density views) facets.

## A worked reading

A representative, realistically-imperfect 768-d corpus (from the concepts doc):

| diagnostic | value | isotropic ref | verdict |
|---|---|---|---|
| mean random-pair cosine | 0.34 | ≈ 0.00 ± 0.036 | strongly anisotropic — a cone |
| effective rank | 47 / 768 | → 768 | severe dimensional collapse |
| top-1 PC variance | 37% (90% in 31 dims) | evenly spread | a few axes dominate |
| IsoScore | 0.27 / 1.0 | 1.0 | space badly under-used |
| NN cosine margin (median) | 0.041 | ≈ 0.11 | thin — top-1 barely beats top-2 |
| hubness (top-1 k-occurrence) | 184 | ≈ 10 | a few hubs dominate retrieval |

Read together: this space *looks* 768-d but lives on ~47 effective axes, packs
unrelated items at cosine ≈ 0.34, and resolves neighbors by only ~0.04 — so
retrieval is fragile and a few hubs absorb most queries. Don't read any single
number alone; they corroborate one story.

## Why it matters downstream

- **Retrieval / RAG** — a compressed similarity range means the right doc is only
  marginally above the wrong ones; thresholds get unreliable, top-k gets noisy,
  hubness over-retrieves a few docs.
- **Clustering / dedup** — crowded regions merge distinct entities; boundaries blur.
- **Classification probes** — within-class collapse + cross-class crowding lowers
  linear separability.
- **Drift / monitoring** — an already-anisotropic baseline leaves little headroom to
  detect real distributional shift.
- **Model selection** — effective rank predicts downstream quality *without task
  labels*, so resolution metrics can rank encoders early.

## Remedies (when ambit says "crowded")

Cheap post-processing first — re-measure against the reference after each:

- **Mean-centering** — subtract `μ`; removes the shared cone direction.
- **All-but-the-top** (Mu, Bhat & Viswanath 2018) — also drop the top few dominant
  directions.
- **Whitening / BERT-flow** — map to a more isotropic distribution as a post-process.
- **Standardize rogue dimensions** before trusting cosine.
- **Contrastive uniformity** (SimCSE) / spread-out or coding-rate objectives — fixes
  at *training* time if you control the model.

Caveat (Steck et al. 2024): whether cosine is meaningful at all is
regularization-dependent — always re-measure against the isotropic reference, not an
absolute threshold.

## How the facets map to the concepts

| ambit facet | the concept |
|---|---|
| density / hotspots | crowded, low-resolution regions — the local face of anisotropy |
| coverage / voids | unused capacity — the spatial face of dimensional collapse |
| topology (kNN graph, bridges) | where distinctness is load-bearing vs. fragile |
| comparison vs reference | the gap from isotropic |
| 3-D views | anisotropy you can rotate and see (broad in X–Y, thin in Z) |
| resolution / isotropy (RES) | the scalar measurement of all the above |
