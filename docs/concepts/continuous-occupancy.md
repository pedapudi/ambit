# Continuous Occupancy — Measuring Crowding Without Bins

> How ambit's central quantity — **how crowded entities are when they land in an
> embedding space** — should be measured. This document develops the geometric
> intuition from scratch, audits the discrete (binned) occupancy measures ambit uses
> today, and then rebuilds occupancy on a continuous foundation: one master object
> (the pair-distance distribution), one per-entity field (the distance to a measure),
> and one structural readout (the merge tree). It doubles as an annotated map of the
> surrounding literature and a set of acceptance tests for evaluating any future
> metric proposed for ambit. The resolution thesis it builds on is developed in
> [Anisotropy, Resolution, and What ambit Measures](./anisotropy-and-resolution.md);
> a prior critical review of the pocket detector is in
> [Resolution loss and dense-pocket detection](../analysis/pocket-detection-and-resolution.md).
> Claims marked *measured* come from the controlled experiments in the
> [appendix](#appendix-the-experiments).

---

## 1. The question ambit answers

ambit exists to measure one thing: **when a corpus of entities is embedded, how
crowded do they land — and what does that crowding cost search and retrieval?**

Everything downstream of an embedding model — nearest-neighbor retrieval, RAG,
dedup, clustering, recommendation — operates on a single primitive: *given a
direction in the space, list the entities near it.* If distinct entities land too
close together, that primitive degrades in a specific, predictable way: impostors
that can't be ranked out of the results, near-duplicates that can't be ranked
against each other, distinct items a query can no longer tell apart. Crowding **is**
the failure mode; occupancy measurement is how ambit sees it coming.

So the occupancy measure is not decoration on the report — it is the report. And
that raises the standard it has to meet: the number that says "this region is
crowded" must be a property of the *data*, not of the measuring instrument.
This document is about a specific way the current instrument leaks into the answer
— **discrete steps** — and what replaces it.

## 2. The geometry, built from scratch

### 2.1 Entities are directions; neighborhoods are caps

Embeddings compared by cosine live on the surface of a unit sphere: each entity is
a point on the sphere, i.e. a **direction**, and similarity is the **angle** between
two directions. A retrieval neighborhood — "everything scoring above 0.8 against
this query" — is a **spherical cap**: the disc of directions within a fixed angle of
the query. This picture is worth internalizing, because every occupancy question
becomes a cap question:

- *Where does the corpus concentrate?* → Which caps hold more entities than their
  share of the sphere's area?
- *What does it leave empty?* → Which caps hold none?
- *How crowded is this entity?* → How small is the cap you must draw around it
  before it stops containing other entities?
- *What will retrieval confuse?* → Which entities share a small cap? They are
  future retrieval collisions: a query landing near one cannot avoid the others.

### 2.2 High dimension makes the well-spread null razor-thin

In two or three dimensions, intuition says "some pairs of random points will happen
to be close." In hundreds of dimensions the opposite is true, and it is the single
most important geometric fact for occupancy work. For points spread uniformly on the
d-sphere, the cosine of a random pair concentrates in a band of width about 1/√d
around zero (for d = 768, a standard deviation of ≈ 0.036). *Measured:* with 4,000
uniform points in 768 dimensions, the entire pairwise-distance distribution sits in
a hair-thin band — and the fraction of pairs at small angles is not merely small,
it is **exactly zero**, in every one of 19 Monte-Carlo draws.

The literature on distance concentration (Beyer, Goldstein, Ramakrishnan & Shaft,
1999; Aggarwal, Hinneburg & Keim, 2001; François, Wertz & Verleysen, 2007) usually
reads this as a curse: all distances look alike, so "nearest" loses meaning. For a
*null-calibrated diagnostic* it is precisely the opposite — a **blessing**. The
well-spread reference is so tight that any mass of close pairs at all is an
unambiguous crowding detection. A corpus does not drift into having close pairs by
chance; close pairs are always structure. (The same reversal appears in the
discrepancy literature: Brauchart, Dick & Pillichshammer
([2026](https://arxiv.org/abs/2604.21340)) call it a blessing of dimensionality for
spherical-cap discrepancy.)

Plain English: **in a high-dimensional embedding space, a healthy corpus has no
close pairs. Every close pair you find is a finding.** The measurement problem is
therefore not "estimate density everywhere" — it is "find and characterize the
excess of close pairs," which is a far easier, sharper question.

### 2.3 What "crowded" must mean operationally

One more distinction, inherited from the
[pocket-detection review](../analysis/pocket-detection-and-resolution.md): the
operational harm is **unresolvability**, not density per se. A dense region whose
members still hold each other at a readable angle is a healthy tight topic; a
sparse region whose members sit at identical angles to everything is still broken.
The occupancy measure should therefore (a) work at the level of *entities and their
scales of confusion*, not cells of space, and (b) always be read against the
isotropic reference, because raw cosine values from learned embeddings are only
interpretable relative to a distribution.

## 3. Where the discrete steps live today

Ambit's current geometric occupancy machinery quantizes twice — first project, then
bin — and fixes a scale at every layer:

| Where | Discretization | Hidden parameters |
|---|---|---|
| Hexbin occupancy figure | hard assignment to a pointy-top hex lattice | 26 columns; lattice origin |
| Density cloud "hot" accent | 48×32 count grid + 97th-percentile cutoff | grid size; grid origin; quantile |
| Density contours | 96×60 count grid, 3-tap blur, marching squares | grid; blur width; iso levels |
| Void detection | 120×90 grid of distance-to-nearest | grid resolution; inset margin |
| Differential occupancy field | per-cell counts vs isotropic reference | cell size; cell origin |
| Isometric voxel lattice | 9×9×9 occupancy | voxel edge; origin |
| Radial shells | ~6 radius-quantile shells | shell count |
| kNN graph (margins, purity, hubness, mutual graph) | fixed k = 10 rank cutoff | k |
| Cluster / separability layer | one flat partition (HDBSCAN or k-means) | the flattening threshold |

The continuous quantities ambit does have — the random-pair cosine sample, the
covariance-spectrum measures (effective rank, participation ratio, IsoScore), the
uniformity scalar — are all **global**: they say *that* the corpus is crowded, not
*where*, *at what scale*, or *which entities*. The local readouts, the ones users
act on, exist only in discretized form. That is the gap this document closes.

## 4. What a discrete step costs

### 4.1 The answer belongs to the lattice, not the data

Any statistic computed from a partition of space inherits two arbitrary choices:
the **scale** of the cells and their **placement**. Geography named this the
**modifiable areal unit problem (MAUP)**. Gehlke & Biehl (1934) showed the scale
choice changes the answer; Openshaw & Taylor (1979), in the canonical experiment,
re-partitioned the same 99 Iowa counties and moved a correlation across essentially
the whole range [−1, +1] — same data, different lines on the map. Openshaw's 1984
monograph is the standard statement, and the placement half (the *zonation effect*)
is the one people forget: it is exactly what a hex lattice's phase or a voxel
grid's origin does.

*Measured, on ambit's own hexbin assignment:* identical synthetic reservoir,
lattice shifted by half a cell → the headline peak-cell count moves **310 → 403**
(±15% from placement alone). Choosing 20 vs 34 columns → peak **574 vs 242** (2.4×
from scale alone). The two numbers the figure promotes hardest — peak count and
occupied-cell fraction — are substantially properties of the lattice.

### 4.2 One hidden scale each, so sub-cell pathology is invisible

Crowding is inherently multi-scale: near-duplicate pockets live at tiny angles,
topic clumps at moderate ones, the anisotropy cone at large ones. Every fixed bin
width, shell count, and neighbor count k silently commits to *one* scale.
*Measured:* a pathological clump of 200 near-identical points dropped inside an
already-dense cell moves the hexbin peak from 340 to 540 — a change
indistinguishable from a healthy tight topic — while a continuous per-entity
crowding field (§6) collapses by a factor of 18 on the same input, an unmissable
signal. The bin cannot see inside itself, and the most dangerous crowding (near
-duplication) is precisely sub-cell.

The fixed k in the kNN layer is the same defect in rank clothing: **a fixed k is a
hidden bin width.** In a dense pocket, 10 neighbors span a tiny radius; in the
sparse rind, a huge one — so "neighborhood" means a different physical scale at
every point, which is exactly the thing a crowding measure must not let drift.

### 4.3 Counts carry no null

"Peak cell n = 340" is not interpretable: there is no statement of what the peak
*would* be for a well-spread corpus of the same size in the same box. A perfectly
uniform cloud still has a hottest cell, and the 97th-percentile accent rule still
paints 3% of a perfectly healthy corpus as "hot." Only the differential field
figure compares to a reference — and it does so per hard cell, inheriting §4.1.
Statistically, the histogram also pays a convergence tax for its discretization:
mean integrated squared error O(n^(−2/3)) against O(n^(−4/5)) for a kernel
estimator at the optimal bandwidths (Scott, 1979; Freedman & Diaconis, 1981) — the
step function is not just fragile, it is *inefficient*.

### 4.4 Step functions can't be trusted, and cells can't name names

A bin count is a step function of the data: an ε-nudge to one point can flip a
count, so there is no stability guarantee of the form "small change in data ⇒
small change in the statistic." The continuous field of §6 carries exactly that
guarantee (provably, in Wasserstein distance), and *measured*: under a 0.1%-of-span
jitter, the field's correlation with itself is 0.99999 while remaining fully smooth.

Cells also suffer the diagnostic version of the **ecological fallacy** (Robinson,
1950): an aggregate over a region does not identify individuals. "This cell holds
40 points" cannot say *which* entities are crowded, cannot rank them, and cannot
distinguish a tight duplicate cluster from a broad dense region — but "which
entities, how badly, at what scale" is the actionable output for a retrieval owner.

### 4.5 The projection compounds it

The 2-D figures bin a **projection**, so they are fragile twice over. A planar
grid measures crowding of the shadow: two far-apart high-dimensional groups can
collide in a cell (false crowding), and genuine crowding orthogonal to the top two
principal components is invisible (missed crowding). For nonlinear projections it
is worse — cluster sizes and inter-cluster densities in t-SNE-style layouts are
explicitly not meaningful (Wattenberg, Viégas & Johnson,
[2016](https://distill.pub/2016/misread-tsne/)), because perplexity-driven local
scaling expands sparse regions and contracts dense ones; density-preserving
variants exist precisely because the standard ones destroy density (den-SNE /
densMAP — Narayan, Berger & Cho, 2021).

The tempting fix — "estimate a proper density in the native space instead" — does
not exist. Kernel density estimation converges at O(n^(−4/(4+d))); at d = 768 that
is indistinguishable from not converging, and the sample sizes required grow
exponentially in d (Silverman, 1986, ch. 4; Scott, 1992). This is the pivotal fork
in the road: **the remedy for fragile bins is not a better density estimate — it is
to stop estimating densities and work with scale-free functionals of interpoint
distances**, whose guarantees carry no dimension factor at all.

## 5. The master object: the pair-distance distribution

### 5.1 The crowding curve — an exact CDF, no bins anywhere

Take every pair of entities and record its distance (equivalently its angle; on
unit vectors ‖x−y‖² = 2−2·cos). The **empirical cumulative distribution of these
pair distances is Ripley's K function** (Ripley, 1976, 1977), the foundational
object of spatial statistics:

> **K(θ) = the fraction of pairs closer than θ**, read as a curve over all θ.

Three properties make it the right spine for ambit:

- **It contains no bins.** It is an exact step-free CDF: sort the pair distances
  once and K at *every* scale is a lookup. No width, no origin, no lattice — the
  two MAUP knobs simply do not exist. (The pair-correlation derivative g(r) does
  reintroduce a smoothing bandwidth — Stoyan & Stoyan, 1994 — which is why K, not
  g, should be primary.)
- **Its null is exact, and the sphere has no edge.** For a well-spread corpus the
  expected K is the area fraction of a spherical cap — computable in closed form
  for any dimension. Better: the edge corrections that dominate practical spatial
  statistics (Ripley's isotropic correction, etc.) exist because observation
  windows have boundaries. The unit sphere is compact and boundary-free, so under
  a rotation-invariant null **there is nothing to correct** — every point sees the
  same geometry. The spherical point-process literature makes both points
  (Robeson, Li & Huang, 2014 — who also show that grid cells on a sphere
  *manufacture* spurious clustering, a direct caution against hexbinning sphere
  data; Lawrence, Baddeley, Milne & Nair, 2016).
- **The whole curve can be tested honestly.** Comparing an empirical curve to a
  null pointwise at many scales is a multiple-testing mistake; the **global rank
  envelope** (Myllymäki, Mrkvička, Grabarnik, Seijo & Hahn, 2017) gives a single
  p-value for "is the entire curve consistent with a well-spread corpus" plus a
  graphical read-out of *which scales* are responsible. Simulating the null is two
  lines (normalize Gaussian samples), and it depends only on (n, d) — cache it.

The interpretation is exactly the retrieval question: K(θ) at small θ counts the
pairs that will collide inside a retrieval neighborhood of that size, and the scale
where the corpus curve first lifts off the null curve is **the scale at which the
space begins to confuse entities**. *Measured* (4,000 points on the 768-sphere, 5%
of them in a tight cap): at angular scales where the null's K is exactly 0 in all
19 Monte-Carlo replicates, the clumped corpus shows K ≈ 2.5×10⁻³ — infinite
signal-to-noise in the only region that matters.

### 5.2 Stolarsky's identity: one number that is every bin at every scale

There is a theorem that closes the argument against lattices. Define the
**spherical-cap discrepancy** of a point set: for *every possible cap* — every
center, every radius — compute (observed fraction inside − expected fraction), and
integrate the square over all caps and all radii. This is precisely "hexbin
occupancy done perfectly": not one lattice at one scale, but all cells at all
positions and all scales simultaneously. **Stolarsky's invariance principle**
(Stolarsky, 1973; modern RKHS proof in Brauchart & Dick,
[2011](https://arxiv.org/abs/1101.4448)) says this integral has a closed form:

> cap discrepancy² = C_d − (mean pairwise distance between the points)

up to explicit constants. The average distance between entities *is* the
all-caps-all-scales occupancy discrepancy, sign-flipped: **the more crowded the
corpus, the shorter the average pair, the worse its cap discrepancy.** One matmul
computes what an infinite family of lattices approximates, with no lattice choice
to defend. And because ambit already samples random-pair cosines for its anisotropy
fingerprint, this number is a one-line transformation of data already in hand.

*Measured:* across 20 null draws (n = 4,000, d = 768) the mean pairwise distance
has a standard deviation of 7×10⁻⁶; the 5%-clump corpus sits 430 standard
deviations below the null mean. As a global crowding alarm it is essentially
noise-free. (One refinement from the discrepancy literature: the classical cap
kernel degenerates slowly as d grows; the balanced large-cap variant of Brauchart,
Dick & Pillichshammer ([2026](https://arxiv.org/abs/2604.21340)) keeps the scalar
comparable across dimensions.)

### 5.3 It was one object all along

Several quantities ambit already computes, or that this document proposes, are the
same object read differently — the pair-distance distribution:

- The **random-pair cosine histogram** (ambit's anisotropy fingerprint) is its
  angular density.
- **Wang–Isola uniformity** ([2020](https://arxiv.org/abs/2005.10242)), already a
  header scalar, is its Laplace transform evaluated at one bandwidth (t = 2).
  Sweeping t traces a smoothed K-curve — the existing metric is one point on the
  curve this section promotes to first-class status.
- The **correlation dimension** of Grassberger & Procaccia (1983) is the log-log
  slope of K at small scales.
- The **Stolarsky scalar** is its mean.

This is the deepest simplification available to ambit: **the corpus's occupancy
signature is the pair-distance distribution; everything global is a functional of
it; the design question is only which functionals to surface.** A metric proposal
that is secretly another functional of the same distribution is not wrong — but it
should be evaluated as a *view*, not as new information (see §10).

## 6. Localizing without cells: the distance to a measure

The curve says *that* and *at what scale*; it does not say *who*. The per-entity
replacement for cell counts is the **distance to a measure** (DTM) of Chazal,
Cohen-Steiner & Mérigot (2011), which in plain English is:

> **the size of the ball an entity needs to gather a fixed share m of the corpus.**

Crowded entity → tiny ball; isolated entity → huge ball. The empirical form could
not be simpler: the root-mean-square distance to the nearest k = m·n sample points
— a smoothed k-nearest-neighbor radius. But the framing upgrade matters:

- **One knob, and it is a mass fraction, not a length.** The parameter m replaces
  the bin width, the grid origin, *and* the fixed k — and because it is a fraction
  of the corpus it is scale-free in n. Reported as a curve over m (which one full
  distance-sort yields for free), even that knob disappears; the reporting m is
  chosen by the smallest clump size worth caring about.
- **It carries the stability guarantee bins lack.** DTM is uniformly Lipschitz in
  the Wasserstein distance between datasets: ‖DTM_P − DTM_Q‖_∞ ≤ m^(−1/2)·W₂(P,Q)
  (Chazal & Michel, [2021](https://arxiv.org/abs/1710.04019)). Small perturbation
  of the corpus ⇒ provably small change in every entity's score, with an explicit
  robustness-to-outliers budget (mass < m cannot move it). This is the theorem-
  shaped version of the jitter experiment in §4.4, and it is the strongest
  guarantee in this entire design space.
- **It subsumes three existing mechanisms.** The k-th-NN distance (Loftsgaarden &
  Quesenberry, 1965) is DTM with all weight on the last neighbor — strictly higher
  variance. Kernel density estimation with a per-query ("balloon") bandwidth
  (Terrell & Scott, 1992) is what DTM's inverse power approximates, without the
  hopeless high-d normalization (report the DTM radius itself in native dimension;
  never exponentiate a 768-dim volume). And the void-detection grid is just DTM
  read from the other tail: **the low tail of the DTM field is the crowding list;
  the high tail is the void list** — one field, both figures, no grid.
- **It names names.** DTM is a continuous score *per entity*, so the report can
  list the crowded items, rank them, and threshold them against the null's own
  spread rather than against a bin size. *Measured:* the 200-point cap clump drags
  the field's 1st percentile from 1.35 to 0.21 (6.5×) while the null's entire
  range is ±0.005 — per-entity detection with enormous margin.

For comparing two corpora (or one corpus embedded two different ways), the
DTM-signature of Brécheteau (2019) turns the field into a distribution with a
subsampling test — a principled two-sample crowding comparison that needs no
alignment between the spaces.

## 7. Structure without thresholds: the merge tree

Between the global curve and the per-entity field sits the structural question:
*is the crowding one broad pile, or many tight pockets — and which entities form
each?* The continuous answer is the oldest data structure in clustering, read
without its usual guillotine: connect entities by shortest links (the **minimum
spanning tree**), and watch components merge as the connection scale grows.

Formally this is the H₀ **persistence** of the density/scale filtration: each
tight group is *born* when its points connect among themselves and *dies* when it
merges into an older group (the elder rule); the difference is its **prominence** —
a continuous number saying how separated-and-tight the pocket is, comparable
across corpora. The persistence diagram is stable under perturbation of the input
(Cohen-Steiner, Edelsbrunner & Harer, 2007), and chaining with §6 makes it stable
in Wasserstein distance when the filtration is built on DTM (Anai, Chazal, Glisse,
Ike, Inakoshi, Tinarrage & Umeda, [2020](https://arxiv.org/abs/1811.04757)).
ToMATo (Chazal, Guibas, Oudot & Skraba, 2013) is the mode-seeking algorithmic
form; Hartigan's cluster tree (1975, 1981) is the statistical object underneath,
with consistency achieved by robust single linkage (Chaudhuri & Dasgupta, 2010).

Two practical notes anchor this to ambit:

- **Ambit already computes this object and throws it away.** HDBSCAN — the default
  clustering backend — internally builds exactly this hierarchy; its "condensed
  tree" is the merge tree, its cluster *stability* (Campello, Moulavi & Sander,
  2013; Campello, Moulavi, Zimek & Sander, 2015) is a mass-weighted normalization
  of the same persistence (the formal bridge is Rolle & Scoccola,
  [2020](https://arxiv.org/abs/2005.09048)), and its GLOSH score is a free
  per-entity outlier readout. The current pipeline flattens all of this to one
  labeling at one implicit threshold and hands the labels to the separability
  panel. Surfacing the tree costs nothing new.
- **It is startlingly informative about exactly the pathology that matters.**
  *Measured:* injecting a 200-point near-duplicate clump produces exactly 199
  short MST edges against 0 for the null — the tree recovers the pocket's
  *cardinality*, not just its existence, and its birth scale says at what angle
  the pocket detaches from the bulk ("these 200 items only merge with the rest at
  distance 1.3, but merge with each other below 0.2").

Higher homology (loops, cavities in coverage) exists in the same framework but
needs specialized dependencies and is only tractable on low-dimensional
projections; it is optional-extra material, not core.

## 8. Counting modes globally: the kernel spectrum

One global signal is invisible to everything above the covariance level: *how many
effectively distinct things does the corpus contain?* Fifty tight clumps spread
across a 500-dimensional subspace look healthy to effective rank, IsoScore, and
even uniformity — the clumps are far apart, the variance is spread — yet the corpus
effectively contains fifty things, and retrieval within any clump is broken.

The continuous, label-free answer is the spectrum of a **nonlinear** kernel matrix.
The **Vendi score** (Friedman & Dieng, [2023](https://arxiv.org/abs/2210.02410)) —
the exponential of the von Neumann entropy of the normalized kernel Gram matrix —
reads directly as the **effective number of dissimilar entities**: m orthogonal
tight clumps score ≈ m. Its Rényi family (Pasarkar & Dieng,
[2024](https://arxiv.org/abs/2310.12952)) profiles imbalance: order ∞ answers "is
one clump eating the corpus," low orders up-weight rare modes. Matrix-based kernel
entropy is the common theory (Giraldo, Rao & Príncipe, 2015); random-feature
approximations scale it past the reservoir if ever needed (Ospanov, Zhang, Jalali,
Cao, Bogdanov & Farnia, [2024](https://arxiv.org/abs/2407.02961)).

Two identities keep this honest — and prune the surrounding literature:

- **With a linear/cosine kernel, the Gram spectrum is the covariance spectrum** (up
  to centering). So the popular self-supervised quality measures — RankMe (Garrido,
  Balestriero, Najman & LeCun, [2023](https://arxiv.org/abs/2210.02885)), NESum,
  stable rank — are re-summaries of eigenvalues ambit already computes. New signal
  requires a nonlinear kernel; only then does the spectrum see mode structure that
  second moments cannot.
- **Order-2 kernel entropy with a Gaussian kernel *is* Wang–Isola uniformity**
  (squaring a Gaussian kernel just doubles its bandwidth), so that point of the
  family is already on ambit's scoreboard. The information ambit lacks lives at
  the other orders (1 and ∞), which require the spectrum rather than the mean.

On a reservoir of a few thousand points the eigendecomposition costs seconds; on a
finite sample the order-1 score needs care (it is capped at n and converges
slowly — truncate the spectrum per Ospanov & Farnia,
[2024](https://arxiv.org/abs/2410.21719); orders 2 and ∞ concentrate well).

## 9. Neighborhoods without k

The kNN layer's fixed k = 10 propagates a hidden scale into margins, purity,
hubness, and the mutual-neighbor graph (§4.2). Three continuous replacements, in
increasing order of novelty:

- **Curves over k instead of points at k.** One full distance sort yields every
  neighbor radius for every k at once; any fixed-k scalar becomes a curve, and the
  parameter evaporates exactly as the bin width did in §5.1.
- **Soft neighborhoods and the effective neighbor count.** Replace the top-k cliff
  with smooth weights (the stochastic-neighbor softmax of Goldberger, Roweis,
  Hinton & Salakhutdinov, 2004 — the same construct as t-SNE's perplexity
  calibration, van der Maaten & Hinton, 2008). Then the **participation ratio of
  an entity's neighbor weights** — (Σw)²/Σw² — is its *effective number of
  neighbors*: a continuous, per-entity crowding score with no k anywhere. Ambit
  already owns this functional (it applies it to covariance eigenvalues); applying
  it to neighbor weights is the same idea one level down, and its lineage runs
  from the inverse participation ratio of physics to Kish's effective sample size
  (1965).
- **Per-entity local intrinsic dimension.** The log-ratio estimator of Levina &
  Bickel (2005) (equivalently the Hill tail-index estimator, 1975; the LID framing
  of Houle, 2017, and Amsaleg, Chelly, Furon, Girard, Houle, Kawarabayashi &
  Nett, 2015) distinguishes the two kinds of "dense" that no count can separate:
  a dense-but-full-rank region (healthy hot topic) versus collapse onto a
  low-dimensional filament (template/duplication pathology). Caveats from
  experiment: use it **per-entity** — the global TwoNN fit (Facco, d'Errico,
  Rodriguez & Laio, [2017](https://www.nature.com/articles/s41598-017-11873-y))
  barely moves under a 5% clump (*measured:* 188.2 → 186.4) — and treat all ID
  estimates as comparative, never absolute, at embedding dimensions (TwoNN reports
  ≈ 188 for genuinely 767-dimensional data).

Hubness (Radovanović, Nanopoulos & Ivanović,
[2010](https://www.jmlr.org/papers/v11/radovanovic10a.html)), already measured by
ambit, is causally downstream of local intrinsic dimension — high-ID regions breed
hubs — so the two readouts should be presented together: hubness says retrieval is
warping; per-entity LID says where the warp originates.

## 10. How to reason about this literature — ambit's acceptance tests

The research behind this document surveyed several dozen candidate metrics. Most
fell to one of five filters, which are worth recording because they are the
fastest way to evaluate the next proposal too:

1. **The spectrum test.** Does it reduce, with a linear kernel, to a functional of
   the covariance/Gram eigenvalues? Then ambit already has it (effective rank,
   participation ratio, IsoScore) and the proposal is a re-summary, not a signal.
   *Felled: RankMe, NESum, stable rank, linear-kernel Vendi; α-ReQ survives only
   as a free extra summary of an existing decomposition.*
2. **The continuity test.** Is it a continuous functional of the data, or a step
   function of bin/rank/partition membership? Discrete steps forfeit stability
   guarantees and reintroduce MAUP. *Felled: anything built on k-means
   pseudo-labels; every fixed-lattice statistic; fixed-k scalars (rescued by
   curves-over-k).*
3. **The null test.** Does it come with a calibrated reference — analytic on the
   sphere, or cheap Monte Carlo — so its value reads as a deviation rather than a
   bare number? *This is what separates K-with-envelope from a raw histogram, and
   the Stolarsky scalar from an uncalibrated "compactness."*
4. **The naming test.** Can it score *entities* (so crowded items can be listed
   and acted on), or only regions/aggregates? Region-only statistics stop one
   step short of the user's action. *This is what promotes DTM, GLOSH, and
   per-entity LID over any cell-based heat.*
5. **The supervision test.** ambit is unsupervised by design: no labels, no gold
   pairs, no quality scores, ever. *Felled: neural-collapse NC1 metrics
   (need classes), LiDAR (needs augmentation pairs), quality-weighted Vendi
   (needs a quality signal) — regardless of their merits elsewhere.*

And one unifying insight to reason about ambit itself: **ambit already computes
the right raw objects — the pair-cosine sample, the covariance spectrum, the
HDBSCAN hierarchy — but currently reads each at a single scale or a single
threshold. Almost every upgrade in this document is the same move applied in a
different place: stop reading the object at one point; read the whole curve.**
Uniformity at one t → the t-sweep; counts in one lattice → all caps at all scales;
the k-th neighbor → the DTM curve over mass; one flat clustering → the full merge
tree. The discrete steps were never separate flaws; they were one flaw, repeated.

## 11. Considered and set aside

| Candidate | Verdict | Reason |
|---|---|---|
| RankMe, NESum, stable rank | redundant | linear-kernel spectrum = covariance spectrum ambit already decomposes (§10.1) |
| α-ReQ (spectral decay exponent) | optional freebie | extra summary of an existing decomposition; power-law fit breaks exactly under collapse |
| Order-2 Gaussian kernel entropy | already present | identical to Wang–Isola uniformity at doubled bandwidth (§8) |
| Global TwoNN intrinsic dimension | comparative only | nearly blind to local clumps; strongly biased at embedding dimension (§9) |
| DANCo | excluded | needs a synthetic calibration bank; awkward under a numpy-only constraint |
| Ambient-space KDE | rejected | O(n^(−4/(4+d))) convergence — meaningless at embedding dimension (§4.5) |
| Pair-correlation g(r) | optional | derivative of K; reintroduces a bandwidth — expose it if shipped |
| H₁/H₂ persistence | optional extra | needs specialized dependencies; tractable only on projections |
| Neural-collapse metrics (NC1, CDNV, VCI), LiDAR, quality-weighted Vendi | excluded | require labels / paired views / quality scores — supervision test (§10.5) |
| k-means-pseudo-label anything | rejected | reintroduces the discontinuity this redesign removes (§10.2) |

## 12. What this implies for the report

The redesign principle is a separation of concerns: **projection and cells are
display; measurement happens in native space, continuously.**

- The **crowding curve** (K vs its null, with a global envelope) becomes the
  canonical global figure: it subsumes the single-scale count figures and
  annotates the *scale* of crowding, which no current figure states.
- The **Stolarsky scalar** joins the header facts: a one-number, null-calibrated
  occupancy discrepancy, computed from the cosine sample already in hand.
- The **DTM field** drives the local layer: its low tail lists the crowded
  entities (replacing quantile-hot accents), its high tail the voids (replacing
  the distance grid); projected figures may keep their geometry but should tint by
  the native-space field rather than by projected cell counts.
- The **merge tree** upgrades the cluster/separability layer: prominence-ranked
  pockets with member lists and birth scales, instead of one flat partition; GLOSH
  arrives free.
- Where a binned *display* is kept for its readability (the hexbin honeycomb is
  genuinely legible), its statistic should be de-fragilized by averaging over
  lattice placements — the **average shifted histogram** (Scott, 1985), which
  removes the zonation effect in a few lines and recovers the kernel convergence
  rate — and its caption must present it as a view, not a measurement.
- Every surfaced number states its null. The sphere's nulls are analytic or
  two-line Monte Carlo; there is no excuse for a bare count.

None of this requires labels, gold pairs, or any dependency beyond numpy; every
component above was cost-checked at reservoir scale (a few thousand points) in
fractions of a second, with the kernel-spectrum eigendecomposition the most
expensive at a few seconds.

## Appendix: the experiments

Two controlled experiments ground the *measured* claims. Sketches suffice to
reproduce; both are pure numpy.

**A. Lattice fragility (2-D, ambit's hexbin assignment).** Draw ~4,000 points from
three Gaussian clumps plus uniform background; replicate the hexbin figure's
lattice assignment (26 columns, pointy-top, nearest-center snap); recompute the
peak-cell count under (i) lattice origin shifts of ¼ and ½ cell, (ii) column
counts 20/26/34, (iii) 0.1%-of-span Gaussian jitter, (iv) injection of 200
near-duplicate points at one clump's edge. Results: peak 310→403 across origin
shifts; 574 vs 242 across column counts; 340→339 under jitter (counts are stable
only when nothing crosses a boundary — the instability is intermittent, which is
worse than constant); duplicate injection reads as 340→540, indistinguishable from
a benign hot cell. The DTM field (m = 1%) under the same jitter: self-correlation
0.99999; under duplicate injection its 1st percentile falls 18×.

**B. Sphere clump (native dimension).** n = 4,000 uniform on S^767 (normalized
Gaussians) versus the same with 200 points drawn inside a cap of angular radius
≈ 0.15. Pairwise statistics via one Gram matrix (float32, ~0.1 s). Results: null
pair-cosine sd = 1/√768 exactly; mean pairwise distance null sd 7×10⁻⁶ with the
clump at z = −430; empirical K(θ) = 0 for all θ below the bulk band in 19/19 null
replicates versus 2.5×10⁻³ for the clumped set; DTM(k=100) 1st percentile 1.350 →
0.207; MST short-edge count 0 → 199 (recovering the 200-point clump); global TwoNN
188.2 → 186.4 (demonstrating its blindness to local pathology).

## References

Grouped; annotations state what each contributes to ambit.

**Spatial statistics and the case against bins**

- Gehlke, C. & Biehl, K. (1934). *Certain effects of grouping upon the size of the
  correlation coefficient in census tract material.* JASA 29(185A). — First
  demonstration that aggregation scale changes statistical answers.
- Openshaw, S. & Taylor, P. (1979). *A million or so correlation coefficients.* In
  *Statistical Applications in the Spatial Sciences.* — The canonical MAUP
  experiment: same data, correlations spanning [−1, +1] across zonations.
- Openshaw, S. (1984). *The Modifiable Areal Unit Problem.* CATMOG 38. — Standard
  monograph; names the scale and zonation effects.
- Robinson, W. (1950). *Ecological correlations and the behavior of individuals.*
  American Sociological Review 15. — Aggregates don't identify individuals; the
  reason cell heat can't name crowded entities.
- Ripley, B. (1976). *The second-order analysis of stationary point processes.*
  J. Applied Probability 13; and (1977) *Modelling spatial patterns.* JRSS-B 39.
  — The K function: the pair-distance CDF as the multi-scale clumping statistic.
- Besag, J. (1977). Discussion of Ripley (1977). JRSS-B 39. — The
  variance-stabilized L(r) transform.
- Stoyan, D. & Stoyan, H. (1994). *Fractals, Random Shapes and Point Fields.*
  Wiley. — Pair-correlation g(r) and its bandwidth caveat.
- Robeson, S., Li, A. & Huang, C. (2014). *Point-pattern analysis on the sphere.*
  Spatial Statistics 10. — Spherical K; grid cells on spheres manufacture spurious
  pattern.
- Lawrence, T., Baddeley, A., Milne, R. & Nair, G. (2016). *Point pattern analysis
  on a region of a sphere.* Stat 5. — Regional spherical K, if a subdomain is ever
  analyzed.
- Myllymäki, M., Mrkvička, T., Grabarnik, P., Seijo, H. & Hahn, U. (2017). *Global
  envelope tests for spatial processes.* JRSS-B 79. — The honest whole-curve test
  against a simulated null.
- Landy, S. & Szalay, A. (1993). *Bias and variance of angular correlation
  functions.* ApJ 412. — Cosmology's mature estimator machinery for the same
  two-point object.

**Discrepancy and the sphere**

- Stolarsky, K. (1973). *Sums of distances between points on a sphere II.* Proc.
  AMS 41. — The invariance principle: mean pair distance = all-caps-all-scales L₂
  discrepancy.
- Brauchart, J. & Dick, J. ([2011](https://arxiv.org/abs/1101.4448)). *A simple
  proof of Stolarsky's invariance principle.* — The RKHS view; the kernel behind
  the scalar.
- Brauchart, J., Dick, J. & Pillichshammer, F.
  ([2026](https://arxiv.org/abs/2604.21340)). *Spherical cap L₂ discrepancy —
  blessing of dimensionality and a balanced large-cap variant.* — High-d behavior;
  the d-comparable variant of the scalar.
- Beyer, K., Goldstein, J., Ramakrishnan, R. & Shaft, U. (1999). *When is "nearest
  neighbor" meaningful?* ICDT; Aggarwal, C., Hinneburg, A. & Keim, D. (2001). *On
  the surprising behavior of distance metrics in high dimensional space.* ICDT;
  François, D., Wertz, V. & Verleysen, M. (2007). IEEE TKDE 19. — Distance
  concentration; read here as the tightness of the null.

**Geometric inference: distance to a measure and persistence**

- Chazal, F., Cohen-Steiner, D. & Mérigot, Q. (2011). *Geometric inference for
  probability measures.* Found. Comput. Math. 11. — DTM: definition, semiconcavity,
  Wasserstein stability.
- Chazal, F. & Michel, B. ([2021](https://arxiv.org/abs/1710.04019)). *An
  introduction to topological data analysis.* Frontiers in AI. — Survey; the
  m^(−1/2)·W₂ stability bound as stated here.
- Chazal, F., Massart, P. & Michel, B. (2016). *Rates of convergence for robust
  geometric inference.* Electron. J. Statist. 10. — Finite-sample rates; the m
  bias/robustness trade-off.
- Brécheteau, C. (2019). *The DTM-signature.* Electron. J. Statist. 13. — A
  subsampling two-sample test on DTM distributions; corpus-vs-corpus crowding
  comparison without alignment.
- Loftsgaarden, D. & Quesenberry, C. (1965). *A nonparametric estimate of a
  multivariate density function.* Ann. Math. Statist. 36. — The k-th-NN density
  DTM improves upon.
- Terrell, G. & Scott, D. (1992). *Variable kernel density estimation.* Ann.
  Statist. 20. — Balloon estimators; where DTM-as-density sits taxonomically.
- Cohen-Steiner, D., Edelsbrunner, H. & Harer, J. (2007). *Stability of
  persistence diagrams.* Discrete Comput. Geom. 37. — Bottleneck stability; the
  merge tree inherits perturbation bounds.
- Chazal, F., Guibas, L., Oudot, S. & Skraba, P. (2013). *Persistence-based
  clustering in Riemannian manifolds.* JACM 60. — ToMATo; prominence-thresholded
  mode-seeking with guarantees.
- Anai, H., Chazal, F., Glisse, M., Ike, Y., Inakoshi, H., Tinarrage, R. & Umeda,
  Y. ([2020](https://arxiv.org/abs/1811.04757)). *DTM-based filtrations.* — Chains
  DTM's Wasserstein stability into the persistence diagram.
- Fasy, B., Lecci, F., Rinaldo, A., Wasserman, L., Balakrishnan, S. & Singh, A.
  (2014). *Confidence sets for persistence diagrams.* Ann. Statist. 42. — Noise
  bands on prominence: the principled "which pockets are real."
- Hartigan, J. (1975). *Clustering Algorithms.* Wiley; (1981). *Consistency of
  single linkage for high-density clusters.* JASA 76. — The cluster tree; why
  naive single linkage fails.
- Chaudhuri, K. & Dasgupta, S. (2010). *Rates of convergence for the cluster
  tree.* NIPS. — Robust single linkage: the consistent estimator.
- Campello, R., Moulavi, D. & Sander, J. (2013). *Density-based clustering based
  on hierarchical density estimates.* PAKDD; Campello, R., Moulavi, D., Zimek, A.
  & Sander, J. (2015). ACM TKDD 10. — HDBSCAN: condensed tree, stability, GLOSH.
- Rolle, A. & Scoccola, L. ([2020](https://arxiv.org/abs/2005.09048)). *Stable and
  consistent density-based clustering via multiparameter persistence.* — The formal
  bridge between HDBSCAN stability and persistence prominence.
- Grassberger, P. & Procaccia, I. (1983). *Measuring the strangeness of strange
  attractors.* Physica D 9. — Correlation dimension = small-scale slope of K.

**Kernel spectra and effective diversity**

- Friedman, D. & Dieng, A. B. ([2023](https://arxiv.org/abs/2210.02410)). *The
  Vendi score.* TMLR. — Effective number of dissimilar items; exp of kernel-spectrum
  entropy.
- Pasarkar, A. & Dieng, A. B. ([2024](https://arxiv.org/abs/2310.12952)). *Cousins
  of the Vendi score.* AISTATS. — The Rényi order family; imbalance profiling.
- Ospanov, A. & Farnia, F. ([2024](https://arxiv.org/abs/2410.21719)). *Do Vendi
  scores converge with finite samples?* — Truncation for trustworthy order-1
  estimates on a reservoir.
- Jalali, M., Li, C. & Farnia, F. (2023). *An information-theoretic evaluation of
  generative models in learning multi-modal distributions.* NeurIPS. — Order-2
  kernel entropy; the identity tying it to uniformity.
- Giraldo, L., Rao, M. & Príncipe, J. (2015). *Measures of entropy from data using
  infinitely divisible kernels.* IEEE Trans. Inf. Theory. — Matrix-based entropy:
  the common theory.
- Ospanov, A., Zhang, J., Jalali, M., Cao, N., Bogdanov, A. & Farnia, F.
  ([2024](https://arxiv.org/abs/2407.02961)). *FKEA.* NeurIPS. — Random-feature
  scaling of kernel-spectrum diversity beyond the reservoir.
- Garrido, Q., Balestriero, R., Najman, L. & LeCun, Y.
  ([2023](https://arxiv.org/abs/2210.02885)). *RankMe.* ICML. — Effective rank as
  SSL quality; here, the exhibit for the linear-kernel redundancy argument.
- Wang, T. & Isola, P. ([2020](https://arxiv.org/abs/2005.10242)). *Understanding
  contrastive representation learning through alignment and uniformity.* ICML. —
  Already in ambit; revealed here as one point of the pair-distance Laplace
  transform.
- Rottach, F., Rudman, W., Rieck, B., Scells, H. & Eickhoff, C.
  ([2025](https://arxiv.org/abs/2511.22150)). *From topology to retrieval.* —
  Geometric + persistence signatures validated against dense-retrieval quality;
  external evidence for the merge-tree direction.

**Neighborhoods, intrinsic dimension, hubness**

- Levina, E. & Bickel, P. (2005). *Maximum likelihood estimation of intrinsic
  dimension.* NIPS. — The per-point log-ratio ID estimator (average inverses, per
  the MacKay–Ghahramani correction).
- Hill, B. (1975). *A simple general approach to inference about the tail of a
  distribution.* Ann. Statist. 3. — The tail-index estimator LID reuses.
- Houle, M. (2017). *Local intrinsic dimensionality I–II.* SISAP; Amsaleg, L.,
  Chelly, O., Furon, T., Girard, S., Houle, M., Kawarabayashi, K. & Nett, M.
  (2015). *Estimating local intrinsic dimensionality.* KDD. — LID as the
  extreme-value view; estimators.
- Facco, E., d'Errico, M., Rodriguez, A. & Laio, A.
  ([2017](https://www.nature.com/articles/s41598-017-11873-y)). *Estimating the
  intrinsic dimension of datasets by a minimal neighborhood information.* Sci.
  Rep. 7. — TwoNN; used here with its measured limitations stated.
- Radovanović, M., Nanopoulos, A. & Ivanović, M.
  ([2010](https://www.jmlr.org/papers/v11/radovanovic10a.html)). *Hubs in space.*
  JMLR 11. — Hubness; causally linked here to local ID.
- Goldberger, J., Roweis, S., Hinton, G. & Salakhutdinov, R. (2004).
  *Neighbourhood components analysis.* NIPS. — The soft-neighbor construct.
- van der Maaten, L. & Hinton, G. (2008). *Visualizing data using t-SNE.* JMLR 9.
  — Perplexity as a continuous k.
- Kish, L. (1965). *Survey Sampling.* Wiley. — Effective sample size; the same
  functional as the effective neighbor count.

**Density estimation and its limits**

- Scott, D. (1979). *On optimal and data-based histograms.* Biometrika 66;
  Freedman, D. & Diaconis, P. (1981). *On the histogram as a density estimator.*
  Z. Wahrsch. Verw. Gebiete 57. — The histogram's convergence tax and bin-width
  rules.
- Scott, D. (1985). *Averaged shifted histograms.* Ann. Statist. 13. — The fix for
  lattice placement, if a binned display is kept.
- Silverman, B. (1986). *Density Estimation for Statistics and Data Analysis.*
  Chapman & Hall; Scott, D. (1992). *Multivariate Density Estimation.* Wiley. —
  Origin sensitivity; the exponential sample cost of KDE in high dimension.
- Wattenberg, M., Viégas, F. & Johnson, I.
  ([2016](https://distill.pub/2016/misread-tsne/)). *How to use t-SNE
  effectively.* Distill. — Densities in nonlinear projections are not meaningful.
- Narayan, A., Berger, B. & Cho, H. (2021). *Assessing single-cell transcriptomic
  variability through density-preserving data visualization.* Nature Biotech. 39.
  — densMAP: the projection to use if a 2-D density panel must be defensible.
