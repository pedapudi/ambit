# ambit — Background & Context

> A self-contained primer on **ambit**: the problem it addresses, the vocabulary it
> uses, the method it runs, the report it produces, and the design philosophy behind
> it. Read this to understand ambit from scratch. The science is developed at length
> in [Anisotropy, Resolution, and What ambit Measures](./concepts/anisotropy-and-resolution.md);
> the theory of the occupancy measurement itself — the geometric intuition, the case
> against binned counts, and the continuous foundation — is in
> [Continuous Occupancy](./concepts/continuous-occupancy.md);
> how to read a generated report is in [the report guide](./guide/interpreting-the-report.md).
> The reference list in [§9](#9-references) is the union of the citations behind the
> claims here.

---

## 1. What ambit is

**ambit measures and visualizes how a dataset occupies an embedding space.** Given a
collection of high-dimensional vectors, it surfaces three things: where the items
**concentrate** (density hotspots), what the data **leaves empty** (coverage voids),
and the **resolution / isotropy** of the space — how much of the space's nominal
capacity the data actually uses, and how distinct its items remain from one another.
It runs either as a terminal **scan** (scalar diagnostics, no rendering) or as a
single self-contained, theme-adaptive **HTML report** with no external requests.

### The core thesis

> **Crowding in an embedding space is a loss of resolution.** When items pack too
> tightly, cosine similarity can no longer tell them apart.

An embedding model is only as useful as its ability to keep *distinct things
distinct*. A space can have high nominal **capacity** — hundreds or thousands of
dimensions — and yet low **effective resolution** in practice: if items collapse into
a narrow region of directions, every pair of them scores high on cosine similarity,
so the signal you care about barely rises above the floor. Anything built on that
geometry — nearest-neighbor retrieval, clustering, deduplication, a RAG retriever —
inherits the ambiguity. ambit's job is to replace *"the embeddings look fine"* with
*"here is exactly how much resolution this dataset has, and where it is spent."*

### Crowding is relatedness-agnostic

The harm is **not** limited to *unrelated* items piling up (impostors that cannot be
ranked out of a search). A set of genuinely *related* items packed too tightly is a
harm too: you lose the **resolution within the group** — the ability to tell *which*
of several relevant items is the best match, to separate near-duplicates from
distinct-but-related neighbors, to rank a cluster's own members. Fine-grained
retrieval, deduplication, re-ranking, and recommendation diversity all live or die on
exactly that within-set resolution. So ambit measures crowding as **local density** —
unsupervised, counting whatever sits in each item's neighborhood — and a tight
neighborhood reads as lost resolution whether the neighbors are impostors or genuine
relatives. The per-item **NN margin** (the cosine gap between an item's first and
second neighbor) is precisely that within-neighborhood resolution.

### The tell

The diagnostic is simple and measurable. Take many random pairs of items and look at
their cosine similarity. In a **healthy** space, unrelated pairs average ≈ 0 (nearly
orthogonal) and a genuinely related pair stands out sharply. In a **crowded** space,
*every* pair — related or not — scores high, so the dynamic range cosine similarity
can express has been compressed. That compression is the phenomenon ambit makes
legible.

Two cautions frame the whole tool:

- **Cosine similarity is not automatically meaningful.** For learned embeddings the
  cosine similarity can be arbitrary — it depends on the model's regularization and
  can even be non-unique (Steck, Ekanadham & Kallus, 2024). A high or low number is
  interpretable only relative to the *distribution* of similarities in the space.
  ambit therefore always reads cosine against a **reference distribution**, never
  against a fixed threshold.
- **A handful of dimensions can dominate the score.** A few high-magnitude "rogue
  dimensions" drive most of the cosine in transformer representations (Timkey & van
  Schijndel, 2021); standardizing them away often restores resolution that raw cosine
  hides.

---

## 2. The vocabulary

### Isotropy ↔ anisotropy (the core axis)

- **Isotropy** (good): variance is spread evenly across directions; the cloud fills
  the space; random pairs are ~orthogonal.
- **Anisotropy** (bad — the "crowding" you observe): embeddings concentrate into a
  narrow **cone**, sharing a dominant direction and looking similar to one another
  regardless of meaning.

Anisotropy was operationalized as the *mean cosine similarity of random pairs*
(Ethayarajh, 2019); contextual representations are strikingly anisotropic in their
upper layers.

### The cone effect & dimensional collapse

- **The cone effect / representation degeneration** — embeddings drift toward a
  shared direction and degenerate into a narrow cone (Gao et al., 2019). A common
  mechanism is a *shared gradient direction* that shifts all embeddings together
  (Biś et al., 2021); the same work shows simple mean-removal restores isotropy.
  Some of the anisotropy arises directly from the self-attention mechanism, not only
  from the training objective (Godey et al., 2024).
- **Dimensional collapse** — the embeddings span only a *lower-dimensional subspace*
  than the model affords; the remaining axes carry almost no variance and are wasted
  capacity (Jing et al., 2022).
- **Neural collapse** — in the terminal phase of classification training, within-class
  variability collapses while class means arrange into a maximally separated simplex
  (Papyan et al., 2020). It is "good" collapse *between* classes but erases
  distinctions *within* a class — exactly the within-set resolution ambit watches.

### Local vs global isotropy (a crucial nuance)

A single global anisotropy scalar can mask local structure. The contextual space is
approximately isotropic *within* clusters and along low-dimensional manifolds even
when it is globally a cone (Cai et al., 2021) — so "crowded globally" and "useful
locally" can coexist. A global isotropy scalar is **non-collapsible** over clusters
(a Simpson's-paradox effect: a globally isotropic cloud can be a union of locally
collapsed cones, and vice versa). Good diagnostics must look at **both scales**,
which is why ambit pairs every global scalar with an unsupervised, multiscale **local
concentration field** that localizes crowded pockets rather than scoring the whole
space at once.

### Alignment & uniformity

Representation quality on the unit hypersphere decomposes into two properties (Wang &
Isola, 2020):

- **Alignment** — semantically similar items map close together.
- **Uniformity** — items spread evenly over the sphere, preserving maximal
  information.

"Low resolution" is **low uniformity**. The two halves are asymmetric in what they
demand. **Uniformity is measurable without labels** — it asks only how the points are
spread, so it can be read off the embeddings alone. **Alignment cannot**: it averages
over a distribution of *positive pairs*, and without knowing which items are supposed
to be close there is nothing to average. ambit is unsupervised, so it **reports
uniformity** (a scalar; more negative = more uniform) and deliberately **does not
claim alignment** — that is the supervised half, out of scope. A measured-uniformity
number is a statement about spread only; it says nothing about whether the right
things ended up together.

### Hubness (the retrieval-side symptom)

A curse-of-dimensionality effect (Radovanović et al., 2010): in crowded
high-dimensional spaces a few points become the nearest neighbor of disproportionately
many queries ("hubs") while others are nobody's neighbor ("anti-hubs"). Hubness
directly degrades kNN retrieval and is a practical fingerprint of poor resolution.
ambit reports the **k-occurrence skew**, and an optional **mutual-kNN** mode recomputes
every neighbor view on the *reciprocal* graph (keeping only edges both endpoints agree
on) — a standard hubness reduction that drops a hub's surplus one-sided connections so
genuine local structure is read apart from the hubness artifact.

### Separability & resolution

- **Separability** is whether a *partition* of the items — provided labels, or ambit's
  own discovered clusters — occupies distinct regions of the space. It is read with a
  centroid-cosine matrix, kNN label purity, silhouette, and a Fisher ratio.
- **Resolution** is the umbrella term for the whole "how distinct are the items"
  facet: global (random-pair cosine, effective rank, IsoScore, uniformity) and local
  (the per-item NN margin and the local-density field).

### A note on the word "crowding"

The same word is also a term of art for **t-SNE** specifically: the difficulty of
placing moderate-distance relationships when squeezing high-dimensional structure into
2-D (van der Maaten & Hinton, 2008). That is a property of the *visualization*, not
the native space. ambit keeps them separate — anisotropy is a property of the
embeddings; the t-SNE crowding problem is an artifact of projecting them — which is
why every 2-D projection ambit draws is paired with native-space diagnostics.

---

## 3. Why it matters downstream

Low resolution silently caps the ceiling of everything built on the embeddings:

- **Retrieval / RAG** — a compressed similarity range means the right document is only
  marginally above the wrong ones; thresholds become unreliable and top-k gets noisy,
  while hubness over-retrieves a few documents.
- **Clustering / dedup** — crowded regions merge distinct entities and blur cluster
  boundaries; near-duplicates and distinct-but-related items become indistinguishable.
- **Monitoring / drift** — if the baseline is already anisotropic, there is little
  headroom to detect a real distributional shift.
- **Model selection** — effective rank is a **label-free** predictor of downstream
  quality (RankMe; Garrido et al., 2023), so resolution metrics can rank candidate
  encoders *before* you have task labels.

Several of these failures live *within* a related set, not just across unrelated ones:
re-ranking the best of many relevant documents, deduplicating near-identical items,
diversifying recommendations. Crowding that is tolerable for coarse retrieval can still
wreck these finer tasks — which is why ambit reports the **local density field** and the
**NN margin** per item, not only a global average. A neighborhood can be made of
perfectly on-topic items and still be *too* crowded to rank.

---

## 4. How ambit measures it

ambit is a small numpy-first Python package. **One in-memory type flows through a
streaming scan, a shared render context, and a registry of figures.** numpy is the
only hard dependency; everything heavier (tabular IO, projection, ANN kNN, GPU,
on-the-fly embedding) is an optional backend selected at runtime, so a missing
dependency disables a capability rather than crashing the core.

### The one type: `EmbeddingSet`

Everything normalizes into a single canonical type, `EmbeddingSet` — the `(n, d)`
float32 vectors plus optional ids, labels, and per-item metadata, with a `metric`
("cosine" or "euclidean") and a normalized flag. Its constructor enforces the
invariants (float32, contiguous, 2-D, finite, matching id/label lengths), so invalid
data fails loudly at construction and no downstream figure has to defend against NaN
or shape bugs. Because everything becomes an `EmbeddingSet`, downstream code never
branches on where the data came from.

### The streaming scan

The scan is **one pass** over the corpus, designed so a multi-million-row dataset is
never fully resident in RAM. It maintains:

- a **streaming covariance** accumulated through the `d × d` scatter (`XᵀX` in
  float64 from float32 grams via BLAS), so the eigenspectrum is **exact over the whole
  corpus** and memory is `O(d²)` regardless of `n`;
- **norm statistics** (mean and standard deviation of the L2 norms);
- a bounded **reservoir sample** (Algorithm R, default ~20k rows) — the working set
  the *visual* layers operate on.

Scalar diagnostics are computed over the **full corpus**; the projections, kNN graph,
clustering, and cosine histogram all run on the **reservoir** (the scatter/3-D views
never draw more than a few thousand points anyway). Two knobs handle the very large
end: routing the covariance accumulation onto a GPU/torch device, and an **approximate
scan** that stops after ~`N` rows (rank and variance converge fast in `n`) while still
reporting the true corpus size when it is cheaply known from file metadata.

### The render context: `Ctx`

From a scan, ambit computes one shared, read-only **render context** (`Ctx`) that
every figure receives:

- **2-D and 3-D projections** of the reservoir (PCA by default; UMAP optional);
- the **full-corpus covariance eigenvalues**;
- a **random-pair cosine sample**;
- the **kNN graph** over the reservoir (indices + distances; `None` if no backend is
  available — figures degrade gracefully);
- **labels** — a provided metadata column if present, else **unsupervised clusters of
  the geometry itself** (HDBSCAN if available, else k-means with a silhouette-picked
  `k`);
- the **hub-skew** scalar.

When mutual-kNN is requested, the reciprocal filter is applied once here, so every
downstream reader (margin, sparsity, purity, hubness, the graph views) sees the same
mutual graph. When a comparison set is supplied, the two-embedding comparison context
hangs off the same `Ctx`.

### The metrics

The scalar diagnostics run on the **original high-dimensional vectors**, not a 2-D
projection. Each captures a different facet of the same phenomenon:

| Metric | What it captures | Healthy direction |
|---|---|---|
| **Mean random-pair cosine** | angular spread / anisotropy fingerprint | ≈ 0 |
| **IsoScore** | how evenly variance fills the dimensions (utilization), in [0,1] | → 1 |
| **Effective rank** | continuous effective dimensionality (entropy of the spectrum) | → full dim |
| **Participation ratio** | effective dimensionality / collapse, from the eigenvalues | → full dim |
| **Dims for 90% variance** | how few axes carry most of the variance | → full dim |
| **Uniformity** (Wang–Isola) | spread on the hypersphere, from the cosine sample | more negative |
| **Hubness (k-occurrence skew)** | a few points dominating neighbor lists | low skew |
| **NN cosine margin** | local retrieval decisiveness (top-1 − top-2) | larger |

Two metrics deserve a note:

- **IsoScore** is derived from the covariance eigenvalues: the eigenvalues are
  normalized to a vector that is all-ones under perfect isotropy, an *isotropy defect*
  measures the distance from that, and the score maps the implied "dimensions
  isotropically used" onto [0,1] (0 = a degenerate line, 1 = a perfect sphere). A
  small-sample variant shrinks the covariance toward an isotropic reference so that a
  genuinely round small cloud is not mistaken for a cone merely because it has fewer
  points than dimensions.
- **Uniformity** is computed straight from the random-pair cosine sample the scan
  already draws (for unit vectors the squared distance is `2 − 2·cos`), so the header
  scalar costs nothing extra. It is read against the uniformity of a synthetic
  isotropic cloud, just as cosine is read against `1/√d`.

Crowding is also measured **locally and as a distribution**. The local concentration
field gives, per item, the mean cosine to its `k` nearest neighbors (a kNN density
estimate, monotone in density so it ranks neighborhoods reliably). It is computed at
**several scales** and read against a **density-matched isotropic reference** (a
uniform cloud at the same point count), so a point counts as crowded only when its
robust z exceeds the reference's *own* upper tail — the cutoff comes from the
reference, not from a fraction of the dataset's own bulk. The distribution of the
field tells the story: one mode near the reference is roomy; the whole mode shifted up
is globally dense; a separated high mode is a crowded pocket (surviving pockets are
split by angular clustering, with a minimum size rejecting scattered false positives).
Because the cosine→density map is strongly nonlinear in high dimensions, the magnitudes
are read as **ranks, not calibrated densities**.

Separability of a partition adds a centroid-cosine matrix (off-diagonal near 1 means
two groups share a direction), kNN label purity (does a neighborhood agree with its
label), silhouette, and a Fisher ratio. For **discovered** clusters — where the
partition is unsupervised and therefore suspect — it additionally reports **stability**
(mean bootstrap adjusted-Rand index across re-clusterings: does the structure
reproduce?) and a **modal count** from the graph-Laplacian eigengap (how many
well-separated modes the spectrum implies). The cosine→geometry read is geometric, not
semantic: tidy clusters can be meaningless and true categories can live on curved
manifolds, which is exactly why the unsupervised branch leads with stability.

### The figures, by family

Figures are **pure functions** of the `Ctx` that emit **token-colored static SVG**.
Every number is computed at generation time and baked into the SVG — nothing is
computed in the browser. They are grouped into facet **families**, each answering a
different occupancy question:

| Family | Question it serves | Reads on |
|---|---|---|
| **MAP / DEN** | where it concentrates | the projected footprint; density peaks, contours, hotspots |
| **COV / 3D** | how much space it uses / its shape | coverage, sparsity, reach, voids; the occupied volume from several axes |
| **RES** | how distinct items are (resolution / isotropy) | native-space diagnostics, **global and local** (the local-density field and crowding map) |
| **CMP** | how two embeddings of the same items differ | neighbor-overlap drift, CKA, MMD / Procrustes, a drift field, a distribution shift |

A topology family (kNN manifold graph, bridge chokepoints) also exists for where
distinctness is load-bearing versus fragile. A given report renders only the figures
**enabled** in its run configuration; the rest are implemented but hidden by default.

---

## 5. Reading the report

The report is **one self-contained HTML file** — CSS, JavaScript, and every SVG inlined,
no external assets, no network requests.

### The header facts

A row of scalar facts sits at the top, computed at generation time:

- **items × dims** (with a note when the scan was approximate),
- **mean L2 norm**,
- **mean pair cosine** (the anisotropy fingerprint, ≈ 0 healthy),
- **isoscore** (variance-fill, → 1 healthy),
- **uniformity** (more negative = more uniform),
- **effective rank** out of the nominal dimension,
- **dims for 90% variance** out of the nominal dimension,
- **groups** (count and whether provided or clustered) when a partition exists,
- **hub skew** when a kNN graph is available.

### The figure cards

Below the facts, each figure is a card titled by its **name**, with a terse method tag,
a one-paragraph note describing what it shows for *this* dataset, a legend, a "reveals"
line, and a "how to read" hovercard in a consistent voice. The report surfaces figures
**by name** — the short family codes (e.g. `RES 01`, `CMP 12`) are cross-reference
shorthand for the documentation, not labels printed in the report.

The default figure set reads top to bottom as the local crowding cloud, the
orthographic triptych, the local density field, the NN-margin, within-vs-between
cosine, the separability panel, the random-pair cosine distribution, cumulative
variance (carrying the IsoScore badge), the eigenvalue scree, radial shell occupancy,
density-peak prominence, and the sparsity field. When a comparison set is present, the
**CMP** figures sort ahead of all of these and lead the report.

### Good/bad is communicated by DIRECTION, not by a fixed hue

This is a deliberate discipline. The *verdict* is carried by **position and
direction**, not by a red/green color:

- In the random-pair cosine histogram, mass to the **right** of the isotropic
  reference is the bad direction (anisotropy); the reference itself sits at 0.
- In the NN-margin histogram, bars at the **right** (large margin) are decisively
  resolved; bars piled at the **left edge** (margin → 0) are near-ties.

Each figure reserves a single **accent** color for the thing it is actually about. A
few figures additionally use semantic tint tokens, but the tint only colors in a
direction that was already legible from the geometry.

### Theme-adaptive

The reason for the direction-not-hue discipline is that the report is
**theme-adaptive**: it ships **16 themes** selectable live from a picker in the header.
Because every SVG is drawn with **CSS color tokens** rather than baked-in colors, a
theme swap re-skins the entire report with **no re-render** — the figures keep their
meaning under any palette. (A small number of figures are interactive: the sparsity
field has a slider to thin a crowded field, and the local crowding cloud is the
report's one rotatable 3-D canvas.)

---

## 6. Comparing two embeddings of the same items

ambit can compare **one dataset embedded two different ways** — the same items run
through two embedding models, encoders, or configurations — to answer *how much*, and
*where*, the two representations differ. The two sets are aligned by **stable id**
(never by row order); the second set is streamed and only the rows matching the first
set's reservoir are kept, so a large comparison set stays bounded. Because alignment is
by id, the comparison fails loudly if the primary set lacks stable ids (two
index-keyed sets would otherwise "match" on every row and pair unrelated items).

The comparison is read at two scales, and the contrast between them is the point:

- **Local: neighbor-overlap drift.** For each item, the fraction of its top-`k`
  neighbors that are **shared** between the two embeddings (1 = identical neighborhood,
  0 = fully reshuffled), reported across scales. This is the **retrieval-relevant**
  signal: it compares neighbor *identities*, so it is **dimension-agnostic** — it works
  even when the two embeddings have different dimensionalities — and it sees local
  reshuffling that a global statistic can miss. It leads the CMP block.
- **Global: linear CKA.** Centered Kernel Alignment (Kornblith et al., 2019) — a
  similarity in [0,1] **invariant to rotation, isotropic scaling, and neuron
  permutation**. It is the exact, scalable headline (it streams, uses `O(d²)` memory,
  and the two sets may differ in dimension). A kernel (RBF) CKA variant captures
  nonlinear structure on a sample.

These two **can disagree, and that disagreement is informative**: CKA is a global
second-moment statistic, so it can read "similar" while every item's neighborhood
reshuffles underneath it — a global re-skin versus a local rebuild. The neighbor
overlap is what tells retrieval whether anything actually moved for the user.

Where the two sets share a dimension, the comparison adds distributional and geometric
measures: **MMD²** and **energy distance** (did the cloud shape move), **Procrustes
disparity** (the residual after best rigid alignment), a **drift field** (each item
pushed through the primary set's PCA basis, showing *where* the representation moved),
and a **distance-distribution shift** (the two cosine distributions side by side).
When the dimensions differ, only the two CKA variants and the neighbor overlap are
defined — the dimension-coupled measures return nothing rather than a meaningless
number.

---

## 7. Design principles

- **Tufte data-ink line art.** A single accent color, minimal chrome, and good/bad
  read by **direction, not hue** — so the figures survive any theme and the ink that
  is present is the data.
- **Unsupervised by construction.** No labels or gold pairs are required: when no
  labels are given, ambit clusters the geometry itself and reports the structure it
  finds (with stability and a modal count to keep an unsupervised partition honest).
  It reports **uniformity**, not alignment — alignment needs supervision and is out of
  scope.
- **No fixed thresholds — read against references.** Cosine is read against the
  isotropic d-sphere reference (`1/√d`), uniformity against a synthetic isotropic
  cloud, and the local density field against a density-matched uniform null. Because
  absolute cosine values are not portable across spaces, nothing is judged against a
  hard cutoff.
- **Self-contained.** One HTML file, no external requests; theme-adaptive across 16
  themes via CSS color tokens, so a theme swap re-skins with no re-render.
- **numpy-first, optional everything else.** The core runs on numpy alone; heavier
  capabilities are opt-in and degrade gracefully. **Configuration is a `Config`
  object plus CLI flags — never environment variables** — so a run is fully described
  by its `Config`.
- **Both scales, always.** Every global scalar is paired with a local view, because a
  single global number is non-collapsible over clusters and can hide or misread a
  crowded pocket.

---

## 8. Usage at a glance

ambit is a **library first**; the `ambit` command is one front end over it. Every CLI
command builds a `Config` and calls the matching library verb, so anything the CLI
does is available programmatically.

### Library

```python
import ambit

# full report -> self-contained, theme-adaptive HTML
rep = ambit.report("embeddings.parquet")
rep.write("report.html")

# just the numbers — a streaming scan + resolution metrics, no rendering
diag = ambit.diagnose("embeddings.parquet")
print(diag.mean_cos, diag.verdict, diag.effective_rank, diag.dims_for_90pct)

# compare one dataset embedded two ways (aligned by id) — adds the CMP figures
ambit.report("set_a.parquet", compare="set_b.parquet", id_col="uuid").write("diff.html")
```

Every verb takes a `Config` or plain keyword overrides, and the pipeline can be driven
stage by stage:

```python
sc   = ambit.scan("embeddings.parquet")       # streaming scan        -> Scan
ctx  = ambit.build_ctx(sc)                     # project, kNN, cluster -> Ctx
html = ambit.build_report(ctx, out="report.html")
la   = ambit.localized_anisotropy(ctx.es.X)    # the local-density field
```

### Command line

```
ambit info   <embeddings>                       # streaming scan -> resolution diagnostics
ambit report <embeddings> --out report.html     # the self-contained HTML report
ambit embed  <dataset> --out vecs.parquet --model <name>   # embed raw items via an OpenAI-compatible endpoint
```

`<embeddings>` is `.npy` / `.npz` / `.parquet` / `.csv` / `.jsonl`, or a directory /
glob of parquet shards (streamed).

### Scaling knobs

- **`--sample N`** — reservoir size for the visual layers (default ~20k).
- **`--approx N`** — cap the scan at ~`N` rows (for very large corpora; the headline
  count still reflects the true size when cheaply known).
- **`--device cpu|auto|cuda|mps`** — route the covariance / kNn kernels onto a GPU.
- **`--knn-backend auto|pynndescent|sklearn|brute|faiss`** — neighbor-graph backend.
- **`--mutual-knn`** — reciprocal (mutual) kNN everywhere, suppressing hubs in every
  neighbor view.
- **`--compare <embeddings> --id-col <col>`** — the two-embedding comparison, aligned
  by id.
- **`--config <json>`** — merge a JSON object onto the `Config` (fields and a figure
  toggle map).

---

## 9. References

**Foundational geometry & origin of the observation**
- Mimno, Thompson (2017). [*The strange geometry of skip-gram with negative sampling*](https://aclanthology.org/D17-1308/). EMNLP 2017.
- Arora, Li, Liang, Ma, Risteski (2016). [*A Latent Variable Model Approach to PMI-based Word Embeddings*](https://aclanthology.org/Q16-1028/). TACL 4 (arXiv:1502.03520).
- van der Maaten, Hinton (2008). [*Visualizing Data using t-SNE*](https://www.jmlr.org/papers/v9/vandermaaten08a.html). JMLR 9 — the t-SNE "crowding problem."
- Radovanović, Nanopoulos, Ivanović (2010). [*Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data*](https://www.jmlr.org/papers/v11/radovanovic10a.html). JMLR 11.

**Anisotropy in (contextual) embeddings**
- Ethayarajh (2019). [*How Contextual are Contextualized Word Representations?*](https://aclanthology.org/D19-1006/) EMNLP-IJCNLP 2019 (arXiv:1909.00512).
- Gao, He, Tan, Qin, Wang, Liu (2019). [*Representation Degeneration Problem in Training Natural Language Generation Models*](https://arxiv.org/abs/1907.12009). ICLR 2019.
- Biś, Podkorytov, Liu (2021). [*Too Much in Common: Shifting of Embeddings in Transformer Language Models and its Implications*](https://aclanthology.org/2021.naacl-main.403/). NAACL-HLT 2021.
- Cai, Huang, Bian, Church (2021). [*Isotropy in the Contextual Embedding Space: Clusters and Manifolds*](https://openreview.net/forum?id=xYGNO86OWDH). ICLR 2021.
- Timkey, van Schijndel (2021). [*All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality*](https://aclanthology.org/2021.emnlp-main.372/). EMNLP 2021 (arXiv:2109.04404).
- Godey, de la Clergerie, Sagot (2024). [*Anisotropy Is Inherent to Self-Attention in Transformers*](https://aclanthology.org/2024.eacl-long.3/). EACL 2024 (arXiv:2401.12143).
- Reif, Yuan, Wattenberg, Viégas, Coenen, Pearce, Kim (2019). [*Visualizing and Measuring the Geometry of BERT*](https://arxiv.org/abs/1906.02715). NeurIPS 2019.

**Collapse & the contrastive lens**
- Wang, Isola (2020). [*Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere*](https://arxiv.org/abs/2005.10242). ICML 2020.
- Papyan, Han, Donoho (2020). [*Prevalence of Neural Collapse during the Terminal Phase of Deep Learning Training*](https://doi.org/10.1073/pnas.2015509117). PNAS 117(40).
- Jing, Vincent, LeCun, Tian (2022). [*Understanding Dimensional Collapse in Contrastive Self-Supervised Learning*](https://arxiv.org/abs/2110.09348). ICLR 2022.

**Metrics of resolution / utilization**
- Roy, Vetterli (2007). [*The Effective Rank: A Measure of Effective Dimensionality*](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/a5p-h05.pdf). EUSIPCO 2007.
- Rudman, Gillman, Rayne, Eickhoff (2022). [*IsoScore: Measuring the Uniformity of Embedding Space Utilization*](https://aclanthology.org/2022.findings-acl.262/). Findings of ACL 2022 (arXiv:2108.07344).
- Garrido, Balestriero, Najman, LeCun (2023). [*RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations by Their Rank*](https://arxiv.org/abs/2210.02885). ICML 2023.
- Yu, Chan, You, Song, Ma (2020). [*Learning Diverse and Discriminative Representations via the Principle of Maximal Coding Rate Reduction (MCR²)*](https://arxiv.org/abs/2006.08558). NeurIPS 2020.

**Two-representation comparison**
- Kornblith, Norouzi, Lee, Hinton (2019). [*Similarity of Neural Network Representations Revisited*](https://arxiv.org/abs/1905.00414) (CKA). ICML 2019.

**Remedies (post-processing toward isotropy)**
- Mu, Bhat, Viswanath (2018). [*All-but-the-Top: Simple and Effective Postprocessing for Word Representations*](https://arxiv.org/abs/1702.01417). ICLR 2018.
- Li, Zhou, He, Wang, Yang, Li (2020). [*On the Sentence Embeddings from Pre-trained Language Models*](https://aclanthology.org/2020.emnlp-main.733/) (BERT-flow). EMNLP 2020 (arXiv:2011.05864).
- Su, Cao, Liu, Ou (2021). [*Whitening Sentence Representations for Better Semantics and Faster Retrieval*](https://arxiv.org/abs/2103.15316) (BERT-whitening).
- Gao, Yao, Chen (2021). [*SimCSE: Simple Contrastive Learning of Sentence Embeddings*](https://aclanthology.org/2021.emnlp-main.552/). EMNLP 2021 (arXiv:2104.08821).
- Rajaee, Pilehvar (2021). [*A Cluster-based Approach for Improving Isotropy in Contextual Embedding Space*](https://aclanthology.org/2021.acl-short.73/). ACL-IJCNLP 2021 (arXiv:2106.01183).
- Rajaee, Pilehvar (2022). [*An Isotropy Analysis in the Multilingual BERT Embedding Space*](https://aclanthology.org/2022.findings-acl.103/). Findings of ACL 2022 (arXiv:2110.04504).
- Zhang, Yu, Kumar, Chang (2017). [*Learning Spread-out Local Feature Descriptors*](https://arxiv.org/abs/1708.06320). ICCV 2017.

**On interpreting cosine itself**
- Steck, Ekanadham, Kallus (2024). [*Is Cosine-Similarity of Embeddings Really About Similarity?*](https://doi.org/10.1145/3589335.3651526) WWW '24 Companion (arXiv:2403.05440).
