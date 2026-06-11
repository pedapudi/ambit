# ambit embedding-space occupancy visualization study

A single standalone, self-contained, theme-adaptive study page
(`index.html`) that explores how to draw **the way a dataset occupies an
embedding space** — its density and hotspots, its coverage and concave reach, its
voids and sparsity, and its surplus/deficit against a reference. Preserved here as
the design record behind ambit's occupancy views: this is an archived exploration,
**not** a TODO list.

Open `index.html` in a browser to view it offline. The page carries the full
16-theme swatch picker (top-right) so every figure recolours live with **no
re-render** — a pure CSS token swap. There is no runtime data layer: every figure
is a static, fit-to-width inline SVG with plausible coordinates baked in.

## The question

> **How much of an embedding space does a dataset occupy, and where are its
> hotspots and voids?**

A scatter of projected embeddings only half-answers this. A corpus can pile into a
single megacluster while leaving most of the reachable space empty; it can wrap a
ring of coverage around an interior hole it never learned to fill; it can
over-represent one region and starve another relative to the **base embedding
space** it was sampled from. The eleven views below read that occupancy structure
directly — each isolating **one** property of how the cloud fills the plane.

## The canonical synthetic dataset

Every figure depicts the **same** projection, so the eleven views compose into one
coherent reading of a single cloud. It is a deliberately legible synthetic
occupancy structure — generated, not measured — chosen so each property has
something unambiguous to surface.

| facet | value |
| --- | --- |
| points | 12,000 embedded items (1 per row of the corpus) |
| source dim | 768-d sentence embeddings (transformer CLS-pooled) |
| projector | UMAP · `n_neighbors=15` · `min_dist=0.1` · cosine |
| structure | 1 megacluster + 2 satellites · 1 thin bridge · diffuse halo |
| voids | 2 interior holes (1 large donut, 1 minor) · sparse outlier tail |
| coverage | occupied area ≈ 61% of convex hull (concave reach) |
| density peak | megacluster core ≈ 18× the diffuse-halo density |
| reference | base embedding space (full 240k-item corpus KDE) for over/under |

The comparison views (VIZ 10–11) score this cloud against the **base embedding
space** — a KDE over the full 240k-item corpus the dataset was drawn from — to read
where it over- vs under-occupies the space.

## How to read the study

- **One page, eleven numbered views.** Each `section.opt` is a first-principles
  technique for one occupancy property, with a short rationale (`.why`), a live
  figure, a technique legend (`.leg`), and a `Reveals:` note naming the occupancy
  insight the viewer gains.
- **The views arc** from the continuous density field (1–3) through its modal and
  connective structure (4–6), to coverage and emptiness (7–9), and finally to the
  signed comparison against the reference space (10–11).
- **`_embed.js`** is the shared iframe-embed shim (carried verbatim from the
  sibling zicato study) so a single view can be inlined elsewhere as a live,
  theme-synced figure; `?bare=1` hides the chrome (`.opt-head` / `.leg` /
  `.reveal`) and shows only the figure.

## The techniques

| # | name | technique | occupancy property it reveals | data it needs |
| --- | --- | --- | --- | --- |
| VIZ 01 | Isodensity contour relief | kde contours | the continuous **shape** of occupancy — nested level-sets read the cloud like a topographic map; tight stacked rings mark the megacluster peak | a 2-D KDE over the point coordinates → contour level-sets at fixed density quantiles |
| VIZ 02 | Hexbin occupancy heatmap | hexbin | the same field as a **countable histogram** — each hex tints by its local point count, so density becomes discrete and the hottest cell is named | per-point coordinates → counts binned onto a hex lattice |
| VIZ 03 | Local-density rank dot field | graded scatter | both the **literal scatter** and its density grading in one mark — cores glow hot, peripheries cool | per-point coordinates + a per-point local-density estimate (kNN density) |
| VIZ 04 | Density-peak prominence ranking | peak ridge | the dataset's **modal structure** — how many distinct hotspots exist, ranked by topographic prominence, separating the one true peak from minor bumps | KDE field → local maxima + prominence (saddle-to-peak height) |
| VIZ 05 | kNN manifold graph | knn mesh | the cloud's **connectivity** — dense regions wire into tight mesh knots, sparse regions thin to single strands | per-point coordinates → k-nearest-neighbour edges |
| VIZ 06 | Bridge betweenness chokepoints | betweenness | the manifold's **chokepoints** — the few load-bearing edges whose removal would split the cloud into disconnected territories | the kNN graph → edge betweenness centrality |
| VIZ 07 | Alpha-hull reach boundary | alpha shape | the **concave frontier** — the true outer extent of the occupied region, carving in at peninsulas the convex hull would over-claim | per-point coordinates → α-shape boundary (+ convex hull for contrast) |
| VIZ 08 | Empty-disc void detection | void discs | the dataset's **interior holes** — the maximal point-free discs inscribed inside the occupied region, naming where coverage is missing | per-point coordinates → largest empty circles (Delaunay / medial-axis) |
| VIZ 09 | Nearest-neighbor sparsity field | nn rings | **local sparsity** as a continuous quantity — each point's nearest-neighbour distance, so dense cores wear tiny rings and the lonely tail wears wide ones | per-point coordinates → nearest-neighbour distance per point |
| VIZ 10 | Differential log-ratio density | surplus/deficit | **where the dataset over- vs under-occupies** the base space — signed log-ratio cells, hot where it crowds beyond the reference, cold where it abandons it | this dataset's KDE + the base-space KDE on a shared grid → `log(p_data/p_ref)` |
| VIZ 11 | Q-Q occupancy curve | q-q curve | whether the dataset is **more concentrated or more even** than the base — its density quantiles plotted against the reference's, bowing off the identity diagonal | sorted density quantiles of the dataset vs the base space |

## Design language

Built in the **zicato** design language (see
`../../../../zicato/docs/design/DESIGN-LANGUAGE.md`): Tufte data-ink line-art,
monospace-forward, one structural accent, theme-adaptive by construction. Every
mark reads its colour from a `--*` CSS token — there is **no hard-coded hex in any
figure** — so the 16-theme picker re-skins the page with no re-render. `good` /
`bad` are earned by direction (denser/better-covered, over- vs under-occupied),
never spent as decoration; the density ramps mix `accent → bad` for cool → hot and
toward `paper`/`panel` for low density. Every figure is fit-to-width
(`viewBox` + `width:100%` + `role="img"`), with no fixed pixel width and no
pan/zoom.

## Files

| file | what it is |
| --- | --- |
| `index.html` | the study — eleven views of one canonical projection, self-contained, default theme `monokai` |
| `_embed.js` | shared iframe-embed shim (`?bare=1` → figure-only) |
| `README.md` | this file |
