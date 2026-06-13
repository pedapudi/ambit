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

## Three dimensions

The eleven 2-D views all flatten the cloud to a plane. A companion section
(`3D 01`–`3D 05` plus one live figure) restores the third coordinate and asks the
same occupancy questions of a **3-D embedding volume**: how much of the reachable
volume the dataset fills, whether its apparent footprint hides anisotropy (broad in
X–Y, thin in Z), and whether an interior hole is a true 3-D **cavity** or a 2-D
coincidence.

### The 3-D canonical specimen — `AMBIT-12K-3D`

Every 3-D figure depicts the **same** baked specimen at the **same** fixed
isometric camera (azimuth 35°, elevation 22°): one dense tilted **megacluster**
core, two detached **satellites**, a thin **bridge** filament spanning depth, a
diffuse low-density **halo** thinning into the page, a large hollow **donut cavity**
flattened in Z, and a sparse **outlier tail**. Depth is read by painter’s
occlusion (back-to-front draw) + nearer = larger / full ink, farther = smaller /
faint — **no bounding 3-D box, no floor grid, no wireframe room**.

| # | name | technique | occupancy property it reveals | data it needs |
| --- | --- | --- | --- | --- |
| 3D 01 | Depth-cued projected 3D scatter | projected scatter | the cloud’s true 3-D **occupancy mass** once depth is restored — a dense megacluster core in front of two detached satellites, the halo thinning into the page, broad in X–Y but visibly thin in Z | baked 3-D point coordinates → fixed-camera projection + depth shading |
| 3D 02 | Orthographic XY / XZ / YZ occupancy triptych | ortho triptych | occupancy **anisotropy** — broad in X–Y, thin in Z; the big donut is a genuine 3-D **cavity** (a hole in two planes, a pancake in the third), not a coincidental 2-D gap | the same 3-D points → three axis-aligned orthographic drops (XY / XZ / YZ) |
| 3D 03 | Isometric voxel-occupancy lattice | iso voxels | the 3-D **occupancy histogram** — occupied voxels fill only **~22%** of the hull volume; the hottest cell is the megacluster core; the donut void is a hollow run punched clean through the lattice | 3-D points binned onto a coarse cubic lattice → occupied-voxel counts |
| 3D 04 | kNN manifold graph in projected 3-space | knn mesh | the manifold’s 3-D **wiring** — dense core knots mesh tightly, the single-strand halo thins out, and the few load-bearing **bridge edges** are the chokepoints whose cut would disconnect the knot | 3-D points → k-nearest-neighbour edges drawn in projected 3-space |
| 3D 05 | Radial shell occupancy profile | shell profile | whether the cloud is a filled ball or a **hollow rind** — occupancy peaks over the core, dips below the uniform expectation at the donut shell, recovers on the rind, then thins into the outlier tail | 3-D points → counts in concentric spherical shells from the centroid |

### The live showcase

| # | name | technique | what it adds | implementation |
| --- | --- | --- | --- | --- |
| 3D · live | Live depth-graded 3D point cloud (drag to rotate) | canvas · drag to rotate | occupancy as a **turnable solid** — orbit the *same* baked points and watch structure-from-motion resolve which clusters are detached in depth and that the donut is a genuine hollow cavity from every angle | dependency-free vanilla `<canvas>` + JS (no npm / no CDN / no external library); reads theme tokens via `getComputedStyle`, so it recolours with the picker like every static figure |

The five static 3-D figures are **fixed-camera SVG projections** (baked coordinates,
one canonical pose) in the same Tufte token-only style as the 2-D views; the live
figure is the one exception that animates, and it stays **fully self-contained** —
no build step, no external dependency.

## Resolution / isotropy

Crowding is **low resolution**: when items pile so close that cosine can no longer
separate them, the space has run out of room to tell them apart. A final section of
five diagnostics reads that crowding directly — see
[anisotropy & resolution](../../concepts/anisotropy-and-resolution.md) for the idea.
Unlike every view above, these are computed on the **original high-dimensional
embeddings** (not the 2-D projection), each scored against an **ideal-isotropic
reference**.

The diagnostic specimen is a deliberately anisotropic 768-d cloud — its summary
stats sit far from the isotropic ideal:

| diagnostic | dataset | isotropic reference |
| --- | --- | --- |
| items × dim | 12,000 × 768-d | — |
| mean random-pair cosine | 0.34 | ≈ 0 |
| effective rank | 47 / 768 | 768 |
| IsoScore | 0.27 | 1.0 |
| nearest-neighbor cosine margin | 0.041 | wide |
| hubness (top-1 k-occurrence) | 184 | ≈ uniform |

| # | name | what it measures |
| --- | --- | --- |
| ISO 01 | Random-pair cosine-similarity histogram | the **cone effect** — how far the random-pair cosine mass has shifted off 0 toward +1, the single most legible crowding signature |
| ISO 02 | Covariance eigenvalue scree & effective rank | whether variance spreads across many axes (isotropic, full rank) or collapses onto a few dominant directions — the structural cause of the cosine crowding |
| ISO 03 | IsoScore space-utilization gauge | the fraction of available dimensions the cloud actually exercises — a single 0–1 read of how much of the space is spent vs wasted |
| ISO 04 | Nearest-neighbor cosine margin distribution | how thin the gap is between a point's nearest and next-nearest neighbour — a narrow margin means neighbours are unresolvable |
| ISO 05 | Within- vs between-cluster cosine separation | whether genuine cluster structure survives the crowding — alignment within a cluster against uniformity between clusters |

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
| `index.html` | the study — eleven 2-D views of one canonical projection plus a six-figure 3-D section (five static + one live cloud), self-contained, default theme `monokai` |
| `_embed.js` | shared iframe-embed shim (`?bare=1` → figure-only) |
| `README.md` | this file |
