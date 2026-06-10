# ambitus

Visualize the embeddings of a dataset to understand how much of an embedding
space it covers — surfacing density hotspots and sparse regions.

## Overview

`ambitus` projects high-dimensional embeddings down to 2D/3D and renders a
density view over the projected points, making it easy to see where a dataset
concentrates in the space and where it leaves the space empty.

The pipeline:

1. **Project** — reduce embeddings to 2D/3D (UMAP / t-SNE / PCA).
2. **Densify** — estimate density over the projection (KDE / hexbin / contours).
3. **Surface** — detect and rank hotspots and coverage gaps.

## Status

Early scaffolding.
