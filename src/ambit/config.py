"""Report configuration — which figures `ambit report` renders.

`FIGURES` maps each figure key (its module/function slug) to enabled/disabled.
Edit this object to taste, or pass a JSON object of the same shape to
`ambit report --config <file.json>` (it is merged over these defaults).

The enabled set below is the curated selection; the rest are implemented and
available but hidden — flip any to True to show it.
"""

FIGURES = {
    # ---- enabled: the curated set -------------------------------------------
    "den_prom":     True,   # DEN 04 · density-peak prominence
    "cov_sparsity": True,   # COV 09 · nearest-neighbor sparsity field
    "d3_trip":      True,   # 3D 02 · orthographic XY/XZ/YZ triptych
    "d3_live":      True,   # 3D · live · drag/zoom 3-D point cloud
    "cos_hist":     True,   # RES 01 · random-pair cosine histogram
    "scree":        True,   # RES 02 · covariance eigenvalue scree
    "res_wb":       True,   # RES 05 · within- vs between-cluster cosine

    # ---- hidden: implemented but disabled (flip to True to show) -------------
    "cloud":        False,  # MAP 01 · projected density cloud
    "den_contour":  False,  # DEN 02 · isodensity contour relief
    "den_hexbin":   False,  # DEN 03 · hexbin occupancy
    "top_knn":      False,  # TOP 05 · kNN manifold graph
    "top_bridge":   False,  # TOP 06 · bridge chokepoints
    "cov_hull":     False,  # COV 07 · reach boundary (hull)
    "cov_void":     False,  # COV 08 · void detection
    "cmp_diff":     False,  # CMP 10 · differential vs reference
    "cmp_qq":       False,  # CMP 11 · Q-Q occupancy curve
    "d3_scatter":   False,  # 3D 01 · depth-cued 3-D scatter
    "d3_voxel":     False,  # 3D 03 · isometric voxel occupancy
    "d3_mesh":      False,  # 3D 04 · kNN mesh in 3-space
    "d3_shell":     False,  # 3D 05 · radial shell occupancy
    "res_margin":   False,  # RES 04 · nearest-neighbor cosine margin
    "res_iso":      False,  # RES 03 · space-utilization gauge (off per review)
}


def enabled(cfg, key, default=True):
    """Whether a figure renders under `cfg` (an override dict over FIGURES)."""
    if cfg is not None and key in cfg:
        return bool(cfg[key])
    return bool(FIGURES.get(key, default))
