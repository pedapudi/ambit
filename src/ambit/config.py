"""ambit run configuration.

`Config` is the single object that describes a run — the figure enable/disable map
plus every scan / projection / device / kNN / embedding setting. Nothing is read
from environment variables: the CLI builds a `Config` from its flags, optionally
merges a `--config` JSON object on top, and passes it down.

A `--config` JSON may set any `Config` field, a nested `"figures"` object, or a
bare figure slug -> bool (treated as a figure toggle).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Union

# default figure visibility — the curated set is on, the rest implemented but hidden.
DEFAULT_FIGURES = {
    # enabled
    "den_prom":     True,   # DEN 04 · density-peak prominence
    "cov_sparsity": True,   # COV 09 · nearest-neighbor sparsity field
    "d3_trip":      True,   # 3D 02 · orthographic XY/XZ/YZ triptych
    "d3_live":      True,   # 3D · live · drag/zoom 3-D point cloud
    "d3_mesh_live": True,   # 3D 04 · live · drag/zoom kNN manifold mesh
    "d3_shell":     True,   # 3D 05 · radial shell occupancy
    "cos_hist":     True,   # RES 01 · random-pair cosine histogram (study ISO 01)
    "scree":        True,   # RES 02 · covariance eigenvalue scree
    "res_wb":       True,   # RES 05 · within- vs between-cluster cosine
    # hidden (flip to True to show)
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
    "d3_mesh":      False,  # 3D 04 · kNN mesh in 3-space (static; live version is d3_mesh_live)
    "res_margin":   False,  # RES 04 · nearest-neighbor cosine margin
    "res_iso":      False,  # RES 03 · space-utilization gauge
}

# kept as a module alias so `ambit.config.FIGURES` still resolves to the defaults
FIGURES = DEFAULT_FIGURES


@dataclass
class Config:
    # ---- ingestion ----
    embedding_col: Optional[str] = None
    id_col: Optional[str] = None
    label_col: Optional[str] = None
    text_col: str = "text"
    metric: str = "cosine"
    # ---- scan ----
    sample: int = 20_000
    pairs: int = 200_000
    batch_rows: int = 50_000
    approx: Optional[int] = None
    device: str = "cpu"                          # cpu | auto | cuda | mps | torch
    # ---- projection / topology / labeling ----
    projector: str = "pca"
    k: int = 10
    knn_backend: str = "auto"                    # auto | pynndescent | sklearn | brute | faiss
    clusters: Union[str, int, None] = "auto"     # "auto" | int | False/None
    # ---- embedding endpoint (no env vars) ----
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    embed_batch: int = 256
    # ---- render ----
    title: str = "ambit — embedding-space occupancy"
    figures: dict = field(default_factory=lambda: dict(DEFAULT_FIGURES))

    def merge(self, overrides) -> "Config":
        """Return a copy with a JSON/dict of overrides applied. Known keys set the
        matching field; a 'figures' object or any other key updates the figure map."""
        fields = set(self.__dataclass_fields__)
        data = asdict(self)
        figs = dict(self.figures)
        for k, v in (overrides or {}).items():
            if k == "figures":
                figs.update(v or {})
            elif k in fields:
                data[k] = v
            else:
                figs[k] = v
        data["figures"] = figs
        return Config(**data)


def enabled(figures, key, default: bool = True) -> bool:
    """Whether a figure renders under a figure map (the dict carries the defaults)."""
    return bool((figures if figures is not None else DEFAULT_FIGURES).get(key, default))
