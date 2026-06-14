---
name: ambit-figures
description: >
  How ambit's report figures work and how to ADD or modify one — the figure
  contract (an @figure-decorated `fig_<slug>(ctx) -> dict` returning num/order/name/
  tech/why/svg/legend/reveal/cls/script), the auto-discovery registry in render.py,
  the SVG helpers (_box/_svg/_local_density), the Ctx fields a figure may read, the
  theme-token design language (no hard-coded hex), the full figure catalog by family
  (MAP/DEN/COV/TOP/CMP/RES/3D), and the step-by-step to add one. Use this when
  creating, editing, enabling/disabling, or debugging a report figure. For the
  surrounding pipeline see ambit-architecture; to run the report see ambit-cli.
version: 1.0.0
---

# ambit figures

The HTML report is a registry of **figures**. Each figure is a small module in
`src/ambit/figures/` that reads the shared `Ctx` and returns a self-describing dict
(SVG + metadata). `render.build_report()` auto-discovers them, calls each, sorts by
`order`, and assembles one self-contained, theme-adaptive HTML file.

## The figure contract

Each figure module exposes one function decorated with `@figure` (imported from
`ambit.render`):

```python
from ..render import figure, _box, _svg, _local_density   # helpers are optional
from .. import metrics
import numpy as np

@figure
def fig_<slug>(ctx) -> dict:
    """One-line description of what this visualizes."""
    return {
        "num":    "FAM 04",   # family code + number, e.g. "DEN 04", "3D 05", "RES 02b"
        "order":  4.0,        # ascending sort key in the report (float allows fine-tuning)
        "name":   "Human-readable title",
        "tech":   "comma, separated, technique tags",
        "why":    "Prose: what it shows and why it matters.",
        "svg":    _svg(W, H, "aria label", "".join(body)),  # the figure as an SVG string
        "legend": '<span><i class="a"></i> accent item</span>...',  # HTML legend fragment
        "reveal": "<b>Reveals:</b> what the viewer should take away.",  # HTML interpretive note
        "cls":    "",         # optional CSS class: "fig-narrow" | "fig-mid" (max-width)
        "script": "",         # optional JS string for interactive figures (collected once)
    }
```

`render.build_report()` calls every enabled figure with the `Ctx`, sorts the
returned dicts by `order`, wraps each in a `<section class="opt">` card
(header = num/name/tech/why; body = svg/legend/reveal), collects all `script`
fields into one trailing `<script>` block, and inlines the theme CSS/JS — producing
a single file with no external requests. The function name is the slug:
`fig_den_prom` → `"den_prom"`.

## The registry (auto-discovery)

In `render.py`:

```python
FIGURES = {}                                   # slug -> callable(Ctx) -> dict

def figure(fn):                                # the decorator
    key = fn.__name__[4:] if fn.__name__.startswith("fig_") else fn.__name__
    FIGURES[key] = fn
    return fn

def _load_figures():                           # imports every figures/*.py at build time
    for mod in pkgutil.iter_modules(figures.__path__):
        try: importlib.import_module(f"...figures.{mod.name}")
        except Exception as e: print(f"ambit: skipping figure {mod.name}: {e}", file=sys.stderr)
```

So **dropping a new `@figure`-decorated module into `figures/` registers it
automatically** — no central list to edit. A broken module is skipped with a stderr
note instead of failing the whole report. A handful of figures (`cloud`, `cos_hist`,
`scree`, `res_cumvar`) are defined directly in `render.py` rather than in `figures/`.

Visibility is decided by `config.DEFAULT_FIGURES` (slug → bool) and
`config.enabled(figures, key)`; `build_report` renders only enabled figures. Toggle
at runtime with `--config '{"figures": {"slug": true}}'` (see ambit-cli).
**`config.DEFAULT_FIGURES` + the `figures/` directory are the source of truth** for
which figures exist and which are on — prefer reading them over any static list.

## What a figure may read from `Ctx`

`scan` (`.n`, `.dim`, `.source`, `.approximate`), `xy` (m,2), `xyz` (m,3), `eigs`
(full-corpus eigenvalues), `cos` (random-pair sample), `knn_idx`/`knn_dist` (m,k),
`labels`/`labels_source`, `hub_skew`, and `ctx.es` (= `scan.sample`, the
L2-normalized reservoir `EmbeddingSet`: `.X`, `.n`, `.dim`). See ambit-architecture
for the full `Ctx`.

**Degrade gracefully.** `knn_*`, `labels`, `hub_skew`, and `xyz` may be `None`
(missing backend, no labels). A figure must return a valid dict with a terse
fallback message, never raise — e.g.:

```python
if ctx.knn_idx is None:
    return {"num": "RES 04", "order": 94, "name": "NN cosine margin", "tech": "knn",
            "why": "Needs a kNN backend.", "svg": _svg(W, H, "unavailable",
            "<text x='20' y='40'>no kNN backend</text>"),
            "legend": "", "reveal": "<b>Reveals:</b> nothing (no kNN).", "cls": ""}
```

## SVG helpers (`render.py`)

- `_box(coords, w, h, pad=20)` → map 2-D data coords into the SVG plot box
  (fit-to-width, y-axis flipped). Most 2-D figures start by `_box(ctx.xy, …)`.
- `_svg(w, h, aria, body)` → wrap body elements in a responsive
  `<svg viewBox=… preserveAspectRatio="xMidYMid meet" role="img" aria-label=…>`.
- `_local_density(P, w, h, gx=48, gy=32)` → coarse per-point density on a grid (for
  accenting the densest cells).

## Design language (non-negotiable for figures)

Built in the **zicato** language: Tufte data-ink line-art, monospace-forward, one
structural accent, theme-adaptive **by construction**.

- **No hard-coded hex.** Every mark reads its color from a CSS token:
  `fill="var(--accent)"`, `stroke="var(--ink-faint)"`, etc. The 16-theme picker
  (`assets/picker.js`) swaps `data-theme` on `<html>` and every figure re-skins live
  with **no re-render** — purely because all colors are tokens (`assets/theme.css`).
- **Tokens:** `--paper` `--panel` `--ink` `--ink-soft` `--ink-faint` `--rule`
  `--rule-soft` (structure); `--good` `--bad` `--caution` `--accent` (semantic).
- **`good`/`bad` are earned by direction** (denser/better-covered, over/under), never
  decoration. Density ramps mix `accent → bad` (cool→hot) and toward `paper`/`panel`
  for low density.
- **Fit-to-width**: `viewBox` + `width:100%` + `role="img"`; no fixed pixel width, no
  pan/zoom. Legend icons are `<i class="a|f|dash|good|bad">` spans styled by the
  theme CSS.
- **Interactive figures** (e.g. `d3_live`, `cov_sparsity`) return a `script` field;
  live/canvas figures read tokens at runtime via `getComputedStyle()` so they recolor
  with the picker too. Everything stays dependency-free (no npm/CDN).

The archived design study `docs/design/embedding-occupancy-study/` is the visual
reference — every technique as a static, theme-adaptive SVG.

## Figure catalog (by family)

Families: **MAP** projected occupancy · **DEN** density · **TOP** topology · **COV**
coverage · **CMP** comparison vs reference · **3D** three-dimensional · **RES**
resolution/isotropy. Enabled-by-default per `DEFAULT_FIGURES` (✓ = on). *(Verify
against `config.py`/`render.py`, which are authoritative.)*

| slug | num | family | shows | default |
|---|---|---|---|---|
| `cloud` | MAP 01 | MAP | projected 2-D density cloud; densest cells accented | — |
| `den_contour` | DEN 02 | DEN | isodensity contour relief (nested level-sets) | — |
| `den_hexbin` | DEN 03 | DEN | hexbin occupancy heatmap; peak cell named | — |
| `den_prom` | DEN 04 | DEN | density-peak prominence ranking (modal structure) | ✓ |
| `top_knn` | TOP 05 | TOP | kNN manifold graph; hub accented | — |
| `top_bridge` | TOP 06 | TOP | bridge betweenness chokepoints | — |
| `cov_hull` | COV 07 | COV | convex-hull reach + occupied/void split | — |
| `cov_void` | COV 08 | COV | largest empty interior discs (voids) | — |
| `cov_sparsity` | COV 09 | COV | per-point 1-NN sparsity field (slider at scale) | ✓ |
| `cmp_diff` | CMP 10 | CMP | log-ratio surplus/deficit vs isotropic reference | — |
| `cmp_qq` | CMP 11 | CMP | Q-Q occupancy curve vs reference | — |
| `d3_scatter` | 3D 01 | 3D | depth-cued static 3-D scatter | — |
| `d3_trip` | 3D 02 | 3D | orthographic XY/XZ/YZ triptych (anisotropy) | ✓ |
| `d3_voxel` | 3D 03 | 3D | isometric voxel occupancy lattice | — |
| `d3_mesh` | 3D 04 | 3D | kNN mesh drawn in projected 3-space | — |
| `d3_shell` | 3D 05 | 3D | radial shell occupancy (filled ball vs hollow rind) | ✓ |
| `d3_live` | 3D · live | 3D | interactive canvas cloud (drag/zoom, kNN-edge toggle) | ✓ |
| `cos_hist` | RES 01 | RES | random-pair cosine distribution vs isotropic spike | ✓ |
| `scree` | RES 02 | RES | covariance eigenvalue scree + effective rank | ✓ |
| `res_cumvar` | RES 02b | RES | cumulative variance vs isotropic diagonal | ✓ |
| `res_iso` | RES 03 | RES | IsoScore space-utilization gauge | ✓ |
| `res_margin` | RES 04 | RES | nearest-neighbor cosine margin (needs kNN) | ✓ |
| `res_wb` | RES 05 | RES | within- vs between-cluster cosine (needs labels) | ✓ |

## Adding a figure (step by step)

1. **Create** `src/ambit/figures/<slug>.py`.
2. **Implement** `@figure def fig_<slug>(ctx):` returning the contract dict
   (`num`/`order`/`name`/`tech`/`why`/`svg`/`legend`/`reveal`/`cls` + optional
   `script`). Use `_box`/`_svg` for the SVG; pull numbers from `ctx` and
   `ambit.metrics`. Color **only** with `var(--token)`.
3. **Degrade gracefully** for any optional `ctx` field you use (return a fallback
   dict if `knn_*`/`labels`/`xyz` is `None`).
4. **Pick a slug/num**: lowercase underscored slug (= module name, no `fig_`); num =
   family code + counter (decimals like `02b` allowed for subsections); set `order`
   to place it.
5. **Register visibility** in `config.DEFAULT_FIGURES`: add `"<slug>": True` (or
   `False` to ship hidden). Discovery is automatic; this only sets the default toggle.
6. **Render to check** — use the real pipeline, not a hand-built `Scan`:

   ```bash
   ambit report vecs.parquet --config '{"figures": {"<slug>": true}}' --out /tmp/r.html
   ```

   or in Python:

   ```python
   from ambit.scan import scan
   from ambit.pipeline import build_ctx
   from ambit.render import build_report
   ctx = build_ctx(scan("vecs.parquet"))
   open("/tmp/r.html", "w").write(build_report(ctx))   # _load_figures() picks up your module
   ```

7. **Open the HTML** in a browser and exercise the theme picker — confirm your
   figure recolors (i.e. you used tokens, not hex) and reads well across themes.
