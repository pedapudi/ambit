"""Render a `Ctx` into a self-contained, theme-adaptive HTML report in the zicato
design language. Figures are token-colored static SVG (so a theme swap re-skins
with no re-render); the 16-theme CSS and the colour picker are vendored from the
study under `assets/`.

New figures are dropped into the `figures/` package as one module each, decorated
with `@figure`; `build_report` auto-loads them (fault-isolated) and orders cards
by their `order` field. This is the extension point the figure-fan-out targets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import metrics

ASSETS = Path(__file__).parent / "assets"

FIGURES = {}  # name -> callable(Ctx) -> dict(num,order,name,tech,why,svg,legend,reveal,cls)


def figure(fn):
    key = fn.__name__[4:] if fn.__name__.startswith("fig_") else fn.__name__
    FIGURES[key] = fn
    return fn


def _load_figures():
    import importlib
    import pkgutil
    import sys
    from . import figures as figpkg
    for mod in pkgutil.iter_modules(figpkg.__path__):
        try:
            importlib.import_module(f"{figpkg.__name__}.{mod.name}")
        except Exception as e:  # one bad figure must not break the whole report
            print(f"ambit: skipping figure {mod.name}: {e}", file=sys.stderr)


# ---------------------------------------------------------------- svg helpers
def _box(coords, w, h, pad=20):
    """Map data coords into the SVG box (y flipped), fit-to-width."""
    c = np.asarray(coords, float)
    mn = c.min(0)
    span = np.maximum(c.max(0) - mn, 1e-9)
    nx = (c[:, 0] - mn[0]) / span[0]
    ny = (c[:, 1] - mn[1]) / span[1]
    return np.column_stack([pad + nx * (w - 2 * pad), pad + (1 - ny) * (h - 2 * pad)])


def _svg(w, h, aria, body):
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="auto" '
            f'preserveAspectRatio="xMidYMid meet" role="img" aria-label="{aria}">{body}</svg>')


def _local_density(P, w, h, gx=48, gy=32):
    bx = np.clip((P[:, 0] / w * gx).astype(int), 0, gx - 1)
    by = np.clip((P[:, 1] / h * gy).astype(int), 0, gy - 1)
    grid = np.zeros((gx, gy))
    np.add.at(grid, (bx, by), 1)
    return grid[bx, by]


# ---------------------------------------------------------------- builtin figures
@figure
def fig_cloud(ctx):
    w, h = 760, 470
    P = _box(ctx.xy, w, h)
    dens = _local_density(P, w, h)
    hot = dens >= np.quantile(dens, 0.97)
    dots = []
    for i in range(len(P)):
        if hot[i]:
            dots.append(f'<circle cx="{P[i,0]:.1f}" cy="{P[i,1]:.1f}" r="1.7" '
                        f'fill="var(--accent)" fill-opacity="0.95"/>')
        else:
            dots.append(f'<circle cx="{P[i,0]:.1f}" cy="{P[i,1]:.1f}" r="1.2" '
                        f'fill="var(--ink-faint)" fill-opacity="0.4"/>')
    return {
        "num": "MAP 01", "order": 1, "name": "Projected density cloud", "tech": "pca · accumulation",
        "why": "The reservoir projected to 2-D; faint ink dots so density reads by accumulation, the densest cells carry the single accent.",
        "svg": _svg(w, h, "Projected embedding cloud; density reads by accumulation", "".join(dots)),
        "legend": '<span><i class="f"></i> point (accumulates)</span>'
                  '<span><i class="a"></i> densest cells (accent)</span>',
        "reveal": "<b>Reveals:</b> where the dataset concentrates in the projected space, and where it leaves the projection empty.",
        "cls": "",
    }


@figure
def fig_cos_hist(ctx):
    w, h, pad = 760, 320, 44
    cos = ctx.cos
    ref = metrics.isotropy_ref(ctx.scan.dim)
    lo, hi, bins = -0.3, 0.9, 48
    cnt, edges = np.histogram(cos, bins=bins, range=(lo, hi))
    top = int(cnt.max()) or 1
    base = h - pad
    bw = (w - 2 * pad) / bins

    def x_of(v):
        return pad + (v - lo) / (hi - lo) * (w - 2 * pad)

    body = [f'<line x1="{pad}" y1="{base}" x2="{w-pad}" y2="{base}" stroke="var(--rule)" stroke-width="1"/>']
    for k in range(bins):
        bh = (cnt[k] / top) * (h - 2 * pad)
        body.append(f'<rect x="{pad+k*bw:.1f}" y="{base-bh:.1f}" width="{bw*0.86:.1f}" '
                    f'height="{bh:.1f}" fill="var(--ink-faint)" fill-opacity="0.75"/>')
    body.append(f'<line x1="{x_of(0):.1f}" y1="{pad}" x2="{x_of(0):.1f}" y2="{base}" '
                f'stroke="var(--ink-faint)" stroke-dasharray="3 3"/>')
    body.append(f'<line x1="{x_of(cos.mean()):.1f}" y1="{pad-2}" x2="{x_of(cos.mean()):.1f}" '
                f'y2="{base}" stroke="var(--accent)" stroke-width="2"/>')
    for t in np.round(np.arange(-0.2, 0.81, 0.1), 1):
        body.append(f'<line x1="{x_of(t):.1f}" y1="{base}" x2="{x_of(t):.1f}" y2="{base+4}" stroke="var(--rule-soft)"/>')
        body.append(f'<text x="{x_of(t):.1f}" y="{base+15}" font-size="9.5" fill="var(--ink-faint)" '
                    f'text-anchor="middle">{t:+.1f}</text>')
    body.append(f'<text x="{x_of(cos.mean()):.1f}" y="{pad-5}" font-size="10" fill="var(--accent)" '
                f'text-anchor="middle">mean {cos.mean():+.3f}</text>')
    return {
        "num": "RES 01", "order": 90, "name": "Random-pair cosine distribution", "tech": "anisotropy fingerprint",
        "why": f"Cosine of {len(cos):,} random pairs against the isotropic reference (0 ± {ref:.3f}). Mass near 0 is isotropic; a shifted lobe is a crowded cone.",
        "svg": _svg(w, h, "Random-pair cosine similarity histogram", "".join(body)),
        "legend": '<span><i class="f"></i> pair counts</span>'
                  '<span><i class="dash"></i> isotropic ref (cos 0)</span>'
                  '<span><i class="a"></i> dataset mean</span>',
        "reveal": "<b>Reveals:</b> the anisotropy of the space — lower mean magnitude means higher resolution between items.",
        "cls": "fig-mid",
    }


@figure
def fig_scree(ctx):
    w, h, pad = 760, 320, 46
    eigs = ctx.eigs
    k = min(len(eigs), 80)
    e = eigs[:k]
    e = e / e.max()
    erank = metrics.effective_rank(eigs)
    base, top = h - pad, pad

    def x_of(i):
        return pad + (i / max(1, k - 1)) * (w - 2 * pad)

    def y_of(v):
        lv = np.log10(max(v, 1e-6))
        return base - (lv - (-6)) / (0 - (-6)) * (base - top)

    pts = " ".join(f"{x_of(i):.1f},{y_of(e[i]):.1f}" for i in range(k))
    body = [f'<polyline points="{pts}" fill="none" stroke="var(--ink)" stroke-width="1.3" vector-effect="non-scaling-stroke"/>']
    xr = x_of(erank)
    body.append(f'<line x1="{xr:.1f}" y1="{top}" x2="{xr:.1f}" y2="{base}" stroke="var(--accent)" stroke-width="2"/>')
    body.append(f'<text x="{xr+4:.1f}" y="{top+12}" font-size="10" fill="var(--accent)">effective rank {erank:.1f}</text>')
    for i in range(0, k + 1, 10):
        body.append(f'<line x1="{x_of(i):.1f}" y1="{base}" x2="{x_of(i):.1f}" y2="{base+4}" stroke="var(--rule-soft)"/>')
        body.append(f'<text x="{x_of(i):.1f}" y="{base+15}" font-size="9.5" fill="var(--ink-faint)" text-anchor="middle">{i}</text>')
    for p in range(0, -7, -2):
        yy = y_of(10.0 ** p)
        body.append(f'<line x1="{pad}" y1="{yy:.1f}" x2="{w-pad}" y2="{yy:.1f}" stroke="var(--rule-soft)" stroke-width="0.6"/>')
        body.append(f'<text x="{pad-6}" y="{yy+3:.1f}" font-size="9" fill="var(--ink-faint)" text-anchor="end">1e{p}</text>')
    return {
        "num": "RES 02", "order": 91, "name": "Covariance eigenvalue scree", "tech": "effective rank",
        "why": f"Normalized eigenvalues (log) over all {ctx.scan.n:,} items. A steep drop means variance is collapsed onto a few axes — low effective dimensionality.",
        "svg": _svg(w, h, "Covariance eigenvalue scree with effective rank", "".join(body)),
        "legend": '<span><i class="f"></i> eigenvalue (log)</span><span><i class="a"></i> effective rank</span>',
        "reveal": f"<b>Reveals:</b> dimensional collapse — here {ctx.scan.dim} nominal dims carry only ≈{erank:.0f} effective.",
        "cls": "fig-mid",
    }


# ---------------------------------------------------------------- report
def _facts(ctx):
    return [
        ("items × dims", f"{ctx.scan.n:,} × {ctx.scan.dim}"),
        ("mean L2 norm", f"{ctx.scan.norm_mean:.3f}"),
        ("mean pair cosine", f"{ctx.cos.mean():+.3f}"),
        ("effective rank", f"{metrics.effective_rank(ctx.eigs):.1f} / {ctx.scan.dim}"),
        ("dims for 90% var", f"{metrics.dims_for_variance(ctx.eigs, 0.9)} / {ctx.scan.dim}"),
    ]


def build_report(ctx, *, out=None, title="ambit — embedding-space occupancy", config=None) -> str:
    from .config import enabled
    _load_figures()
    style = (ASSETS / "theme.css").read_text(encoding="utf-8")
    picker = (ASSETS / "picker.js").read_text(encoding="utf-8")
    facts = "".join(f'<div class="kv"><span class="k">{k}</span><span class="v">{v}</span></div>'
                    for k, v in _facts(ctx))
    active = [fn for key, fn in FIGURES.items() if enabled(config, key)]
    metas = sorted((fn(ctx) for fn in active), key=lambda d: d.get("order", 999))
    cards = []
    for f in metas:
        cards.append(
            f'<section class="opt"><div class="opt-head">'
            f'<span class="num">{f["num"]}</span><span class="name">{f["name"]}</span>'
            f'<span class="tech">{f["tech"]}</span><span class="why">{f["why"]}</span></div>'
            f'<div class="opt-body"><figure class="{f.get("cls","")}">{f["svg"]}</figure>'
            f'<div class="leg">{f["legend"]}</div><div class="reveal">{f["reveal"]}</div></div></section>')
    html = (
        '<!DOCTYPE html>\n<html lang="en" data-theme="monokai">\n<head>\n'
        '<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<title>{title}</title>\n<style>{style}</style>\n</head>\n<body>\n'
        f'<header><h1>{title}</h1><span class="crumb">ambit / report</span>'
        '<span class="spacer"></span><span class="theme-pick-label">theme</span>'
        '<span id="theme-picker"></span></header>\n<main>\n'
        '<h2>occupancy</h2><div class="lede">How this dataset occupies its embedding space — '
        'where it concentrates, how much of the space it uses, and how distinct its items are.</div>\n'
        f'<div class="sample">{facts}</div>\n<div id="options">{"".join(cards)}</div>\n</main>\n'
        f'<footer>generated by ambit · {ctx.scan.source}</footer>\n'
        f'<script>{picker}</script>\n</body></html>\n')
    if out:
        Path(out).write_text(html, encoding="utf-8")
    return html
