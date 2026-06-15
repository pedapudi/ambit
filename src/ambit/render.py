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


_ISO_CARD = (
    '<span class="hc-card" role="tooltip">'
    '<span class="hc-h">IsoScore = {score:.3f}</span>'
    '<span class="hc-sub">uniformity of embedding-space use · Rudman et&nbsp;al. 2022</span>'
    '<p><b>0</b> = all variance on a single axis (a degenerate line); '
    '<b>1</b> = variance spread equally over every dimension (a perfect sphere). '
    'Real text embeddings sit low.</p>'
    '<div class="hc-math">'
    '<div class="hc-intro">from the d covariance eigenvalues λ (variance along each principal axis):</div>'
    '<div class="hc-eq">v = √d · λ / ‖λ‖₂ &nbsp;<span class="hc-note">— normalize: isotropic ⇒ v = (1,…,1)</span></div>'
    '<div class="hc-eq">δ = ‖v − 1‖₂ / √(2(d − √d)) &nbsp;<span class="hc-note">— isotropy defect, 0…1</span></div>'
    '<div class="hc-eq">n = d − δ²·(d − √d) &nbsp;<span class="hc-note">— dims isotropically used</span></div>'
    '<div class="hc-eq">IsoScore = (n² − d) / (d² − d)</div>'
    '</div>'
    '<div class="hc-foot">this dataset · d = {d} · δ = {defect:.2f} · n = {n_iso:.0f}</div>'
    '</span>'
)

_ISO_POS = {
    "br":  ("hc--up", "right:5%;top:57%;"),                       # in-chart lower-right, opens up
    "top": ("", "left:50%;top:4%;transform:translateX(-50%);"),  # gauge header, opens down
}


def _isoscore_hc(eigs, *, pos="br", big=False):
    """Isoscore readout + a design-language hovercard explaining the metric and its
    exact formula. Returns one HTML <span> to drop into a `.hc-fig` wrapper."""
    score, defect, n_iso, d = metrics.isoscore_parts(eigs)
    up, style = _ISO_POS[pos]
    card = _ISO_CARD.format(score=score, defect=defect, n_iso=n_iso, d=d)
    cls = ("hc-big " if big else "") + up
    return (f'<span class="hc {cls}" tabindex="0" role="button" '
            f'aria-label="IsoScore {score:.3f} — uniformity of embedding-space use; activate for the formula." '
            f'style="{style}">isoscore = {score:.2f}<i class="hc-i" aria-hidden="true">i</i>{card}</span>')


def _hc_fig(svg, hc):
    """Wrap an SVG with an absolutely-positioned hovercard overlay."""
    return f'<div class="hc-fig">{svg}{hc}</div>'


# Explicit display order, by figure registry key. Listed figures render in this
# sequence; any enabled figure not listed sorts after, by its own `order`.
_DISPLAY_ORDER = [
    "res_cmap",    # RES 07 · local crowding cloud
    "d3_trip",     # 3D 02 · orthographic triptych
    "res_field",   # RES 06 · local concentration field
    "res_margin",  # RES 04 · nearest-neighbor cosine margin
    "res_wb",      # RES 05 · within- vs between-cluster cosine
    "cos_hist",    # RES 01 · random-pair cosine distribution
    "res_cumvar",  # RES 02b · cumulative variance
    "scree",       # RES 02 · covariance eigenvalue scree
    "d3_shell",    # 3D 05 · radial shell occupancy
    "den_prom",    # DEN 04 · density-peak prominence
    "cov_sparsity",# COV 09 · nearest-neighbor sparsity field
]


# ---- "how to read" interpretation hovercards, keyed by figure registry key ----
# One per figure, in the same popover language as the IsoScore explainer. Centralized
# here (not in each figure module) so the guidance reads as one consistent voice; any
# figure without an entry falls back to its own why/reveal text.
_INTERP = {
    "cos_hist": ("Random-pair cosine", "global anisotropy fingerprint",
        "<p>Cosine between random pairs of items. In a healthy space unrelated pairs are near-orthogonal, "
        "so the mass sits in a narrow spike at <b>0</b>.</p>"
        "<p>Mass shifted right (toward +1) means even unrelated items look alike — the space is anisotropic, "
        "or crowded. Compare the peak to the dashed isotropic reference (centered at 0, width ≈ 1/&#8730;d).</p>"
        '<div class="hc-foot">A small positive mean is normal for real text embeddings; what matters is how '
        "far right of the reference it sits.</div>"),
    "scree": ("Eigenvalue scree", "dimensional collapse",
        "<p>Covariance eigenvalues, largest first, on a log axis — the variance carried by each principal "
        "direction.</p>"
        '<p>A <span class="hc-good">gentle slope</span> means variance is spread over many axes (high '
        'effective rank, healthy). A <span class="hc-bad">steep cliff</span> in the first few means variance '
        "has collapsed onto a handful of directions.</p>"
        '<div class="hc-foot">Effective rank (annotated) is the continuous count of dimensions actually in '
        "use.</div>"),
    "res_cumvar": ("Cumulative variance", "how evenly the space is used · carries the IsoScore",
        "<p>Running fraction of total variance against the number of dimensions included.</p>"
        '<p>A curve that hugs the <span class="hc-good">diagonal</span> means variance is spread evenly '
        '(isotropic). A curve that <span class="hc-bad">shoots up early</span> means a few dimensions carry '
        "almost everything. The &ldquo;dims for 90%&rdquo; marker and the IsoScore badge quantify it — hover "
        "the IsoScore itself for its formula.</p>"),
    "res_margin": ("Nearest-neighbor margin", "retrieval decisiveness",
        "<p>Per item, the cosine gap between its #1 and #2 nearest neighbor.</p>"
        '<p>Mass piled at the <span class="hc-bad">floor (margin &#8594; 0)</span> means near-ties, where the '
        "index can barely separate the best match from the runner-up — brittle retrieval. A "
        '<span class="hc-good">fat right tail</span> means many items have a decisively separated best '
        "neighbor.</p>"
        '<div class="hc-foot">Compare the accent median rule to the dashed isotropic-reference median.</div>'),
    "res_wb": ("Within vs between cosine", "are the groups geometrically separable?",
        "<p>Two distributions: cosine of pairs <em>within</em> the same group vs <em>between</em> groups.</p>"
        '<p><span class="hc-good">Clean separation</span> (within shifted right of between, little overlap) '
        'means groups occupy distinct directions. <span class="hc-bad">Heavy overlap</span> means the labels '
        "are not separable by geometry alone.</p>"),
    "res_field": ("Local concentration field", "localized crowding, as a distribution",
        "<p>For each item, the mean cosine to its k nearest neighbors — how tightly its local neighborhood "
        "collapses — drawn against a synthetic isotropic reference (the faint ghost).</p>"
        '<p><span class="hc-look">Read the shape:</span> one mode at the floor near the reference = roomy; the '
        "whole mode shifted far right = globally crowded (a union of cones); a separated high mode across a "
        "valley = a crowded pocket (one bump each).</p>"
        '<div class="hc-foot">The gap from the isotropic reference to the dataset bulk is the global '
        "crowding.</div>"),
    "res_cmap": ("Local crowding cloud", "read color and shape, not position",
        '<p>The reservoir projected to 3-D, recolored and reshaped by native crowding: '
        '<span class="hc-good">flat green dots</span> are the least-crowded items, '
        '<span class="hc-bad">red pyramids</span> the most (taller = more crowded).</p>'
        '<p><span class="hc-look">Position does not show crowding.</span> It is a PCA projection of the top-3 '
        "variance directions; about 98% of each item's true 768-d neighbors do not survive it, so crowded and "
        "open points look equally clumped. Read crowding from <b>color and shape only</b>.</p>"
        "<p>What the layout <em>does</em> show: whether the pyramids cluster in one region (a localized "
        "pocket) or spread evenly (global crowding). Toggle <b>kNN edges</b> to wire each point to its true "
        "neighbors — they fan across the projection rather than staying local, which is the same projection "
        "limit made visible.</p>"
        '<div class="hc-foot">Color is relative rank — in a globally crowded space the green dots are still '
        "cramped, just less than the median. RES 06 shows the absolute level.</div>"),
    "den_prom": ("Density-peak prominence", "where the data piles up",
        "<p>Hotspots in the projected cloud, scored by how far they rise above their surroundings.</p>"
        "<p>Tall, isolated peaks are genuine concentrations; a flat field is evenly spread. Prominence "
        "separates a real peak from a gentle rise.</p>"),
    "cov_sparsity": ("Nearest-neighbor sparsity", "open space vs packing",
        "<p>Each point is a ring sized by the distance to its nearest neighbor.</p>"
        '<p><span class="hc-good">Large rings</span> (the isolated decile, highlighted) mark open space and '
        "cleanly separated points; tiny rings mark tight packing. Drag the slider to thin a crowded field.</p>"),
    "d3_live": ("Live 3-D cloud", "the occupied volume, by cluster",
        "<p>The projected reservoir as a turnable solid, each point colored by its cluster. Drag to rotate, "
        "scroll or pinch to zoom; it auto-spins when idle.</p>"
        '<p>Look at how clusters sit in the volume and whether the cloud is <span class="hc-bad">thin in one '
        "axis</span> (anisotropy you can see). Toggle kNN edges to overlay the neighbor graph; the lower-left "
        "gnomon tracks orientation.</p>"
        '<div class="hc-foot">A global-variance projection — like RES 07, position shows the gross shape, not '
        "fine local structure.</div>"),
    "d3_trip": ("Orthographic triptych", "the cloud from three axes",
        "<p>The same 3-D cloud viewed straight down the X, Y, and Z axes.</p>"
        '<p>Compare the three silhouettes: a cloud that is <span class="hc-bad">flat in one panel</span> '
        "occupies fewer effective dimensions there. Round in all three is more isotropic occupancy.</p>"),
    "d3_shell": ("Radial shell occupancy", "how the cloud fills outward",
        "<p>Concentric shells from the centroid, shaded by how many points fall in each.</p>"
        "<p>Most embedding clouds are a thin spherical shell (norms cluster at one radius) — even shading all "
        "the way around. Lopsided shells indicate directional structure.</p>"),
}


def _interpret_hc(key, meta):
    """A header 'how to read' hovercard for one figure — same popover as the IsoScore
    explainer, opened from an info icon in the card header. Falls back to the figure's
    own why/reveal when no curated entry exists."""
    num = meta.get("num", "")
    spec = _INTERP.get(key)
    if spec:
        h, sub, body = spec
    else:
        h, sub = meta.get("name", "How to read"), meta.get("tech", "")
        rev = meta.get("reveal", "")
        body = f'<p>{meta.get("why", "")}</p>' + (f"<p>{rev}</p>" if rev else "")
    return (f'<span class="hc hc-head" tabindex="0" role="button" '
            f'aria-label="How to read {num} — activate for guidance on interpreting this figure">'
            f'<i class="hc-i" aria-hidden="true">i</i>'
            f'<span class="hc-card" role="tooltip">'
            f'<span class="hc-h">{h}</span><span class="hc-sub">{sub}</span>{body}</span></span>')


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
    # Mirrors the study's ISO 01: a smooth random-pair cosine *density* over the
    # full [-1, +1] axis (0 dead-centre = isotropic), with the analytic isotropic
    # d-sphere reference drawn as a razor spike at 0, the anisotropy-gap wedge
    # between 0 and the dataset mean, and an accent mean tick. An isotropic space
    # sits symmetric on 0; a crowded cone shifts its whole mass toward +1.
    w, h = 760, 470
    L, R, T, B = 70, 720, 80, 392
    XLO, XHI = -1.0, 1.0

    cos = np.asarray(ctx.cos, float)
    n = int(cos.size)
    mean = float(cos.mean())
    sd = float(cos.std())
    tail = float(np.quantile(cos, 0.99))
    dim = int(getattr(ctx.scan, "dim", 0) or 0)
    sd_ref = float(metrics.isotropy_ref(dim)) if dim else 0.02

    def X(v):
        return L + (v - XLO) / (XHI - XLO) * (R - L)

    # ---- numpy-only KDE: fine histogram smoothed by a Gaussian kernel ----
    nb = 400
    edges = np.linspace(XLO, XHI, nb + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(cos, bins=edges, density=True)
    dx = centers[1] - centers[0]
    bw = max(sd * n ** (-0.2), 2.5 * dx)                 # Scott's rule, floored to stay smooth
    ksig = bw / dx
    half = int(np.ceil(ksig * 4))
    kx = np.arange(-half, half + 1)
    kern = np.exp(-0.5 * (kx / ksig) ** 2)
    kern = kern / kern.sum()
    dens = np.convolve(counts, kern, mode="same")
    dmax = float(dens.max()) or 1.0

    DATA_TOP = B - 0.80 * (B - T)                        # dataset peak reaches 80% height
    REF_TOP = B - 0.98 * (B - T)                         # reference spike a touch taller

    def Yd(d):
        return B - (d / dmax) * (B - DATA_TOP)

    def Yr(r):
        return B - r * (B - REF_TOP)

    # density y at the mean (for the accent tick + circle)
    dens_mean = float(np.interp(mean, centers, dens))
    mx = X(mean)
    my = Yd(dens_mean)

    body = []

    # vertical gridlines every 0.2
    for g in np.arange(-1.0, 1.0001, 0.2):
        body.append(f'<line x1="{X(g):.1f}" y1="{T}" x2="{X(g):.1f}" y2="{B}" '
                    f'stroke="var(--rule-soft)" stroke-width="0.7"/>')

    # light crowding fill under the whole dataset curve (tinted toward +1)
    curve = [(X(centers[i]), Yd(dens[i])) for i in range(nb)]
    fill_d = (f"M {X(XLO):.1f} {B:.1f} "
              + " ".join(f"L {x:.2f} {y:.2f}" for x, y in curve)
              + f" L {X(XHI):.1f} {B:.1f} Z")
    body.append(f'<path d="{fill_d}" fill="color-mix(in srgb, var(--bad) 11%, transparent)" stroke="none"/>')

    # anisotropy-gap wedge: area under the curve between cos=0 and the mean
    a, b = (0.0, mean) if mean >= 0 else (mean, 0.0)
    idx = np.where((centers >= a) & (centers <= b))[0]
    if idx.size >= 2:
        wc = [(X(centers[i]), Yd(dens[i])) for i in idx]
        wedge = (f"M {wc[0][0]:.1f} {B:.1f} "
                 + " ".join(f"L {x:.2f} {y:.2f}" for x, y in wc)
                 + f" L {wc[-1][0]:.1f} {B:.1f} Z")
        body.append(f'<path d="{wedge}" fill="color-mix(in srgb, var(--bad) 24%, transparent)" stroke="none"/>')

    # isotropic d-sphere reference: analytic N(0, 1/√dim) razor spike at 0
    gref = np.linspace(-0.28, 0.28, 225)
    rref = np.exp(-0.5 * (gref / sd_ref) ** 2)
    ref_pts = " ".join(f"{X(gref[i]):.2f} {Yr(rref[i]):.2f}" for i in range(gref.size))
    body.append(f'<polyline points="{ref_pts}" fill="none" stroke="var(--ink-faint)" '
                f'stroke-width="1" stroke-dasharray="3 3" vector-effect="non-scaling-stroke"/>')

    # cos = 0 axis (faint dashed rule up the spike)
    body.append(f'<line x1="{X(0):.1f}" y1="{Yr(1.0):.1f}" x2="{X(0):.1f}" y2="{B}" '
                f'stroke="var(--ink-faint)" stroke-width="0.9" stroke-dasharray="3 3"/>')

    # dataset density curve (the one accent)
    line_d = " ".join(f"{x:.2f} {y:.2f}" for x, y in curve)
    body.append(f'<polyline points="{line_d}" fill="none" stroke="var(--accent)" '
                f'stroke-width="1.4" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>')

    # baseline
    body.append(f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="var(--rule)" stroke-width="1"/>')

    # mean tick + circle on the curve, with annotation to the right
    body.append(f'<line x1="{mx:.1f}" y1="{B}" x2="{mx:.1f}" y2="{my:.1f}" '
                f'stroke="var(--accent)" stroke-width="2.2"/>')
    body.append(f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="2.8" fill="var(--accent)"/>')
    lab_anchor = "start" if mx < R - 170 else "end"
    lab_dx = 8 if lab_anchor == "start" else -8
    body.append(f'<text x="{mx + lab_dx:.1f}" y="{my - 6:.1f}" fill="var(--accent)" font-size="11" '
                f'font-weight="700" text-anchor="{lab_anchor}" '
                f'style="font-variant-numeric:tabular-nums">mean cos = {mean:+.2f}</text>')
    body.append(f'<text x="{mx + lab_dx:.1f}" y="{my + 8:.1f}" fill="var(--ink-faint)" font-size="9" '
                f'text-anchor="{lab_anchor}" style="font-variant-numeric:tabular-nums">'
                f'sd ≈ {sd:.2f} · right tail → {tail:.2f}</text>')

    # isotropic reference label (left gutter is empty for a positive cone) + leader
    body.append('<text x="300" y="100" fill="var(--ink-faint)" font-size="10" '
                'text-anchor="end">isotropic d-sphere reference</text>')
    body.append(f'<text x="300" y="113" fill="var(--ink-faint)" font-size="9" text-anchor="end" '
                f'style="font-variant-numeric:tabular-nums">N(0, 1/√{dim}) · sd ≈ {sd_ref:.3f}</text>')
    body.append(f'<line x1="308" y1="103" x2="{X(0)-3:.1f}" y2="{Yr(1.0)+2:.1f}" '
                f'stroke="var(--ink-faint)" stroke-width="0.7" stroke-dasharray="2 2"/>')

    # x-axis ticks (-1.0 … +1.0) + minor ticks + label
    for g in np.arange(-1.0, 1.0001, 0.2):
        body.append(f'<line x1="{X(g):.1f}" y1="{B}" x2="{X(g):.1f}" y2="{B+7}" '
                    f'stroke="var(--rule)" stroke-width="1"/>')
        body.append(f'<text x="{X(g):.1f}" y="{B+20}" fill="var(--ink-faint)" font-size="10.5" '
                    f'text-anchor="middle" style="font-variant-numeric:tabular-nums">'
                    f'{("%+.1f" % g) if abs(g) > 1e-9 else "0"}</text>')
    for g in np.arange(-0.9, 1.0, 0.2):
        body.append(f'<line x1="{X(g):.1f}" y1="{B}" x2="{X(g):.1f}" y2="{B+4}" '
                    f'stroke="var(--rule-soft)" stroke-width="0.8"/>')
    body.append(f'<text x="{(L+R)/2:.1f}" y="{B+38}" fill="var(--ink-faint)" font-size="9.5" '
                f'text-anchor="middle">cosine similarity</text>')

    # title row + verdict (accent = the dataset's own signal)
    if mean <= 3 * sd_ref:
        verdict = f"near-isotropic · mean cos = {mean:+.2f}"
    elif mean <= 0.15:
        verdict = f"mild anisotropy · mean cos = {mean:+.2f}"
    else:
        verdict = f"anisotropic cone · mean cos = {mean:+.2f}"
    body.insert(0, f'<text x="{L}" y="26" fill="var(--ink-soft)" font-size="11" text-anchor="start" '
                   f'style="font-variant-numeric:tabular-nums">random-pair cosine density · '
                   f'{dim}-d · ~{n//1000}k pairs</text>')
    body.insert(1, f'<text x="{R}" y="26" fill="var(--accent)" font-size="11" font-weight="700" '
                   f'text-anchor="end" style="font-variant-numeric:tabular-nums">{verdict}</text>')

    aria = (f"Random-pair cosine-similarity density over {n:,} pairs of the "
            f"{dim}-dimensional embeddings, on a full -1 to +1 cosine axis. The dataset "
            f"density is an accent hump centred at cosine {mean:+.2f} (sd {sd:.2f}, right "
            f"tail to {tail:.2f}); a tall dashed isotropic d-sphere reference spike sits at "
            f"cosine 0 with sd {sd_ref:.3f}. The shaded wedge between cosine 0 and the accent "
            f"mean tick is the anisotropy gap — the further right the mass, the more crowded "
            f"and less resolvable random items are.")
    return {
        "num": "RES 01", "order": 90, "name": "Random-pair cosine distribution", "tech": "cosine density",
        "why": f"Cosine of {n:,} random pairs as a smooth density on the full -1…+1 axis. An "
               f"isotropic space sits symmetric on 0 (analytic ref N(0, 1/√{dim}), sd ≈ {sd_ref:.3f}); "
               f"this dataset's mass sits at mean cos {mean:+.2f} — the shaded wedge is the anisotropy gap.",
        "svg": _svg(w, h, aria, "".join(body)),
        "legend": '<span><i class="a"></i> dataset random-pair cosine density</span>'
                  '<span><i class="dash"></i> isotropic d-sphere reference — N(0, 1/√d)</span>'
                  '<span><i class="dash"></i> cos = 0 axis</span>'
                  '<span><i class="a"></i> accent tick — dataset mean cosine</span>',
        "reveal": "<b>Reveals:</b> <b>anisotropy</b> / the cone effect — how far the dataset's "
                  "random-pair mass sits to the right of the isotropic reference at 0. The wider that "
                  "gap, the more every random pair looks alike and the less items are resolvable.",
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
        "why": f"Each principal axis's share of the variance, largest first, on a log scale, over all {ctx.scan.n:,} items. The shape is the read: a gentle, gradual slope means variance is spread across many axes (the space is fully used); a steep cliff in the first few means it has collapsed onto a handful of directions — high nominal dimensionality but low effective rank, the geometry behind crowding.",
        "svg": _svg(w, h, "Covariance eigenvalue scree with effective rank", "".join(body)),
        "legend": '<span><i class="f"></i> eigenvalue (log)</span><span><i class="a"></i> effective rank</span>',
        "reveal": f"<b>Reveals:</b> dimensional collapse — here {ctx.scan.dim} nominal dims carry only ≈{erank:.0f} effective.",
        "cls": "fig-mid",
    }


# ---------------------------------------------------------------- report
def _facts(ctx):
    items = f"{ctx.scan.n:,} × {ctx.scan.dim}"
    if getattr(ctx.scan, "approximate", False):
        items += f" (≈{ctx.scan.scanned:,} sampled)"
    f = [
        ("items × dims", items),
        ("mean L2 norm", f"{ctx.scan.norm_mean:.3f}"),
        ("mean pair cosine", f"{ctx.cos.mean():+.3f}"),
        ("isoscore", f"{metrics.isoscore(ctx.eigs):.3f}"),
        ("effective rank", f"{metrics.effective_rank(ctx.eigs):.1f} / {ctx.scan.dim}"),
        ("dims for 90% var", f"{metrics.dims_for_variance(ctx.eigs, 0.9)} / {ctx.scan.dim}"),
    ]
    if ctx.labels is not None:
        ng = len(set(map(str, ctx.labels.tolist())))
        f.append(("groups", f"{ng} · {ctx.labels_source or 'labeled'}"))
    if getattr(ctx, "hub_skew", None) is not None:
        f.append(("hub skew", f"{ctx.hub_skew:.1f}"))
    return f


def build_report(ctx, *, out=None, title="ambit — embedding-space occupancy", config=None) -> str:
    from .config import Config, DEFAULT_FIGURES, enabled
    figures = config.figures if isinstance(config, Config) else (config if config is not None else DEFAULT_FIGURES)
    _load_figures()
    style = (ASSETS / "theme.css").read_text(encoding="utf-8")
    picker = (ASSETS / "picker.js").read_text(encoding="utf-8")
    facts = "".join(f'<div class="kv"><span class="k">{k}</span><span class="v">{v}</span></div>'
                    for k, v in _facts(ctx))
    active = [(key, fn) for key, fn in FIGURES.items() if enabled(figures, key)]
    _rank = {k: i for i, k in enumerate(_DISPLAY_ORDER)}
    metas = sorted((dict(fn(ctx), _key=key) for key, fn in active),
                   key=lambda d: _rank.get(d.get("_key"), 10_000 + d.get("order", 999)))
    cards = []
    for f in metas:
        hc = _interpret_hc(f.get("_key", ""), f)
        cards.append(
            f'<section class="opt"><div class="opt-head">'
            f'<span class="num">{f["num"]}</span><span class="name">{f["name"]}</span>{hc}'
            f'<span class="tech">{f["tech"]}</span></div>'
            f'<div class="opt-body"><figure class="{f.get("cls","")}">{f["svg"]}</figure>'
            f'<div class="leg">{f["legend"]}</div><div class="reveal">{f["reveal"]}</div></div></section>')
    figscripts = "".join(f.get("script", "") for f in metas)
    figscript_block = f'<script>{figscripts}</script>\n' if figscripts else ''
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
        f'{figscript_block}<script>{picker}</script>\n</body></html>\n')
    if out:
        Path(out).write_text(html, encoding="utf-8")
    return html
