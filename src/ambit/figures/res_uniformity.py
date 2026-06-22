"""Uniformity on the hypersphere (Wang & Isola, 2020) — how evenly the dataset
spreads over the sphere, read against the isotropic (uniform-on-sphere) reference.
Uniformity is `U = log E exp(-t·‖x−y‖²)` over random pairs (t=2); for unit vectors
‖x−y‖² = 2−2·cos, so it is a function of the random-pair cosines ambit already samples
(`ctx.cos`). More negative = more uniform = better occupancy.

This is the uniformity half of the alignment/uniformity decomposition the concept note
references. (Alignment — the other half — needs positive pairs and is out of ambit's
unsupervised scope.) In `--compare` / series mode the figure draws the trajectory
along the uniformity axis for the same items embedded two (or more) ways; standalone
it is the dataset vs the reference.
"""

from __future__ import annotations

import numpy as np

from ..render import figure, _svg
from .. import metrics


@figure
def fig_res_uniformity(ctx):
    w, h = 760, 300
    L, R = 80, 680
    axy = 188

    dim = int(getattr(ctx.scan, "dim", 0) or 0)
    u_data = metrics.uniformity_from_cos(ctx.cos)
    u_ref = metrics.uniformity_ref(dim) if dim else u_data

    # optional compare/series: a list of (label, uniformity) along the trajectory
    series = getattr(getattr(ctx, "cmp", None), "uniformity_series", None)
    pts = list(series) if series else [(getattr(ctx, "label_a", "this dataset"), u_data)]

    us = [u for _, u in pts] + [u_ref]
    lo, hi = min(us), max(us)
    rng = max(hi - lo, 0.5)
    lo -= 0.18 * rng
    hi += 0.18 * rng
    span = hi - lo

    def X(u):                                   # more uniform (more negative) -> right
        return R - (u - lo) / span * (R - L)

    body = []
    body.append(f'<text x="{L}" y="34" fill="var(--ink-soft)" font-size="12">'
                f'uniformity on the unit sphere · U = log E exp(−2‖x−y‖²) · {dim}-d</text>')
    body.append(f'<text x="{R}" y="34" fill="var(--ink-faint)" font-size="11" text-anchor="end">'
                f'more negative = more uniform</text>')

    # axis
    body.append(f'<line x1="{L}" y1="{axy}" x2="{R}" y2="{axy}" stroke="var(--rule)" stroke-width="1"/>')
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        u = lo + frac * span
        xx = X(u)
        body.append(f'<line x1="{xx:.1f}" y1="{axy}" x2="{xx:.1f}" y2="{axy+6}" stroke="var(--rule-soft)" stroke-width="0.8"/>')
        body.append(f'<text x="{xx:.1f}" y="{axy+19}" fill="var(--ink-faint)" font-size="9" '
                    f'text-anchor="middle" style="font-variant-numeric:tabular-nums">{u:.2f}</text>')
    body.append(f'<text x="{L}" y="{axy+40}" fill="var(--ink-faint)" font-size="9.5">less uniform · crowded</text>')
    body.append(f'<text x="{R}" y="{axy+40}" fill="var(--ink-faint)" font-size="9.5" text-anchor="end">more uniform · isotropic</text>')

    # isotropic reference
    rx = X(u_ref)
    body.append(f'<line x1="{rx:.1f}" y1="{axy-54}" x2="{rx:.1f}" y2="{axy+8}" '
                f'stroke="var(--ink-faint)" stroke-width="1" stroke-dasharray="3 3"/>')
    body.append(f'<text x="{rx:.1f}" y="{axy-58}" fill="var(--ink-faint)" font-size="10" '
                f'text-anchor="middle">isotropic reference</text>')

    # uniformity gap (dataset -> reference) for the single-dataset read
    if not series:
        dx = X(u_data)
        x0, x1 = min(dx, rx), max(dx, rx)
        body.append(f'<rect x="{x0:.1f}" y="{axy-10}" width="{x1-x0:.1f}" height="20" '
                    f'fill="color-mix(in srgb, var(--bad) 14%, transparent)"/>')
        body.append(f'<text x="{(x0+x1)/2:.1f}" y="{axy+74}" fill="var(--ink-faint)" font-size="9.5" '
                    f'text-anchor="middle" style="font-variant-numeric:tabular-nums">'
                    f'uniformity gap = {u_data - u_ref:+.2f}</text>')

    # dataset mark(s) / trajectory
    prev = None
    for k, (lab, u) in enumerate(pts):
        xx = X(u)
        if prev is not None:                    # trajectory arrow base->next
            body.append(f'<line x1="{prev:.1f}" y1="{axy-26}" x2="{xx:.1f}" y2="{axy-26}" '
                        f'stroke="var(--accent)" stroke-width="1.4" opacity="0.6"/>')
        body.append(f'<line x1="{xx:.1f}" y1="{axy-18}" x2="{xx:.1f}" y2="{axy+8}" '
                    f'stroke="var(--accent)" stroke-width="2.4"/>')
        body.append(f'<circle cx="{xx:.1f}" cy="{axy-18:.1f}" r="3.2" fill="var(--accent)"/>')
        dyl = -32 if k % 2 == 0 else 96
        body.append(f'<text x="{xx:.1f}" y="{axy+dyl if dyl>0 else axy+dyl}" fill="var(--accent)" '
                    f'font-size="10.5" font-weight="700" text-anchor="middle" '
                    f'style="font-variant-numeric:tabular-nums">{lab} · {u:.2f}</text>')
        prev = xx

    aria = (f"Wang–Isola uniformity of the embeddings on the unit sphere: U = {u_data:.2f} "
            f"(more negative = more uniform), against the isotropic {dim}-sphere reference "
            f"U = {u_ref:.2f}; the gap of {u_data - u_ref:+.2f} is how far the dataset sits from a "
            f"perfectly uniform spread.")
    return {
        "num": "RES 08", "order": 91.7, "name": "Uniformity on the hypersphere", "tech": "Wang–Isola uniformity",
        "why": f"How evenly the data spreads over the unit sphere (Wang–Isola uniformity, "
               f"U = log E exp(−2‖x−y‖²) over random pairs). Here U = {u_data:.2f} vs the isotropic "
               f"reference {u_ref:.2f} — a gap of {u_data - u_ref:+.2f}; more negative is more uniform. "
               f"This is the uniformity half of the alignment/uniformity pair (alignment needs positive "
               f"pairs and is outside ambit's unsupervised scope).",
        "svg": _svg(w, h, aria, "".join(body)),
        "legend": '<span><i class="a"></i> dataset uniformity</span>'
                  '<span><i class="dash"></i> isotropic reference</span>'
                  '<span><i class="r"></i> uniformity gap</span>',
        "reveal": "<b>Reveals:</b> occupancy of the whole sphere as a single number — how far the "
                  "dataset sits from a perfectly uniform spread. In compare mode the trajectory shows "
                  "which embedding is more uniform (further right is more uniform).",
        "cls": "fig-mid",
    }
