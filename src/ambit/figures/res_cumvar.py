"""RES 02b — Cumulative variance & dimensional concentration. The covariance
eigenvalues, *integrated*: cumulative fraction of variance vs dimension, drawn
against the isotropic y = x diagonal (what a full-rank, equal-variance space would
give). The gap above the diagonal is the concentration — the further the curve bows
up, the fewer dimensions actually carry the variance. This is the companion to the
scree (RES 02): it reconciles the single-number summaries the scree leaves implicit
(effective rank vs participation ratio vs dims-for-90%), which can disagree wildly
on a heavy-tailed spectrum.
"""

from __future__ import annotations

import numpy as np

from ..render import figure, _svg, _isoscore_hc, _hc_fig
from .. import metrics


@figure
def fig_res_cumvar(ctx):
    w, h = 760, 470
    L, R, T, B = 70, 718, 84, 392

    eigs = np.asarray(ctx.eigs, float)
    eigs = eigs[eigs > 0]
    dim = int(getattr(ctx.scan, "dim", eigs.size) or eigs.size)
    n = int(eigs.size)
    if n == 0 or dim <= 1:
        note = ('<text x="380" y="235" font-size="12" fill="var(--ink-faint)" '
                'text-anchor="middle">no covariance spectrum available</text>')
        return {"num": "RES 02b", "order": 91.5, "name": "Cumulative variance & dimensional concentration",
                "tech": "spectral concentration",
                "why": "Cumulative variance vs dimension against the isotropic diagonal.",
                "svg": _svg(w, h, "Cumulative variance unavailable: no spectrum.", note),
                "legend": '<span class="leg">no spectrum</span>',
                "reveal": "<b>Reveals:</b> dimensional concentration once a spectrum is available.",
                "cls": "fig-mid"}

    total = float(eigs.sum()) or 1.0
    cum = np.cumsum(eigs) / total                       # cumulative variance, descending sort
    cumfull = np.ones(dim)                               # pad rank-deficient tail at 1.0
    cumfull[:min(n, dim)] = cum[:dim]

    erank = float(metrics.effective_rank(eigs))
    pratio = float(metrics.participation_ratio(eigs))
    d50 = int(metrics.dims_for_variance(eigs, 0.5))
    d90 = int(metrics.dims_for_variance(eigs, 0.9))

    XHI = float(dim)

    def X(d):
        return L + (float(d) / XHI) * (R - L)

    def Y(f):
        return B - float(np.clip(f, 0.0, 1.0)) * (B - T)

    body = []

    # ---- gridlines (quarter dims, quarter variance) ----
    for q in (0.25, 0.5, 0.75):
        body.append(f'<line x1="{X(q*dim):.1f}" y1="{T}" x2="{X(q*dim):.1f}" y2="{B}" '
                    f'stroke="var(--rule-soft)" stroke-width="0.6"/>')
        body.append(f'<line x1="{L}" y1="{Y(q):.1f}" x2="{R}" y2="{Y(q):.1f}" '
                    f'stroke="var(--rule-soft)" stroke-width="0.6"/>')

    # ---- isotropic full-utilization diagonal (0,0)->(dim,1) ----
    body.append(f'<line x1="{X(0):.1f}" y1="{Y(0):.1f}" x2="{X(dim):.1f}" y2="{Y(1):.1f}" '
                f'stroke="var(--ink-faint)" stroke-width="1" stroke-dasharray="4 3" '
                f'vector-effect="non-scaling-stroke"/>')

    # ---- concentration gap: area between the dataset curve and the diagonal ----
    curve_pts = [(X(i + 1), Y(cumfull[i])) for i in range(dim)]
    gap = (f"M {X(0):.1f} {Y(0):.1f} "
           + " ".join(f"L {x:.2f} {y:.2f}" for x, y in curve_pts)
           + f" L {X(dim):.1f} {Y(1):.1f} Z")
    body.append(f'<path d="{gap}" fill="color-mix(in srgb, var(--bad) 15%, transparent)" stroke="none"/>')

    # ---- 50% / 90% variance crosshairs (operational dimensionality) ----
    for frac, dk, soft in ((0.5, d50, True), (0.9, d90, False)):
        sw = "0.8" if soft else "1.0"
        op = "0.6" if soft else "0.9"
        body.append(f'<line x1="{L}" y1="{Y(frac):.1f}" x2="{X(dk):.1f}" y2="{Y(frac):.1f}" '
                    f'stroke="var(--ink-faint)" stroke-width="{sw}" stroke-dasharray="2 3" opacity="{op}"/>')
        body.append(f'<line x1="{X(dk):.1f}" y1="{B}" x2="{X(dk):.1f}" y2="{Y(frac):.1f}" '
                    f'stroke="var(--ink-faint)" stroke-width="{sw}" stroke-dasharray="2 3" opacity="{op}"/>')
        body.append(f'<circle cx="{X(dk):.1f}" cy="{Y(frac):.1f}" r="2.4" fill="var(--ink-soft)"/>')
        body.append(f'<text x="{X(dk)+6:.1f}" y="{Y(frac)+12:.1f}" font-size="9.5" fill="var(--ink-soft)" '
                    f'text-anchor="start" style="font-variant-numeric:tabular-nums">'
                    f'{int(frac*100)}% · {dk} dims</text>')

    # ---- dataset cumulative curve (the one accent) ----
    line = " ".join(f"{x:.2f} {y:.2f}" for x, y in curve_pts)
    body.append(f'<polyline points="{line}" fill="none" stroke="var(--accent)" stroke-width="1.6" '
                f'stroke-linejoin="round" vector-effect="non-scaling-stroke"/>')

    # ---- effective-rank marker (caution): where the entropy-rank lands vs the curve ----
    er_x = X(min(erank, dim))
    body.append(f'<line x1="{er_x:.1f}" y1="{T}" x2="{er_x:.1f}" y2="{B}" '
                f'stroke="var(--caution)" stroke-width="1.6" stroke-dasharray="5 3"/>')
    er_anchor = "start" if er_x < R - 150 else "end"
    er_dx = 5 if er_anchor == "start" else -5
    body.append(f'<text x="{er_x+er_dx:.1f}" y="{T+13:.1f}" font-size="10" font-weight="700" '
                f'fill="var(--caution)" text-anchor="{er_anchor}" style="font-variant-numeric:tabular-nums">'
                f'effective rank {erank:.0f}</text>')
    body.append(f'<text x="{er_x+er_dx:.1f}" y="{T+26:.1f}" font-size="8.5" '
                f'fill="var(--ink-faint)" text-anchor="{er_anchor}" style="font-variant-numeric:tabular-nums">'
                f'participation ratio {pratio:.0f}</text>')

    # ---- baseline + axes ----
    body.append(f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="var(--rule)" stroke-width="1"/>')
    body.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{B}" stroke="var(--rule)" stroke-width="1"/>')
    for q in (0.0, 0.25, 0.5, 0.75, 1.0):
        body.append(f'<line x1="{X(q*dim):.1f}" y1="{B}" x2="{X(q*dim):.1f}" y2="{B+6}" stroke="var(--rule)" stroke-width="1"/>')
        body.append(f'<text x="{X(q*dim):.1f}" y="{B+19}" font-size="9.5" fill="var(--ink-faint)" '
                    f'text-anchor="middle" style="font-variant-numeric:tabular-nums">{int(round(q*dim))}</text>')
        body.append(f'<line x1="{L-6}" y1="{Y(q):.1f}" x2="{L}" y2="{Y(q):.1f}" stroke="var(--rule)" stroke-width="1"/>')
        body.append(f'<text x="{L-9}" y="{Y(q)+3:.1f}" font-size="9.5" fill="var(--ink-faint)" '
                    f'text-anchor="end" style="font-variant-numeric:tabular-nums">{int(q*100)}%</text>')
    body.append(f'<text x="{(L+R)/2:.1f}" y="{B+36}" font-size="9.5" fill="var(--ink-faint)" '
                f'text-anchor="middle">dimensions (sorted by variance)</text>')
    body.append(f'<text x="22" y="{(T+B)/2:.1f}" font-size="9" fill="var(--ink-faint)" text-anchor="middle" '
                f'transform="rotate(-90 22 {(T+B)/2:.1f})">cumulative variance</text>')

    # diagonal label (parallels the diagonal, kept left of the effective-rank rule)
    dlx, dly = X(0.6 * dim), Y(0.52)
    body.append(f'<text x="{dlx:.1f}" y="{dly:.1f}" font-size="9" fill="var(--ink-faint)" '
                f'text-anchor="middle" transform="rotate(-25 {dlx:.1f} {dly:.1f})">'
                f'isotropic · full utilization</text>')

    # ---- verdict (good/caution/bad by how many dims hold 90%) ----
    r90 = d90 / max(1, dim)
    if r90 >= 0.6:
        verdict, vcol = f"well spread · {d90}/{dim} dims hold 90%", "--good"
    elif r90 >= 0.25:
        verdict, vcol = f"moderately concentrated · {d90}/{dim} dims hold 90%", "--caution"
    else:
        verdict, vcol = f"concentrated · {d90}/{dim} dims hold 90%", "--bad"
    body.insert(0, f'<text x="{L}" y="26" font-size="11" fill="var(--ink-soft)" text-anchor="start" '
                   f'style="font-variant-numeric:tabular-nums">cumulative variance · '
                   f'{dim}-d covariance</text>')
    body.insert(1, f'<text x="{R}" y="26" font-size="11" font-weight="700" fill="var({vcol})" '
                   f'text-anchor="end" style="font-variant-numeric:tabular-nums">{verdict}</text>')

    aria = (f"Cumulative fraction of covariance variance versus dimension for the {dim}-dimensional "
            f"space, sorted by variance, against the isotropic y=x diagonal. The dataset curve reaches "
            f"50% of variance by dimension {d50} and 90% by dimension {d90}; the effective rank is "
            f"{erank:.0f} and the participation ratio {pratio:.0f}. The shaded area between the curve and "
            f"the diagonal is the concentration — a larger gap means fewer dimensions carry the variance.")
    svg = _svg(w, h, aria, "".join(body))
    hc = _isoscore_hc(eigs, pos="br")            # isoscore = the gap-to-diagonal, as a 0..1 scalar
    return {
        "num": "RES 02b", "order": 91.5,
        "name": "Cumulative variance & dimensional concentration", "tech": "spectral concentration",
        "why": f"The eigenspectrum integrated: cumulative variance vs dimension against the isotropic "
               f"diagonal. Here 50% of variance lands in {d50} dims and 90% in {d90} of {dim}; effective "
               f"rank {erank:.0f} but participation ratio only {pratio:.0f} — the heavy tail inflates the "
               f"entropy rank, so the curve tells the fuller story. The IsoScore (hover) is this gap as a "
               f"single 0..1 scalar.",
        "svg": _hc_fig(svg, hc),
        "legend": '<span><i class="a"></i> cumulative variance (dataset)</span>'
                  '<span><i class="dash"></i> isotropic diagonal — full utilization</span>'
                  '<span><i class="r"></i> concentration gap</span>'
                  '<span><i class="c"></i> effective rank</span>'
                  '<span><i class="a"></i> isoscore — hover for the formula</span>',
        "reveal": "<b>Reveals:</b> how many dimensions actually carry the variance — the bow above the "
                  "diagonal is the concentration. When effective rank and participation ratio disagree, "
                  "this curve shows why: a few dominant axes plus a long low-variance tail.",
        "cls": "fig-mid",
    }
