"""RES 09b — Resolution bandwidth (expected collisions vs query noise).

The operational-units companion to the crowding curve: the same pair sample,
transformed by the exact confusion kernel. For each query-noise scale sigma, the
curve shows the expected number of competitors that out-score an intended target
(Gaussian query channel; union-bound conservative under any competitor
correlation). Where the curve crosses one expected collision is the corpus's
**resolution bandwidth** sigma* — the noise budget printed in the header facts,
here shown with its provenance. The uniform-null curve underneath is what a
well-spread corpus of the same size and dimension would tolerate; the horizontal
gap between the two crossings is the budget crowding has already spent.
"""

from __future__ import annotations

import numpy as np

from ..render import figure, _svg
from .. import occupancy as occ


@figure
def fig_res_bandwidth(ctx):
    W, H = 920, 470
    L, R, T, B = 84.0, 856.0, 54.0, 380.0

    cos = np.asarray(ctx.cos, np.float64)
    dim = int(getattr(ctx.scan, "dim", 0) or 0)
    n_items = int(getattr(ctx.scan, "n", len(cos)) or len(cos))
    if cos.size < 100 or dim < 2:
        note = (f'<text x="{W/2:.0f}" y="{H/2:.0f}" fill="var(--ink-faint)" font-size="12" '
                f'text-anchor="middle">needs a pair-cosine sample</text>')
        return {"num": "RES 09b", "order": 90.55, "name": "Resolution bandwidth", "tech": "confusion curve",
                "why": "No pair sample available to build the curve.",
                "svg": _svg(W, H, "Resolution bandwidth (no data)", note),
                "legend": "", "reveal": "<b>Reveals:</b> the query-noise budget.", "cls": ""}

    null_cos = occ.null_pair_cos(dim, min(len(cos), 100_000), seed=0)
    sig_grid = np.geomspace(1e-3, 3.0, 120)
    c_data = np.array([occ.expected_collisions(cos, n_items, s) for s in sig_grid])
    c_null = np.array([occ.expected_collisions(null_cos, n_items, s) for s in sig_grid])
    s_star = occ.sigma_star(cos, n_items, tol=1.0)
    s_null = occ.sigma_star(null_cos, n_items, tol=1.0)

    y_hi = max(float(c_data.max()), float(c_null.max()), 10.0)
    y_lo = 1e-4
    lx0, lx1 = np.log10(sig_grid[0]), np.log10(sig_grid[-1])
    ly0, ly1 = np.log10(y_lo), np.log10(y_hi)

    def Xc(s):
        return L + (np.log10(max(s, 1e-9)) - lx0) / (lx1 - lx0) * (R - L)

    def Yc(c):
        return B - (np.log10(np.clip(c, y_lo, None)) - ly0) / (ly1 - ly0) * (B - T)

    def poly(cv, stroke, width, dash=""):
        pts = " ".join(f"{Xc(s):.1f},{Yc(c):.1f}" for s, c in zip(sig_grid, cv) if c > 0)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return (f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="{width}"{d} '
                f'vector-effect="non-scaling-stroke"/>')

    body = []
    body.append(f'<text x="{L}" y="32" fill="var(--ink-soft)" font-size="12">resolution bandwidth · expected '
                f'competitors out-scoring the target, per entity, vs query-noise scale σ</text>')
    body.append(f'<text x="{R}" y="32" fill="var(--ink-faint)" font-size="11" text-anchor="end">'
                f'{n_items:,} items · exact pairwise law, union bound</text>')

    # axes: log-log
    body.append(f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="var(--rule)" stroke-width="1"/>')
    body.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{B}" stroke="var(--rule)" stroke-width="1"/>')
    for dec in range(int(np.ceil(ly0)), int(np.floor(ly1)) + 1):
        yy = Yc(10.0 ** dec)
        lab = f"{10**dec:g}" if -1 <= dec <= 3 else f"1e{dec}"
        body.append(f'<line x1="{L}" y1="{yy:.1f}" x2="{R}" y2="{yy:.1f}" stroke="var(--rule-soft)" '
                    f'stroke-width="0.5" vector-effect="non-scaling-stroke"/>')
        body.append(f'<text x="{L-8}" y="{yy+3.2:.1f}" fill="var(--ink-faint)" font-size="9" text-anchor="end" '
                    f'style="font-variant-numeric:tabular-nums">{lab}</text>')
    for s in (0.001, 0.01, 0.1, 1.0):
        xx = Xc(s)
        body.append(f'<line x1="{xx:.1f}" y1="{B}" x2="{xx:.1f}" y2="{B+6}" stroke="var(--rule-soft)" stroke-width="0.7"/>')
        body.append(f'<text x="{xx:.1f}" y="{B+18}" fill="var(--ink-faint)" font-size="9" text-anchor="middle" '
                    f'style="font-variant-numeric:tabular-nums">{s:g}</text>')
    body.append(f'<text x="{(L+R)/2:.1f}" y="{B+36}" fill="var(--ink-soft)" font-size="9.5" text-anchor="middle">'
                f'query-noise scale σ (log) · more noise tolerated →</text>')

    # tolerance line at 1 expected collision
    y1v = Yc(1.0)
    body.append(f'<line x1="{L}" y1="{y1v:.1f}" x2="{R}" y2="{y1v:.1f}" stroke="var(--ink-soft)" '
                f'stroke-width="1" stroke-dasharray="6 4" vector-effect="non-scaling-stroke"/>')
    body.append(f'<text x="{R}" y="{y1v-6:.1f}" fill="var(--ink-soft)" font-size="9.5" text-anchor="end">'
                f'tolerance · 1 expected collision</text>')

    # the noise budget spent: gap between the two crossings
    xs, xn = Xc(s_star), Xc(s_null)
    if xn > xs + 2:
        body.append(f'<rect x="{xs:.1f}" y="{T}" width="{xn-xs:.1f}" height="{B-T}" '
                    f'fill="color-mix(in srgb, var(--bad) 10%, transparent)"/>')
        body.append(f'<text x="{(xs+xn)/2:.1f}" y="{T+16}" fill="var(--bad)" font-size="10" '
                    f'text-anchor="middle">budget spent by crowding</text>')

    body.append(poly(c_null, "var(--ink-faint)", 1.2, dash="4 3"))
    body.append(poly(c_data, "var(--accent)", 2.2))

    # crossings
    for sv, col, lab, anchor in ((s_star, "var(--accent)", f"σ* = {s_star:.3f}", "end"),
                                 (s_null, "var(--ink-faint)", f"uniform σ* = {s_null:.3f}", "start")):
        xx = Xc(sv)
        body.append(f'<line x1="{xx:.1f}" y1="{y1v:.1f}" x2="{xx:.1f}" y2="{B}" stroke="{col}" '
                    f'stroke-width="1.3" stroke-dasharray="3 3" vector-effect="non-scaling-stroke"/>')
        body.append(f'<circle cx="{xx:.1f}" cy="{y1v:.1f}" r="3.4" fill="{col}"/>')
        dx = -7 if anchor == "end" else 7
        body.append(f'<text x="{xx+dx:.1f}" y="{y1v+16:.1f}" fill="{col}" font-size="10.5" font-weight="700" '
                    f'text-anchor="{anchor}" style="font-variant-numeric:tabular-nums">{lab}</text>')

    frac = s_star / s_null if s_null > 0 else 1.0
    aria = (f"Resolution bandwidth: expected collisions per entity as a function of query-noise scale, "
            f"log-log, for the dataset and the uniform null; the dataset crosses one expected collision "
            f"at sigma {s_star:.3f} versus {s_null:.3f} for the null — it retains {frac:.0%} of the "
            f"noise budget a well-spread corpus of this size would have.")
    return {
        "num": "RES 09b", "order": 90.55, "name": "Resolution bandwidth", "tech": "confusion curve",
        "why": (f"The pair sample transformed into operational units: at each query-noise scale σ, the "
                f"expected number of competitors that out-score the intended target (exact pairwise law, "
                f"conservative in aggregate). The crossing at one expected collision is the header's σ* — "
                f"this corpus keeps {frac:.0%} of the noise budget a perfectly spread corpus of the same "
                f"size and dimension would have."),
        "svg": _svg(W, H, aria, "".join(body)),
        "legend": ('<span><i class="a"></i> this dataset</span>'
                   '<span><i class="dash"></i> uniform-sphere null</span>'
                   '<span><i class="r"></i> noise budget spent by crowding</span>'),
        "reveal": ("<b>Reveals:</b> the <b>query-noise budget</b> — how much perturbation the corpus absorbs "
                   "before entities become interchangeable, and how much of that budget crowding has already "
                   "spent relative to a well-spread corpus. Scope: intra-corpus confusability."),
        "cls": "fig-mid",
    }
