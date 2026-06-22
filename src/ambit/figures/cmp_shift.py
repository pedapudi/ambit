"""CMP 14 — Distance-distribution shift. Overlays the two random-pair cosine densities
(A vs the compare set) on one axis, in the style of RES 01, with the gap between their
means shaded. Answers "did the *whole* similarity distribution move, not just per
item." Works at any dims (each set's cosines are within-set). Reads only `ctx.cmp`.
"""

from __future__ import annotations

import numpy as np

from ..render import figure, _svg
from .. import metrics


def _kde(samples, grid, bw):
    s = np.asarray(samples, float)
    if s.size == 0:
        return np.zeros_like(grid)
    d = (grid[:, None] - s[None, :]) / bw
    return np.exp(-0.5 * d * d).sum(1) / (s.size * bw * np.sqrt(2 * np.pi))


def _placeholder():
    w, h = 760, 300
    body = (f'<text x="{w/2:.0f}" y="{h/2:.0f}" text-anchor="middle" font-size="12" '
            f'fill="var(--ink-faint)">needs a second embedding — run with --compare</text>')
    return {"num": "CMP 14", "order": 14, "name": "Distance-distribution shift", "tech": "A vs B cosine",
            "why": "Whether the whole random-pair cosine distribution shifted between two embeddings.",
            "svg": _svg(w, h, "Distance-distribution shift unavailable without a second embedding.", body),
            "legend": '<span><i class="f"></i> needs --compare</span>',
            "reveal": "<b>Reveals:</b> nothing yet — pass a second embedding with --compare.", "cls": ""}


@figure
def fig_cmp_shift(ctx):
    cmp = getattr(ctx, "cmp", None)
    if cmp is None:
        return _placeholder()

    w, h = 760, 470
    L, R, T, B = 70, 720, 70, 392
    cosA = metrics.random_pair_cosine(cmp.A, n_pairs=120_000, normalized=True)
    cosB = metrics.random_pair_cosine(cmp.B, n_pairs=120_000, normalized=True)
    mA, mB = float(cosA.mean()), float(cosB.mean())

    def X(v):
        return L + (v + 1.0) / 2.0 * (R - L)

    grid = np.linspace(-1.0, 1.0, 360)
    bw = max(0.02, 0.9 * min(cosA.std(), cosB.std()) * (len(cosA)) ** (-0.2))
    dA, dB = _kde(cosA, grid, bw), _kde(cosB, grid, bw)
    dmax = max(dA.max(), dB.max(), 1e-9)

    def Y(d):
        return B - (np.asarray(d, float) / dmax) * (B - T) * 0.92

    xs = X(grid)
    body = []
    body.append(f'<text x="{L}" y="34" fill="var(--ink-soft)" font-size="12">'
                f'random-pair cosine · A vs {cmp.label_b} (same {len(cmp.ids):,} items)</text>')

    # gridlines + axis
    for g in np.arange(-1.0, 1.0001, 0.2):
        body.append(f'<line x1="{X(g):.1f}" y1="{T}" x2="{X(g):.1f}" y2="{B}" stroke="var(--rule-soft)" stroke-width="0.7"/>')
    body.append(f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="var(--rule)" stroke-width="1"/>')
    for g in np.arange(-1.0, 1.0001, 0.2):
        body.append(f'<line x1="{X(g):.1f}" y1="{B}" x2="{X(g):.1f}" y2="{B+6}" stroke="var(--rule)" stroke-width="1"/>')
        body.append(f'<text x="{X(g):.1f}" y="{B+19}" fill="var(--ink-faint)" font-size="10" '
                    f'text-anchor="middle" style="font-variant-numeric:tabular-nums">'
                    f'{("%+.1f" % g) if abs(g)>1e-9 else "0"}</text>')
    body.append(f'<text x="{(L+R)/2:.1f}" y="{B+36}" fill="var(--ink-faint)" font-size="9.5" '
                f'text-anchor="middle">cosine similarity</text>')

    # shaded gap between the two means
    x0, x1 = sorted((X(mA), X(mB)))
    body.append(f'<rect x="{x0:.1f}" y="{T}" width="{x1-x0:.1f}" height="{B-T}" '
                f'fill="color-mix(in srgb, var(--accent) 9%, transparent)"/>')

    # A density (faint) and B density (accent)
    aA = " ".join(f"{('M' if i==0 else 'L')} {xs[i]:.1f},{Y(dA[i]):.1f}" for i in range(len(grid)))
    aB = " ".join(f"{('M' if i==0 else 'L')} {xs[i]:.1f},{Y(dB[i]):.1f}" for i in range(len(grid)))
    body.append(f'<path d="{aA}" fill="none" stroke="var(--ink-faint)" stroke-width="1.5" vector-effect="non-scaling-stroke"/>')
    body.append(f'<path d="{aB}" fill="none" stroke="var(--accent)" stroke-width="1.6" vector-effect="non-scaling-stroke"/>')

    # mean ticks
    for mv, col, lab in ((mA, "var(--ink-faint)", "A"), (mB, "var(--accent)", cmp.label_b)):
        xx = X(mv)
        body.append(f'<line x1="{xx:.1f}" y1="{B}" x2="{xx:.1f}" y2="{T+10}" stroke="{col}" stroke-width="1.6" stroke-dasharray="3 3"/>')
        body.append(f'<text x="{xx:.1f}" y="{T+6}" fill="{col}" font-size="10" text-anchor="middle" '
                    f'style="font-variant-numeric:tabular-nums">{lab} {mv:+.2f}</text>')

    dmean = mB - mA
    callout = f"Δmean cos = {dmean:+.3f}"
    if cmp.mmd2 is not None:
        callout += f" · 1−MMD² = {np.clip(1.0 - cmp.mmd2, 0.0, 1.0):.3f}"
    body.append(f'<text x="{R}" y="34" fill="var(--accent)" font-size="11" font-weight="700" '
                f'text-anchor="end" style="font-variant-numeric:tabular-nums">{callout}</text>')

    aria = (f"Random-pair cosine densities for embedding A (faint) and {cmp.label_b} (accent) over the "
            f"same {len(cmp.ids):,} items, on a −1 to +1 cosine axis. A's mean is {mA:+.2f} and "
            f"{cmp.label_b}'s is {mB:+.2f}; the shaded band between the means is the distributional shift "
            f"(Δmean cos {dmean:+.3f}).")
    return {
        "num": "CMP 14", "order": 14, "name": "Distance-distribution shift", "tech": "A vs B cosine",
        "why": f"Did the whole similarity distribution move, not just individual items? A's random-pair "
               f"cosine sits at mean {mA:+.2f}, {cmp.label_b}'s at {mB:+.2f} (Δ {dmean:+.3f}); a leftward "
               f"shift toward 0 means the second embedding spread unrelated items further apart "
               f"(less anisotropic).",
        "svg": _svg(w, h, aria, "".join(body)),
        "legend": '<span><i class="f"></i> A — random-pair cosine</span>'
                  f'<span><i class="a"></i> {cmp.label_b} — random-pair cosine</span>'
                  '<span><i class="a"></i> shaded gap between the means</span>',
        "reveal": "<b>Reveals:</b> whether the global similarity distribution shifted — the gap between "
                  "the two means is the distributional move, complementing the per-item drift field.",
        "cls": "fig-mid",
    }
