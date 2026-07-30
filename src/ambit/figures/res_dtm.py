"""RES 10 — Per-entity crowding field (distance to a measure).

Each entity's score is the radius of the ball it needs to gather a fixed share of
the corpus (DTM; Wasserstein-stable — the guarantee cell counts lack). The figure
draws the field's exact ECDF: the low tail is the crowded-entity list (named, with
ids), the high tail the most isolated entities (voids). A shaded band marks the
1st–99th percentile range of the field for a uniform corpus of matched size and
dimension — a healthy field hugs that band; a low-tail spike escaping left of it is
crowding with a name.
"""

from __future__ import annotations

import numpy as np

from ..render import figure, _svg
from .. import crowding as cr


def _idstr(v, width: int = 16) -> str:
    s = str(v)
    return s if len(s) <= width else s[: width - 1] + "…"


@figure
def fig_res_dtm(ctx):
    W, H = 920, 520
    L, R, T, B = 250.0, 856.0, 56.0, 430.0
    M_FRAC, CAP = 0.02, 6000

    es = ctx.es
    X = np.asarray(es.X)
    dim = int(getattr(ctx.scan, "dim", X.shape[1] if X.ndim == 2 else 0) or 0)
    if X.ndim != 2 or len(X) < 32:
        note = (f'<text x="{W/2:.0f}" y="{H/2:.0f}" fill="var(--ink-faint)" font-size="12" '
                f'text-anchor="middle">needs a reservoir sample</text>')
        return {"num": "RES 10", "order": 90.6, "name": "Per-entity crowding field", "tech": "distance to a measure",
                "why": "No reservoir available to compute the field.",
                "svg": _svg(W, H, "Per-entity crowding field (no data)", note),
                "legend": "", "reveal": "<b>Reveals:</b> which entities are crowded, by name.", "cls": ""}

    rng = np.random.default_rng(0)
    if len(X) > CAP:
        sub = rng.choice(len(X), CAP, replace=False)
    else:
        sub = np.arange(len(X))
    Xs = X[sub]
    ids = np.asarray(es.ids)[sub] if es.ids is not None else sub

    field = cr.dtm(Xs, m_frac=M_FRAC)
    k_used = max(2, int(np.ceil(M_FRAC * len(Xs))))
    lo_null, hi_null = cr.dtm_null_band(len(Xs), dim, m_frac=M_FRAC, seed=0)

    order = np.argsort(field)
    xs = field[order]
    ys = (np.arange(len(xs)) + 1) / len(xs)

    x_min = min(float(xs[0]), lo_null) * 0.92
    x_max = max(float(xs[-1]), hi_null) * 1.04
    span = max(x_max - x_min, 1e-9)

    def Xc(v):
        return L + (v - x_min) / span * (R - L)

    def Yc(f):
        return B - f * (B - T)

    body = []
    body.append(f'<text x="{L}" y="34" fill="var(--ink-soft)" font-size="12">'
                f'per-entity crowding field · DTM radius to hold {M_FRAC*100:.0f}% of the corpus '
                f'(k={k_used}) · smaller = more crowded</text>')
    body.append(f'<text x="{R}" y="34" fill="var(--ink-faint)" font-size="11" text-anchor="end">'
                f'{len(Xs):,} entities · chord units</text>')

    # axes
    body.append(f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="var(--rule)" stroke-width="1"/>')
    body.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{B}" stroke="var(--rule)" stroke-width="1"/>')
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        v = x_min + frac * span
        xx = Xc(v)
        body.append(f'<line x1="{xx:.1f}" y1="{B}" x2="{xx:.1f}" y2="{B+6}" stroke="var(--rule-soft)" stroke-width="0.7"/>')
        body.append(f'<text x="{xx:.1f}" y="{B+18}" fill="var(--ink-faint)" font-size="9" text-anchor="middle" '
                    f'style="font-variant-numeric:tabular-nums">{v:.2f}</text>')
        yy = Yc(frac)
        body.append(f'<line x1="{L}" y1="{yy:.1f}" x2="{R}" y2="{yy:.1f}" stroke="var(--rule-soft)" '
                    f'stroke-width="0.4" vector-effect="non-scaling-stroke"/>')
        body.append(f'<text x="{L-8}" y="{yy+3.2:.1f}" fill="var(--ink-faint)" font-size="9" text-anchor="end" '
                    f'style="font-variant-numeric:tabular-nums">{frac:.2f}</text>')
    body.append(f'<text x="{(L+R)/2:.1f}" y="{B+34}" fill="var(--ink-soft)" font-size="9.5" text-anchor="middle">'
                f'DTM radius · ← crowded · isolated →</text>')
    body.append(f'<text x="{L-40}" y="{(T+B)/2:.1f}" fill="var(--ink-soft)" font-size="9.5" text-anchor="middle" '
                f'transform="rotate(-90 {L-40} {(T+B)/2:.1f})">fraction of entities ≤ radius</text>')

    # uniform-null band
    bx0, bx1 = Xc(lo_null), Xc(hi_null)
    body.append(f'<rect x="{bx0:.1f}" y="{T}" width="{max(bx1-bx0, 1.5):.1f}" height="{B-T}" '
                f'fill="color-mix(in srgb, var(--good) 12%, transparent)"/>')
    body.append(f'<text x="{(bx0+bx1)/2:.1f}" y="{T+14}" fill="var(--good)" font-size="9.5" '
                f'text-anchor="middle">uniform reference band</text>')

    # the ECDF (decimated for SVG size)
    step = max(1, len(xs) // 480)
    pts = " ".join(f"{Xc(v):.1f},{Yc(f):.1f}" for v, f in zip(xs[::step], ys[::step]))
    body.append(f'<polyline points="{pts}" fill="none" stroke="var(--accent)" stroke-width="2.2" '
                f'vector-effect="non-scaling-stroke"/>')

    # crowded-entity callouts (low tail), left panel
    n_call = min(6, len(order))
    body.append(f'<text x="24" y="{T+6}" fill="var(--ink-soft)" font-size="10.5" font-weight="700">'
                f'most crowded entities</text>')
    for r in range(n_call):
        i = order[r]
        v = field[i]
        yy = T + 24 + r * 18
        body.append(f'<circle cx="{Xc(v):.1f}" cy="{Yc((r+1)/len(xs)):.1f}" r="3.0" fill="var(--bad)"/>')
        body.append(f'<text x="24" y="{yy:.1f}" fill="var(--ink)" font-size="9.5" '
                    f'style="font-variant-numeric:tabular-nums">{_idstr(ids[i])} · {v:.3f}</text>')
    # isolated (void) side, bottom of the left panel
    body.append(f'<text x="24" y="{T+24+n_call*18+16}" fill="var(--ink-soft)" font-size="10.5" '
                f'font-weight="700">most isolated (voids)</text>')
    for r in range(min(3, len(order))):
        i = order[-(r + 1)]
        yy = T + 24 + n_call * 18 + 34 + r * 18
        body.append(f'<text x="24" y="{yy:.1f}" fill="var(--ink)" font-size="9.5" '
                    f'style="font-variant-numeric:tabular-nums">{_idstr(ids[i])} · {field[i]:.3f}</text>')

    p1 = float(np.percentile(field, 1))
    below = int((field < lo_null).sum())
    aria = (f"Per-entity crowding field: the exact CDF of each entity's distance-to-measure radius "
            f"(share {M_FRAC*100:.0f}% of the corpus), against the uniform reference band "
            f"[{lo_null:.2f}, {hi_null:.2f}]; 1st percentile {p1:.3f}; {below} entities sit left of "
            f"the band; the most crowded and most isolated entities are listed by id.")
    return {
        "num": "RES 10", "order": 90.6, "name": "Per-entity crowding field", "tech": "distance to a measure",
        "why": (f"Every entity scored by the radius it needs to gather {M_FRAC*100:.0f}% of the corpus "
                f"(distance to a measure — continuous, provably stable under data perturbation). The low "
                f"tail names the crowded entities; the high tail the voids; the shaded band is where a "
                f"uniform corpus of this size and dimension lives."),
        "svg": _svg(W, H, aria, "".join(body)),
        "legend": ('<span><i class="a"></i> DTM field (CDF)</span>'
                   '<span><i class="g"></i> uniform reference band</span>'
                   '<span><i class="r"></i> most crowded entities</span>'),
        "reveal": (f"<b>Reveals:</b> <b>which entities</b> are crowded, by name — {below} sit below the "
                   f"uniform band — replacing per-cell heat with a per-entity, null-calibrated score."),
        "cls": "",
    }
