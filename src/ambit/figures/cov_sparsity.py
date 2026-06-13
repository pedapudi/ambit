from ..render import figure, _box, _svg, _local_density
from .. import metrics
import numpy as np


@figure
def fig_cov_sparsity(ctx):
    W, H = 1180, 760
    pad = 70
    plot_w = 902  # data panel right edge; legend lives to the right

    P = _box(ctx.xy, plot_w, H, pad=pad)
    m = len(P)

    kd = getattr(ctx, "knn_dist", None)
    if kd is None or kd.ndim < 2 or kd.shape[1] < 1:
        # degrade: faint cloud + centered note, never crash
        dots = "".join(
            f'<circle cx="{P[i,0]:.1f}" cy="{P[i,1]:.1f}" r="1.2" '
            f'fill="var(--ink-faint)" fill-opacity="0.4"/>'
            for i in range(m)
        )
        note = (f'<text x="{W/2:.1f}" y="{H/2:.1f}" fill="var(--ink-faint)" '
                f'font-size="13" text-anchor="middle">needs kNN backend</text>')
        return {
            "num": "COV 09", "order": 9,
            "name": "Nearest-neighbor sparsity field", "tech": "nn distance",
            "why": "Per-point nearest-neighbor distance would size each ring; larger NN distance means more open space and cleaner separation.",
            "svg": _svg(W, H, "Nearest-neighbor sparsity field (needs kNN backend)", dots + note),
            "legend": '<span><i class="f"></i> point (no kNN)</span>',
            "reveal": "<b>Reveals:</b> nothing yet — kNN distances are unavailable.",
            "cls": "",
        }

    d0 = np.asarray(kd[:, 0], float)
    d0 = np.nan_to_num(d0, nan=0.0)

    med = float(np.median(d0))
    p90 = float(np.quantile(d0, 0.90))
    dmax = float(d0.max()) if d0.max() > 0 else 1.0

    # radius scale: map NN distance -> svg pixels. Cap at a generous physical max
    # so the very widest rings stay legible; median lands near a comfortable size.
    R_MAX = 26.0
    scale_hi = max(p90 * 1.6, dmax * 0.85, 1e-9)

    def r_of(dist):
        return float(np.clip(dist / scale_hi, 0.0, 1.0) * R_MAX)

    isolated = d0 >= p90  # top decile = most space = GOOD resolution

    body = []
    # header captions
    body.append(
        f'<text x="{pad}" y="30.0" fill="var(--ink-soft)" font-size="12" '
        f'text-anchor="start">nearest-neighbor sparsity field · ring radius = '
        f"each point's 1-NN distance</text>"
    )
    body.append(
        f'<text x="{plot_w-2:.1f}" y="30.0" fill="var(--good)" font-size="12" '
        f'text-anchor="end">large NN distance = high resolution · good separation</text>'
    )
    # panel divider between data field and legend
    body.append(
        f'<line x1="{plot_w}" y1="54" x2="{plot_w}" y2="712" stroke="var(--rule-soft)" '
        f'stroke-width="0.8" vector-effect="non-scaling-stroke"/>'
    )

    # median reference circle (dashed, neutral) — "normal spacing"
    r_med = max(r_of(med), 3.0)
    cx_ref, cy_ref = plot_w * 0.50, H * 0.50
    body.append(
        f'<circle cx="{cx_ref:.1f}" cy="{cy_ref:.1f}" r="{r_med:.1f}" fill="none" '
        f'stroke="var(--ink-faint)" stroke-dasharray="2 5" stroke-width="1" opacity="0.7"/>'
    )

    # rings — neutral crowded points first, isolated good rings on top
    neutral, good = [], []
    for i in range(m):
        x, y = P[i, 0], P[i, 1]
        r = r_of(d0[i])
        # tiny center dot anchors each item
        dot = (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.0" '
               f'fill="var(--ink-faint)" fill-opacity="0.5"/>')
        if isolated[i]:
            ring = (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{max(r,2.5):.1f}" fill="none" '
                    f'stroke="var(--good)" stroke-width="1.2" opacity="0.9"/>')
            good.append(dot + ring)
        else:
            # crowded / typical: small neutral ring, never bad
            rr = max(r, 1.2)
            ring = (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rr:.1f}" fill="none" '
                    f'stroke="var(--ink-soft)" stroke-width="0.9" opacity="0.55"/>')
            neutral.append(dot + ring)
    body.append('<g>' + ''.join(neutral) + '</g>')
    body.append('<g>' + ''.join(good) + '</g>')

    # ---- radius scale legend (right rail) ----
    lx = (plot_w + W) / 2  # legend column center
    body.append(
        f'<g><text x="{lx:.1f}" y="80.0" fill="var(--ink-soft)" font-size="11" '
        f'text-anchor="middle">1-NN RING</text>'
        f'<text x="{lx:.1f}" y="94.0" fill="var(--ink-faint)" font-size="9.5" '
        f'text-anchor="middle">radius = NN distance</text>'
        f'<text x="{lx:.1f}" y="108.0" fill="var(--good)" font-size="9.5" '
        f'text-anchor="middle">wider = higher resolution</text>'
    )

    # quantitative axis: a vertical ruler with evenly spaced ticks in distance units
    ax_x = lx - 74
    ax_top, ax_bot = 148.0, 408.0
    n_tk = 5
    body.append(
        '<g font-family="ui-monospace,Menlo,Consolas,monospace" '
        'font-variant-numeric="tabular-nums">'
        f'<line x1="{ax_x:.0f}" y1="{ax_top:.0f}" x2="{ax_x:.0f}" y2="{ax_bot:.0f}" '
        f'stroke="var(--rule-soft)" stroke-width="0.8" vector-effect="non-scaling-stroke"/>'
    )
    for t in range(n_tk):
        frac = t / (n_tk - 1)
        ty = ax_top + frac * (ax_bot - ax_top)
        dist_val = frac * scale_hi
        body.append(
            f'<line x1="{ax_x-4:.0f}" y1="{ty:.0f}" x2="{ax_x:.0f}" y2="{ty:.0f}" '
            f'stroke="var(--rule-soft)" stroke-width="0.8" vector-effect="non-scaling-stroke"/>'
            f'<text x="{ax_x-8:.1f}" y="{ty+3.5:.1f}" fill="var(--ink-faint)" font-size="9" '
            f'text-anchor="end">{dist_val:.3f}</text>'
        )
    body.append('</g>')

    # exemplar rings at core / median / p90, sized by the same scale
    def exemplar(cy, dist, color, dash, label):
        rr = max(r_of(dist), 1.6)
        d = f' stroke-dasharray="{dash}"' if dash else ''
        return (
            f'<circle cx="{lx:.0f}" cy="{cy:.0f}" r="{rr:.1f}" fill="none" '
            f'stroke="{color}" stroke-width="1.2"{d}/>'
            f'<circle cx="{lx:.0f}" cy="{cy:.0f}" r="1.4" fill="var(--ink-faint)"/>'
            f'<text x="{lx:.1f}" y="{cy+rr+15:.1f}" fill="{color}" font-size="9.5" '
            f'text-anchor="middle">{label}</text>'
        )

    body.append(exemplar(168, 0.0, "var(--ink-soft)", "", f"dense core ≈ {0.0:.3f}"))
    body.append(exemplar(230, med, "var(--ink-faint)", "3 3", f"median ≈ {med:.3f} (normal)"))
    body.append(exemplar(330, p90, "var(--good)", "", f"p90 ≈ {p90:.3f} · open space"))

    body.append(
        f'<text x="{lx:.1f}" y="448.0" fill="var(--ink-faint)" font-size="9" '
        f'text-anchor="middle" font-family="ui-monospace,Menlo,Consolas,monospace" '
        f'font-variant-numeric="tabular-nums">core {0.0:.3f} · median {med:.3f} · '
        f'p90 {p90:.3f}</text></g>'
    )

    n_iso = int(isolated.sum())
    return {
        "num": "COV 09", "order": 9,
        "name": "Nearest-neighbor sparsity field", "tech": "nn distance",
        "why": (f"Each point wears an open ring whose radius is its distance to its "
                f"nearest neighbor (1-NN ∈ 1-cos). The {n_iso} top-decile most-isolated "
                f"items (NN ≥ {p90:.3f}) claim open space and get the good token — large "
                f"distance is high resolution, not a defect."),
        "svg": _svg(W, H,
                    "Nearest-neighbor sparsity field: ring radius equals each point's "
                    "distance to its nearest neighbor; the most-isolated top-decile points "
                    "wear good-token rings because larger separation means higher resolution.",
                    "".join(body)),
        "legend": ('<span><i class="f"></i> typical point (neutral ring)</span>'
                   '<span><i class="dash"></i> median NN reference</span>'
                   '<span><i class="g"></i> top-decile isolated (good)</span>'),
        "reveal": (f"<b>Reveals:</b> where the corpus leaves breathing room — the widest "
                   f"rings mark items the embedding resolves cleanly, while tight clusters "
                   f"of bare dots mark crowded, harder-to-separate regions."),
        "cls": "",
    }
