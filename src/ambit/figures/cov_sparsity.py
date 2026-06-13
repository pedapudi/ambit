"""COV 09 — Nearest-neighbor sparsity field. Each reservoir point is a ring sized by
its 1-NN distance; the most-isolated decile (largest NN distance = most open space =
GOOD resolution) reads var(--good), the rest neutral. At 100k+ the field is crowded,
so the points are split into shuffled groups toggled by a #-samples slider.
"""

from __future__ import annotations

import numpy as np

from ..render import figure, _svg, _box

GROUPS = 40


@figure
def fig_cov_sparsity(ctx):
    W, H = 1180, 760
    pad = 70
    plot_w = 902
    P = _box(ctx.xy, plot_w, H, pad=pad)
    m = len(P)

    kd = getattr(ctx, "knn_dist", None)
    if kd is None or getattr(kd, "ndim", 0) < 2 or kd.shape[1] < 1:
        dots = "".join(f'<circle cx="{P[i,0]:.1f}" cy="{P[i,1]:.1f}" r="1.2" '
                       f'fill="var(--ink-faint)" fill-opacity="0.4"/>' for i in range(m))
        note = (f'<text x="{W/2:.1f}" y="{H/2:.1f}" fill="var(--ink-faint)" font-size="13" '
                f'text-anchor="middle">needs kNN backend</text>')
        return {"num": "COV 09", "order": 9, "name": "Nearest-neighbor sparsity field", "tech": "nn distance",
                "why": "Per-point nearest-neighbor distance sizes each ring; larger = more open space, cleaner separation.",
                "svg": _svg(W, H, "Nearest-neighbor sparsity field (needs kNN backend)", dots + note),
                "legend": '<span><i class="f"></i> point (no kNN)</span>',
                "reveal": "<b>Reveals:</b> nothing yet — kNN distances are unavailable.", "cls": ""}

    d0 = np.nan_to_num(np.asarray(kd[:, 0], float), nan=0.0)
    med = float(np.median(d0))
    p90 = float(np.quantile(d0, 0.90))
    dmax = float(d0.max()) or 1.0
    R_MAX = 22.0
    scale_hi = max(p90 * 1.6, dmax * 0.85, 1e-9)

    def r_of(dist):
        return float(np.clip(dist / scale_hi, 0.0, 1.0) * R_MAX)

    isolated = d0 >= p90

    head = [
        f'<text x="{pad}" y="30" fill="var(--ink-soft)" font-size="12">nearest-neighbor sparsity field · '
        f"ring radius = each point's 1-NN distance</text>",
        f'<text x="{plot_w-2:.1f}" y="30" fill="var(--good)" font-size="12" text-anchor="end">'
        f'large NN distance = high resolution · good separation</text>',
        f'<line x1="{plot_w}" y1="54" x2="{plot_w}" y2="712" stroke="var(--rule-soft)" stroke-width="0.8"/>',
        f'<circle cx="{plot_w*0.5:.1f}" cy="{H*0.5:.1f}" r="{max(r_of(med),3.0):.1f}" fill="none" '
        f'stroke="var(--ink-faint)" stroke-dasharray="2 5" stroke-width="1" opacity="0.7"/>',
        f'<text x="{plot_w*0.5:.1f}" y="{H*0.5 + max(r_of(med),3.0) + 13:.1f}" fill="var(--ink-faint)" '
        f'font-size="9" text-anchor="middle">median spacing</text>',
    ]

    # rings split into shuffled groups so the slider can thin the field
    rng = np.random.default_rng(0)
    order = rng.permutation(m)
    groups = [[] for _ in range(GROUPS)]
    for rank, i in enumerate(order):
        x, y = P[i, 0], P[i, 1]
        if isolated[i]:
            r = max(r_of(d0[i]), 2.0)
            groups[rank % GROUPS].append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="none" stroke="var(--good)" stroke-width="1.25"/>')
        else:
            groups[rank % GROUPS].append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{min(r_of(d0[i]),3.0):.1f}" '
                f'fill="var(--ink-faint)" fill-opacity="0.30"/>')
    grp_svg = "".join(f'<g class="cov09-grp" data-i="{g}">{"".join(groups[g])}</g>' for g in range(GROUPS))
    svg = _svg(W, H, "Nearest-neighbor sparsity field; rings on the most-isolated decile read good",
               "".join(head) + grp_svg)

    # default: thin to ~6000 visible points when the field is large
    vis = max(4, min(GROUPS, int(round(GROUPS * min(1.0, 6000.0 / max(1, m))))))
    ctrl = ('<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;'
            'font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px;color:var(--ink-faint)">'
            f'samples <input id="cov09-range" type="range" min="1" max="{GROUPS}" value="{vis}" '
            'style="flex:0 0 220px;accent-color:var(--accent)"> '
            '<span id="cov09-count"></span></div>')
    script = ("(function(){var T=%d,G=%d;var r=document.getElementById('cov09-range'),"
              "c=document.getElementById('cov09-count');if(!r)return;"
              "var gs=document.querySelectorAll('.cov09-grp');"
              "function ap(){var v=+r.value;for(var i=0;i<gs.length;i++)gs[i].style.display=(i<v)?'':'none';"
              "if(c)c.textContent=Math.round(T*v/G).toLocaleString()+' of '+T.toLocaleString()+' points';}"
              "r.addEventListener('input',ap);ap();})();") % (m, GROUPS)

    return {
        "num": "COV 09", "order": 9, "name": "Nearest-neighbor sparsity field", "tech": "nn distance · slider",
        "why": "Each point is a ring sized by its 1-NN distance; the most-isolated decile (most open space) reads good. The slider thins the field — useful at 100k where it is crowded.",
        "svg": ctrl + svg, "script": script,
        "legend": '<span><i class="g"></i> isolated decile — most distinct (good)</span>'
                  '<span><i class="f"></i> typical spacing (neutral)</span>'
                  '<span><i class="dash"></i> median spacing</span>',
        "reveal": "<b>Reveals:</b> where the space is open vs. packed — the good rings mark the best-separated points; drag the slider to thin a crowded field.",
        "cls": "",
    }
