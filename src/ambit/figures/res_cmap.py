"""RES 07 — Local crowding map. The reservoir projection recolored by each item's
local crowding score (the signed, multiscale robust z of its neighborhood
concentration vs the dataset's own field). Crowded-relative-to-typical items light
up var(--bad); the most-open items read var(--good); pockets (if any) are ringed.

This is the spatial companion to RES 06: RES 06 says *how much* and *what kind* of
crowding (global vs pocketed, against the isotropic reference); RES 07 says *where*
it sits in the projection. Both read the same `local_anisotropy.for_ctx` result.
"""

from __future__ import annotations

import numpy as np

from ..render import figure, _svg, _box
from .. import local_anisotropy

GROUPS = 40


@figure
def fig_res_cmap(ctx):
    W, H = 1180, 760
    pad, plot_w = 70, 902
    P = _box(ctx.xy, plot_w, H, pad=pad)
    m = len(P)

    la = local_anisotropy.for_ctx(ctx)
    s = np.asarray(la.score, float)
    s = np.nan_to_num(s, nan=0.0)
    s_hi = max(float(np.quantile(s, 0.97)), 1e-6)
    s_lo = min(float(np.quantile(s, 0.03)), -1e-6)

    pocket_of = np.full(m, -1, int)
    for pi, p in enumerate(la.pockets):
        pocket_of[np.asarray(p.members, int)] = pi
    gc = float(la.global_crowding)

    def dot(i):
        x, y = P[i, 0], P[i, 1]
        if pocket_of[i] >= 0:                                   # pocket member: ringed
            return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="var(--bad)" fill-opacity="0.85" '
                    f'stroke="var(--bad)" stroke-width="0.8"/>')
        si = s[i]
        if si >= 0:
            t = min(si / s_hi, 1.0)
            return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{1.3 + 1.3 * t:.2f}" '
                    f'fill="var(--bad)" fill-opacity="{0.18 + 0.66 * t:.2f}"/>')
        t = min(-si / -s_lo, 1.0)
        return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.3" '
                f'fill="var(--good)" fill-opacity="{0.18 + 0.5 * t:.2f}"/>')

    rng = np.random.default_rng(0)
    order = rng.permutation(m)
    groups = [[] for _ in range(GROUPS)]
    for rank, i in enumerate(order):
        groups[rank % GROUPS].append(dot(int(i)))
    grp_svg = "".join(f'<g class="res07-grp" data-i="{g}">{"".join(groups[g])}</g>' for g in range(GROUPS))

    crowded = gc > 8.0
    read = (f"{len(la.pockets)} crowded pocket(s) ringed" if la.pockets else
            ("globally crowded — color is each item's crowding relative to the dataset's own typical neighborhood"
             if crowded else "roomy — little local crowding structure"))
    head = (f'<text x="{pad}" y="26" fill="var(--ink-soft)" font-size="13" font-weight="700">'
            f'local crowding map · projection recolored by each item&#8217;s neighborhood concentration</text>'
            f'<text x="{pad}" y="44" fill="var(--ink-faint)" font-size="11">'
            f'<tspan fill="var(--bad)">crowded</tspan> vs <tspan fill="var(--good)">open</tspan>, '
            f'relative to the dataset bulk · {read}</text>'
            f'<line x1="{plot_w}" y1="60" x2="{plot_w}" y2="{H - 48}" stroke="var(--rule-soft)" stroke-width="0.8"/>')

    # vertical color-key on the right gutter
    kx, ky0, ky1 = plot_w + 40, 150, H - 150
    key = [f'<text x="{kx:.1f}" y="{ky0 - 16:.1f}" font-size="10" fill="var(--ink-soft)" '
           f'text-anchor="middle">crowding</text>']
    nseg = 28
    for j in range(nseg):
        f = j / (nseg - 1)                                       # 0 bottom (open) .. 1 top (crowded)
        y = ky1 - f * (ky1 - ky0)
        if f >= 0.5:
            col, op = "--bad", 0.18 + 0.66 * ((f - 0.5) / 0.5)
        else:
            col, op = "--good", 0.18 + 0.5 * ((0.5 - f) / 0.5)
        key.append(f'<rect x="{kx - 9:.1f}" y="{y - (ky1 - ky0) / nseg:.1f}" width="18" '
                   f'height="{(ky1 - ky0) / nseg + 1:.1f}" fill="var({col})" fill-opacity="{op:.2f}"/>')
    key += [f'<text x="{kx + 16:.1f}" y="{ky0 + 4:.1f}" font-size="9" fill="var(--bad)">crowded (+z)</text>',
            f'<text x="{kx + 16:.1f}" y="{(ky0 + ky1) / 2 + 3:.1f}" font-size="9" fill="var(--ink-faint)">typical</text>',
            f'<text x="{kx + 16:.1f}" y="{ky1 + 2:.1f}" font-size="9" fill="var(--good)">open (−z)</text>']
    key_svg = "".join(key)

    svg = _svg(W, H, "Reservoir projection recolored by each item's local crowding score; crowded items "
               "read bad-tinted, open items good-tinted, pockets ringed.", head + grp_svg + key_svg)

    vis = max(4, min(GROUPS, int(round(GROUPS * min(1.0, 6000.0 / max(1, m))))))
    ctrl = ('<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;'
            'font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px;color:var(--ink-faint)">'
            f'samples <input id="res07-range" type="range" min="1" max="{GROUPS}" value="{vis}" '
            'style="flex:0 0 220px;accent-color:var(--accent)"> <span id="res07-count"></span></div>')
    script = ("(function(){var T=%d,G=%d;var r=document.getElementById('res07-range'),"
              "c=document.getElementById('res07-count');if(!r)return;"
              "var gs=document.querySelectorAll('.res07-grp');"
              "function ap(){var v=+r.value;for(var i=0;i<gs.length;i++)gs[i].style.display=(i<v)?'':'none';"
              "if(c)c.textContent=Math.round(T*v/G).toLocaleString()+' of '+T.toLocaleString()+' points';}"
              "r.addEventListener('input',ap);ap();})();") % (m, GROUPS)

    legend = ('<span><i class="r"></i> crowded (above the dataset bulk)</span>'
              '<span><i class="g"></i> open (below it)</span>'
              + ('<span><i class="r"></i> ringed = crowded pocket</span>' if la.pockets else ''))

    return {"num": "RES 07", "order": 95.5,
            "name": "Local crowding map", "tech": "field z · projection · slider",
            "why": "The 2-D projection recolored by each item's local crowding score (signed multiscale z of "
                   "its neighborhood concentration vs the dataset's own field). It places the crowding RES 06 "
                   "summarizes — where the tight neighborhoods sit, and whether they form coherent pockets.",
            "svg": ctrl + svg, "script": script,
            "legend": legend,
            "reveal": "<b>Reveals:</b> the spatial layout of crowding. A coherent bad-tinted blob is a "
                      "localized pocket; an even bad wash is global crowding with no pocket; good-tinted "
                      "regions are the open, well-resolved parts of the space.",
            "cls": ""}
