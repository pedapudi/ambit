"""RES 11 — Crowding pockets (the merge tree, read without a threshold).

Entities are connected by their shortest bridges (single linkage on the
mutual-reachability metric — the construction inside the default clustering
backend, surfaced instead of flattened). Each tight group is *born* when it first
holds together and *dies* when it merges into an older group (the elder rule); the
bar spans birth → death on the cosine-distance axis and its length is the pocket's
**prominence**. Long bars born near zero are near-duplicate pockets — the crowding
that hurts retrieval most; sample member ids are listed so the finding is
actionable. No flat cut is chosen anywhere.
"""

from __future__ import annotations

import numpy as np

from ..render import figure, _svg
from .. import crowding as cr


def _idstr(v, width: int = 14) -> str:
    s = str(v)
    return s if len(s) <= width else s[: width - 1] + "…"


@figure
def fig_res_pockets(ctx):
    W, H = 920, 520
    L, R, T, B = 90.0, 700.0, 64.0, 470.0

    es = ctx.es
    X = np.asarray(es.X)
    if X.ndim != 2 or len(X) < 64:
        note = (f'<text x="{W/2:.0f}" y="{H/2:.0f}" fill="var(--ink-faint)" font-size="12" '
                f'text-anchor="middle">needs a reservoir sample</text>')
        return {"num": "RES 11", "order": 90.7, "name": "Crowding pockets", "tech": "merge tree",
                "why": "No reservoir available to build the merge tree.",
                "svg": _svg(W, H, "Crowding pockets (no data)", note),
                "legend": "", "reveal": "<b>Reveals:</b> how many over-tight pockets exist, and who is in them.", "cls": ""}

    pk, _ = cr.pockets(X, min_size=8, max_n=4096, max_pockets=10, seed=0)
    all_ids = np.asarray(es.ids) if es.ids is not None else np.arange(len(X))

    body = []
    body.append(f'<text x="{L}" y="28" fill="var(--ink-soft)" font-size="12">'
                f'crowding pockets · merge-tree prominence (birth → merge into the bulk)</text>')
    body.append(f'<text x="{L}" y="46" fill="var(--ink-faint)" font-size="10">'
                f'no threshold chosen · {min(len(X), 4096):,} entities · cosine distance · '
                f'labels list size, scales, and sample member ids</text>')

    if not pk:
        body.append(f'<text x="{(L+R)/2:.0f}" y="{(T+B)/2:.0f}" fill="var(--good)" font-size="13" '
                    f'text-anchor="middle">no prominent tight pockets — the corpus merges as one bulk</text>')
        aria = "Crowding pockets: no prominent tight pockets; the corpus merges as one bulk."
        return {
            "num": "RES 11", "order": 90.7, "name": "Crowding pockets", "tech": "merge tree",
            "why": ("The merge tree of the reservoir (single linkage on mutual reachability) contains no "
                    "component that both forms tightly and stays separate — no over-tight pockets."),
            "svg": _svg(W, H, aria, "".join(body)),
            "legend": '<span><i class="g"></i> healthy: one bulk</span>',
            "reveal": "<b>Reveals:</b> no separated tight pockets at any scale.",
            "cls": "",
        }

    d_max = max(p["death"] for p in pk) * 1.08

    def Xc(v):
        return L + v / max(d_max, 1e-9) * (R - L)

    # axis
    body.append(f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="var(--rule)" stroke-width="1"/>')
    n_t = 5
    for t in range(n_t + 1):
        v = d_max * t / n_t
        xx = Xc(v)
        body.append(f'<line x1="{xx:.1f}" y1="{B}" x2="{xx:.1f}" y2="{B+6}" stroke="var(--rule-soft)" stroke-width="0.7"/>')
        body.append(f'<text x="{xx:.1f}" y="{B+18}" fill="var(--ink-faint)" font-size="9" text-anchor="middle" '
                    f'style="font-variant-numeric:tabular-nums">{v:.2f}</text>')
        body.append(f'<line x1="{xx:.1f}" y1="{T}" x2="{xx:.1f}" y2="{B}" stroke="var(--rule-soft)" '
                    f'stroke-width="0.4" vector-effect="non-scaling-stroke"/>')
    body.append(f'<text x="{(L+R)/2:.1f}" y="{B+34}" fill="var(--ink-soft)" font-size="9.5" text-anchor="middle">'
                f'cosine distance (1 − cos) · pocket forms at the left end, merges into the bulk at the right end</text>')

    # bars, most prominent first
    n = len(pk)
    row_h = min(30.0, (B - T - 10) / max(n, 1))
    for r, p in enumerate(pk):
        y = T + 8 + r * row_h
        x0, x1 = Xc(p["birth"]), Xc(p["death"])
        hot = r == 0
        col = "var(--bad)" if hot else "var(--accent)"
        body.append(f'<rect x="{x0:.1f}" y="{y:.1f}" width="{max(x1-x0, 2.0):.1f}" height="{row_h*0.46:.1f}" '
                    f'rx="2" fill="{col}" fill-opacity="{0.92 if hot else 0.75}"/>')
        body.append(f'<line x1="{x0:.1f}" y1="{y-2:.1f}" x2="{x0:.1f}" y2="{y+row_h*0.46+2:.1f}" '
                    f'stroke="{col}" stroke-width="1.6"/>')
        sample = " · ".join(_idstr(all_ids[m]) for m in p["members"][:3])
        txt = (f'n={p["size"]} · forms at {p["birth"]:.2f} '
               f'· holds for {p["prominence"]:.2f} · {sample}')
        # flip the label to the left of the bar when it would run off the card
        if x1 + 8 + 6.2 * len(txt) > W - 14:
            lx, anchor = x0 - 8, "end"
        else:
            lx, anchor = x1 + 8, "start"
        body.append(f'<text x="{lx:.1f}" y="{y+row_h*0.36:.1f}" fill="var(--ink)" font-size="9.5" '
                    f'text-anchor="{anchor}" style="font-variant-numeric:tabular-nums;paint-order:stroke" '
                    f'stroke="var(--paper)" stroke-width="3">{txt}</text>')

    top = pk[0]
    aria = (f"Crowding pockets from the merge tree: {n} prominent tight pockets ranked by prominence; "
            f"the top pocket has {top['size']} members, forms at cosine distance {top['birth']:.2f} and "
            f"only merges into the bulk at {top['death']:.2f}; sample member ids are listed per pocket.")
    return {
        "num": "RES 11", "order": 90.7, "name": "Crowding pockets", "tech": "merge tree · elder rule",
        "why": ("Tight groups read off the merge tree with no flat cut: each bar spans the scale range over "
                "which a pocket exists as its own thing — born tight on the left, absorbed into the bulk on "
                "the right. Long bars born near zero are near-duplicate pockets; member ids make each one "
                "actionable."),
        "svg": _svg(W, H, aria, "".join(body)),
        "legend": ('<span><i class="r"></i> most prominent pocket</span>'
                   '<span><i class="a"></i> other pockets</span>'),
        "reveal": (f"<b>Reveals:</b> <b>how many</b> over-tight pockets exist ({n}), how tight each is, "
                   f"which entities belong to each, and the scale at which each detaches from the bulk."),
        "cls": "",
    }
