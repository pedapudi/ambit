"""CMP — Neighbor-overlap drift. The *local*, retrieval-relevant comparison: for each
item, how much of its top-k nearest-neighborhood is the **same** in the two embeddings.
This is the counterweight to CKA (CMP 12): CKA is a global second-moment statistic
dominated by the top variance directions, so it can read "barely changed" while the
neighborhoods that actually drive retrieval reshuffle underneath it. Neighbor overlap
sees that reshuffling directly, localizes it, and — because it compares neighbor
*identities* — is dimension-agnostic (it works even when d_A ≠ d_B). Reads only `ctx.cmp`.
"""

from __future__ import annotations

import numpy as np

from ..render import figure, _svg, _box


def _placeholder():
    w, h = 760, 300
    body = (f'<text x="{w/2:.0f}" y="{h/2:.0f}" text-anchor="middle" font-size="12" '
            f'fill="var(--ink-faint)">needs a second embedding — run with --compare</text>')
    return {"num": "CMP 12a", "order": 11.5, "name": "Neighbor-overlap drift", "tech": "kNN retention",
            "why": "How much each item's nearest-neighborhood is preserved between two embeddings.",
            "svg": _svg(w, h, "Neighbor-overlap drift unavailable without a second embedding.", body),
            "legend": '<span><i class="f"></i> needs --compare</span>',
            "reveal": "<b>Reveals:</b> nothing yet — pass a second embedding with --compare.", "cls": ""}


@figure
def fig_cmp_overlap(ctx):
    cmp = getattr(ctx, "cmp", None)
    if cmp is None:
        return _placeholder()

    ov = np.asarray(cmp.nbr_overlap, float)
    mean = float(cmp.nbr_overlap_mean)
    k = int(cmp.nbr_k)
    W, H = 760, 470

    # ---- left panel: retention histogram --------------------------------------
    L, R, T, B = 60, 360, 70, 392
    nb = 26
    edges = np.linspace(0.0, 1.0, nb + 1)
    counts, _ = np.histogram(np.clip(ov, 0, 1), bins=edges)
    cmax = max(int(counts.max()), 1)

    def hx(v):
        return L + v * (R - L)

    def hy(c):
        return B - (c / cmax) * (B - T)

    body = []
    body.append(f'<text x="{L}" y="34" fill="var(--ink-soft)" font-size="12">'
                f'neighbor retention · share of each item\'s top-{k} neighbors kept</text>')
    body.append(f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="var(--rule)" stroke-width="1"/>')
    bw = (R - L) / nb
    for i in range(nb):
        if counts[i] <= 0:
            continue
        x = hx(edges[i])
        # deeper accent where MORE reshuffled (low retention) — matches the map ramp
        change = 1.0 - 0.5 * (edges[i] + edges[i + 1])
        pct = int(round(20 + change * 80))
        body.append(f'<rect x="{x:.1f}" y="{hy(counts[i]):.1f}" width="{bw-1:.1f}" '
                    f'height="{B-hy(counts[i]):.1f}" fill="color-mix(in srgb, var(--accent) {pct}%, transparent)"/>')
    for g in (0.0, 0.25, 0.5, 0.75, 1.0):
        body.append(f'<line x1="{hx(g):.1f}" y1="{B}" x2="{hx(g):.1f}" y2="{B+5}" stroke="var(--rule)" stroke-width="1"/>')
        body.append(f'<text x="{hx(g):.1f}" y="{B+18}" fill="var(--ink-faint)" font-size="9.5" '
                    f'text-anchor="middle" style="font-variant-numeric:tabular-nums">{g:.2f}</text>')
    body.append(f'<text x="{L}" y="{B+34}" fill="var(--ink-faint)" font-size="9">← reshuffled</text>')
    body.append(f'<text x="{R}" y="{B+34}" fill="var(--ink-faint)" font-size="9" text-anchor="end">retained →</text>')
    mx = hx(np.clip(mean, 0, 1))
    body.append(f'<line x1="{mx:.1f}" y1="{B}" x2="{mx:.1f}" y2="{T}" stroke="var(--accent)" stroke-width="2"/>')
    body.append(f'<text x="{mx:.1f}" y="{T-4}" fill="var(--accent)" font-size="11" font-weight="700" '
                f'text-anchor="middle" style="font-variant-numeric:tabular-nums">mean {mean:.2f}</text>')

    # ---- right panel: where the neighborhoods reshuffled -----------------------
    px0, py0, pw, ph = 400, 70, 320, 322
    xy = np.asarray(cmp.xyA, float)
    chg = 1.0 - np.clip(ov, 0, 1)
    rng = np.random.default_rng(0)
    if len(xy) > 4000:
        sel = rng.choice(len(xy), 4000, replace=False)
        xy, chg = xy[sel], chg[sel]
    P = _box(xy, pw, ph, pad=16) + np.array([px0, py0])
    order = np.argsort(chg)                                  # draw most-changed last (on top)
    body.append(f'<text x="{px0}" y="34" fill="var(--ink-soft)" font-size="12">'
                f'where it reshuffled · deeper = neighborhood changed more</text>')
    for i in order:
        pct = int(round(8 + float(chg[i]) * 92))
        r = 1.3 + 2.2 * float(chg[i])
        body.append(f'<circle cx="{P[i,0]:.1f}" cy="{P[i,1]:.1f}" r="{r:.1f}" '
                    f'fill="color-mix(in srgb, var(--accent) {pct}%, transparent)"/>')
    body.append(f'<rect x="{px0}" y="{py0}" width="{pw}" height="{ph}" fill="none" '
                f'stroke="var(--rule-soft)" stroke-width="0.6"/>')

    byk = " · ".join(f"@{kk} {v:.2f}" for kk, v in sorted(cmp.nbr_overlap_by_k.items()))
    body.append(f'<text x="{px0}" y="{py0+ph+22}" fill="var(--ink-faint)" font-size="9.5" '
                f'style="font-variant-numeric:tabular-nums">mean retention {byk}</text>')

    cka = cmp.cka_linear
    aria = (f"Neighbor-overlap drift between embedding A and {cmp.label_b} over {len(cmp.ids):,} id-aligned "
            f"items: a histogram of per-item top-{k} neighbor retention (mean {mean:.2f}) and a map of A's "
            f"projection with each point shaded by how much its neighborhood reshuffled. For contrast, global "
            f"linear CKA is {cka:.3f}.")
    return {
        "num": "CMP 12a", "order": 11.5, "name": "Neighbor-overlap drift", "tech": "kNN retention · local",
        "why": f"The *local* comparison: for each item, how much of its top-{k} nearest-neighborhood is the "
               f"same in A and {cmp.label_b}. Mean retention {mean:.2f} ({byk}). This is the retrieval-relevant "
               f"counterpart to global CKA ({cka:.3f}) — CKA can stay high while neighborhoods reshuffle, so a "
               f"low retention next to a high CKA means the change is local and fine-grained (exactly what hurts "
               f"retrieval). Dimension-agnostic: it compares neighbor identities, so it works even when dims differ.",
        "svg": _svg(W, H, aria, "".join(body)),
        "legend": '<span><i class="a"></i> deeper = neighborhood reshuffled more</span>'
                  '<span><i class="a"></i> mean retention (accent rule)</span>',
        "reveal": "<b>Reveals:</b> <b>where retrieval structure actually moved</b> — items whose top-k "
                  "neighbors changed. Read it against CKA: high CKA + low retention = a fine-grained, local "
                  "change that a global similarity score smooths over.",
        "cls": "",
    }
