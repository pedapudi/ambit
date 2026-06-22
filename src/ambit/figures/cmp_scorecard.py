"""CMP 12 — Representational drift (CKA & distances). A compact scorecard of how
similar two embeddings of the *same items* are, on one
[0,1] track each. Linear CKA is the exact, whole-reservoir headline; the kernel/
distribution estimates (RBF CKA, 1−MMD²) are sampled; Procrustes disparity reads the
other way (higher = more change). Reads only `ctx.cmp`; a placeholder when absent.
"""

from __future__ import annotations

import numpy as np

from ..render import figure, _svg


def _placeholder():
    w, h = 760, 200
    body = (f'<text x="{w/2:.0f}" y="{h/2:.0f}" text-anchor="middle" font-size="12" '
            f'fill="var(--ink-faint)">needs a second embedding — run with --compare</text>')
    return {"num": "CMP 12", "order": 12, "name": "Representational drift (CKA & distances)",
            "tech": "CKA · Procrustes",
            "why": "How similar two embeddings of the same items are (CKA, MMD, Procrustes). "
                   "Run with --compare to diff a second set aligned by id.",
            "svg": _svg(w, h, "Representational drift unavailable without a second embedding.", body),
            "legend": '<span><i class="f"></i> needs --compare</span>',
            "reveal": "<b>Reveals:</b> nothing yet — pass a second embedding with --compare.", "cls": ""}


@figure
def fig_cmp_scorecard(ctx):
    cmp = getattr(ctx, "cmp", None)
    if cmp is None:
        return _placeholder()

    rows = [("linear CKA", cmp.cka_linear, "--accent", "exact · whole reservoir")]
    if cmp.cka_rbf is not None:
        rows.append(("RBF CKA", cmp.cka_rbf, "--accent", "kernel · sampled ≈"))
    if cmp.mmd2 is not None:
        rows.append(("1 − MMD²", float(np.clip(1.0 - cmp.mmd2, 0.0, 1.0)), "--accent", "distribution · sampled ≈"))
    if cmp.procrustes_disp is not None:
        rows.append(("Procrustes disparity", float(np.clip(cmp.procrustes_disp / 2.0, 0.0, 1.0)),
                     "--bad", f"rigid residual = {cmp.procrustes_disp:.2f} · higher = more change"))

    w = 760
    top = 86
    rh = 46
    h = top + rh * len(rows) + (40 if not cmp.same_dim else 18)
    L, R = 230, 700                                   # track extent
    body = []
    body.append(f'<text x="40" y="34" fill="var(--ink-soft)" font-size="13" font-weight="700">'
                f'representational drift · A vs {cmp.label_b}</text>')
    body.append(f'<text x="40" y="52" fill="var(--ink-faint)" font-size="11">'
                f'two embeddings of the same {len(cmp.ids):,} items, aligned by id · '
                f'CKA is invariant to rotation, scale and neuron permutation</text>')
    body.append(f'<text x="40" y="68" fill="var(--caution)" font-size="10">'
                f'global second-moment similarity — read against the neighbor-overlap drift (the local view)</text>')
    body.append(f'<text x="{R}" y="34" fill="var(--ink-faint)" font-size="10" text-anchor="end">'
                f'1.0 = identical representation</text>')

    for i, (name, val, tok, note) in enumerate(rows):
        cy = top + i * rh + rh / 2
        # track 0..1
        body.append(f'<line x1="{L}" y1="{cy:.1f}" x2="{R}" y2="{cy:.1f}" stroke="var(--rule-soft)" stroke-width="3"/>')
        body.append(f'<line x1="{L}" y1="{cy-7:.1f}" x2="{L}" y2="{cy+7:.1f}" stroke="var(--rule)" stroke-width="1"/>')
        body.append(f'<line x1="{R}" y1="{cy-7:.1f}" x2="{R}" y2="{cy+7:.1f}" stroke="var(--rule)" stroke-width="1"/>')
        vx = L + float(np.clip(val, 0.0, 1.0)) * (R - L)
        # lollipop: stem + head
        body.append(f'<line x1="{L}" y1="{cy:.1f}" x2="{vx:.1f}" y2="{cy:.1f}" stroke="var({tok})" stroke-width="3"/>')
        body.append(f'<circle cx="{vx:.1f}" cy="{cy:.1f}" r="5.5" fill="var({tok})"/>')
        body.append(f'<text x="40" y="{cy-3:.1f}" fill="var(--ink)" font-size="12" font-weight="700">{name}</text>')
        body.append(f'<text x="40" y="{cy+12:.1f}" fill="var(--ink-faint)" font-size="9">{note}</text>')
        body.append(f'<text x="{R+0:.1f}" y="{cy-9:.1f}" fill="var({tok})" font-size="13" font-weight="700" '
                    f'text-anchor="end" style="font-variant-numeric:tabular-nums">{val:.3f}</text>')

    if not cmp.same_dim:
        body.append(f'<text x="40" y="{h-18}" fill="var(--caution)" font-size="10.5">'
                    f'dims differ (d_A ≠ d_B) — MMD, energy and Procrustes need equal dims; '
                    f'only CKA is shown (it is dimension-agnostic).</text>')

    aria = (f"Representational-drift scorecard comparing embedding A to {cmp.label_b} over "
            f"{len(cmp.ids):,} id-aligned items. Linear CKA {cmp.cka_linear:.3f} on a 0–1 track where "
            f"1 is an identical representation; "
            + (", ".join(f"{n} {v:.3f}" for n, v, _, _ in rows[1:]) if len(rows) > 1 else "")
            + ".")
    return {
        "num": "CMP 12", "order": 12, "name": "Representational drift (CKA & distances)",
        "tech": "CKA · Procrustes",
        "why": f"How much the representation moved from A to {cmp.label_b}, over {len(cmp.ids):,} "
               f"id-aligned items. Linear CKA = {cmp.cka_linear:.3f} (1 = the same geometry up to rotation, "
               f"scale and permutation; lower = genuinely different) — invariant to those nuisances, so it ignores the "
               f"nuisances that make raw coordinate diffs meaningless. But CKA is a *global* second-moment "
               f"statistic dominated by the top variance directions: it can stay high while local "
               f"neighborhoods reshuffle, so read it against the neighbor-overlap drift (the local view).",
        "svg": _svg(w, h, aria, "".join(body)),
        "legend": '<span><i class="a"></i> CKA / similarity (1 = identical)</span>'
                  '<span><i class="b"></i> Procrustes disparity (higher = more change)</span>',
        "reveal": "<b>Reveals:</b> how much the representation differs between the two embeddings *globally* — a low CKA "
                  "with a large isotropy change is a genuinely different geometry. A high CKA does <b>not</b> mean "
                  "retrieval is unchanged: pair it with the neighbor-overlap drift, which can be low "
                  "(neighborhoods reshuffled) even when CKA is high.",
        "cls": "",
    }
