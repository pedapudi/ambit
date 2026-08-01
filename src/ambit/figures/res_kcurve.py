"""RES 09 — The crowding curve (pair-closeness CDF vs its nulls).

K at each cosine scale = the fraction of random pairs at least that similar — an
exact ECDF of the pair sample, no bins anywhere. Read against two references drawn
with matched sample size: the **uniform-sphere null** (its envelope over replicates)
and the **anisotropy-conditioned null** (angular central Gaussian with the corpus's
own covariance spectrum — the corpus's cone without its clustering). Data above the
ACG curve is crowding *beyond* what anisotropy explains. The scale where the data
first exceeds the uniform envelope is annotated: that is where the space begins to
confuse entities. The footer carries the two continuous-occupancy scalars: the
Stolarsky occupancy-discrepancy z and the resolution bandwidth sigma*.
"""

from __future__ import annotations

import numpy as np

from ..render import figure, _svg
from .. import occupancy as occ


@figure
def fig_res_kcurve(ctx):
    W, H = 920, 520
    L, R, T, B = 74.0, 856.0, 56.0, 420.0

    cos = np.asarray(ctx.cos, np.float64)
    dim = int(getattr(ctx.scan, "dim", 0) or 0)
    n_items = int(getattr(ctx.scan, "n", len(cos)) or len(cos))
    if cos.size < 100 or dim < 2:
        note = (f'<text x="{W/2:.0f}" y="{H/2:.0f}" fill="var(--ink-faint)" font-size="12" '
                f'text-anchor="middle">needs a pair-cosine sample</text>')
        return {"num": "RES 09", "order": 90.5, "name": "Crowding curve", "tech": "pair-closeness CDF",
                "why": "No pair sample available to build the curve.",
                "svg": _svg(W, H, "Crowding curve (no data)", note),
                "legend": "", "reveal": "<b>Reveals:</b> the scale at which crowding begins.", "cls": ""}

    n_pairs = len(cos)
    floor = 0.5 / n_pairs

    # grid: from the bulk (data ~always closer than this) up to cos -> 1
    c_lo = float(np.quantile(cos, 0.02))
    grid = np.linspace(c_lo, 0.9995, 260)
    k_data = occ.exceedance(cos, grid)
    # the honest whole-curve test: a global rank envelope (the pointwise band
    # false-alarms at the bulk edge on pure nulls — measured); liftoff is only
    # annotated when the global test rejects
    p_glob, lift_cos, _, _, env_max = occ.rank_envelope(
        cos, dim, grid=grid[::-1], reps=99, seed=0)
    env_max = env_max[::-1]
    _, env_mean = occ.null_envelope(dim, n_pairs, grid, reps=19, seed=0)
    acg_cos = occ.acg_pair_cos(ctx.eigs, min(n_pairs, 100_000), seed=0)
    k_acg = occ.exceedance(acg_cos, grid)

    s_scalar, s_z = occ.stolarsky_z(cos, dim, reps=24, seed=0)
    sig = occ.sigma_star(cos, n_items, tol=1.0)

    # ---- log-y mapping ------------------------------------------------------
    y_lo = np.log10(floor)
    def Y(k):
        return B - (np.log10(max(k, floor)) - y_lo) / (0.0 - y_lo) * (B - T)
    def X(c):
        return L + (c - grid[0]) / (grid[-1] - grid[0]) * (R - L)

    def poly(kv, stroke, width, dash=""):
        pts = " ".join(f"{X(c):.1f},{Y(k):.1f}" for c, k in zip(grid, kv))
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return (f'<polyline points="{pts}" fill="none" stroke="{stroke}" '
                f'stroke-width="{width}"{d} vector-effect="non-scaling-stroke"/>')

    body = []
    body.append(f'<text x="{L}" y="28" fill="var(--ink-soft)" font-size="12">'
                f'crowding curve · fraction of pairs at least this similar</text>')
    body.append(f'<text x="{L}" y="46" fill="var(--ink-faint)" font-size="10">'
                f'exact CDF — no bins, no lattice · log scale · {n_pairs:,} sampled pairs · {dim}-d</text>')

    # axes + log gridlines
    body.append(f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="var(--rule)" stroke-width="1"/>')
    body.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{B}" stroke="var(--rule)" stroke-width="1"/>')
    dec = 0
    while 10.0 ** (-dec) >= floor:
        k = 10.0 ** (-dec)
        yy = Y(k)
        body.append(f'<line x1="{L}" y1="{yy:.1f}" x2="{R}" y2="{yy:.1f}" stroke="var(--rule-soft)" '
                    f'stroke-width="0.5" vector-effect="non-scaling-stroke"/>')
        lab = "1" if dec == 0 else f"10⁻{dec}" if dec < 10 else f"1e-{dec}"
        body.append(f'<text x="{L-8}" y="{yy+3.2:.1f}" fill="var(--ink-faint)" font-size="9" '
                    f'text-anchor="end" style="font-variant-numeric:tabular-nums">{lab}</text>')
        dec += 1
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        c = grid[0] + frac * (grid[-1] - grid[0])
        xx = X(c)
        body.append(f'<line x1="{xx:.1f}" y1="{B}" x2="{xx:.1f}" y2="{B+6}" stroke="var(--rule-soft)" stroke-width="0.7"/>')
        body.append(f'<text x="{xx:.1f}" y="{B+18}" fill="var(--ink-faint)" font-size="9" text-anchor="middle" '
                    f'style="font-variant-numeric:tabular-nums">{c:+.2f}</text>')
    body.append(f'<text x="{(L+R)/2:.1f}" y="{B+34}" fill="var(--ink-soft)" font-size="9.5" text-anchor="middle">'
                f'pair cosine — the size of a retrieval neighborhood (a cap on the sphere) → '
                f'crowding lives to the right</text>')

    # excess shading: where data exceeds the uniform envelope
    lift = np.flatnonzero(k_data > env_max)     # excess region (display shading)
    if lift.size:
        seg = []
        for c, kd, ke in zip(grid[lift[0]:], k_data[lift[0]:], env_max[lift[0]:]):
            seg.append((X(c), Y(kd), Y(max(ke, floor))))
        top = " ".join(f"{x:.1f},{yd:.1f}" for x, yd, _ in seg)
        bot = " ".join(f"{x:.1f},{ye:.1f}" for x, _, ye in reversed(seg))
        body.append(f'<polygon points="{top} {bot}" fill="color-mix(in srgb, var(--bad) 16%, transparent)"/>')

    body.append(poly(env_mean, "var(--ink-faint)", 1.0, dash="2 3"))
    body.append(poly(env_max, "var(--ink-faint)", 1.2, dash="5 3"))
    body.append(poly(k_acg, "var(--good)", 1.4))
    body.append(poly(k_data, "var(--accent)", 2.2))

    # liftoff marker — annotated only when the whole-curve rank test rejects
    if lift_cos is not None:
        xx = X(lift_cos)
        body.append(f'<line x1="{xx:.1f}" y1="{T}" x2="{xx:.1f}" y2="{B}" stroke="var(--bad)" '
                    f'stroke-width="1.2" stroke-dasharray="4 3" vector-effect="non-scaling-stroke"/>')
        anchor = "end" if xx > (L + R) / 2 else "start"
        dx = -6 if anchor == "end" else 6
        body.append(f'<text x="{xx+dx:.1f}" y="{T+14}" fill="var(--bad)" font-size="10.5" font-weight="700" '
                    f'text-anchor="{anchor}" style="paint-order:stroke" stroke="var(--paper)" '
                    f'stroke-width="3">crowding begins ≈ cos {lift_cos:+.2f} · global p = {p_glob:.3f}</text>')
        verdict = (f"exceeds the alpha-critical global envelope from cos {lift_cos:+.2f} "
                   f"(whole-curve rank test, p = {p_glob:.3f})")
    else:
        body.append(f'<text x="{R}" y="{T+14}" fill="var(--good)" font-size="10.5" font-weight="700" '
                    f'text-anchor="end">globally consistent with uniformity (p = {p_glob:.2f})</text>')
        verdict = f"is globally consistent with uniformity (whole-curve rank test, p = {p_glob:.2f})"

    # footer scalars — two stacked lines so they can never collide
    body.append(f'<text x="{L}" y="{H-34}" fill="var(--ink-soft)" font-size="10.5" '
                f'style="font-variant-numeric:tabular-nums">occupancy discrepancy (Stolarsky): mean chord '
                f'{s_scalar:.4f} · z = {s_z:+,.0f} vs uniform null</text>')
    body.append(f'<text x="{L}" y="{H-16}" fill="var(--ink-soft)" font-size="10.5" '
                f'style="font-variant-numeric:tabular-nums">resolution bandwidth σ* = {sig:.3f} '
                f'— query noise at ≤1 expected collision</text>')

    aria = (f"Crowding curve: the fraction of random pairs above each cosine, on a log scale, "
            f"for the dataset against a uniform-sphere null envelope and an anisotropy-matched "
            f"reference; the data {verdict}; occupancy-discrepancy z {s_z:+.0f}; resolution "
            f"bandwidth {sig:.3f}.")
    return {
        "num": "RES 09", "order": 90.5, "name": "Crowding curve", "tech": "pair-closeness CDF · Ripley K",
        "why": ("The exact cumulative pair-closeness curve (no bins, no lattice) read against two nulls: "
                "the uniform sphere's α-critical global envelope (dashed; a whole-curve rank test — the "
                "liftoff mark appears only when it rejects, with its p printed) and the corpus's own "
                "anisotropy cone without its clustering (solid reference). Height above the cone "
                "reference is crowding that anisotropy cannot explain; the marked scale is where the "
                "space starts confusing entities."),
        "svg": _svg(W, H, aria, "".join(body)),
        "legend": ('<span><i class="a"></i> this dataset</span>'
                   '<span><i class="g"></i> anisotropy-matched reference</span>'
                   '<span><i class="dash"></i> α-critical global envelope (rank test)</span>'
                   '<span><i class="r"></i> excess close pairs (crowding)</span>'),
        "reveal": ("<b>Reveals:</b> the <b>scale</b> at which crowding begins, and how much of it is "
                   "explained by the anisotropy cone versus genuine clustering — plus the two continuous "
                   "occupancy scalars (Stolarsky discrepancy z, resolution bandwidth σ*)."),
        "cls": "",
    }
