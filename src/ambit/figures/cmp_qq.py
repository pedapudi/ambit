from ..render import figure, _box, _svg, _local_density
from .. import metrics
import numpy as np
import math


def _norm_ppf(p, sd):
    """Inverse CDF of N(0, sd) via scipy if present, else Acklam's rational approx."""
    p = np.asarray(p, float)
    try:
        from scipy.stats import norm
        return norm.ppf(p) * sd
    except Exception:
        pass
    # Acklam's algorithm for the standard-normal quantile, then scale by sd.
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    out = np.empty_like(p)
    lo = p < plow
    hi = p > phigh
    mid = ~(lo | hi)
    # lower tail
    if lo.any():
        q = np.sqrt(-2 * np.log(p[lo]))
        out[lo] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                  ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    # upper tail
    if hi.any():
        q = np.sqrt(-2 * np.log(1 - p[hi]))
        out[hi] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                   ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    # central region
    if mid.any():
        q = p[mid] - 0.5
        r = q * q
        out[mid] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
                   (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    return out * sd


@figure
def fig_cmp_qq(ctx):
    W, H = 612, 560
    L, R, T, B = 92, 520, 60, 488          # plot square in svg coords
    cos = np.asarray(ctx.cos, float)
    sd = float(metrics.isotropy_ref(ctx.scan.dim))   # ~N(0, 1/dim) reference

    # --- matched-quantile sampling over the central body (clip extreme tails) ---
    probs = np.linspace(0.01, 0.99, 65)
    data_q = np.quantile(cos, probs)                 # empirical dataset cosine quantiles
    ref_q = _norm_ppf(probs, sd)                     # isotropic-reference cosine quantiles

    # shared cosine-value axis: cover both series symmetrically, round to nice bounds
    vmax = float(max(np.abs(data_q).max(), np.abs(ref_q).max(), 4 * sd))
    vmax = math.ceil(vmax * 10) / 10.0               # nice 0.1 step
    vmin = -vmax
    span = vmax - vmin

    def X(v):  # reference quantile -> x (horizontal)
        return L + (np.asarray(v) - vmin) / span * (R - L)

    def Y(v):  # dataset quantile -> y (vertical, flipped)
        return B - (np.asarray(v) - vmin) / span * (B - T)

    rx = X(ref_q)
    dy = Y(data_q)
    # diagonal endpoints (value v maps to (X(v), Y(v)))
    dx0, dy0 = float(X(vmin)), float(Y(vmin))
    dx1, dy1 = float(X(vmax)), float(Y(vmax))

    body = []

    # ---- over-concentration band: where data quantile > reference quantile -------
    # (dataset cosine larger than isotropic -> more crowded -> over-concentrated).
    # Build a polygon: along the curve forward, back along its diagonal projection.
    over = data_q > ref_q
    if over.any():
        fwd, bwd = [], []
        for i in range(len(probs)):
            if over[i]:
                fwd.append(f"{rx[i]:.1f},{dy[i]:.1f}")
                # the parity point on the diagonal at this reference quantile
                bwd.append(f"{rx[i]:.1f},{float(Y(ref_q[i])):.1f}")
        pts = " ".join(fwd) + " " + " ".join(reversed(bwd))
        body.append(f'<polygon points="{pts}" '
                    f'fill="color-mix(in srgb, var(--bad) 22%, var(--paper))" stroke="none"/>')

    # ---- hairline gridlines at 0.05 cosine increments on both axes ---------------
    ticks_minor = np.round(np.arange(vmin, vmax + 1e-9, 0.05), 3)
    body.append('<g stroke="var(--rule-soft)" stroke-width="0.6" vector-effect="non-scaling-stroke">')
    for t in ticks_minor:
        xx = float(X(t))
        body.append(f'<line x1="{xx:.1f}" y1="{T}" x2="{xx:.1f}" y2="{B}"/>')
        yy = float(Y(t))
        body.append(f'<line x1="{L}" y1="{yy:.1f}" x2="{R}" y2="{yy:.1f}"/>')
    body.append('</g>')

    # ---- plot-square baseline rules (two axes, no frame) -------------------------
    body.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{B}" stroke="var(--rule)" vector-effect="non-scaling-stroke"/>')
    body.append(f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="var(--rule)" vector-effect="non-scaling-stroke"/>')

    # ---- major ticks + mono tabular labels at 0.1 cosine increments --------------
    ticks_major = np.round(np.arange(vmin, vmax + 1e-9, 0.1), 2)
    for t in ticks_major:
        xx = float(X(t))
        body.append(f'<line x1="{xx:.1f}" y1="{B}" x2="{xx:.1f}" y2="{B+6}" '
                    f'stroke="var(--ink-faint)" vector-effect="non-scaling-stroke"/>')
        body.append(f'<text x="{xx:.1f}" y="{B+18}" fill="var(--ink-faint)" font-size="10" '
                    f'text-anchor="middle" style="font-variant-numeric:tabular-nums">{t:+.1f}</text>')
        yy = float(Y(t))
        body.append(f'<line x1="{L-6}" y1="{yy:.1f}" x2="{L}" y2="{yy:.1f}" '
                    f'stroke="var(--ink-faint)" vector-effect="non-scaling-stroke"/>')
        body.append(f'<text x="{L-9}" y="{yy+3.5:.1f}" fill="var(--ink-faint)" font-size="10" '
                    f'text-anchor="end" style="font-variant-numeric:tabular-nums">{t:+.1f}</text>')

    # ---- minor ticks at the 0.05 midpoints (short marks on the axes) -------------
    body.append('<g stroke="var(--ink-faint)" vector-effect="non-scaling-stroke">')
    for t in ticks_minor:
        if round(t * 10) % 1 == 0 and abs(t * 10 - round(t * 10)) < 1e-6 and round(t, 2) in set(np.round(ticks_major, 2)):
            continue
        xx = float(X(t))
        body.append(f'<line x1="{xx:.1f}" y1="{B}" x2="{xx:.1f}" y2="{B+3}"/>')
        yy = float(Y(t))
        body.append(f'<line x1="{L-3}" y1="{yy:.1f}" x2="{L}" y2="{yy:.1f}"/>')
    body.append('</g>')

    # ---- identity diagonal: dashed reference rule (parity = isotropic) -----------
    body.append(f'<line x1="{dx0:.1f}" y1="{dy0:.1f}" x2="{dx1:.1f}" y2="{dy1:.1f}" '
                f'stroke="var(--ink-faint)" stroke-dasharray="3 3" vector-effect="non-scaling-stroke"/>')
    mxd, myd = (dx0 + dx1) / 2 + 60, (dy0 + dy1) / 2 - 60
    body.append(f'<text x="{mxd:.1f}" y="{myd:.1f}" fill="var(--ink-faint)" font-size="9.5" '
                f'transform="rotate(-45 {mxd:.1f} {myd:.1f})" text-anchor="middle">identity · isotropic parity</text>')

    # ---- widest-gap caliper: max vertical separation curve->diagonal -------------
    gaps = data_q - ref_q
    gi = int(np.argmax(np.abs(gaps)))
    gx = float(rx[gi])
    gy_curve = float(dy[gi])
    gy_diag = float(Y(ref_q[gi]))
    body.append(f'<line x1="{gx:.1f}" y1="{gy_diag:.1f}" x2="{gx:.1f}" y2="{gy_curve:.1f}" '
                f'stroke="var(--bad)" stroke-width="1" stroke-dasharray="2 2" vector-effect="non-scaling-stroke"/>')
    body.append(f'<text x="{gx-9:.1f}" y="{(gy_curve+gy_diag)/2:.1f}" fill="var(--bad)" font-size="9.5" '
                f'text-anchor="end" style="font-variant-numeric:tabular-nums">'
                f'Δ{gaps[gi]:+.2f} widest gap</text>')

    # ---- Q-Q curve + decile dots -------------------------------------------------
    poly = " ".join(f"{rx[i]:.1f},{dy[i]:.1f}" for i in range(len(probs)))
    body.append(f'<polyline points="{poly}" fill="none" stroke="var(--ink)" stroke-width="1.4" '
                f'stroke-linejoin="round" vector-effect="non-scaling-stroke"/>')
    for q in np.linspace(0.1, 0.9, 9):
        i = int(np.argmin(np.abs(probs - q)))
        body.append(f'<circle cx="{rx[i]:.1f}" cy="{dy[i]:.1f}" r="2.2" fill="var(--ink)" '
                    f'vector-effect="non-scaling-stroke"/>')

    # ---- extreme-tail endpoints: fine markers (sparsest / densest) ---------------
    body.append(f'<circle cx="{rx[0]:.1f}" cy="{dy[0]:.1f}" r="2.3" fill="none" stroke="var(--good)" '
                f'stroke-width="1.2" vector-effect="non-scaling-stroke"/>')
    body.append(f'<text x="{rx[0]+10:.1f}" y="{dy[0]-8:.1f}" fill="var(--ink-soft)" font-size="9.5" '
                f'style="font-variant-numeric:tabular-nums">q0.01 · sparsest pairs — most room</text>')
    body.append(f'<circle cx="{rx[-1]:.1f}" cy="{dy[-1]:.1f}" r="2.3" fill="none" stroke="var(--bad)" '
                f'stroke-width="1.2" vector-effect="non-scaling-stroke"/>')
    body.append(f'<text x="{rx[-1]-8:.1f}" y="{dy[-1]+15:.1f}" fill="var(--ink-soft)" font-size="9.5" '
                f'text-anchor="end" style="font-variant-numeric:tabular-nums">q0.99 · densest pairs — crowded</text>')

    # ---- axis titles -------------------------------------------------------------
    body.append(f'<text x="{(L+R)/2:.0f}" y="530" fill="var(--ink-soft)" font-size="11" '
                f'text-anchor="middle">reference cosine quantile  (isotropic N(0, 1/dim) · sd={sd:.3f})</text>')
    body.append(f'<text x="22" y="{(T+B)/2:.0f}" fill="var(--ink-soft)" font-size="11" text-anchor="middle" '
                f'transform="rotate(-90 22 {(T+B)/2:.0f})">dataset cosine quantile  (random pairs)</text>')

    # ---- header read -------------------------------------------------------------
    frac_over = float(over.mean())
    body.append(f'<text x="{L}" y="26" fill="var(--ink-soft)" font-size="11">matched cosine quantiles (q,q)</text>')
    body.append(f'<text x="{L}" y="40" fill="var(--ink-faint)" font-size="9.5">'
                f'45° diagonal = cosine spectrum of an isotropic corpus</text>')
    tone = "var(--bad)" if frac_over >= 0.5 else "var(--good)"
    verdict = "over-concentrated" if frac_over >= 0.5 else "more dispersed than isotropic"
    body.append(f'<text x="{L}" y="54" fill="{tone}" font-size="11" font-weight="700" '
                f'style="font-variant-numeric:tabular-nums">{frac_over*100:.0f}% of quantiles above identity · {verdict}</text>')

    aria = ("Q-Q occupancy curve: cosine quantiles of the dataset versus the isotropic "
            f"reference N(0, 1/dim) with sd {sd:.3f}. The matched-quantile curve sits above the "
            "45-degree identity diagonal across the body of the distribution, meaning random pairs "
            "in this dataset are systematically more similar (more concentrated) than an isotropic "
            "corpus; the region where it over-concentrates is band-shaded, the widest vertical gap "
            "is calipered, and both axes carry hairline gridlines and ticks at 0.05 and 0.1 cosine "
            "increments.")

    legend = ('<span><i class="r"></i> Q-Q curve (dataset vs isotropic)</span>'
              '<span><i class="b"></i> over-concentration band</span>'
              '<span><i class="f"></i> identity · isotropic parity</span>')

    reveal = ("<b>Reveals:</b> whether random-pair similarities are more crowded than an isotropic "
              "corpus of the same dimensionality — the curve bowing above identity quantifies the "
              "anisotropy as a shift in the whole cosine spectrum, not just the mean.")

    return {"num": "CMP 11", "order": 11, "name": "Q-Q occupancy curve", "tech": "quantiles",
            "why": ("Quantile-quantile of the random-pair cosine sample against the isotropic reference "
                    "N(0, 1/dim): a curve above the identity diagonal means the dataset packs pairs "
                    "closer together than an isotropic corpus would, across the whole distribution."),
            "svg": _svg(W, H, aria, "".join(body)),
            "legend": legend, "reveal": reveal, "cls": ""}
