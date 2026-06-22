"""Localized density — the unsupervised, multiscale local-concentration field.

What this measures is *local density*: per item, how concentrated its k-nearest-
neighbor neighborhood is (the mean cosine to its k nearest ≈ a k-NN density estimate
in cosine units). It is monotone in density, so it ranks neighborhoods reliably; the
magnitudes are cosine, not calibrated densities (the cosine→density map is strongly
nonlinear in high d). Crowding is *local*: a dataset can be globally diffuse while a
sub-population is crammed together, and a global scalar is non-collapsible over
clusters, so it hides or misreads the pocket. (See the concept note for the geometry.)

We read the DISTRIBUTION of that field at several scales:

  - one mode near the reference       -> roomy / uniform
  - the whole mode shifted far up      -> globally crowded (the "dandelion")
  - a separated high mode               -> a crowded pocket

Detection is calibrated against a synthetic *isotropic reference* (a density-matched
uniform cloud): per scale, a point is crowded when its robust z exceeds the
reference's own upper-tail threshold, unioned across scales, with the angular
clustering + minimum pocket size rejecting scattered false positives. The threshold
comes from the reference, not from a fraction of the dataset's own bulk peak, so a
small but cleanly separated pocket is not swamped by a tall bulk. The reference also
gives the *global* crowding (bulk vs the uniform baseline).

Caveats it does not yet correct for: k-NN density estimation in ~768-d suffers
distance concentration (compressed dynamic range), hubness, and fixed-k boundary
bias; magnitudes should be read as ranks, not absolute densities. Fully unsupervised;
labels, when present, add a neighborhood-relevance overlay (handled by the caller).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import knn
from . import metrics

DEFAULT_SCALES = (10, 50, 200)


@dataclass
class Pocket:
    members: np.ndarray         # reservoir indices in this crowded pocket
    size: int
    concentration: float        # mean local concentration (mean cosine to k-NN)
    margin: float               # mean NN cosine margin (top1 - top2) -> resolution
    isoscore_star: float        # collapsed sliver (low) vs round dense ball (high), n<d-robust
    scale: int                  # k at which the pocket is most anomalous
    z: float                    # peak crowding score in the pocket


@dataclass
class LocalAnisotropy:
    field: np.ndarray           # (m,) per-item local density (mean cos to k-NN) at scale_star
    score: np.ndarray           # (m,) signed robust z at scale_star (single scale; for map color)
    scale_star: int             # headline scale (most separated field)
    scales: tuple
    per_scale: dict             # k -> (m,) local density per scale
    iso_ref: dict               # k -> (n_ref,) isotropic-reference density (the null)
    bulk: float                 # dataset field median at scale_star
    iso_bulk: float             # isotropic-reference median at scale_star
    global_crowding: float      # (bulk - iso_bulk)/iso_mad : how far the bulk sits above uniform
    multimodal: bool            # a separated dense pocket survived detection + clustering?
    valley: Optional[float]     # field value splitting bulk from the crowded set (figure)
    crowded_fraction: float     # fraction of items past the reference-calibrated threshold
    pockets: list               # list[Pocket], densest -> least order


def for_ctx(ctx, **kw):
    """Compute (and memoize on the report Ctx) the localized-anisotropy result.

    RES 06 (the field histogram) and RES 07 (the crowding map) read the *same*
    object, so it is computed once on first access and cached on the Ctx. All the
    numbers come from this generation step — the figures only draw them."""
    cached = getattr(ctx, "_local_aniso", None)
    if cached is None:
        cached = localized_anisotropy(ctx.es.X, labels=getattr(ctx, "labels", None),
                                      mutual=getattr(ctx, "mutual_knn", False), **kw)
        try:
            ctx._local_aniso = cached
        except Exception:
            pass
    return cached


def _topk_sims(X: np.ndarray, kmax: int, block: int = 2048):
    """For each row, the top-`kmax` cosine similarities (descending, self excluded)
    and their column indices. Blocked exact kNN — O(m^2 d), fine on the reservoir."""
    m = X.shape[0]
    kmax = int(min(kmax, m - 1))
    sims = np.empty((m, kmax), np.float32)
    idx = np.empty((m, kmax), np.int32)
    Xt = np.ascontiguousarray(X.T)
    for s in range(0, m, block):
        e = min(s + block, m)
        S = (X[s:e] @ Xt).astype(np.float32, copy=False)
        S[np.arange(e - s), s + np.arange(e - s)] = -2.0          # drop self
        part = np.argpartition(-S, kmax - 1, axis=1)[:, :kmax]
        rows = np.arange(e - s)[:, None]
        vals = S[rows, part]
        order = np.argsort(-vals, axis=1)
        sims[s:e] = vals[rows, order]
        idx[s:e] = part[rows, order]
    return sims, idx


def _mad(a):
    med = float(np.median(a))
    return med, float(1.4826 * np.median(np.abs(a - med)))


def _mutual_mean(sims_k, recip_k):
    """Mean cosine over the *reciprocal* neighbors within the top-k (a hubness-corrected
    density estimate). Rows with no reciprocal neighbor fall back to the least-dense
    value (the column min), so anti-hubs read as sparse rather than NaN."""
    cnt = recip_k.sum(1)
    with np.errstate(invalid="ignore"):
        masked = np.where(recip_k, sims_k, 0.0)
        m = np.where(cnt > 0, masked.sum(1) / np.maximum(cnt, 1), np.nan)
    if np.isnan(m).any():
        finite = m[np.isfinite(m)]
        fill = float(finite.min()) if finite.size else 0.0
        m = np.where(np.isnan(m), fill, m)
    return m


def _angular_components(X, dist_thr):
    """Split a set of (crowded) unit vectors into pockets by angular proximity. The
    threshold sits between within-pocket and between-pocket cosine distance, so a
    single tight cone stays one pocket and distinct cones separate (unlike a k-means
    with a floor of >=3 clusters, which over-segments)."""
    n = len(X)
    if n <= 6000:
        try:
            from sklearn.cluster import AgglomerativeClustering
            ac = AgglomerativeClustering(n_clusters=None, metric="cosine",
                                         linkage="average", distance_threshold=dist_thr)
            return ac.fit_predict(X)
        except Exception:
            pass
    # greedy fallback (memory-safe, order-dependent): grow pocket seeds
    cen = []
    lab = np.full(n, -1, int)
    c0 = 1.0 - dist_thr
    for i in range(n):
        best, bestc = -1, c0
        for j, c in enumerate(cen):
            s = float(X[i] @ c)
            if s > bestc:
                bestc, best = s, j
        if best < 0:
            cen.append(X[i].copy()); lab[i] = len(cen) - 1
        else:
            lab[i] = best
            cen[best] = cen[best] + (X[i] - cen[best]) / 8.0     # nudge the seed
    return lab


def localized_anisotropy(X, *, labels: Optional[np.ndarray] = None,
                         scales=DEFAULT_SCALES, threshold: float = 3.0,
                         n_ref: int = 4000, seed: int = 0,
                         mutual: bool = False) -> LocalAnisotropy:
    X = np.asarray(X, np.float32)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    m, d = X.shape
    scales = tuple(int(k) for k in scales if 2 <= k < m)
    if not scales:
        scales = (max(2, m // 4),)
    kmax = max(scales)

    sims, idx = _topk_sims(X, kmax)
    if mutual:
        recip = knn.reciprocal_mask(idx)                       # mutual neighbors only
        per_scale = {k: _mutual_mean(sims[:, :k], recip[:, :k]) for k in scales}
    else:
        per_scale = {k: sims[:, :k].mean(1) for k in scales}

    # isotropic reference: a uniform same-density sample (k scaled to hold k/N fixed)
    rng = np.random.default_rng(seed)
    nref = int(min(n_ref, m))
    R = rng.standard_normal((nref, d)).astype(np.float32)
    R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
    kref = {k: max(2, int(round(k * nref / m))) for k in scales}
    rsims, ridx = _topk_sims(R, max(kref.values()))
    if mutual:
        rrecip = knn.reciprocal_mask(ridx)                     # mutual neighbors in the null too
        iso_ref = {k: _mutual_mean(rsims[:, :kref[k]], rrecip[:, :kref[k]]) for k in scales}
    else:
        iso_ref = {k: rsims[:, :kref[k]].mean(1) for k in scales}

    # per-scale robust z for the dataset, and for the isotropic reference (the null)
    zs, zref = {}, {}
    for k in scales:
        med, mad = _mad(per_scale[k])
        zs[k] = (per_scale[k] - med) / (mad or 1e-9)
        rmed, rmad = _mad(iso_ref[k])
        zref[k] = (iso_ref[k] - rmed) / (rmad or 1e-9)

    # headline scale = the one whose field has the most extended upper tail (MAD units)
    def sep(k):
        med, mad = _mad(per_scale[k])
        return (np.quantile(per_scale[k], 0.99) - med) / (mad or 1e-9)
    scale_star = max(scales, key=sep)
    field = per_scale[scale_star]
    med_s, mad_s = _mad(field)
    bulk = float(med_s)
    iso_med, iso_mad = _mad(iso_ref[scale_star])
    global_crowding = float((bulk - iso_med) / (iso_mad or 1e-9))

    # map coloring: signed robust z at the headline scale. A single scale (not the
    # max over several correlated scales) keeps the null honest — uniform data stays
    # near zero instead of being inflated by a maximum-of-correlated-z's tail.
    score = zs[scale_star]
    best_k = np.array(scales)[np.argmax(np.stack([zs[k] for k in scales], 1), 1)]

    # ---- reference-calibrated, multiscale detection of crowded points ----
    # For each scale the isotropic reference supplies the null distribution of the
    # field with no local structure; a point is crowded at scale k if its robust z
    # exceeds the reference's own upper-tail threshold there. We union across scales
    # (splitting a small expected-false budget between them) and rely on the angular
    # clustering + minimum pocket size below to reject the scattered, near-orthogonal
    # false positives that any tail threshold lets through.
    #
    # This replaces a histogram valley whose threshold was 2% of the *bulk peak
    # height*. At a fixed reservoir of m points that bar is ~2% of the tallest bin, so
    # a small pocket — tens of items spread over a few bins — sits below it however
    # cleanly it is separated, and the valley-walk swallows it. On a large (m≈20000)
    # reservoir the bar is ~24 items/bin while the real pockets peak at ~8 items/bin,
    # so they read as "0 pockets". (It is not that the reservoir changes size; the bar
    # was simply anchored to the wrong quantity.) The reference null is a fixed,
    # density-matched yardstick keyed to whether a point is denser than the reference
    # ever gets — not to the bulk's height — so small separated pockets surface, and a
    # uniformly-crowded "dandelion" yields none because its z-tail matches the reference's.
    alpha = 0.01
    per = alpha / max(len(scales), 1)
    crowded = np.zeros(m, bool)
    zstar = {}
    for k in scales:
        zstar[k] = float(np.quantile(zref[k], 1.0 - per))
        crowded |= zs[k] > zstar[k]
    valley = float(med_s + zstar[scale_star] * mad_s)     # field split, for the figure

    pockets = []
    cr = np.where(crowded)[0]
    if cr.size >= max(8, int(m * 0.002)):
        thr_d = float(np.clip(1.0 - 0.5 * field[cr].mean(), 0.35, 0.95))
        plabs = _angular_components(X[cr], thr_d)
        for lab in sorted(set(int(x) for x in plabs)):
            if lab < 0:
                continue
            mem = cr[plabs == lab]
            if mem.size < max(5, int(m * 0.001)):
                continue
            cov = np.cov(X[mem].T) if mem.size > 1 else np.eye(d)
            iss = metrics.isoscore_star(metrics.eigs_from_cov(cov), int(mem.size))
            sub = mem[np.argmax(score[mem])]
            pockets.append(Pocket(
                members=mem, size=int(mem.size),
                concentration=float(field[mem].mean()),
                margin=float(np.mean(sims[mem, 0] - sims[mem, 1])),
                isoscore_star=float(iss), scale=int(best_k[sub]),
                z=float(score[mem].max())))
        pockets.sort(key=lambda p: -p.z)

    # a separated dense pocket survived detection + clustering; report the fraction
    # of items that landed in a pocket (the scattered, unclustered tail is dropped)
    multimodal = len(pockets) > 0
    crowded_fraction = float(sum(int(p.size) for p in pockets) / m)

    return LocalAnisotropy(field=field, score=score, scale_star=int(scale_star),
                           scales=scales, per_scale=per_scale, iso_ref=iso_ref,
                           bulk=bulk, iso_bulk=float(iso_med),
                           global_crowding=global_crowding, multimodal=bool(multimodal),
                           valley=(float(valley) if valley is not None else None),
                           crowded_fraction=crowded_fraction, pockets=pockets)
