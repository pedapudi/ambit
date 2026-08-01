"""Training-time regularizers and batch mining from the occupancy measures.

The measurement layer (occupancy.py, crowding.py) diagnoses crowding; this module
turns the same mathematics into gradient signal and batch composition for anyone
training or adapting an embedding model on a corpus ambit has profiled. The sound
workflow (measure first, cheapest fix wins, held-out verdicts) is developed in
docs/concepts/continuous-occupancy.md §13; the short version:

1. **Diagnose** with a report; record the liftoff scale and sigma* (the header's
   resolution bandwidth). If crowding is cone-dominated, try the linear fixes
   before training at all; if it is duplicate pockets, fix the data, not the model.
2. **Mine** — `resolution_weights` oversamples the crowded entities into batches;
   `mine_confusable_negatives` drafts negatives from the confusable window with a
   false-negative guard (each anchor's top-m base-model neighbors are never
   negatives: the window is exactly where unlabeled true relatives live).
3. **Regularize** — `confusion_loss` at the *measured* scale spends gradient only
   inside the confusable window (the Φ kernel's gradient dies exponentially beyond
   ~3σ, so far pairs — the global structure — are untouched by construction), while
   `preservation_loss` anchors who-is-similar-to-whom to the frozen base model:

       L = task_loss + lam_c * confusion_loss(z, sigma, exclude=pos_or_guard)
                     + lam_p * preservation_loss(z, z_base)

4. **Verify on held-out data the training never saw**, with the full report and
   the compare (CMP) figures — never with the one functional you optimized.

Losses accept torch tensors (differentiable; torch imported lazily, stays an
optional dependency) or numpy arrays (for tests and analysis). Mining utilities
are numpy and run on the corpus/reservoir, not the batch.
"""

from __future__ import annotations

import numpy as np

from . import occupancy as occ


# ---------------------------------------------------------------- backend shim
def _is_torch(x) -> bool:
    return type(x).__module__.split(".")[0] == "torch"


def _phi_t(z):
    import torch
    return 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))


def _normalize(z, eps=1e-12):
    if _is_torch(z):
        return z / z.norm(dim=-1, keepdim=True).clamp_min(eps)
    z = np.asarray(z, np.float64)
    return z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), eps)


def _pair_chords(zn):
    """(B, B) chord distances from L2-normalized rows. The floor before sqrt is a
    gradient guard: at exactly 0 (the diagonal, or true duplicates) sqrt has an
    infinite derivative and masked entries would still poison backward with
    0·inf = NaN."""
    if _is_torch(zn):
        g = (zn @ zn.T).clamp(-1.0, 1.0)
        return (2.0 - 2.0 * g).clamp_min(1e-12).sqrt()
    g = np.clip(zn @ zn.T, -1.0, 1.0)
    return np.sqrt(np.clip(2.0 - 2.0 * g, 1e-12, None))


# ---------------------------------------------------------------- losses
def confusion_loss(z, sigma: float, exclude=None):
    """Mean pairwise retrieval-confusion probability at query-noise scale `sigma`:
    Φ(−‖zᵢ−zⱼ‖ / 2σ) over ordered pairs (the batch estimate of C(σ)/(n−1)).

    `z`: (B, d) embeddings (normalized internally). `exclude`: optional (B, B)
    boolean mask of pairs to EXCLUDE from the penalty — set it for positive pairs
    and for the false-negative guard (each anchor's top-m base-model neighbors);
    the diagonal is always excluded. Differentiable in torch; the gradient acts
    only on pairs inside the confusable window (≲3σ apart) and vanishes
    exponentially beyond it, so far pairs are untouched by construction.

    Choose `sigma` from measurement (the corpus's sigma* / liftoff scale), not by
    sweep folklore; see occupancy.sigma_star."""
    zn = _normalize(z)
    r = _pair_chords(zn)
    s = 2.0 * max(float(sigma), 1e-12)
    if _is_torch(zn):
        import torch
        p = _phi_t(-r / s)
        keep = ~torch.eye(len(zn), dtype=torch.bool, device=p.device)
        if exclude is not None:
            keep = keep & ~exclude.to(dtype=torch.bool, device=p.device)
        n = keep.sum().clamp_min(1)
        return (p * keep).sum() / n
    p = occ._phi(-r / s)
    keep = ~np.eye(len(zn), dtype=bool)
    if exclude is not None:
        keep &= ~np.asarray(exclude, bool)
    return float(p[keep].mean()) if keep.any() else 0.0


def preservation_loss(z, z_ref, temperature: float = 0.05):
    """Local-structure distillation: per anchor, the KL divergence from the frozen
    reference model's in-batch similarity distribution to the current one
    (softmax over cosine / temperature, self excluded), averaged over anchors.

    This is the neighbor-overlap comparison metric in differentiable form — it
    pins *who is similar to whom* while `confusion_loss` widens margins inside
    the crowded window. `z_ref` should come from the frozen base model on the
    same batch (and should not require grad)."""
    zn, rn = _normalize(z), _normalize(z_ref)
    if _is_torch(zn):
        import torch
        neg_inf = torch.finfo(zn.dtype).min
        eye = torch.eye(len(zn), dtype=torch.bool, device=zn.device)
        ls = ((zn @ zn.T) / temperature).masked_fill(eye, neg_inf).log_softmax(-1)
        with torch.no_grad():
            pt = ((rn @ rn.T) / temperature).masked_fill(eye, neg_inf).softmax(-1)
        return (pt * (pt.clamp_min(1e-12).log() - ls)).sum(-1).mean()
    eye = np.eye(len(zn), dtype=bool)

    def rows(m):
        m = np.where(eye, -np.inf, m / temperature)
        m = m - m.max(-1, keepdims=True)
        e = np.exp(m)
        return e / e.sum(-1, keepdims=True)

    pt, ps = rows(rn @ rn.T), rows(zn @ zn.T)
    return float((pt * (np.log(np.maximum(pt, 1e-12)) - np.log(np.maximum(ps, 1e-12)))).sum(-1).mean())


def uniformity_loss(z, t: float = 2.0):
    """Wang–Isola uniformity, log E exp(−t‖zᵢ−zⱼ‖²) over ordered pairs — the
    classical spreading term, provided for parity. `confusion_loss` is the same
    family with operational units for its one knob (t = 1/(8σ²)); prefer it when
    a measured sigma exists."""
    zn = _normalize(z)
    r = _pair_chords(zn)
    if _is_torch(zn):
        import torch
        keep = ~torch.eye(len(zn), dtype=torch.bool, device=r.device)
        return torch.log((torch.exp(-t * r * r)[keep]).mean())
    keep = ~np.eye(len(zn), dtype=bool)
    return float(np.log(np.exp(-t * r * r)[keep].mean()))


# ---------------------------------------------------------------- batch mining
def guard_mask(topk_idx, rows) -> np.ndarray:
    """(B, B) exclude mask for `confusion_loss` from precomputed base-model top-k
    neighbor indices (`knn.topk_cosine` on the base embedding, computed once per
    round). A pair is guarded — excluded from the penalty — when either batch
    row's corpus index appears in the other's top-k: the confusable window is
    exactly where unlabeled true relatives live, and the batch analogue of the
    miner's false-negative guard keeps the loss from pushing them apart.

    `topk_idx`: (n, k) corpus-level neighbor indices; `rows`: (B,) corpus indices
    of the batch rows. Symmetric, diagonal False."""
    topk_idx = np.asarray(topk_idx)
    rows = np.asarray(rows, np.int64)
    sub = topk_idx[rows]                                    # (B, k) corpus ids
    m = (sub[:, None, :] == rows[None, :, None]).any(-1)    # j in top-k of i
    m |= m.T
    np.fill_diagonal(m, False)
    return m


def resolution_weights(Xn, sigma: float, floor: float = 0.25, block: int = 2048) -> np.ndarray:
    """Sampling weights over the corpus/reservoir that oversample the entities in
    trouble: proportional to per-entity expected collision counts at `sigma`,
    mixed with a uniform floor (`floor` of the mass) so the healthy bulk keeps
    anchoring the geometry. Normalized to sum to 1; feed to a weighted sampler."""
    c = occ.collision_counts(np.asarray(Xn), sigma, block=block)
    total = c.sum()
    prop = c / total if total > 0 else np.full(len(c), 1.0 / len(c))
    w = (1.0 - floor) * prop + floor / len(c)
    return w / w.sum()


def mine_confusable_negatives(Xn, *, cos_window, guard_top_m: int = 5,
                              per_anchor: int = 8, anchors=None,
                              block: int = 2048, seed: int = 0):
    """Draft negative pairs from each anchor's confusable window, with the
    false-negative guard.

    `cos_window` = (lo, hi): candidates must have cosine to the anchor inside the
    window — set it from measurement (lo = the crowding curve's liftoff cosine;
    hi = just below the alignment scale of true pairs). The guard excludes each
    anchor's `guard_top_m` nearest neighbors under THIS (base) embedding: the
    window is exactly where unlabeled true relatives live, and pushing those
    apart is the classic silent failure of hard-negative mining.

    Returns (anchor_idx, negative_idx) int arrays, up to `per_anchor` sampled
    negatives per anchor. Runs blocked over the reservoir; numpy only."""
    Xn = np.ascontiguousarray(np.asarray(Xn), np.float32)
    n = len(Xn)
    lo, hi = float(cos_window[0]), float(cos_window[1])
    if not lo < hi:
        raise ValueError(f"cos_window must be (lo, hi) with lo < hi, got {cos_window!r}")
    rng = np.random.default_rng(seed)
    anchors = np.arange(n) if anchors is None else np.asarray(anchors)
    Xt = np.ascontiguousarray(Xn.T)
    out_a, out_n = [], []
    for s in range(0, len(anchors), block):
        idx = anchors[s:s + block]
        G = Xn[idx] @ Xt
        G[np.arange(len(idx)), idx] = 2.0                        # self out of the running
        # guard: the top-m nearest (besides self, which now ranks first at 2.0)
        # are never negatives — take m+1 so self does not eat a guard slot
        if guard_top_m > 0:
            mm = int(min(guard_top_m + 1, n - 1))
            top = np.argpartition(-G, mm - 1, axis=1)[:, :mm]
            G[np.arange(len(idx))[:, None], top] = 2.0
        for r in range(len(idx)):
            cand = np.flatnonzero((G[r] >= lo) & (G[r] <= hi))
            if cand.size == 0:
                continue
            take = cand if cand.size <= per_anchor else rng.choice(cand, per_anchor, replace=False)
            out_a.extend([int(idx[r])] * len(take))
            out_n.extend(int(t) for t in take)
    return np.asarray(out_a, np.int64), np.asarray(out_n, np.int64)
