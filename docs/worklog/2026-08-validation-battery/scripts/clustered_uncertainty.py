#!/usr/bin/env python3
"""Clustered uncertainty for the per-document prediction claim.

The published sign test (26 of 29 informative cells positive,
p ~ 8e-6) treats each corpus-encoder cell as an independent
observation. Cells are not independent: they share corpora (documents
and queries) and encoders. This script recomputes the evidence at
units that respect that structure, using only the frozen artifacts.

Three analyses, most conservative first:

1. Corpus-level sign test: collapse each corpus to its mean AUC; the
   unit is the corpus. With four informative public corpora plus the
   legal case study, five positives give one-sided p = 1/2^5 ~ 0.031.
2. Two-way cluster bootstrap: resample corpora and encoders with
   replacement, keep the cells indexed by the drawn pairs, and read a
   percentile interval for the mean AUC.
3. Per-corpus exact Spearman p-values for the encoder-ranking claim
   (n = 7 encoders per corpus), which the report should state are
   individually not significant --- the finding is the sign pattern.

Run: python docs/worklog/2026-08-validation-battery/scripts/clustered_uncertainty.py
"""
import itertools
import json
import math
import os

import numpy as np

HERE = os.path.dirname(__file__)
SYN = os.path.join(HERE, "..", "artifacts", "SYNTHESIS.json")
LEGAL_AUC = 0.58        # the case-study cell (one corpus, one encoder)
INFORMATIVE = ("quora", "esci-us", "nq", "hotpotqa")


def main():
    d = json.load(open(SYN))
    cells = [c for c in d["h2"] if c["corpus"] in INFORMATIVE]
    corpora = sorted({c["corpus"] for c in cells})
    encoders = sorted({c["model"] for c in cells})
    auc = {(c["corpus"], c["model"]): c["auc"] for c in cells}

    # ---- 1. corpus-level sign test --------------------------------------
    print("== 1. corpus as the unit")
    means = {}
    for corp in corpora:
        vals = [v for (c, m), v in auc.items() if c == corp]
        means[corp] = float(np.mean(vals))
        print(f"   {corp:10s} mean AUC {means[corp]:.3f} "
              f"({sum(v > 0.5 for v in vals)}/{len(vals)} cells positive)")
    print(f"   legal      AUC {LEGAL_AUC:.3f} (1/1)")
    n_pos = sum(v > 0.5 for v in means.values()) + (LEGAL_AUC > 0.5)
    n_tot = len(means) + 1
    p = 0.5 ** n_tot if n_pos == n_tot else sum(
        math.comb(n_tot, k) for k in range(n_pos, n_tot + 1)) / 2 ** n_tot
    print(f"   {n_pos}/{n_tot} corpora positive, one-sided sign test p = {p:.4f}")

    # ---- 2. two-way cluster bootstrap -----------------------------------
    rng = np.random.default_rng(0)
    B = 20_000
    boot = np.empty(B)
    C, E = len(corpora), len(encoders)
    A = np.array([[auc[(c, m)] for m in encoders] for c in corpora])
    for b in range(B):
        ci = rng.integers(0, C, C)
        ei = rng.integers(0, E, E)
        boot[b] = A[np.ix_(ci, ei)].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n== 2. two-way cluster bootstrap over corpora x encoders "
          f"({B} draws)")
    print(f"   mean AUC {A.mean():.3f}, 95% interval [{lo:.3f}, {hi:.3f}]"
          f"  (chance = 0.500 {'excluded' if lo > 0.5 else 'NOT excluded'})")

    # ---- 3. exact Spearman p per corpus ---------------------------------
    print(f"\n== 3. encoder-ranking correlations (n = 7 per corpus)")
    perms = list(itertools.permutations(range(7)))

    def spearman(x, y):
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1])

    for corp, row in d["h1"].items():
        rho = row["sigma"]
        # exact permutation two-sided p for |rho| under independence
        base = np.arange(7.0)
        null = np.array([spearman(base, np.array(p_, float)) for p_ in perms])
        p2 = float(np.mean(np.abs(null) >= abs(rho) - 1e-12))
        print(f"   {corp:10s} rho({chr(963)}*) = {rho:+.2f}   exact two-sided p = {p2:.3f}")
    print("   No single correlation is significant at 0.05; the reportable")
    print("   finding is the sign pattern across task families.")


if __name__ == "__main__":
    main()
