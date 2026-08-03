import glob, json
import numpy as np

def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])

def auc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    s = score[order]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]: j += 1
        if j > i: ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0: return float("nan")
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

cells = {}
for f in glob.glob("/home/sunil/validation-runs/*.score.json"):
    base = f.replace(".score.json", "")
    parts = base.split("/")[-1].split(".")
    corpus, model = parts[0], ".".join(parts[1:])
    r = json.load(open(base + ".readout.json"))
    s = json.load(open(f))
    cells.setdefault(corpus, {})[model] = (r, s)

print("=== H1: within-corpus Spearman across encoders ===")
h1 = {}
for c, mm in sorted(cells.items()):
    if len(mm) < 4: continue
    rows = [(r["sigma_star"], r["sigma_star"]/r["sigma_star_uniform"],
             r["mean_pair_cos"], r["effective_rank"], s["ndcg@10"]) for r, s in mm.values()]
    ss, ratio, mc, er, nd = map(list, zip(*rows))
    sp_s, sp_r = spearman(ss, nd), spearman(ratio, nd)
    sp_m, sp_e = spearman(mc, nd), spearman(er, nd)
    h1[c] = {"n": len(rows), "sigma": sp_s, "ratio": sp_r, "meancos": sp_m, "effrank": sp_e}
    print(f"{c:12s} n={len(rows)}  Sp(sigma*)={sp_s:+.3f}  Sp(ratio)={sp_r:+.3f}  ctrl(meancos)={sp_m:+.3f}  ctrl(effrank)={sp_e:+.3f}")

print()
print("=== H2: per-cell collision->failure AUC ===")
h2 = []
for c, mm in sorted(cells.items()):
    for m, (r, s) in sorted(mm.items()):
        try:
            z = np.load(f"/home/sunil/validation-runs/{c}.{m}.cc.npz", allow_pickle=False)
        except FileNotFoundError:
            continue
        uu = list(z["uuid"]); cc = z["cc"]
        pos = {u: i for i, u in enumerate(uu)}
        fail = [pos[d] for d in s["h2"]["fail_docs"] if d in pos]
        ok = [pos[d] for d in s["h2"]["ok_docs"] if d in pos]
        if len(fail) < 10 or len(ok) < 10: continue
        idx = np.array(fail + ok)
        lab = np.array([True]*len(fail) + [False]*len(ok))
        a = auc(cc[idx], lab)
        h2.append((c, m, a, len(fail), len(ok)))
        print(f"{c:12s} {m[:32]:34s} AUC={a:.3f}  (fail={len(fail)}, ok={len(ok)})")
if h2:
    aucs = [x[2] for x in h2]
    print(f"H2 across {len(h2)} cells: mean AUC {np.mean(aucs):.3f}, >0.5 in {sum(a>0.5 for a in aucs)}/{len(h2)} cells")
json.dump({"h1": h1, "h2": [{"corpus": c, "model": m, "auc": a, "n_fail": nf, "n_ok": no} for c, m, a, nf, no in h2]},
          open("/home/sunil/validation-runs/SYNTHESIS.json", "w"), indent=1)
