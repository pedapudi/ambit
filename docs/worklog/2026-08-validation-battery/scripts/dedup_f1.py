import glob, json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.expanduser("~/ambit/experiments/validation_grid"))
from score_cell import load_map, load_queries, load_qrels, QUERY_PREFIX
from sentence_transformers import SentenceTransformer

model_dir, key, out = sys.argv[1], sys.argv[2], sys.argv[3]
uuids, X = load_map(os.path.expanduser(f"~/maps/validation/quora/{model_dir}"))
qids, qtexts = load_queries(os.path.expanduser("~/datasets/validation/quora"))
rels = load_qrels(os.path.expanduser("~/datasets/validation/quora-qrels/test.tsv"))
keep = [i for i, q in enumerate(qids) if q in rels]
qids = [qids[i] for i in keep]; qtexts = [qtexts[i] for i in keep]
m = SentenceTransformer(os.path.expanduser(f"~/models/validation/{model_dir}"),
                        device="cuda:0", model_kwargs={"torch_dtype": "bfloat16"})
m.max_seq_length = 512
Q = m.encode([QUERY_PREFIX[key] + t for t in qtexts], batch_size=256,
             normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
Xt = torch.tensor(X, device="cuda:0"); Qt = torch.tensor(Q, device="cuda:0")
pos = {u: i for i, u in enumerate(uuids)}
# gold duplicate pairs (qid, docid) with rel>=1; predictions: cos >= thr
gold = {(q, d) for q, r in rels.items() for d, s in r.items() if s >= 1}
res = {}
K = 50
vals = torch.zeros(len(Q), K); idxs = torch.zeros(len(Q), K, dtype=torch.int64)
for s in range(0, len(Q), 512):
    S = Qt[s:s+512] @ Xt.T
    v, ix = torch.topk(S, K, dim=1)
    vals[s:s+512] = v.cpu(); idxs[s:s+512] = ix.cpu()
vals = vals.numpy(); idxs = idxs.numpy()
for thr in (0.80, 0.85, 0.90, 0.95, 0.98):
    pred = set()
    for i, q in enumerate(qids):
        for v, j in zip(vals[i], idxs[i]):
            if v >= thr:
                pred.add((q, uuids[j]))
    tp = len(pred & gold)
    p = tp / max(len(pred), 1); r = tp / max(len(gold), 1)
    f1 = 2*p*r/max(p+r, 1e-9)
    res[str(thr)] = {"precision": round(p, 4), "recall": round(r, 4),
                     "f1": round(f1, 4), "n_pred": len(pred)}
res["n_gold_pairs"] = len(gold)
json.dump(res, open(out, "w"), indent=1)
print(model_dir, json.dumps(res, indent=1))
