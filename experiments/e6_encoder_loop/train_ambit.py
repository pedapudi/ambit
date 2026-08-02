#!/usr/bin/env python3
"""Unsupervised, measurement-driven fine-tuning of an embedding encoder (E6).

No labels, no positives: the objective is built entirely from ambit's readouts
on the base embedding of the fixed subset —

  L = confusion_loss(z_batch, sigma = measured sigma*)   [guarded]
      + lambda_p * preservation_loss(z_batch, z_base_batch)

with batches drawn by `resolution_weights` (crowded items oversampled) and the
false-negative guard built from the base model's top-m neighbors
(`training.guard_mask`), so the loss never pushes apart pairs the base model
itself considers each other's nearest relatives. Held-out rows (manifest split
== "heldout") are excluded from training and from the sampling weights.

Two stages, one script:
  --lora   LoRA adapters (default r=16, alpha=32) on attention + MLP projections
  --full   full fine-tune (no adapters), lower default lr

The saved output is always a merged, servable checkpoint (vLLM-compatible), so
re-embedding uses the unchanged embed_corpus.py pipeline.

GPU host only. Do not launch without the operator's go-ahead.

  python train_ambit.py --model ~/models/qwen3-embedding-0.6b \
      --subset subset200k --data ~/datasets/Nemotron-Pretraining-Legal-v1 \
      --sigma 0.1226 --lora --out ckpt-r1-lora
"""
import argparse, glob, json, os, time

import numpy as np


def load_texts(data_dir, manifest, max_chars):
    """uuid -> text for the manifest's train rows, streamed from the dataset
    parquet (all four subsets share the (uuid, text) schema)."""
    import pyarrow.parquet as pq
    want = {u for u, s in zip(manifest["uuid"], manifest["split"]) if s == "train"}
    texts = {}
    for f in sorted(glob.glob(os.path.join(data_dir, "*", "*.parquet"))):
        for b in pq.ParquetFile(f).iter_batches(batch_size=2000, columns=["uuid", "text"]):
            for u, t in zip(b.column("uuid").to_pylist(), b.column("text").to_pylist()):
                if u in want and u not in texts:
                    texts[u] = str(t)[:max_chars]
        if len(texts) == len(want):
            break
    missing = len(want) - len(texts)
    if missing:
        raise SystemExit(f"{missing} train uuids not found under {data_dir}")
    return texts


def last_token_pool(hidden, attention_mask):
    """Embedding = hidden state of the last non-pad token (the serving config's
    pooling); assumes right padding."""
    import torch
    idx = attention_mask.sum(1) - 1
    return hidden[torch.arange(len(hidden), device=hidden.device), idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--subset", required=True, help="dir from make_subset.py")
    ap.add_argument("--data", required=True, help="dataset root (subset dirs of parquet)")
    ap.add_argument("--sigma", type=float, required=True,
                    help="measured sigma* of the CURRENT round's embedding")
    ap.add_argument("--out", required=True)
    stage = ap.add_mutually_exclusive_group(required=True)
    stage.add_argument("--lora", action="store_true")
    stage.add_argument("--full", action="store_true")
    ap.add_argument("--lr", type=float, default=None, help="default: 1e-4 lora / 2e-5 full")
    ap.add_argument("--lambda-p", type=float, default=0.3)
    ap.add_argument("--ref-npy", default=None,
                    help="preservation references aligned to the manifest (use "
                         "truncation-matched base embeddings — see make_refs.py; "
                         "default: the subset base.npy, which embeds FULL texts "
                         "and turns preservation into a length distillation)")
    ap.add_argument("--mined-frac", type=float, default=0.5,
                    help="fraction of each batch drawn as mined confusable "
                         "anchor/negative pairs (guarded window pairs), so the "
                         "confusion term sees in-window pairs every step; the "
                         "rest is resolution-weighted")
    ap.add_argument("--mine-window-hi", type=float, default=0.98)
    ap.add_argument("--guard-top-m", type=int, default=5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=256,
                    help="rows per FORWARD. The pair losses see only within-batch "
                         "pairs, and the dry run showed small pair batches let the "
                         "model overfit in-batch pairs while destroying global "
                         "geometry — keep this large (gradient checkpointing makes "
                         "it affordable); --accum does NOT substitute, it averages "
                         "independent small pair batches")
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--no-grad-checkpoint", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-chars", type=int, default=8000)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    a = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    from ambit import knn
    from ambit import training as tr

    # optional DDP (torchrun): each rank forwards its own batch, embeddings are
    # gathered across ranks so the pair losses see the FULL world batch — pair
    # statistics scale with world size (the dry run showed bigger pair batches
    # are strictly better), and DDP averages the gradients.
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if ws > 1:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

    torch.manual_seed(a.seed + 7919 * rank)
    manifest = json.load(open(os.path.join(a.subset, "manifest.json")))
    base = np.load(os.path.join(a.subset, "base.npy")).astype(np.float32)
    train_rows = np.flatnonzero(np.asarray(manifest["split"], object) == "train")

    # measurement-derived batch machinery, from TRAIN rows only
    w = tr.resolution_weights(base[train_rows], a.sigma, floor=0.25)
    topk = knn.topk_cosine(base[train_rows], a.guard_top_m)
    texts = load_texts(a.data, manifest, a.max_chars)
    train_uuid = [manifest["uuid"][i] for i in train_rows]
    if a.ref_npy:
        refs = np.load(a.ref_npy).astype(np.float32)
        if len(refs) != len(manifest["uuid"]):
            raise SystemExit("--ref-npy must be aligned to the manifest")
        ref_all = torch.tensor(refs[train_rows])
    else:
        ref_all = torch.tensor(base[train_rows])

    # mined confusable pairs (train-row indices), from the measured window
    liftoff = float(json.load(open(os.path.join(a.subset, "round0.json")))["liftoff_cos"]) \
        if os.path.exists(os.path.join(a.subset, "round0.json")) else 0.77
    m_a, m_n = tr.mine_confusable_negatives(
        base[train_rows], cos_window=(liftoff, a.mine_window_hi),
        guard_top_m=a.guard_top_m, per_anchor=4, seed=a.seed)
    print(f"mined {len(m_a)} guarded window pairs "
          f"(window {liftoff:.3f}-{a.mine_window_hi})", flush=True)

    if ws > 1:
        device = f"cuda:{local_rank}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model, padding_side="right")
    model = AutoModel.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(device)
    if not a.no_grad_checkpoint:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    if a.lora:
        from peft import LoraConfig, get_peft_model
        if not a.no_grad_checkpoint:
            model.enable_input_require_grads()      # checkpointing + frozen embed layer
        cfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.05,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                         "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
    model.train()
    if ws > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank])
    lr = a.lr or (1e-4 if a.lora else 2e-5)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    steps = int(a.epochs * len(train_rows) / (a.batch * a.accum * ws))
    sampler = np.random.default_rng(a.seed + 7919 * rank)
    if rank == 0:
        print(f"{'LoRA' if a.lora else 'full'} | {len(train_rows)} train rows | "
              f"sigma={a.sigma} lambda_p={a.lambda_p} lr={lr} | {steps} optimizer "
              f"steps | world {ws} (pair batch {a.batch * ws})", flush=True)

    t0, run_c, run_p = time.time(), 0.0, 0.0
    for step in range(steps):
        opt.zero_grad()
        for _ in range(a.accum):
            # exactly a.batch distinct rows on EVERY rank — all_gather requires
            # identical shapes across ranks (variable-size batches deadlock NCCL)
            n_pair_rows = int(a.mined_frac * a.batch) if len(m_a) else 0
            picks = sampler.choice(len(m_a), n_pair_rows // 2, replace=False) \
                if n_pair_rows else np.empty(0, np.int64)
            rows = np.unique(np.concatenate([m_a[picks], m_n[picks]])) \
                if n_pair_rows else np.empty(0, np.int64)
            fill = np.ones(len(train_rows), bool)
            fill[rows] = False
            w_fill = w * fill
            rest = sampler.choice(len(train_rows), a.batch - len(rows),
                                  replace=False, p=w_fill / w_fill.sum())
            rows = np.concatenate([rows, rest])
            batch_texts = [texts[train_uuid[r]] for r in rows]
            enc = tok(batch_texts, padding=True, truncation=True,
                      max_length=a.max_tokens, return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state
            z = last_token_pool(h, enc["attention_mask"]).float()
            if ws > 1:
                # gather the world batch; only the local slice carries grad,
                # DDP's gradient averaging supplies the cross-rank terms
                zs = [torch.zeros_like(z) for _ in range(ws)]
                dist.all_gather(zs, z)
                zs[rank] = z
                rt = torch.tensor(rows, device=device)
                rs = [torch.zeros_like(rt) for _ in range(ws)]
                dist.all_gather(rs, rt)
                z = torch.cat(zs)
                rows_w = torch.cat(rs).cpu().numpy()
            else:
                rows_w = rows
            ex_mask = tr.guard_mask(topk, rows_w)
            ex_mask |= (rows_w[:, None] == rows_w[None, :])   # same row on 2 ranks
            np.fill_diagonal(ex_mask, False)
            guard = torch.tensor(ex_mask, device=device)
            l_c = tr.confusion_loss(z, a.sigma, exclude=guard)
            l_p = tr.preservation_loss(z, ref_all[rows_w].to(device))
            ((l_c + a.lambda_p * l_p) / a.accum).backward()
            run_c += float(l_c) / a.accum
            run_p += float(l_p) / a.accum
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        if rank == 0 and (step + 1) % a.log_every == 0:
            k = a.log_every
            print(f"step {step+1}/{steps}  conf {run_c/k:.5f}  pres {run_p/k:.5f}  "
                  f"{(step+1)*a.batch*a.accum*ws/(time.time()-t0):.0f} ex/s", flush=True)
            run_c = run_p = 0.0

    if ws > 1:
        model = model.module
        dist.barrier()
    if rank == 0:
        if a.lora:
            model = model.merge_and_unload()
        model.save_pretrained(a.out, safe_serialization=True)
        tok.save_pretrained(a.out)
        json.dump(vars(a), open(os.path.join(a.out, "train_args.json"), "w"), indent=1)
        print(f"saved merged checkpoint -> {a.out}", flush=True)
    if ws > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
