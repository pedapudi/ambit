#!/usr/bin/env python3
"""Guardrail experiment: supervised fine-tuning with ambit's safety terms.

The prior supervised fine-tune (in-batch contrastive on question/section
pairs) gained R@1 but silently collapsed multi-hop retrieval — the failure
class the preservation term and the false-negative guard exist to prevent.
This trainer runs the supervised objective in two arms with identical code:

  control:    L = InfoNCE(q, p) alone            (--lambda-p 0 --no-guard)
  guardrail:  L = InfoNCE(q, p)
              + lambda_p * L_pres(docs vs frozen base)
              with the in-batch false-negative guard: batch documents whose
              BASE-model cosine to my positive exceeds the measured liftoff
              are masked out of my denominator (they are likely unlabeled
              true relatives, not negatives).

A frozen copy of the base encoder runs alongside the trained one to supply
the base embeddings for both the guard mask and the preservation reference
(same batch, same truncation — no length mismatch by construction).

DDP-capable (torchrun), same conventions as train_ambit.py.

  CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 train_guardrail.py \
    --model ~/models/qwen3-embedding-0.6b --pairs ~/ft-legal-data/train_ecfr.jsonl \
    --guard-cos 0.8165 --lambda-p 0.3 --out ~/e6-ckpt-guardrail
"""
import argparse, json, os, time

import numpy as np


def last_token_pool(hidden, attention_mask):
    import torch
    idx = attention_mask.sum(1) - 1
    return hidden[torch.arange(len(hidden), device=hidden.device), idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", required=True, help="jsonl with anchor/positive")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lambda-p", type=float, default=0.3)
    ap.add_argument("--no-guard", action="store_true")
    ap.add_argument("--guard-cos", type=float, default=0.8165,
                    help="measured liftoff cosine under the BASE model")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=128, help="pairs per rank per step")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--holdout-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    a = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    from ambit import training as tr

    ws = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if ws > 1:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed + 7919 * rank)

    rows = [json.loads(l) for l in open(a.pairs)]
    rng = np.random.default_rng(a.seed)
    hold = set(rng.choice(len(rows), int(a.holdout_frac * len(rows)), replace=False))
    train_rows = [r for i, r in enumerate(rows) if i not in hold]
    if rank == 0:
        print(f"{len(train_rows)} train pairs | {len(hold)} held out "
              f"| guard {'OFF' if a.no_guard else f'cos>={a.guard_cos}'} "
              f"| lambda_p {a.lambda_p}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model, padding_side="right")
    model = AutoModel.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(device)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    frozen = AutoModel.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(device)
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)
    if ws > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr)

    def embed(m, texts, grad):
        enc = tok(texts, padding=True, truncation=True, max_length=a.max_tokens,
                  return_tensors="pt").to(device)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            h = m(**enc).last_hidden_state
            z = last_token_pool(h, enc["attention_mask"]).float()
        return F.normalize(z, dim=-1)

    steps = int(a.epochs * len(train_rows) / (a.batch * ws))
    sampler = np.random.default_rng(a.seed + 7919 * rank)
    t0, run_t, run_p = time.time(), 0.0, 0.0
    for step in range(steps):
        idx = sampler.choice(len(train_rows), a.batch, replace=False)
        qs = [f"Instruct: Given a question about U.S. federal regulations, "
              f"retrieve the Code of Federal Regulations section that governs "
              f"the answer.\nQuery: {train_rows[i]['anchor']}" for i in idx]
        ds = [train_rows[i]["positive"][:8000] for i in idx]

        zq = embed(model, qs, grad=True)
        zd = embed(model, ds, grad=True)
        with torch.no_grad():
            bd = embed(frozen, ds, grad=False)

        logits = (zq @ zd.T) / a.temperature
        if not a.no_guard:
            base_sim = bd @ bd.T                       # base-model doc geometry
            guard = (base_sim >= a.guard_cos)
            guard.fill_diagonal_(False)
            logits = logits.masked_fill(guard, torch.finfo(logits.dtype).min)
        l_task = F.cross_entropy(logits, torch.arange(len(zq), device=device))
        l_p = tr.preservation_loss(zd, bd) if a.lambda_p > 0 else zq.new_zeros(())
        loss = l_task + a.lambda_p * l_p
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        run_t += float(l_task); run_p += float(l_p)
        if rank == 0 and (step + 1) % a.log_every == 0:
            k = a.log_every
            print(f"step {step+1}/{steps}  task {run_t/k:.4f}  pres {run_p/k:.5f}  "
                  f"{(step+1)*a.batch*ws/(time.time()-t0):.0f} pairs/s", flush=True)
            run_t = run_p = 0.0

    if ws > 1:
        model = model.module
        dist.barrier()
    if rank == 0:
        model.save_pretrained(a.out, safe_serialization=True)
        tok.save_pretrained(a.out)
        json.dump(vars(a), open(os.path.join(a.out, "train_args.json"), "w"),
                  indent=1)
        print(f"saved checkpoint -> {a.out}", flush=True)
    if ws > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
