#!/usr/bin/env python3
"""v3 — MULTI-POSITIVE fine-tune of Qwen3-Embedding-0.6B for legal retrieval.

Fixes the multi-hop regression of v1/v2 (single-positive MNRL) by training a
multi-positive InfoNCE: every gold of a query is a positive in the SAME row (no
flatten), and positives are identified by GLOBAL gold-uuid so a doc that is a gold
of query q is scored as a positive for q even when it occupies another query's
positive slot in the batch (true multi-positive; no cross-query false negatives —
and 82% of golds here are shared by >=2 queries, so this matters a lot).

Columns (forced order): anchor, positive_1..3, negative_1..5, label=[n_pos,puid1..3].
Loss wrapped in MatryoshkaLoss(1024/768/512/256). DDP-safe: fixed shapes, single
backward (NO GradCache), per-rank batch (gradients synced by DDP).

NOTE (result): this experiment did NOT close the multi-hop gap (set-Recall@10 stayed
~flat vs v2). The loss is correct; the limit is data composition — only ~3.7% of the
training queries are multi-positive. See FINETUNE.md ("v3").

  UNIT_TEST=1 python train_v3.py   -> pure-tensor mask test, exit
  SMOKE=1 CUDA_VISIBLE_DEVICES=0 python train_v3.py -> 3-step plumbing test, exit
  NCCL_CUMEM_ENABLE=0 torchrun --nproc_per_node=2 train_v3.py  -> full run
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

ECFR_INSTRUCT = ("Given a question about U.S. federal regulations, retrieve the Code of "
                 "Federal Regulations section that governs the answer.")
OPEN_INSTRUCT = ("Given a question about U.S. law and regulations, retrieve the document(s) "
                 "that answer it.")
DATA = "train-data"
OUT = "qwen3-emb-legal-v3"
COLS = ["anchor", "positive_1", "positive_2", "positive_3",
        "negative_1", "negative_2", "negative_3", "negative_4", "negative_5", "label"]


class MultiPositiveInfoNCE(nn.Module):
    """sentence_features order = COLS minus label = [anchor, pos1..3, neg1..5] (9).
    label = (B,4) long: [n_pos, puid1, puid2, puid3]  (padded positive cols -> -1)."""

    def __init__(self, model, temperature: float = 0.03):
        super().__init__()
        self.model = model
        self.tau = temperature

    def forward(self, sentence_features, labels):
        assert len(sentence_features) == 9, f"expected 9 feature cols, got {len(sentence_features)}"
        reps = [self.model(sf)["sentence_embedding"] for sf in sentence_features]
        a = F.normalize(reps[0], dim=-1)
        pos = torch.stack([F.normalize(r, dim=-1) for r in reps[1:4]], dim=1)   # (B,3,d)
        neg = torch.stack([F.normalize(r, dim=-1) for r in reps[4:9]], dim=1)   # (B,5,d)
        return self._compute(a, pos, neg, labels)

    def _compute(self, a, pos, neg, labels):
        dev = a.device
        B, d = a.shape
        labels = labels.to(dev).long()
        n_pos = labels[:, 0]                                            # (B,)
        gold = labels[:, 1:4]                                           # (B,3) puids, padded -1
        col = torch.arange(3, device=dev)
        pos_real = col[None, :] < n_pos[:, None]                        # (B,3)

        docs = torch.cat([pos.reshape(B * 3, d), neg.reshape(B * 5, d)], 0)            # (M,d)
        M = docs.shape[0]
        doc_real = torch.cat([pos_real.reshape(B * 3),
                              torch.ones(B * 5, dtype=torch.bool, device=dev)], 0)     # (M,)
        doc_is_pos = torch.cat([torch.ones(B * 3, dtype=torch.bool, device=dev),
                                torch.zeros(B * 5, dtype=torch.bool, device=dev)], 0)  # (M,)
        doc_puid = torch.cat([gold.reshape(B * 3),
                              torch.full((B * 5,), -1, dtype=torch.long, device=dev)], 0)  # (M,)

        sims = (a @ docs.T) / self.tau                                  # (B,M)
        gq = torch.where(pos_real, gold, torch.full_like(gold, -2))     # (B,3) valid golds; pad -> -2
        match = (doc_puid[None, :, None] == gq[:, None, :]).any(-1)     # (B,M) doc's uuid is a gold of q
        pos_mask = match & doc_is_pos[None, :] & doc_real[None, :]      # numerator: positive docs that are q's golds
        denom_mask = doc_real[None, :].expand(B, M)                     # denominator: all real docs

        ninf = torch.finfo(sims.dtype).min
        lse_denom = torch.logsumexp(sims.masked_fill(~denom_mask, ninf), dim=1)
        lse_num = torch.logsumexp(sims.masked_fill(~pos_mask, ninf), dim=1)
        return (lse_denom - lse_num).mean()


def unit_test():
    torch.manual_seed(0)
    loss = MultiPositiveInfoNCE(model=None, temperature=0.05)
    B, d = 3, 8
    a = F.normalize(torch.randn(B, d), dim=-1)
    pos = F.normalize(torch.randn(B, 3, d), dim=-1)
    neg = F.normalize(torch.randn(B, 5, d), dim=-1)
    # q0: golds {10,11}; q1: gold {11} (shares 11 with q0); q2: gold {12}
    labels = torch.tensor([[2, 10, 11, -1], [1, 11, -1, -1], [1, 12, -1, -1]])
    # reproduce the internal masks
    n_pos = labels[:, 0]; gold = labels[:, 1:4]
    pos_real = torch.arange(3)[None, :] < n_pos[:, None]
    doc_real = torch.cat([pos_real.reshape(B * 3), torch.ones(B * 5, dtype=torch.bool)])
    doc_is_pos = torch.cat([torch.ones(B * 3, dtype=torch.bool), torch.zeros(B * 5, dtype=torch.bool)])
    doc_puid = torch.cat([gold.reshape(B * 3), torch.full((B * 5,), -1)])
    gq = torch.where(pos_real, gold, torch.full_like(gold, -2))
    match = (doc_puid[None, :, None] == gq[:, None, :]).any(-1)
    pos_mask = match & doc_is_pos[None, :] & doc_real[None, :]
    # doc indices: 0..2=q0 pos, 3..5=q1 pos, 6..8=q2 pos, 9..23=negs
    assert set(torch.where(pos_mask[0])[0].tolist()) == {0, 1, 3}, "q0 should own its 10,11 AND q1's shared 11"
    assert set(torch.where(pos_mask[1])[0].tolist()) == {1, 3}, "q1's gold 11 is also q0's pos1 (no false neg)"
    assert set(torch.where(pos_mask[2])[0].tolist()) == {6}, "q2 only gold 12"
    assert not pos_mask[:, 9:].any(), "no negative should ever be a positive"
    assert not pos_mask[0, 2] and not pos_mask[1, 4], "padded positive cols excluded"
    l = loss._compute(a, pos, neg, labels)
    assert torch.isfinite(l) and l.item() >= 0, f"loss not valid: {l}"
    print(f"UNIT TEST PASSED — masks correct (cross-query shared gold counted as positive); loss={l.item():.4f}")


def build_dataset():
    from datasets import load_dataset

    def add_instruct(e):
        instr = ECFR_INSTRUCT if e["src"] == "ecfr" else OPEN_INSTRUCT
        return {"anchor": f"Instruct: {instr}\nQuery: {e['anchor']}"}

    ds = load_dataset("json", data_files=f"{DATA}/train_v3.jsonl", split="train")
    ds = ds.map(add_instruct)
    ds = ds.select_columns(COLS)          # force column order; drops 'src'
    return ds.shuffle(seed=0)


def main():
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                        SentenceTransformerTrainingArguments)
    from sentence_transformers.losses import MatryoshkaLoss

    smoke = os.environ.get("SMOKE") == "1"
    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"},
        processor_kwargs={"padding_side": "left"},
    )
    model.max_seq_length = 512

    ds = build_dataset()
    if smoke:
        ds = ds.select(range(64))
    print(f"v3 train: {len(ds)} rows | cols {ds.column_names}", flush=True)

    base = MultiPositiveInfoNCE(model, temperature=0.03)
    loss = MatryoshkaLoss(model, base, matryoshka_dims=[1024, 768, 512, 256])

    args = SentenceTransformerTrainingArguments(
        output_dir=OUT + ("-smoke" if smoke else ""),
        num_train_epochs=2,
        max_steps=3 if smoke else -1,
        per_device_train_batch_size=8 if smoke else 128,
        learning_rate=1e-5,
        warmup_ratio=0.05,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="no" if smoke else "epoch",
        save_total_limit=1,
        dataloader_drop_last=True,
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss)
    trainer.train()
    if smoke:
        print("SMOKE TEST PASSED — ST plumbing OK (labels (B,4), 9 feature cols, loss + backward ran)", flush=True)
        return
    model.save_pretrained(OUT)
    print("saved fine-tuned model ->", OUT, flush=True)


if __name__ == "__main__":
    if os.environ.get("UNIT_TEST") == "1":
        unit_test()
    else:
        main()
