#!/usr/bin/env python3
"""Fine-tune Qwen3-Embedding-0.6B for legal question->document retrieval (v2).
Full fine-tune, asymmetric (query instruction baked into the anchor, matching the
eval harness), MultipleNegativesRankingLoss (in-batch negatives gathered across GPUs
+ 5 mined hard negatives per row) wrapped in MatryoshkaLoss (1024/768/512/256).
Trains BOTH sources (eCFR + Set B synthetic) as one uniform 7-column schema so
multi-GPU DDP stays in sync — run mine_synth_negs.py first to give the synthetic rows
their hard negatives (see FINETUNE.md, "Multi-GPU lessons")."""
import os
import torch
from datasets import load_dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments)
from sentence_transformers.losses import MultipleNegativesRankingLoss, MatryoshkaLoss

# instructions MUST match the eval harness (run_eval.py / run_open_eval.py)
ECFR_INSTRUCT = ("Given a question about U.S. federal regulations, retrieve the Code of "
                 "Federal Regulations section that governs the answer.")
OPEN_INSTRUCT = ("Given a question about U.S. law and regulations, retrieve the document(s) "
                 "that answer it.")
DATA = "train-data"
OUT = "qwen3-emb-legal-v2"


def instruct(ds, instr):
    return ds.map(lambda e: {"anchor": f"Instruct: {instr}\nQuery: {e['anchor']}"})


def main():
    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"},
        processor_kwargs={"padding_side": "left"},
    )
    model.max_seq_length = 512

    ds_ecfr = instruct(load_dataset("json", data_files=f"{DATA}/train_ecfr.jsonl", split="train"), ECFR_INSTRUCT)
    ds_synth = instruct(load_dataset("json", data_files=f"{DATA}/train_synth.jsonl", split="train"), OPEN_INSTRUCT)
    print(f"train_ecfr: {len(ds_ecfr)} (cols {ds_ecfr.column_names}) | train_synth: {len(ds_synth)}", flush=True)

    base = MultipleNegativesRankingLoss(model, gather_across_devices=True)  # pools in-batch negatives across GPUs (single uniform dataset)
    loss = MatryoshkaLoss(model, base, matryoshka_dims=[1024, 768, 512, 256])

    args = SentenceTransformerTrainingArguments(
        output_dir=OUT,
        num_train_epochs=2,
        per_device_train_batch_size=128,
        learning_rate=2e-5,
        warmup_ratio=0.05,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        dataloader_drop_last=True,
        report_to="none",
    )
    # Both datasets share the SAME 7-column schema (anchor, positive, negative_1..5),
    # so the DDP ranks always receive same-shaped batches. v1 trained on ds_ecfr ALONE
    # because train_synth then had 0 hard-neg columns; mixing the two shapes desynced
    # the ranks -> NCCL hang at a fixed step. mine_synth_negs.py removes that mismatch.
    trainer = SentenceTransformerTrainer(
        model=model, args=args,
        train_dataset={"ecfr": ds_ecfr, "synth": ds_synth},
        loss=loss,
    )
    trainer.train()
    model.save_pretrained(OUT)
    print("saved fine-tuned model ->", OUT, flush=True)


if __name__ == "__main__":
    main()
