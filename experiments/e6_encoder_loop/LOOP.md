# E6 encoder loop — run log

Unsupervised measurement-driven fine-tuning of the legal-corpus encoder
(Qwen3-Embedding-0.6B, Nemotron-Pretraining-Legal-v1, 1M docs). One round =
train → re-embed the fixed 200k subset → measure → license the next step.

**Objective (no labels anywhere):** `confusion_loss` at the measured σ* with
the base-neighbor guard, + λ_p·`preservation_loss` against frozen base vectors;
batches drawn by `resolution_weights`. Held-out 20% never touched by training,
weights, or mining — verdicts only.

**Licensing rule:** continue only while σ* falls AND held-out neighbor
overlap@10 ≥ 0.90. Stop on overlap collapse, or when the envelope no longer
separates data from the anisotropy-matched reference at duplicate scales.

**Gold labels** (eCFR eval sets) are used exclusively in `score_final.py`,
after all rounds are frozen — blind-then-score.

## Rounds

| round | vehicle | σ* | σ*/uniform | liftoff | env p | z | top pocket prom | overlap@10 (held-out) | flagged-cohort med. collisions | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | base (measured, 1M) | 0.1226 | 0.83 | ≈0.82 | 0.005 | −4211 | 0.090 | 1.00 | — | licenses training (beyond-cone clumping) |
| A0 | adapter (stored vectors) | | | | | | | | | |
| 1 | LoRA r1 | | | | | | | | | |
| 2 | (per rule) | | | | | | | | | |
| F | full FT | | | | | | | | | |

Round-0 numbers are the base 1M measurements from the TR case study; subset
(200k) round-0 numbers are re-measured by `measure_round.py --round 0
--emb subset200k/base.npy` for like-for-like comparison and recorded here
before any training round.

## Dry-run findings (20k subset, 2026-08-01, pre-GPU)

Plumbing validated end-to-end (make_subset → weights/guard → adapter train →
measure_round → licensing verdict). Subset round-0 readouts are consistent with
the 1M case study (liftoff 0.79, z −3554, DTM p1 0.966, the 204-doc pocket
appearing as 165 in the subsample; σ* 0.154 at n=20k, as expected from the
n-dependence of the collision budget).

1. **Small pair batches are destructive.** Linear adapter, batch 256: training
   loss fell (0.019 → 0.009) while every global readout degraded — mean pair
   cosine 0.235 → 0.288, σ* 0.154 → 0.147, median collisions 0.96 → 1.75,
   held-out overlap@10 0.90 — the map overfits within-batch pairs and destroys
   out-of-batch structure. Uniform vs resolution-weighted sampling made no
   difference (weighting is not the cause). The licensing rule catches this
   round and rejects it, as designed.
2. **Large pair batches are safe but the adapter is a near-no-op.** Batch 2048:
   held-out overlap 0.982, mean cosine 0.243, collisions ≈ flat, and the
   near-duplicate pocket unchanged (169 members, prominence 0.063) — a linear
   map cannot unfold a real near-duplicate pocket. This is the measured
   capacity floor of the ladder (adapter < LoRA < full fine-tune) and the
   reason encoder tuning is the actual E6 test.
3. **Consequence for the encoder trainer:** gradient accumulation does not
   substitute for batch size (accumulated micro-batches have separate pair
   statistics). train_ambit.py therefore defaults to batch 256 per forward with
   gradient checkpointing, and larger is better where memory allows.

## Final (1M, TR parity)

- full re-embed + report: pending
- blind-then-score (`score_final.py`): pending
- regression check (open-eval multi-hop all-gold@10): pending

## Gate discipline

No GPU job (training or embedding) is launched without the operator's explicit
go-ahead; measurement rounds run local/CPU.
