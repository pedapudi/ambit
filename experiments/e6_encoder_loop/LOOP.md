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
| 0-s | base (200k subset) | 0.1337 | 0.84 | 0.769 | 0.010 | −3565 | 0.085 (n=294) | 1.00 | 2.58 (408 held-out) | baseline for all loop rounds |
| A0 | adapter (stored vectors) | | | | | | | | | |
| 1 | LoRA (σ=0.1337, λ_p=0.3, batch 256, 625 steps) | 0.1324 | 0.83 | 0.758 | 0.010 | −3775 | 0.091 (n=319) | 0.924 | 3.01 | **REJECTED** — σ* fell, cohort collisions rose |
| 2 | LoRA + fixes (matched refs, mined pairs) | 0.1339 | 0.84 | 0.763 | 0.010 | −3545 | 0.082 (n=307) | 0.973 | 2.51 | **accepted, marginal** — cohort −2.8%, no degradation; σ* flat → capacity/step-limited; Stage A closed |
| F | full FT (same objective, DDP world 2, pair batch 512) | 0.1338 | 0.84 | 0.763 | 0.010 | −3552 | 0.083 (n=275) | 0.965 | 2.53 | **accepted, marginal** — matches LoRA; capacity is not the binding constraint |

**Stage B post-mortem.** Full fine-tuning with the same guarded objective
reproduces the LoRA result (cohort −2%, σ* flat, no degradation) despite ~40×
the trainable parameters and a 512-row pair batch. Capacity is therefore not
what limits resolution recovery on this corpus: the top pocket is textual
near-duplication, and the preservation constraint (correctly) refuses to
separate items whose inputs are near-identical. This is the training-side
confirmation of the staged protocol's routing rule — duplicate pathology is a
data-repair problem; training at the measured scale is safe but cannot
substitute for deduplication.

**Round-1 post-mortem.** Training loss decomposed as conf ≈ 1e-5 (≈2 active
window pairs per 256-row batch — the confusion term was starved) against
pres ≈ 0.03; and the preservation reference was the stored base vectors of
*full* documents while the trainer embeds 512-token truncations, so the
gradient was dominated by a text-length distillation rather than geometry
repair. Round-2 fixes: (a) truncation-matched references — base model embeds
the same 512-token views, via the running base server; (b) pair-aware
batching — half of each batch drawn from `mine_confusable_negatives`
(window = liftoff cosine to 0.98, guarded), so the confusion term sees real
in-window pairs every step.

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
