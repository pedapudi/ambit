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
| A0 | adapter (stored vectors, batch 2048, 300 steps) | 0.1331 | 0.83 | 0.770 | 0.010 | −3665 | 0.064 (n=17; pocket split) | 0.979 | 2.82 | **REJECTED** — σ* fell, cohort +9%; a single linear map cannot serve the mixture of scales |
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

## Final (1M, TR parity) — complete 2026-08-02

- **Full 1M re-embed** with the accepted full fine-tune (two vllm replicas,
  ~500k docs/stream, same sampling as the base map). Parity measurement
  (`measure_map.py`, reservoir 20k): σ* 0.1227 (base 0.1226), liftoff 0.817
  (envelope p = 0.01), z −3554, effective rank 728, IsoScore 0.089, mean pair
  cos 0.2354, hubness 1.88, top pocket 198 @ prominence 0.090 (base 204 @
  0.090). **Every readout reproduces base within noise** — the tuned model is
  geometry-preserving at 10⁶ docs and the 200k held-out loop verdicts
  transfer to full scale.
- **Blind-then-score** (labels used here only): full FT R@1 0.288→0.301,
  MRR@10 0.467→0.475; LoRA flat (0.286, 0.466); supervised ceiling 0.355.
  H2 preview: base collision counts predict failure participation AUC 0.578
  vs 0.551 top-1-cosine control (n=953 gold docs, fail = rank>10, σ=0.145
  eCFR-matched); cohort-level repair limited — consistent with the
  duplication finding.
- Regression check (open-eval multi-hop): not run — deferred; the geometry
  parity result (identical map) bounds the regression risk that motivated it.

## Dedup routing experiment (run 2026-08-02, `dedup_routing.py`)

The repair the measurement prescribed, executed label-blind (canonical =
lexicographically-first uuid per near-dup group; qrels mapped mechanically)
and scored on the frozen eval. **Double dissociation across ambit's scales:**

| threshold | distractor arm (foreign near-dups removed) | in-corpus arm (eCFR merged) |
|---|---|---|
| 0.8165 = liftoff (confusable window) | **+12 fixed / 0 broken, McNemar p = 0.0005; R@10 0.727→0.739** | 26 fixed / 57 broken, p = 0.0009 — *harmful* (merges legally distinct formulaic sections; 11.6k of 35k docs) |
| 0.95 (true duplicates) | 0 / 0 — no effect | **9 fixed / 2 broken, p = 0.065; R@10 0.750→0.757** |

Reading: interference from *foreign* material lives in the confusable
window (69,445 of 555k distractors!) and removing it yields the significant
gain training could not; *within* a formulaic corpus the liftoff scale
over-merges and only the true-duplicate scale helps. The instrument's scale
structure — liftoff = confusable onset, ≈0.95+ = duplicate boundary —
prescribes different repairs, and the frozen labels confirm each mapping.

## Guardrail experiment (run 2026-08-02, `train_guardrail.py`)

Identical supervised recipe (in-batch InfoNCE on 45k question/section pairs,
2 epochs, lr 2e-5, DDP world 2) in two arms: **control** (task loss only) vs
**guarded** (+ λ_p·L_pres against the frozen base on the same truncated
views, + in-batch false-negative guard masking batch documents whose BASE
cosine to the anchor's positive ≥ liftoff 0.8165).

**Label-free gates (held-out 200k subset), read before any qrel:** guarded
dominates control on every readout — overlap@10 0.650 vs 0.537, σ* 0.1459
vs 0.1360 (base 0.1337 — first genuine σ* gain in the program), flagged-
cohort collisions 0.47 vs 2.10 (base 2.58), z −1515 vs −2217.

**Labels (frozen evals):**
- Set A (closed, 1000 q): control R@1 .303 / MRR .501; guarded .299 / .503
  (base .288/.467) — the guardrails cost nothing on the target task.
- Set B (open, 1496 q): guarded wins everywhere — single R@1 .382 vs .340,
  MRR .597 vs .563; **multi-doc all-gold@10 .339 vs .223 (+52% rel., z≈5)**;
  set-R@10 .593 vs .489. The unguarded arm reproduces the historical v1
  multi-hop collapse (base ≈.48 → .22); the guard halves the damage while
  improving everything else.

**Conclusion:** ambit's guardrails make supervised domain fine-tuning safer
at zero cost to the supervised gain, and the instrument's label-free
held-out gates select the better model before any label is consulted.

## Significance tests (run 2026-08-02, `/tmp/sig_tests.py` on the GPU host)

- **Cohort collision reduction: significant and uniform.** Paired per-item on
  the 408 held-out flagged items at fixed σ: LoRA improved 406/408 (sign-test
  p = 2.5e-118), median Δ −0.071, bootstrap 95% CI [−0.076, −0.067]; full FT
  356/408 (p = 8.4e-57), median Δ −0.056, CI [−0.062, −0.049].
- **Retrieval gain: not significant.** McNemar exact on paired R@1 outcomes:
  118 vs 131 discordant, p = 0.447; R@10 identical (6 vs 6, p = 1.0); ΔMRR@10
  +0.0075, bootstrap 95% CI [−0.008, +0.023]. Read as
  retrieval-neutral-to-positive at n = 1,000 queries.
- **Collision→failure AUC: above chance, not above control.** AUC 0.578,
  bootstrap 95% CI [0.536, 0.622] (excludes 0.5); margin over top-1-cosine
  control +0.028, paired CI [−0.017, +0.073]. Cross-corpus H2 (Part II) is
  the designed test of the control comparison.
- **σ\* invariance is measured, not noise-masked.** Eight independent
  200k-pair samples: base 0.13361 ± 0.00012, full FT 0.13368 ± 0.00013,
  paired Δ +0.00007 (paired sd ≈ 0).

## Gate discipline

No GPU job (training or embedding) is launched without the operator's explicit
go-ahead; measurement rounds run local/CPU.
