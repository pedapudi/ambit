# Work log — ambit validation battery (2026-08-01 → 2026-08-02)

A detailed journal of the full experiment battery: what was run, in what
order, what broke, what was fixed, and what each experiment concluded.
Supporting evidence is checked in beside this file: `artifacts/` holds the
result JSONs and trimmed logs exactly as produced on the compute hosts;
`scripts/` holds the analysis scripts that were run ad hoc (everything else
lives in the repo under `experiments/`). The complete run of commits for the
period is listed at the end.

Hosts (properties, per the TR's appendix): a 16-core/32-thread CPU
workstation (tier-1/2/3 battery), a 4-GPU workstation (96 GB per GPU;
everything from E6 on), and a light coordinator that holds the repo and
builds the TR. Compute placement changed mid-battery after two coordinator
sessions were OOM-killed — see Incidents.

---

## Day 1 — Part I: the internal battery (E1–E5)

Ran from `experiments/tier1_sensitivity.py`, `tier2_power.py`,
`tier3_scaling.py`; verdicts recorded in `docs/tr/validation-plan.md`.

**E1a (reservoir size).** σ* flat across reservoirs 1k–32k
(0.0559–0.0564) once corpus-n semantics were fixed — the sweep initially
showed drift that turned out to be an *experiment-script* bug (the script
passed the reservoir size instead of the corpus size into `sigma_star`);
ambit itself was correct. Verdict: sufficiency from 1k rows, stronger than
the ≥4k acceptance bar.

**E1b (pair-sample size).** sd(σ*) tracks 1/√P as predicted
(0.0126→0.0013 over P=10⁴→10⁶). Finding with a reporting rule: z grows
with P (it is a test statistic, not an effect size), so the report must
state z at the standard P.

**E1c (DTM mass fraction).** Pocket/bulk separation collapses at the
predicted blur boundary m ≈ k/n; the "choose m below the smallest pocket
share" rule became measured fact.

**E1d (merge-tree parameters).** The sweep found a real instrument bug:
single-linkage chaining could leave the tightest pocket unreported (no real
split ⇒ only the root cluster ⇒ excluded). Fixed with a root-candidate
fallback (`crowding.pockets`), regression test added; 15/16 grid cells then
recover the planted pocket exactly.

**E1e (seed replication).** 20/20 detection at exact size after the E1d
fix; z = −46.7±4.0 clumped vs −0.14±1.00 uniform. Two carried findings:
pointwise envelopes false-alarm on *every* uniform seed (making E4a
mandatory), and uniform corpora report negligible-prominence pockets
(motivating E4b's floor).

**E4a (global rank-envelope test).** Implemented
`occupancy.rank_envelope` (Myllymäki-style, ERL tie-breaking). Measured:
0/40 false liftoffs on nulls where the pointwise band fired on all 40;
envelope-only gating still over-fired ~4×, so liftoff is gated on the
global p. Case-study correction: the legal corpus's real liftoff is
cos ≈ 0.82 at p = 0.005 (the earlier +0.07 reading was the bulk-edge
artifact).

**E4b (null calibration of the whole instrument).** 400 pure nulls:
rank-test rejection 1.25% @ nominal 1%, gated liftoff 6% @ 5%, |z|>3 at
0.75%. Pocket-prominence null p99 = 0.009 shipped as a per-report
simulated floor. **E4c:** z's null-replicate default raised 24→64 (|z| was
inflated ~20% at 24).

**E2 (detection power at matched 1% FA).** The continuous layer at full
power in every pocket cell including a single 1%-of-corpus pocket where
mean-cosine/hubness/kNN-distance baselines sit at the false-alarm floor.
Honest cells printed: IsoScore also full-power on single coherent pockets
but cannot localize; z weak on small pockets; the pocket detector correctly
non-alarming on a pure cone.

**E3 (scaling).** Scan linear in bytes at ~2 GB/s (10⁷×1024 = 41 GB in
18.6 s); everything downstream reservoir-bound and constant in n; d enters
via d² scan / d³ eigendecomposition. Sorted-shards failure mode
demonstrated: approximate scans miss a last-rows pocket (p = 0.18–0.45)
that the full scan detects (p = 0.010).

**E5 (noise-model robustness).** σ*/C corpus ranking robust under
moderately structured noise (rank-25 subspace: 0.99 order agreement with
isotropic), degraded under extreme misspecification (rank-2: 0.79;
PC-aligned: 0.82) — recorded as a measured scope caveat.

---

## Day 1–2 — E6: the measurement-driven fine-tuning loop

Corpus: Nemotron-Pretraining-Legal-v1 (1M docs, 4 subsets), base encoder
0.6B instruction-tuned, d=1024. Fixed stratified 200k measurement subset,
20% held out, never touched by mining/weights/training. Harness:
`experiments/e6_encoder_loop/` (subset builder, trainer, per-round
measurement, re-embedding, scoring).

**Round 0 (baseline, `artifacts/round0.json`).** σ* 0.1337 (uniform
0.1597), liftoff 0.769 (envelope p=0.01), z −3565, DTM p1 0.965, one
pocket above floor (294 members, prominence 0.085), flagged cohort (top 1%
by base collisions; 408 held-out members) median 2.58 collisions.
Consistent with the separately measured 1M case study — the subset is a
faithful proxy.

**Dry run finding (20k, CPU).** Small pair batches are destructive: at
batch 256 a linear adapter's training loss fell while every global readout
degraded (overlap 0.90, mean cosine 0.235→0.288) — batch-level
overfitting of pair statistics. At batch 2048 the adapter is safe but a
near-no-op. Consequence baked into the trainer: large per-forward batches
with gradient checkpointing; gradient accumulation is *not* a substitute
(accumulated micro-batches have separate pair statistics).

**Round 1 — LoRA, rejected by measurement (`artifacts/round1.json`).**
Unsupervised objective (confusion loss at measured σ* under the
false-negative guard + λ_p·preservation). Training losses decreased;
held-out measurements degraded: σ* 0.1324, cohort +17% (3.01), pocket
enlarged, overlap 0.924. Post-mortem found two objective defects: the
preservation reference used stored full-document base vectors while the
trainer embedded 512-token truncations (a length distillation), and random
batches starved the confusion term (~2 in-window pairs per batch).

**Round 2 — LoRA with fixes, accepted-marginal
(`artifacts/round2.json`).** Truncation-matched references (base model
embeds the same 512-token views — `make_refs.py`) and pair-aware batching
(half of each batch from `mine_confusable_negatives`, window = liftoff
0.769→0.98, 8,518 guarded pairs). Result: no degradation, overlap 0.973,
cohort −2.8% (2.51), σ* flat.

**Round 3 — full fine-tune, DDP (`artifacts/round3.json`).** Trainer
extended to torchrun DDP with cross-rank gathered pair statistics (world
batch 512; the local slice keeps its gradient; measured 3.3× throughput on
2 GPUs). First DDP launch deadlocked NCCL — variable-size batches from
`np.unique` truncation made `all_gather` shapes mismatch across ranks;
fixed by emitting exactly batch-size distinct rows per rank. Result:
matches LoRA (cohort 2.53, overlap 0.965, σ* flat) despite ~40× the
trainable parameters — capacity is not the binding constraint.

**A0 — linear adapter at 200k, rejected (`artifacts/roundA0.json`).**
σ* fell (0.1331) and cohort worsened 9% (2.82) at overlap 0.979 — a single
linear map cannot serve the mixture of scales. Completes the capacity
ladder: adapter rejected, LoRA ≈ full FT accepted-marginal.

**1M parity (`artifacts/map1M.json`).** Full corpus re-embedded with the
accepted fine-tune (two vllm replicas, ~500k docs/stream): every readout
reproduces base within noise (σ* 0.1227 vs 0.1226, liftoff 0.817, effective
rank 728, IsoScore 0.089, top pocket 198@0.090 vs 204@0.090). The 200k
held-out verdicts transfer to full scale.

**Blind-then-score (`artifacts/scores_r3*.json`).** Labels enter only
here. Full FT: R@1 0.288→0.301, MRR@10 0.467→0.475; LoRA flat. **Not
significant** (McNemar p=0.447, 118 vs 131 discordant; ΔMRR CI
[−0.008,+0.023]) → stated as retrieval-neutral-to-positive.

**Significance tests (`artifacts/sig_tests.log`, `scripts/sig_tests.py`).**
Per-item cohort improvement is uniform and overwhelming: LoRA 406/408
held-out flagged items improved (sign test p=2.5e-118, median Δ −0.071 CI
[−0.076,−0.067]); full FT 356/408 (p=8.4e-57). Collision→failure AUC 0.578
CI [0.536,0.622] (above chance) vs control +0.028 CI [−0.017,+0.073] (not
individually significant). σ* invariance is measured, not noise-masked:
paired Δ +0.00007 vs per-sample sd 0.00012 over eight pair samples.

**E6 conclusion.** The loop works as an adjudicator (rejects a
misspecified round that training loss endorsed; verifies safe rounds;
subset verdicts transfer to 1M), the corpus's crowding is duplication that
training must not repair, and the honest task effect of the unsupervised
objective is neutral-to-positive.

---

## Day 2 — Dedup routing (the repair the instrument prescribed)

Script `experiments/e6_encoder_loop/dedup_routing.py`; results
`artifacts/dedup_routing.json` (threshold = liftoff 0.8165) and
`artifacts/dedup_routing_95.json` (0.95). Label-blind canonicalization
(connectivity groups over GPU-exact kNN; lexicographically-first canonical;
qrels mapped mechanically); scored on 1,000 frozen queries.

| threshold | foreign distractors removed | target corpus merged |
|---|---|---|
| 0.8165 (confusable window) | **+12 fixed / −0 broken, p=0.0005**; R@10 0.727→0.739 | +26/−57, p=0.0009 — harmful (11.6k of 35k docs merged; legally distinct sections conflated) |
| 0.95 (true duplicates) | 0/0 | +9/−2, p=0.065; R@10 0.750→0.757 |

Double dissociation: the confusable-window scale prescribes *foreign*
cleanup; only the true-duplicate scale licenses *in-corpus* merging. Both
mappings confirmed by labels the dedup rule never saw. Scale of the
finding: 69,445 of 555k distractors were confusable-window duplicates.

---

## Day 2 — Guardrail experiment (supervised fine-tuning with ambit gates)

Script `experiments/e6_encoder_loop/train_guardrail.py`. Identical
supervised recipe (in-batch InfoNCE over 45,465 question/section pairs, 2
epochs, DDP world 2) in two arms: control (task loss only) vs guarded
(+λ_p·L_pres against the frozen base on identical truncated views, + batch
documents whose *base* cosine to the anchor's positive ≥ liftoff masked
from the contrastive denominator).

**Label-free gates first** (`artifacts/sup{ctl,grd}.round.json`): guarded
dominates control on every readout — overlap 0.650 vs 0.537, σ* 0.1459 vs
0.1360 (base 0.1337; the program's first genuine σ* gain), cohort 0.47 vs
2.10, z −1515 vs −2217. Prediction recorded before any label was read.

**Labels** (`artifacts/sup*.seta.txt`, `sup*.setb.txt`):
- Set A (closed, 1,000 q): control R@1 .303/MRR .501; guarded .299/.503
  (base .288/.467) — guardrails cost nothing on the target task.
- Set B (open, 1,496 q): guarded wins everywhere — single-doc R@1 .382 vs
  .340, MRR .597 vs .563; **multi-doc all-gold@10 .339 vs .223 (+52%
  relative; ≈5 standard errors)**; set-R@10 .593 vs .489. The control
  reproduces the historical multi-hop collapse (≈.48 → .22); the guard
  halves the damage.

**Conclusion.** Ambit's guardrails make supervised domain adaptation safer
at no cost to its gains, and the label-free held-out gate selects the
better model before any judgment is consulted. This is the program's
headline positive for the fine-tuning claim.

---

## Day 2 — External validation grid (Part II, single-day slice)

Infrastructure `experiments/validation_grid/`: `embed_cell.py` (BEIR or
jsonl.gz corpora, per-model document prefixes, house parquet schema),
`measure_map.py` (readouts; frozen via `FREEZE.sha256` before qrels),
`collisions_cell.py` (exact per-document collision field on GPU at the
cell's measured σ*), `score_cell.py` (query embedding, exact search,
nDCG@10 / recall@100, per-relevant-doc failure labels), `run_cell.sh` /
`queue_worker.sh` (orchestration), `--start-shard` (collision-free
two-GPU splits of one cell's embedding).

Corpora: TREC-COVID (171k), Quora (523k), NQ (2.68M), ESCI-US (1.2M) +
legal (1M, in hand); HotpotQA (5.2M) deferred by decision (queue
preserved). Encoders: the 0.6B case-study encoder, bge-large-en-v1.5,
embeddinggemma-300m, arctic-embed-m-v1.5, e5-base-v2, MiniLM-L6-v2,
mpnet-base-v2.

Build/debug notes, in order encountered: the CLI `hf download` failed on
the gated gemma repo even after license acceptance (fine-grained-token
path; the python `snapshot_download` worked — the license was never the
problem); qwen3's sentence-transformers config defaults to a 32k sequence
length (OOM at batch — capped via `--max-seq-length`); the collision
kernel allocated four block×n temporaries (40 GB spikes at NQ scale —
rewritten as a single in-place buffer chain); tilde paths in queue files
don't expand inside variables (ESCI scoring — fixup queue with absolute
paths); pilot cell validated the whole pipeline against public numbers
(bge-large on TREC-COVID nDCG@10 0.743 ≈ the published ≈0.75).

**Results (`artifacts/SYNTHESIS.json`), all 28 cells:**

- **H2 (per-item collision → failure), confirmed cross-corpus.** Every
  cell with a non-degenerate outcome is above chance: Quora 7/7 (AUC
  0.66–0.74), ESCI 7/7 (0.57–0.69, after a query-id loader bug that silently
  scored zero queries was caught and fixed), NQ 7/7 (0.51–0.62), legal
  0.578 → 22/22 positive (sign-test p ≈ 2×10⁻⁷). TREC-COVID's ≈0.5 cells are an outcome artifact (50 queries
  × hundreds of relevant docs ⇒ 98% of judged docs "fail at rank 10" —
  the label saturates).
- **H1 (σ* ranks encoders within a corpus), rejected as universal;
  replaced by a scope law.** Positive only on pure duplicate
  matching (Quora +0.75, mean-cosine control inverted); null on product
  search (ESCI −0.07); negative on knowledge tasks (NQ −0.54,
  TREC-COVID −0.32). Retrieval = semantic alignment × geometric
  headroom; ambit measures only the second factor (unsupervised by
  design), so σ* predicts outcomes exactly where alignment is roughly
  constant across encoders — and anti-predicts where a weak encoder "wins"
  geometry by being uniformly spread and semantically wrong.
- **Dedup-F1 on annotated duplicates (`artifacts/quora.*.dedupf1.json`).**
  Duplicate-pair prediction precision rises monotonically along the
  threshold ladder (case-study encoder: 0.29 @0.80 → 0.93 @0.95 → 0.98
  @0.98) — ground-truth confirmation of the scale semantics the routing
  experiment relied on; the paraphrase mass sits in the confusable window,
  consistent with the dissociation.

---

## Incidents and operational lessons

- **Two coordinator-host OOM kills.** Root cause: the coordinator's
  scratchpad is a 14 GB RAM-backed tmpfs; staging multi-GB relays there
  plus running 200k-vector measurement locally exhausted 27 GB RAM. All
  heavy compute moved to the GPU host; relay copies deleted immediately.
- **`pkill -f "vllm serve"` collateral.** A cleanup pattern matched the
  operator's own two serving instances and killed them alongside the
  experiment's replicas. Both restored within minutes from their exact
  bash-history launch commands; lesson recorded (bracketed pkill patterns,
  target by PID).
- **NCCL deadlock** from variable-size `all_gather` (fixed: fixed-size
  batches); **ssh+nohup self-match** pkill footguns (three occurrences;
  bracket-pattern convention adopted); **buffered stdout** under nohup
  hiding progress (flush=True convention).
- Monitors with explicit failure-signature alternations were used for
  every long-running job; queue rebalancing kept all four GPUs saturated
  through the tail of the grid.

## Where each conclusion lives

- `docs/tr/ambit-technical-report.tex` — the report: loop section, routing
  and guardrail sections, calibration/power/scaling, formula glossary.
- `docs/tr/validation-plan.md` — per-experiment verdict tables (E1–E6),
  Part II design, deferral notes.
- `experiments/e6_encoder_loop/LOOP.md` — round-by-round loop log with
  post-mortems, significance tests, guardrail and dedup results.
- `~/validation-runs/` on the GPU host — per-cell readout/score/collision
  JSONs with the freeze ledger (`FREEZE.sha256`).

## Commits for the period (oldest last)

```
3769d6a grid: --start-shard for collision-free two-GPU cell splits
3dec0fc validation plan: HotpotQA grid row deferred (queue preserved)
07dcc72 TR: routing execution (double dissociation) and guardrail sections
d6596cf E6: guardrail experiment results — the headline positive
aafc078 TR: glossary rewritten for complete term-by-term coverage
92594f7 TR: formula glossary reorganized with labeled bullets
0764ca4 grid: cap ST max_seq_length; in-place collision kernel
d8bf71f grid: embeddinggemma as seventh encoder
a40aceb grid: jsonl.gz query loader for ESCI
3461742 Validation grid pipeline: embed/measure/collisions/score per cell
e963858 E6: dedup routing experiment + guardrail trainer
9bfe5c1 TR: formula glossary appendix
37d251b E6: significance tests; TR claims calibrated to them
9be9c96 TR: encoder-loop and retrieval-behavior sections
aec65fa E6: case-study-parity map measurement script
76f2999 E6: adapter rejected at 200k scale; ladder verdicts finalized
1172f95 validation plan: E6 verdict recorded
db93460 E6: round-3 full-FT verdict — capacity not the binding constraint
619e198 E6 trainer: DDP with cross-rank gathered pair statistics
9b079e8 E6: round-1 rejected, round-2 accepted-marginal; Stage A closed
3fff924 E6 round-2 fixes: matched refs, pair-aware batching
b010af9 E6: resumable subset re-embedding script
05edfc1 E6: record 200k-subset round-0 baseline
2df2464 E6 encoder loop: unsupervised measurement-driven harness
9552d6a TR: appendix on its own page, labeled Appendix A/B/C
d7a72d0 TR: formal prose pass; discussion/engineering to appendix
e97487e TR: report readouts reproduced natively
1d4cfe0 TR: name and cite Nemotron-Pretraining-Legal-v1
a4935e1 TR methodology: datasets first, environment to appendix
6e25010 TR: scaling section (E3), property-based environments
b83b246 TR + plan: tier-2 results integrated
0307cf7 Pockets: null-calibrated prominence floor; tier-3 runner
6e7d4d4 Experiments: tier-2 runner (E2, E4b)
9748b60 occupancy.rank_envelope: whole-curve liftoff test (E4a)
3bb9530 Technical report: separate the tool from the report artifact
a2ee734 Validation plan: tier-1 results and verdicts; defaults frozen
48a3776 crowding.pockets: root-candidate fallback (E1d fix)
975eee8 Experiments: tier-1 sensitivity runner (E1a-e, E4c, E5)
c8a3738 Validation plan: internal battery + external methodology
```

## 2026-08-13 — Conditioned-reference defect found by external review, fixed, audited

An external review observed that the anisotropy-conditioned reference
(the ACG, sampled from the centered covariance spectrum) is antipodally
symmetric, so its expected pair cosine is zero and it cannot reproduce a
mean-direction cone. Verified by direct experiment: against a synthetic
cone matched to the legal corpus (mean pair cosine +0.25), the centered
ACG recovers under 1% of the data's pair mass at cos 0.2 (K = 0.001
against 0.810). On the real corpus the published reference curve was
confirmed to be the centered ACG and sat 1-2 orders of magnitude below
the data throughout — the figure caption's "tracks through
mid-similarities" was never true of the plotted curves.

Blast radius (verified from the consumer map): the ACG fed only the
comparison curve in the crowding-curve figure and the attribution prose.
The rank envelope, liftoff, Stolarsky z, DTM, pockets, and sigma-star
all use the analytic uniform null and are unaffected; no grid readout
touches the ACG.

Fix: `conditioned_pair_cos` (occupancy.py) — the corpus's own Gaussian
fit, normalized: x = (mu + Sigma^1/2 g)/||.||, the projected normal.
Audit (experiments/null_audit.py):
- tracking: max K-error 0.006–0.029 on mean cones (centered ACG:
  0.40–1.00); ties on zero-mean ellipses; two-cone mixture only partly
  absorbed (0.109) — multi-cluster structure still shows as excess.
- composite-null calibration: even with (mu, Sigma) refitted inside
  every bootstrap replicate, the fitted reference's tail test
  false-alarms at 15% for nominal 5%. The reference is therefore
  descriptive only; significance stays with the uniform-null envelope.
- absorption: a planted pocket of up to 20% of the corpus leaves the
  fitted reference's near-duplicate tail at exactly zero; detection of
  5%+ pockets is 100%.

On the legal corpus the corrected reference matches the mean pair cosine
to four decimals (+0.2354 vs +0.2356), tracks K(t) through cos 0.3, and
the data's excess beyond it begins near cos 0.4 — earlier than the old
prose implied — growing to orders of magnitude in the near-duplicate
tail. TR updated: figure curve regenerated, reference renamed
cone-conditioned, roles separated (description vs significance), and the
zero-mean ACG's limitation recorded in §6. Four calibration tests added
(tests/test_conditioned_null.py), including the one that would have
caught the defect.

## 2026-08-13 — Clustered uncertainty for the per-document claim

The published sign test treated 29 corpus-encoder cells as independent;
they share collections and encoders. Recomputed from the frozen
artifacts (scripts/clustered_uncertainty.py):
- corpus as the unit: all five collections positive on average
  (per-corpus mean AUC 0.53-0.69), one-sided sign test p = 0.031;
- two-way cluster bootstrap over collections x encoders: mean AUC 0.603,
  95% interval [0.544, 0.665], chance excluded;
- encoder-ranking Spearmans (n = 7 per collection): none individually
  significant (exact two-sided p for +-0.75 is 0.066); the supportable
  finding is the sign pattern across task families.
TR updated: the independent-cells p is now labeled optimistic and
presented beside the clustered numbers; both external results are framed
as strong preliminary evidence pending independent replication.
