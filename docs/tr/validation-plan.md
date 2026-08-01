# Validation plan for the technical report

> Two parts. **Part I** is the internal experiment battery the technical report
> needs before it can face a critical reviewer: sensitivity ablations, an
> internal baseline comparison, scaling curves, and statistical-rigor upgrades
> — all runnable from the artifact alone, no labels anywhere. **Part II** is
> the recommended methodology for **external validation** — the study that
> tests whether the unsupervised readouts predict measured retrieval outcomes.
> Part II consumes relevance judgments *only to score predictions*, never as
> input to the tool; it is designed to be run independently of the tool's
> authors. Each experiment states its purpose, design, artifact, and
> acceptance criterion, so a run either upgrades the report or produces an
> honest limitation to print.

---

## Part I — Internal experiments (for the technical report)

### E1. Sensitivity ablations ("every knob gets a curve")

The report claims its parameters are either scale-free or benign. Each claim
becomes a measured curve.

**E1a — Reservoir size.** Sweep the reservoir n_r ∈ {1k, 2k, 4k, 8k, 16k}
against a fixed 10^6-row synthetic corpus (and the real corpus of the case
study). Readouts: σ*, liftoff cosine, occupancy z, DTM 1st-percentile,
pocket count/size. Artifact: readout-vs-n_r curves with ±sd over 10 seeds.
*Accept if* readouts are flat within their seed noise from n_r ≥ 4k (else
the default must change and the report must say so).

**E1b — Pair-sample size.** Sweep P ∈ {10^4 … 10^6}. Artifact: sd of
liftoff, z, and σ* vs P, against the 1/√P reference line. *Accept if*
concentration follows √P and the default P = 2×10^5 puts headline readouts'
sampling noise well below their reported effects.

**E1c — DTM mass fraction.** Sweep m ∈ {0.005, 0.01, 0.02, 0.05, 0.1} on
benchmarks with planted pockets of size k ∈ {20, 50, 200, 800}. Artifact: a
heat map of low-tail separation (pocket p1 / bulk p50) over (m, k), showing
the predicted blur boundary at m ≈ k/n. *Accept if* the boundary matches
prediction; the report then states the rule "choose m below the smallest
pocket share you care about" as a measured fact.

**E1d — Merge-tree parameters.** Sweep min-pocket-size ∈ {4, 8, 16, 32} and
the core-distance k over the pocket grid of E2. Artifact: recovery
precision/recall of planted pockets per setting. *Accept if* defaults sit on
a plateau, not a cliff.

**E1e — Seed replication of every headline number.** All "measured" scalars
in the report re-run over ≥ 20 seeds; report mean ± sd. Cheap; kills the
single-seed objection wholesale.

### E2. Internal baseline comparison ("why not a simpler dashboard?")

**Design.** A detection-power study on synthetic corpora, no labels. Pathology
grid: planted pockets varying (count ∈ {1, 5, 20}) × (size ∈ {20, 50, 200})
× (tightness: median intra-pocket cosine ∈ {0.99, 0.93, 0.7}), plus cone
strengths, low-rank collapse, and pure nulls; ≥ 50 replicates per cell.
**Detectors compared:** ambit's continuous layer (liftoff, z, DTM tail,
pocket prominence, σ* deficit) versus the pre-existing suite each computed
alone (mean pair cosine, IsoScore, effective rank, uniformity, hubness skew,
a kNN-distance histogram with a quantile threshold). Thresholds for *every*
detector are calibrated on pure-null replicates at a fixed false-alarm rate
(1%), so the comparison is power-vs-power at matched specificity.
**Artifact:** detection-rate tables/curves per pathology family; a
localization score (fraction of planted members recovered) for the detectors
that claim to name items. *Accept if* the continuous layer dominates or ties
everywhere it claims to, and the report prints the cells where a simple
baseline suffices (there will be some — cone detection needs nothing fancy —
and saying so is credibility).

### E3. Scaling curves ("one timing point is not a systems claim")

Time and peak RSS versus n ∈ {10^4 … 10^7} at d = 1024, and versus
d ∈ {128 … 4096} at n = 10^6, separated into scan / context / figure
phases; single-thread and default-BLAS both. Include the reservoir-vs-full
tradeoff: full-scan readouts versus `--approx` sub-scans, error against the
full-scan reference as a function of rows scanned — including the known
failure mode (label-sorted shards biasing an approximate reservoir), printed
as a warning with its measurement. Artifact: log-log curves plus a
where-it-breaks table (the O(n_r²) pieces, memory ceilings).

### E4. Statistical-rigor upgrades

**E4a — Global envelope for liftoff.** Implement the rank-envelope test
(Myllymäki et al. 2017) beside the pointwise band; report global p-values
for the benchmark and case study. Artifact: pointwise vs global liftoff
comparison; the report's claims then rest on the honest test.

**E4b — Null-calibration of the whole instrument.** Run the complete report
on ≥ 200 pure-null corpora across (n, d) grid: measure the false-positive
rate of liftoff detection, pocket reporting, and |z| > 3 against nominal
rates. Artifact: a calibration table. *Accept if* empirical false-positive
rates match nominal within binomial error; any miscalibration gets printed
and fixed, in that order.

**E4c — Uncertainty of the z's null sd.** Sweep null-replicate count for the
Stolarsky z (currently 24); report the sd-of-the-sd and its effect on |z|
for the case study. Cheap footnote-grade fix.

### E5. Noise-model robustness of σ* (internal half of the query question)

σ* assumes isotropic query noise. Simulate structured alternatives — noise
confined to a random low-rank subspace (rank ∈ {1%, 10% of d}), noise
aligned with the corpus's top principal directions, heavy-tailed noise — and
recompute the *ranking* of the E2 corpus grid by σ* under each model.
Artifact: rank-correlation of corpus orderings across noise models.
*Accept if* orderings are stable (σ* is then defensible as a comparative
instrument under model misspecification); if not, the report's scope
statement gains a measured caveat.

### E6 (stretch). Training section with a real encoder

Replace the linear adapter with a LoRA-tuned public encoder on a public
corpus, same protocol (measured σ target, guarded mining, held-out
verdicts). This overlaps Part II infrastructure; run it after Part II
exists so the task metrics come for free.

**Estimated cost of Part I:** E1/E4/E5 are CPU-days at most (mostly
embarrassingly parallel over seeds); E2 is the largest synthetic compute
(grid × replicates) but each cell is seconds; E3 needs one large machine and
disk for the 10^7 corpus; E6 needs one GPU.

---

## Tier-1 results (run 2026-08-01; 32-core CPU host; ~4 min/batch)

| exp | verdict | result |
|---|---|---|
| E1a | **accept** | with corpus-n semantics, σ\* is flat across reservoirs: 0.0559–0.0562 for n\_r ∈ [1k, 16k] vs 0.0564 at 32k; DTM p1 invariant (0.365); pocket detection 10/10 at every size. Sufficiency from n\_r = 1,000 — stronger than the ≥4k bar. |
| E1b | **accept, with a reporting rule** | sd(σ\*) tracks 1/√P (0.0126→0.0013 over P = 10⁴→10⁶); liftoff ultra-stable. **But z grows with P** (it is a test statistic, not an effect size) — the report must state z at the standard P. |
| E1c | **accept** | separation collapses at the predicted m ≈ k/n boundary (k=200 detects through m=0.05, dies at 0.1). The "choose m below the smallest pocket share" rule is measured fact. |
| E1d | **accept after fix** | the sweep found a real bug: single-linkage chaining could leave the tightest pocket unreported (no real split ⇒ only the root cluster ⇒ excluded). Fixed (root-candidate fallback); 15/16 cells recover exactly; the remaining cell plants a pocket *smaller than min\_size*, which is by definition not reportable as a pocket (it appears in the DTM low tail — documented semantics). |
| E1e | **accept after fix** | 20/20 detection at exact size 200.0±0.0 (was 17/20 pre-fix); z = −46.7±4.0 clumped vs −0.14±1.00 uniform (perfect calibration). Two carried findings: uniform corpora trigger spurious pointwise-envelope "liftoff" at the bulk edge in every seed (**E4a is now required, not optional**), and uniform corpora report negligible-prominence pockets (E4b to set the null-calibrated floor). |
| E4c | **accept, default changed** | \|z\| inflated ~20% at 24 null replicates (sd ±8); default raised to 64 (−43±4.5). |
| E5 | **accept as scope** | σ\*/C corpus ranking robust to moderately structured noise (rank-25: 0.99 order agreement with isotropic) and degraded under extreme misspecification (rank-2: 0.79; PC-aligned: 0.82) — a measured caveat for the scope section. |

**Defaults frozen** for Part II: reservoir 20k (≫ the measured 1k sufficiency),
P = 2×10⁵, m = 0.02, min pocket size 8 (with root-candidate fallback), z at 64
null replicates.

## Tier-2 results (run 2026-08-01)

| exp | verdict | result |
|---|---|---|
| E4a | **accept, shipped** | global rank-envelope test with ERL tie-breaking; measured: 0/40 false liftoffs on nulls (pointwise band fired on all 40); liftoff gated on the ERL p (the α-critical envelope alone over-fires ~4×). Benchmark rejects at min p with liftoff 0.93. **Case-study correction:** the legal corpus's liftoff is p=0.005 at cos ≈ 0.82; the earlier +0.07 was the bulk-edge artifact. Figure + hovercard updated. |
| E4b | **accept** | null calibration on 400 nulls: rank-test rejection 1.25% @ nominal 1%; gated liftoff 6% @ nominal 5%; \|z\|>3 at 0.75%; **pocket-prominence null p99 = 0.009** → shipped as a per-report simulated floor in the pockets figure (suppresses within-null pockets; closes the E1e finding). |
| E2 | **accept** | at matched 1% FA (measured 0.010–0.013 for every detector): the continuous layer (rank p, σ\*, DTM ratio, pocket prominence) at **full power in every pocket cell** incl. a single 1% pocket where mean-cos/hubness/kNN-distance sit near the FA floor. Honest cells recorded: IsoScore also full-power on single coherent pockets (eigenvalue past the null spectrum edge) but cannot localize and misses the mean-shift cone; z weak on small pockets; pocket detector correctly non-alarming on a cone. |

Remaining: E3 (scaling — running), E6 (deferred; needs sign-off), Part II pilot.

## Part II — External validation methodology (to be run independently)

**The question:** do ambit's unsupervised readouts, computed from document
embeddings alone, predict retrieval outcomes measured with relevance
judgments the tool never sees?

### Design principles

1. **Blind, then score.** All readouts are computed and frozen (files
   hashed) before any outcome metric is computed. Judgments touch nothing
   upstream of scoring.
2. **Freeze the instrument first.** Ambit's defaults (m, P, n_r, pocket
   size) are fixed by Part I *before* Part II begins; no parameter may be
   revisited after outcomes are seen. This is the leakage that silently
   invalidates studies of this shape.
3. **Vary the encoder within a corpus.** The primary comparisons hold the
   corpus (hence topics and judgments) fixed and vary the embedding —
   different encoders, plus degraded variants of one encoder (dimension
   truncation, quantization, early training checkpoints). This creates wide,
   interpretable variation in geometry with the outcome measured on
   identical queries, and avoids the corpus-level confounds that pollute
   across-dataset correlations.
4. **Preregister.** Hypotheses, readouts, outcome metrics, and analysis code
   are written down before the first outcome is computed. Exploratory
   findings go in a separate, labeled section and are confirmed on held-out
   corpora.

### Materials

- **Corpora:** ≥ 5 public retrieval collections with relevance judgments,
  spanning genre and scale (general web/QA, biomedical, legal or financial,
  code, argumentation — the standard heterogeneous-retrieval benchmark
  suites cover this), each ≥ 10^5 documents where possible.
- **Embeddings:** ≥ 4 encoders spanning quality tiers (small general-purpose,
  large general-purpose, instruction-tuned, and one deliberately weak or
  degraded family), embedding **documents only** for the readouts. Per
  corpus×encoder cell: one frozen embedding file.
- **Readouts per cell (all unsupervised):** σ* and its uniform-null ratio;
  liftoff cosine with global-envelope p; occupancy z; DTM low-tail mass and
  1st percentile; pocket count / total pocket mass / max prominence; hubness
  skew; the pre-existing global suite as covariates (mean cosine, IsoScore,
  effective rank, uniformity).
- **Outcomes per cell:** nDCG@10 and recall@100 with the collection's
  judgments and queries under one fixed retrieval setup (exact search;
  optionally one ANN configuration as a robustness check). For collections
  with duplicate annotations, dedup F1 as a second task.

### Hypotheses (the preregistered core)

- **H1 (corpus-level, primary).** Within each corpus, across encoders, σ*
  (and liftoff) correlate with nDCG@10 (Spearman; one test per corpus,
  combined by Fisher's method). Prediction: positive, and stronger than any
  single baseline covariate; test the increment with partial correlations.
- **H2 (item-level, the distinctive claim).** Within a cell, per-document
  expected-collision counts predict *which documents* participate in
  retrieval failures (a judged-relevant document outranked at its own
  queries by non-relevant neighbors). Score as a retrieval-failure
  classifier: AUC / precision-at-k of collision counts against per-document
  failure labels derived from the run. No baseline in the suite can even
  enter this comparison at item granularity except the raw kNN-distance —
  include it as the control.
- **H3 (pathology ordering).** Where the duplicate-vs-diffuse-collapse
  distinction occurs among degraded encoder variants (quantization tends to
  produce one, truncation the other), the confusion-functional ordering —
  not the uniformity ordering — matches the outcome ordering. This is the
  single falsifiable prediction the redundancy analysis (report §6.4)
  licenses.
- **H4 (comparison layer).** Across encoder pairs on a fixed corpus, the
  CMP neighbor-overlap between their document embeddings predicts the
  |Δ nDCG| between them better than global representational similarity
  (CKA) does.
- **H5 (query-model translation).** For each cell, estimate the empirical
  query "noise" directly: the distribution of query-to-judged-relevant
  distances. Test the report's translation — queries whose distance to
  their relevant document exceeds the σ*-implied fade bound should account
  for a disproportionate share of failures. This is the direct audit of the
  isotropic-channel assumption, and its failure modes are as informative as
  its success.

### Analysis and reporting rules

- Within-corpus comparisons are primary; across-corpus pooling is secondary
  and must model corpus as a group factor (Simpson's traps are the norm at
  this altitude — the report's own §2.3 argument, one level up).
- Significance by permutation within corpus; effect sizes with confidence
  intervals; all preregistered readouts reported regardless of outcome —
  negative cells printed with the same prominence as positive ones (house
  rule).
- Robustness appendix: repeat H1/H2 under the ANN configuration; repeat
  with recall@100; drop-one-corpus jackknife.
- Contamination checks: note encoders whose training data plausibly includes
  a corpus; flag, don't silently drop.
- Deliverable: a per-hypothesis results table, the σ*-vs-nDCG scatter
  (color = corpus, shape = encoder family), the H2 AUC table, and a
  plain-language verdict per hypothesis suitable for pasting into the
  technical report's evaluation section.

### What success and failure look like

Success upgrades the report's §9 from "the outstanding scientific step" to a
results section, and turns σ* from an internally-consistent scalar into a
validated predictor. Partial failure is expected and useful: H5 in
particular is likely to expose the isotropic model's limits and would
motivate the structured-noise extension already sketched in §9. Total
failure of H1/H2 would mean the geometry-to-outcome link is weaker than the
theory suggests — which would be the most publishable finding of all, and
would be reported as such.
