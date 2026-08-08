# Rounds — per-round outputs of the encoder loop

One *round* is: train a candidate, re-embed the fixed 200k measurement
subset, measure it, and apply the continuation rule to the held-out
fraction. This directory holds what each round produced. The narrative —
which round was accepted, which was rejected, and why — lives in
`experiments/e6_encoder_loop/LOOP.md` in the code repository, which is
authoritative wherever it disagrees with a filename here.

Filenames were originally terse (`r1.npy`, `roundA0.json`,
`serve_rF.log`). They were renamed 2026-08-08 to say what they are; the
mapping is at the bottom.

## Subdirectories

| directory | holds |
|---|---|
| `readouts/` | the ambit readout for each candidate, measured on the held-out split — the numbers the continuation rule reads |
| `held-out-subsets/` | the exact row indices held out for each candidate, so a verdict can be re-derived |
| `training-logs/` | training, measurement, evaluation, and significance-test runs |
| `retrieval-scores/` | blind-then-score retrieval results, including a second seed for the rounds that were re-scored |
| `routing/` | the deduplication routing experiment at both thresholds — the liftoff scale and the true-duplicate scale (0.95) |
| `serving-logs/`, `serving-configs/` | vLLM instances stood up to evaluate each candidate |
| `scripts/` | `run-round.sh`, `run-arm.sh` — the drivers |

## The candidates

| name used here | what it was |
|---|---|
| `linear-adapter` | the smallest trainee, a single linear map. Rejected on the real corpus: sigma* falls and flagged-cohort collisions worsen. |
| `lora-round1-rejected` | first LoRA round. Vetoed by the gate — training loss fell while held-out geometry degraded, caused by a reference mismatch (preservation computed against full-document embeddings while the trainer saw truncated views). |
| `lora-round2-accepted` | the corrected LoRA round. |
| `round3` | the third round. Left generically named because its role is described in `LOOP.md` rather than inferable from the file. |
| `full-finetune-accepted` | the full fine-tune, the top rung of the capacity ladder. |
| `supervised-control-arm` / `supervised-guarded-arm` | the guardrail experiment: one supervised recipe run twice, differing only in the guard terms. |

`set-a` / `set-b` in the arm evaluation logs are the two evaluation sets:
the targeted retrieval set (where no cost is claimed) and the untargeted
open-corpus set (where protection is claimed).

## Name mapping

| was | is |
|---|---|
| `round1.json`, `round2.json`, `round3.json`, `roundA0.json` | `readouts/readout-{lora-round1-rejected,lora-round2-accepted,round3,linear-adapter}.json` |
| `map1M.json` | `readouts/readout-full-corpus-1M-accepted-tune.json` |
| `supctl.round.json`, `supgrd.round.json` | `readouts/readout-supervised-{control,guarded}-arm.json` |
| `r1.npy`, `r2.npy`, `r3.npy`, `a0.npy` | `held-out-subsets/subset-*.npy` |
| `run_r1.log` … `run_rF.log`, `a0.log`, `arm_sup*.log` | `training-logs/train-*.log` |
| `measure_*.log`, `eval_r*.log`, `sig_tests.log`, `sup*.set?.log` | `training-logs/` |
| `scores_r*.json` | `retrieval-scores/scores-*.json` |
| `dedup_routing*.{json,log}` | `routing/routing-at-*-threshold*.{json,log}` |
| `serve_*.log`, `vllm_*.yaml` | `serving-logs/`, `serving-configs/` |
