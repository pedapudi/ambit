# Data index — what lives here, and what is safe to delete

Rewritten 2026-08-08 for the `~/ambit-evals` layout (previously a flat
home directory). Sizes measured the same day.

**KEEP** — evidence, ground truth, or expensive to recreate.
**REGEN** — reproducible from KEEP items plus the code in `~/ambit`.
**SCRATCH** — delete freely.

Total under `~/ambit-evals`: ~119 G.

## External grid (~85 G)

| path | size | what | tag |
|---|---|---|---|
| `external-grid/readouts/` | 144 K | frozen per-cell geometry, computed before any label | **KEEP — irreplaceable** |
| `external-grid/FREEZE.sha256` | 8 K | hash chain proving the freeze preceded scoring | **KEEP — irreplaceable** |
| `external-grid/collision-fields/` | 2.8 G | per-document expected collisions; the per-item claim rests on these | KEEP |
| `external-grid/scores/` | 7.2 M | retrieval metrics and per-document failure labels | KEEP |
| `external-grid/synthesis/` | 12 K | aggregate correlations and AUCs | REGEN (from scores + collision fields) |
| `external-grid/dedup-f1/` | 16 K | duplicate-threshold ladder on Quora | KEEP (tiny) |
| `external-grid/maps/` | 82 G | the embeddings themselves | REGEN — **expensive** (HotpotQA alone is 44 G / ~2 h on four GPUs) |
| `external-grid/{queues,logs,scripts}/` | 260 K | how the runs were driven | KEEP (tiny) |

Deleting `external-grid/maps/` reclaims 82 G — two thirds of everything
here — and costs only re-embedding time. Everything the report cites
survives without it. Nothing else in this section is worth deleting.

## Encoder loop (~18 G)

| path | size | what | tag |
|---|---|---|---|
| `encoder-loop/checkpoints/full-finetune-accepted` | 1.2 G | accepted full fine-tune | KEEP (report artifact) |
| `encoder-loop/checkpoints/lora-round2-accepted` | 1.2 G | accepted LoRA; the capacity-ladder rung | KEEP |
| `encoder-loop/checkpoints/lora-round1-rejected` | 1.2 G | the round the gate vetoed | KEEP — a rejected round is evidence |
| `encoder-loop/checkpoints/supervised-{control,guarded}-arm` | 2.4 G | the guardrail experiment's two arms | KEEP (only meaningful as a pair) |
| `encoder-loop/measurement-subset-200k/` | 1.6 G | the fixed subset every round is judged on | KEEP |
| `encoder-loop/rounds/` | 4.6 G | per-round measurements and training logs | KEEP |
| `encoder-loop/retrieval-eval/` | 2.1 G | blind-then-score evaluations per model | KEEP |
| `encoder-loop/tuned-map-1M/` | 3.9 G | accepted tune re-embedded at full scale | REGEN (from the checkpoint) |
| `encoder-loop/corpus-texts{,-heldout}/` | 208 M | raw text fed to the trainer | REGEN (from `~/datasets`) |

## Legal case study (~16 G)

| path | size | what | tag |
|---|---|---|---|
| `case-study-legal/maps-by-model/base-1024-1M` | 3.9 G | the case study proper; every legal figure comes from this | **KEEP — never delete** |
| `case-study-legal/maps-by-model/v3-supervised-1024-1M` | 3.9 G | supervised upper reference | KEEP |
| `case-study-legal/maps-by-model/sup{ctl,grd}-1M` | 7.8 G | guardrail arms at full scale | KEEP (pair) |
| `case-study-legal/logs/` | 28 K | figure data for the report | KEEP (tiny) |

## Reports (443 M)

| path | size | what | tag |
|---|---|---|---|
| `reports/` | 443 M | 66 generated HTML reports | SCRATCH — rendered views of stored readouts |

## Inputs, not held here

`~/models` (95 G) and `~/datasets` (20 G) are shared inputs. The
validation encoders and corpora under them are REGEN via HuggingFace
download; `~/models/qwen3-embedding-0.6b` is **KEEP — never delete**,
since it is the case-study encoder with its serving config and pooling
sidecars.

`~/ft-legal`, `~/ft-legal-data` (25 G) are the supervised legal
fine-tunes — a separate line of work, cited by the report only as a
reference point, and deliberately left outside this tree.

## If space is ever needed

In order of what to remove first:

1. `reports/` — 443 M, pure rendering, regenerate in minutes.
2. `encoder-loop/tuned-map-1M/` — 3.9 G, one embedding pass from a kept
   checkpoint.
3. `external-grid/maps/` — 82 G, the only large reclaim, at the cost of
   hours of GPU time to rebuild.

Do not delete anything tagged KEEP without reading the README in the
directory concerned. The readouts and the freeze chain in particular
cannot be regenerated in any meaningful sense: recomputing them produces
new numbers that no longer predate the labels.
