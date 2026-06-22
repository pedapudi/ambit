#!/usr/bin/env bash
# End-to-end driver for the legal-retrieval-audit example. Each stage is independent;
# run the ones you need. Assumes an OpenAI-compatible embeddings endpoint is serving
# Qwen3-Embedding (e.g. vLLM:  vllm serve <model> --task embed  -> :8081/v1) and that
# the Nemotron-Pretraining-Legal-v1 dataset is available locally.
set -euo pipefail

ENDPOINT=${ENDPOINT:-http://localhost:8081/v1}
DATA=${DATA:-~/datasets/Nemotron-Pretraining-Legal-v1}
PY=${PY:-python}

# 1) EMBED the occupancy map: 4 subdomains -> ~1M docs at native 1024-d (the split
#    used in the demonstration). --dims 768 would Matryoshka-truncate + renorm.
while read -r sub target prefix; do
  $PY embed_corpus.py --subset "Nemotron-Pretraining-Legal-$sub" --target "$target" \
      --prefix "$prefix" --dims 1024 --out-dir ./emb-1024-1M --base-url "$ENDPOINT"
done <<'SUBS'
California-Code-Of-Regulations 57523 careg
Case-Law-Summary 53137 caselaw
CaseHOLD 444670 casehold
eCFR-QA 444670 ecfrqa
SUBS

# 2) OCCUPANCY MAP — ambit's own diagnostics + the per-label breakdown.
ambit info  ./emb-1024-1M --embedding-col embedding
ambit report ./emb-1024-1M --embedding-col embedding --label-col subset --out report.html
$PY analyze_by_label.py ./emb-1024-1M --label-col subset --strip-prefix Nemotron-Pretraining-Legal-

# 3) EVAL SET A — citation-grounded eCFR question->section retrieval.
$PY build_eval.py --data "$DATA" --out eval/questions.jsonl --corpus-out eval/ecfr_corpus.jsonl
$PY run_eval.py --eval-dir eval --base-url "$ENDPOINT" --split eval \
    --distractor-dir ./emb-1024-1M --exclude-label eCFR-QA   # open-corpus variant (optional)

# 4) EVAL SET B — open-corpus synthetic (single + multi-doc), authored by LLM subagents.
$PY build_open_candidates.py --out eval-open/candidates.jsonl
$PY build_open_inputs.py                       # -> single.jsonl, clusters_{sibling,semantic}.jsonl
#   ... slice the pools and spawn authoring agents that read AUTHORING_{SINGLE,MULTI}.md
#   (throttle to small concurrent waves), writing eval-open/authored/*.out.jsonl ...
$PY merge_open_eval.py                          # label + split + validate -> questions_open.jsonl
$PY run_open_eval.py --eval-dir eval-open --map-dir ./emb-1024-1M --base-url "$ENDPOINT" --split eval

# 5) FINE-TUNE (see FINETUNE.md for full training details + the before/after results).
$PY build_train_pairs.py        # derive holdout-disjoint (query, positive) pairs (eCFR-QA + Set B)
$PY mine_hard_negs.py           # add within-family sibling hard negatives via the base endpoint
torchrun --nproc_per_node=2 train.py   # full FT, MNRL + Matryoshka, single uniform dataset
# Then serve the tuned weights under a NEW --served-model-name (cache keys on model id, so a new
# name re-embeds instead of reusing base vectors) and re-run run_eval.py / run_open_eval.py + ambit.
