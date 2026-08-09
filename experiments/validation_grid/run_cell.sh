#!/bin/bash
# One grid cell end to end: embed -> measure (freeze) -> collisions -> score.
# Usage: run_cell.sh <corpus-name> <model-dir-name> <model-key> <gpu> [queries-dir] [qrels-file]
set -e
C=$1; M=$2; KEY=$3; GPU=${4:-0}
QDIR=${5:-~/datasets/validation/$C}
QRELS=${6:-~/datasets/validation/$C-qrels/test.tsv}
EVALS=~/ambit/evals/external-grid
MAP=$EVALS/maps/$C/$M
RUNS=$EVALS
PY=~/git/vllm/.venv/bin/python
APY=~/ambit/code/.venv/bin/python
mkdir -p $RUNS/readouts $RUNS/collision-fields $RUNS/scores

CUDA_VISIBLE_DEVICES=$GPU $PY ~/ambit/code/experiments/validation_grid/embed_cell.py \
  --corpus ~/datasets/validation/$C --model ~/models/validation/$M \
  --model-key $KEY --out $MAP --device cuda:0

cd ~/ambit && PYTHONPATH=src $APY experiments/e6_encoder_loop/measure_map.py \
  --map $MAP --out $RUNS/readouts/$C.$M.readout.json
sha256sum $RUNS/readouts/$C.$M.readout.json >> $RUNS/FREEZE.sha256   # freeze before qrels

SIGMA=$($APY -c "import json;print(json.load(open('$RUNS/readouts/$C.$M.readout.json'))['sigma_star'])")
CUDA_VISIBLE_DEVICES=$GPU $PY ~/ambit/code/experiments/validation_grid/collisions_cell.py \
  --map $MAP --sigma $SIGMA --out $RUNS/collision-fields/$C.$M.cc.npz --device cuda:0

CUDA_VISIBLE_DEVICES=$GPU $PY ~/ambit/code/experiments/validation_grid/score_cell.py \
  --map $MAP --queries $QDIR --qrels $QRELS \
  --model ~/models/validation/$M --model-key $KEY \
  --out $RUNS/scores/$C.$M.score.json --device cuda:0

echo "CELL COMPLETE: $C x $M"
