# Eval cases (generated artifacts)

The question sets this example produces. Field schemas + the label vocabulary are
in [../DATASHEET.md](../DATASHEET.md).

| file | set | rows | built by |
|---|---|---|---|
| `questions.jsonl` | A — eCFR citation-grounded | 5,000 (eval/train split) | `../build_eval.py` |
| `questions_open.jsonl` | B — open-corpus synthetic (single + multi-hop) | 4,987 | `../build_open_candidates.py` → `../build_open_inputs.py` → authoring (`../AUTHORING_*.md`) → `../merge_open_eval.py` |
| `baseline_ranks.json` | A — per-query gold rank (with-instruction) | 1,000 eval rows | `../run_eval.py` (a sample result) |

**Not checked in** — large and fully regenerable from the dataset + endpoint:

- the eCFR retrieval corpus `ecfr_corpus.jsonl` (35,173 sections, ~175 MB) —
  regenerate with `../build_eval.py`.
- the ~1.035M-doc open-corpus index (1M map embeddings + the 35k eCFR `.npy`) —
  regenerate with `../embed_corpus.py` and the first `../run_eval.py` /
  `../run_open_eval.py` run (which embed + **cache** the corpus, keyed by model).

So to evaluate against these question sets: regenerate the corpus once (above), then
run the matching `run_*.py` — corpus and query embeddings are cached, so re-runs are
instant and only a new model triggers re-embedding.
