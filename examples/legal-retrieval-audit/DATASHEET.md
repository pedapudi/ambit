# Eval datasheet — legal question→document retrieval

Two complementary retrieval eval sets accompany this example. Both measure whether
an embedding model, given a natural-language legal **question**, retrieves the
**document(s)** that answer it. Every question carries labels so the set is
self-describing — you can slice metrics by question type, reasoning type, source,
and difficulty without re-reading the questions.

Citations, section numbers, the `§` symbol, and CFR/Part/Title locators are
**stripped from every question** by construction, so the task tests *semantic*
retrieval, not string-matching a number.

---

## Set A — eCFR citation-grounded (`eval/questions.jsonl`)

Ground truth is *extracted from the data*, not synthesized: each NVIDIA eCFR-QA
item cites the CFR section its answer rests on; we map that citation to the eCFR
document containing the section and use it as the gold target.

- **Corpus:** the 35,173 eCFR section documents (`eval/ecfr_corpus.jsonl`).
- **Reservoir:** 5,000 questions; `split` ∈ {eval, train} (disjoint, so training
  pairs never leak into the measurement). We over-generate and use a slice.
- **Build:** `build_eval.py`. Gold is unique-match only (ambiguous citations dropped).

Record schema:

| field | meaning |
|---|---|
| `qid` | stable id (`ecfrqa-<hash>`) |
| `question` | citation-stripped natural-language question |
| `gold_uuid` | the eCFR document that governs the answer |
| `gold_title`, `gold_section` | the cited Title / section (provenance only; not in the question) |
| `split` | `eval` or `train` |

## Set B — open-corpus synthetic (`eval-open/questions_open.jsonl`)

Questions authored from the cross-domain documents themselves (by an LLM
subagents, grounded in the real text — no external model), to test retrieval over
a **mixed** corpus and over **multi-document** reasoning.

- **Corpus (open):** the 1M map docs (CA-Regs, Case-Law-Summary, CaseHOLD,
  eCFR-QA) + the 35k eCFR sections ≈ 590k docs — all already embedded.
- **Size:** ~5,000 questions — ~2,500 single-doc, ~2,500 multi-doc.
- **Build:** `build_open_candidates.py` → `build_open_inputs.py` → authoring
  subagents → `merge_open_eval.py`.

Record schema:

| field | meaning |
|---|---|
| `qid` | `open-s-*` (single) / `open-m-*` (multi) |
| `question` | the authored question (no locators) |
| `gold_uuids` | list of document(s) that must be retrieved |
| `n_gold` | 1 for single-doc; 2–3 for multi-doc |
| `question_kind` | `single-doc` \| `multi-doc` |
| `grounding` | how gold was formed: `synthetic-single`, `sibling-cluster` (same CFR Part), `semantic-cluster` (embedding neighbors) |
| `source_subsets` | source(s) of the gold doc(s): CA-Regs / eCFR / Case-Law-Summary / CaseHOLD |
| `reasoning_type` | what the question tests (vocab below) |
| `note` | one-line description of the question's nature, written by the author |
| `split` | `eval` or `train` (stratified by kind×grounding) |

### `reasoning_type` vocabulary

Single-doc: `lookup`, `definition`, `requirement`, `scope`, `procedure`,
`numeric_threshold`, `exception`.

Multi-doc: `synthesis` (combine facts across docs), `comparison` (contrast docs),
`cross_reference` (one doc defines/triggers, another elaborates), `aggregation`
(items split across docs), `conditional` (answer depends on facts in different docs).

A multi-doc question is authored so that **every** gold document is necessary —
it cannot be answered from any single member alone.

---

## Metrics (`run_eval.py`)

- **Single-doc:** Recall@k, MRR@10, median/mean rank.
- **Multi-doc:** Recall@k over the gold *set* (fraction of required docs in top-k)
  and all-gold@k (were *all* required docs retrieved in top-k).
- Query embeddings are **cached** (`eval*/cache/`), keyed by model + id list, so a
  re-eval never re-embeds unchanged questions or corpus — only a new model does.

## Caveats (read before trusting a number)

- **Synthetic label noise (Set B).** A question authored from a document may also
  be answerable by a sibling; gold credits the source doc(s) only. Treat single-doc
  Recall@1 as a *lower bound* on useful retrieval.
- **Citation-grounded noise (Set A).** Some "misses" are sibling sections that also
  answer the question — same lower-bound caveat.
- **Multi-doc necessity** is enforced by the author's judgment, not proven; the
  `note` records the intended dependency.
- Distributions of `kind` / `grounding` / `reasoning_type` are printed by
  `merge_open_eval.py` at build time — consult that output for the exact mix.
