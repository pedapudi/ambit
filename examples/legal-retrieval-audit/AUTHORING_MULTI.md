# Authoring task — multi-hop legal retrieval questions

You author MULTI-HOP evaluation questions for a legal-document retrieval benchmark.
Quality and discipline matter more than speed. You will be told an INPUT file and
an OUTPUT file.

IMPORTANT: Do ALL of this YOURSELF in this single process. Do NOT spawn sub-agents
and do NOT use the Agent/Task tools — author every question sequentially yourself.
(Self-parallelizing multiplies API concurrency and trips rate limits.)

The INPUT is JSONL; each line is `{cluster_id, type, members:[{uuid, subset, text}, ...]}`
— a cluster of 2–3 related legal documents. `type` is:
- "sibling" = sections from the same regulatory Part (closely related federal regs)
- "semantic" = semantically related documents, possibly across sources (CA-Regs,
  eCFR, Case-Law-Summary, CaseHOLD)

For EACH cluster, write ONE natural-language question that genuinely REQUIRES
information from ALL member documents to answer fully — it must NOT be answerable
from any single member alone.

Patterns: synthesis (combine a definition/condition in one with a
requirement/consequence in another), comparison (contrast how members treat an
issue), cross_reference (one defines/triggers, another elaborates), aggregation
(gather items split across members), conditional (answer depends on facts in
different members).

Rules (critical):
- Before finalizing, verify EACH member contributes something necessary. If a
  single member answers it alone, rewrite it.
- No citations, section numbers, the `§` symbol, CFR/Part/Title numbers, or
  document headings/titles. Semantic phrasing only.
- Phrase naturally, as a practitioner would ask.
- `reasoning_type`: exactly ONE of: synthesis, comparison, cross_reference,
  aggregation, conditional.
- `note`: one line explaining why ALL the documents are needed.

OUTPUT: write JSONL to the given output path — one line per cluster, IN THE SAME
ORDER, each object exactly:
`{"cluster_id": <id>, "gold_uuids": [<uuid of EVERY member>], "question": <string>, "reasoning_type": <vocab>, "note": <string>, "cluster_type": <the input "type">}`

Before finishing, verify: line count equals the input; each `gold_uuids` matches
its cluster's member count (2 or 3); no question contains `§`, `CFR`, `Part `, or a
section locator.

Return ONLY a one-line summary: count written, the reasoning_type distribution,
and any clusters skipped with the reason.
