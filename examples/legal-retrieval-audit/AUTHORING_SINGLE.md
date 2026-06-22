# Authoring task — single-doc legal retrieval questions

You author evaluation questions for a legal-document retrieval benchmark. Quality
and discipline matter more than speed. You will be told an INPUT file and an
OUTPUT file.

IMPORTANT: Do ALL of this YOURSELF in this single process. Do NOT spawn sub-agents
and do NOT use the Agent/Task tools — author every question sequentially yourself.
(Self-parallelizing multiplies API concurrency and trips rate limits.)

The INPUT is JSONL; each line is `{uuid, subset, text}` — one legal document.
`subset` is the source:
- CA-Regs / California-Code-Of-Regulations = California regulation text
- eCFR = U.S. Code of Federal Regulations (a federal regulation section)
- Case-Law-Summary = a summary of a court case
- CaseHOLD = a case-holding item (a citing context + candidate holdings)

For EACH document, write ONE natural-language question for which THIS document is
the single best answer in a large mixed legal corpus.

Rules (critical):
- Answerable primarily from THIS document's content.
- SPECIFIC enough to distinguish it from thousands of similar documents — anchor it
  on a concrete distinguishing detail (an entity, condition, threshold, or topic
  actually in the text).
- Phrase it as a practitioner/user would ask.
- Do NOT include any citation, section number, the `§` symbol, CFR/Part/Title
  numbers, or the document's heading/title identifiers. You may name the subject
  matter, never the locator. (No tokens like `§`, `CFR`, `Part 52`, `1.23`.)
- Vary phrasing and question types; avoid a repetitive template.
- `reasoning_type`: exactly ONE of: lookup, definition, requirement, scope,
  procedure, numeric_threshold, exception.
- `note`: one line stating what the question tests.

OUTPUT: write JSONL to the given output path — one line per input document, IN THE
SAME ORDER, each object exactly:
`{"uuid": <uuid from input>, "question": <string>, "reasoning_type": <vocab>, "note": <string>}`

Before finishing, verify: line count equals the input; no question contains `§`,
`CFR`, `Part `, or a number like `1.23`.

Return ONLY a one-line summary: count written, the reasoning_type distribution,
and any documents skipped with the reason.
