# Technical report review viewer

A local page-by-page viewer for the built report that records inline
comments to `docs/tr/review-comments.json`, where they can be read
straight out of the repository.

## Use

```
python3 tools/tr-review/server.py
```

Then open <http://127.0.0.1:8711>. First run renders the PDF to page
images (about 20 seconds for 60 pages); later runs reuse them and
re-render only when the PDF is newer.

- **Click anywhere on a page** to drop a pin, type the comment, Save
  (or Cmd/Ctrl+Enter).
- **Click a pin or a card** to select it and jump to the other.
- **Mark done / Delete** on each card; the *Open* filter hides
  everything already handled.

Comments save immediately — there is nothing to export, and closing the
browser loses nothing.

Options: `--pdf <path>` to review a different build, `--port`, `--dpi`.

## The comment file

```json
[{"id": "c1786332117154",
  "page": 1,
  "x": 0.4996, "y": 0.6431,
  "text": "Tighten this sentence — it runs long.",
  "status": "open",
  "created": "2026-08-09 20:21"}]
```

`x` and `y` are fractions of the page width and height, so they stay
correct at any zoom or dpi. Rows are kept sorted by page and then by
vertical position, which means the file reads top-to-bottom in document
order.

Only `status: "open"` rows are outstanding. Marking one done leaves it
in the file as a record rather than deleting it.

## Notes

- The server binds to `127.0.0.1` only and has no authentication; it is
  a local tool, not something to expose.
- `.tr-review-pages/` holds the rendered images and is git-ignored.
- Requires `pdftoppm` (poppler-utils). No Python dependencies beyond the
  standard library.
