#!/usr/bin/env python3
"""Local review server for the technical report.

Serves a page-by-page viewer of the built PDF and stores inline comments
as JSON on disk, so they can be read straight out of the repository.

  python3 tools/tr-review/server.py            # http://127.0.0.1:8711
  python3 tools/tr-review/server.py --port 9000 --pdf path/to.pdf

Page images are rendered once with pdftoppm into a scratch directory and
re-rendered whenever the PDF is newer. Standard library only.
"""
import argparse, json, os, re, shutil, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_PDF = os.path.join(ROOT, "docs", "tr", "ambit-technical-report.pdf")
COMMENTS = os.path.join(ROOT, "docs", "tr", "review-comments.json")
PAGES_DIR = os.path.join(ROOT, ".tr-review-pages")
LOCK = threading.Lock()


def render_pages(pdf, dpi=130):
    """Render the PDF to PNGs, skipping work when they are already current."""
    stamp = os.path.join(PAGES_DIR, ".stamp")
    if os.path.exists(stamp) and os.path.getmtime(stamp) >= os.path.getmtime(pdf):
        return sorted(f for f in os.listdir(PAGES_DIR) if f.endswith(".png"))
    if shutil.which("pdftoppm") is None:
        sys.exit("pdftoppm not found (install poppler-utils)")
    shutil.rmtree(PAGES_DIR, ignore_errors=True)
    os.makedirs(PAGES_DIR, exist_ok=True)
    print(f"rendering {os.path.basename(pdf)} at {dpi} dpi ...", flush=True)
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), pdf,
                    os.path.join(PAGES_DIR, "page")], check=True)
    open(stamp, "w").close()
    pages = sorted(f for f in os.listdir(PAGES_DIR) if f.endswith(".png"))
    print(f"{len(pages)} pages ready", flush=True)
    return pages


def load_words(pdf):
    """Word boxes per page, so a pin can be resolved to the text under it.

    Returns {page_number: (width, height, [(x0, y0, x1, y1, word), ...])}.
    """
    out = subprocess.run(["pdftotext", "-bbox", pdf, "-"],
                         capture_output=True, text=True, check=True).stdout
    pages, page_no = {}, 0
    for chunk in re.split(r"<page\b", out)[1:]:
        page_no += 1
        dims = re.match(r'[^>]*width="([\d.]+)"\s+height="([\d.]+)"', chunk)
        if not dims:
            continue
        words = [(float(a), float(b), float(c), float(d),
                  w.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">"))
                 for a, b, c, d, w in re.findall(
                     r'<word xMin="([\d.-]+)" yMin="([\d.-]+)" '
                     r'xMax="([\d.-]+)" yMax="([\d.-]+)">([^<]*)</word>', chunk)]
        pages[page_no] = (float(dims.group(1)), float(dims.group(2)), words)
    return pages


def group_lines(words):
    """Cluster words into visual lines by their vertical centre.

    Bands cannot simply be taken from one word's box: adjacent lines
    overlap slightly, and sorting the union by x interleaves two lines
    into nonsense. Clustering on the centre with a tolerance below one
    line height keeps them apart.
    """
    if not words:
        return []
    heights = sorted(w[3] - w[1] for w in words)
    tol = max(1.0, 0.45 * heights[len(heights) // 2])
    lines, cur, cur_c = [], [], None
    for w in sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        c = (w[1] + w[3]) / 2
        if cur_c is None or abs(c - cur_c) <= tol:
            cur.append(w)
            cur_c = c if cur_c is None else (cur_c * (len(cur) - 1) + c) / len(cur)
        else:
            lines.append(sorted(cur, key=lambda w: w[0]))
            cur, cur_c = [w], c
    if cur:
        lines.append(sorted(cur, key=lambda w: w[0]))
    return lines


def anchor_text(pages, page, xf, yf, span=14):
    """The line under a fractional (x, y), with a little context either side."""
    if page not in pages:
        return ""
    W, H, words = pages[page]
    lines = group_lines(words)
    if not lines:
        return ""
    x, y = xf * W, yf * H

    def vdist(line):
        top, bot = min(w[1] for w in line), max(w[3] for w in line)
        return 0.0 if top <= y <= bot else min(abs(top - y), abs(bot - y))

    line = min(lines, key=vdist)
    k = min(range(len(line)), key=lambda i: abs((line[i][0] + line[i][2]) / 2 - x))
    a, b = max(0, k - span), min(len(line), k + span + 1)
    text = " ".join(w[4] for w in line[a:b])
    return ("… " if a else "") + text + (" …" if b < len(line) else "")


def load_comments():
    if not os.path.exists(COMMENTS):
        return []
    try:
        return json.load(open(COMMENTS))
    except json.JSONDecodeError:
        return []


def save_comments(rows):
    os.makedirs(os.path.dirname(COMMENTS), exist_ok=True)
    tmp = COMMENTS + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rows, fh, indent=1)
    os.replace(tmp, COMMENTS)


class Handler(BaseHTTPRequestHandler):
    pages = []
    words = {}

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = open(os.path.join(os.path.dirname(__file__), "index.html")).read()
            return self._send(200, html, "text/html; charset=utf-8")
        if self.path == "/api/pages":
            return self._send(200, json.dumps(self.pages))
        if self.path == "/api/comments":
            with LOCK:
                return self._send(200, json.dumps(load_comments()))
        if self.path.startswith("/api/anchor?"):
            q = dict(p.split("=", 1) for p in self.path.split("?", 1)[1].split("&"))
            return self._send(200, json.dumps({"anchor": anchor_text(
                self.words, int(q.get("page", 1)),
                float(q.get("x", 0)), float(q.get("y", 0)))}))
        m = re.fullmatch(r"/pages/([A-Za-z0-9_.-]+\.png)", self.path)
        if m and os.path.exists(os.path.join(PAGES_DIR, m.group(1))):
            data = open(os.path.join(PAGES_DIR, m.group(1)), "rb").read()
            return self._send(200, data, "image/png")
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, json.dumps({"error": "bad json"}))
        if self.path == "/api/comments":
            row = {
                "id": payload.get("id") or f"c{int(time.time()*1000)}",
                "page": int(payload.get("page", 1)),
                "x": float(payload.get("x", 0)),
                "y": float(payload.get("y", 0)),
                "text": str(payload.get("text", "")).strip(),
                "anchor": payload.get("anchor") or anchor_text(
                    self.words, int(payload.get("page", 1)),
                    float(payload.get("x", 0)), float(payload.get("y", 0))),
                "status": payload.get("status", "open"),
                "created": payload.get("created") or time.strftime("%Y-%m-%d %H:%M"),
            }
            with LOCK:
                rows = [r for r in load_comments() if r["id"] != row["id"]]
                if payload.get("delete"):
                    save_comments(rows)
                    return self._send(200, json.dumps({"ok": True, "deleted": row["id"]}))
                rows.append(row)
                rows.sort(key=lambda r: (r["page"], r["y"]))
                save_comments(rows)
            return self._send(200, json.dumps(row))
        self._send(404, json.dumps({"error": "not found"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=DEFAULT_PDF)
    ap.add_argument("--port", type=int, default=8711)
    ap.add_argument("--dpi", type=int, default=130)
    a = ap.parse_args()
    if not os.path.exists(a.pdf):
        sys.exit(f"no such pdf: {a.pdf}")
    Handler.pages = render_pages(os.path.abspath(a.pdf), a.dpi)
    Handler.words = load_words(os.path.abspath(a.pdf))
    print(f"text index: {len(Handler.words)} pages")
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"review server on http://127.0.0.1:{a.port}")
    print(f"comments -> {os.path.relpath(COMMENTS, ROOT)}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
