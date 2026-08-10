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
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"review server on http://127.0.0.1:{a.port}")
    print(f"comments -> {os.path.relpath(COMMENTS, ROOT)}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
