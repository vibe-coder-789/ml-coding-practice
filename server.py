#!/usr/bin/env python3
"""Local practice server.

    ./.venv/bin/python server.py          # http://127.0.0.1:8000

Serves the problem list and runs submissions in a subprocess (see runner.py).

SCOPE: this executes arbitrary Python on this machine — that is the feature, and
it is why the socket is bound to 127.0.0.1 and never 0.0.0.0. Do not put it on a
network. Progress is kept in progress.json next to this file; delete it to reset.
"""
import json
import os
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import runner
import tasks

HERE = os.path.dirname(os.path.abspath(__file__))
PROGRESS = os.path.join(HERE, "progress.json")
EXAMPLES = os.path.join(HERE, "examples.json")
HOST, PORT = "127.0.0.1", int(os.environ.get("PORT", "8000"))


def load_examples():
    try:
        with open(EXAMPLES) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def load_progress():
    try:
        with open(PROGRESS) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def save_progress(p):
    tmp = PROGRESS + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(p, fh, indent=1, sort_keys=True)
    os.replace(tmp, PROGRESS)


class Handler(BaseHTTPRequestHandler):
    server_version = "practice/1.0"

    def log_message(self, fmt, *args):          # quieter than the default
        if "/api/run" in (args[0] if args else ""):
            sys.stderr.write("  run  %s\n" % (args[0],))

    # ---------------------------------------------------------------- helpers
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        raw = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj))

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    # ---------------------------------------------------------------- routes
    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            with open(os.path.join(HERE, "app.html"), "rb") as fh:
                return self._send(200, fh.read(), "text/html; charset=utf-8")
        if path == "/cmtest.html":            # editor smoke-test page
            fp = os.path.join(HERE, "cmtest.html")
            if os.path.isfile(fp):
                with open(fp, "rb") as fh:
                    return self._send(200, fh.read(), "text/html; charset=utf-8")
        if path.startswith("/vendor/"):
            name = os.path.basename(path)          # no traversal: basename only
            fp = os.path.join(HERE, "vendor", name)
            if not os.path.isfile(fp):
                return self._json({"error": "not found"}, 404)
            ctype = ("text/css" if name.endswith(".css")
                     else "application/javascript") + "; charset=utf-8"
            with open(fp, "rb") as fh:
                return self._send(200, fh.read(), ctype)
        if path == "/api/tasks":
            return self._json({
                "tasks": [tasks.public(t) for t in tasks.TASKS],
                "chapters": [list(c) for c in tasks.CHAPTERS],
                "books": tasks.BOOKS,
                "projects": tasks.PROJECTS,
                "book_titles": tasks.BOOK_TITLES,
                "progress": load_progress(),
                "examples": load_examples(),
                "python": sys.version.split()[0],
            })
        if path.startswith("/api/solution/"):
            t = tasks.BY_ID.get(path.rsplit("/", 1)[-1])
            if not t:
                return self._json({"error": "unknown task"}, 404)
            return self._json({fw: tasks.reference(t, fw) for fw in t["frameworks"]})
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self._body()
        except ValueError:
            return self._json({"error": "bad JSON"}, 400)

        if path == "/api/run":
            task_id = body.get("task", "")
            code = body.get("code", "")
            framework = body.get("framework", "torch")
            if not code.strip():
                return self._json({"error": "nothing to run — the editor is empty"})
            result = runner.run(task_id, code, framework)

            if result.get("accepted"):
                p = load_progress()
                entry = p.setdefault(task_id, {"solved": False, "attempts": 0})
                entry["attempts"] += 1
                entry["solved"] = True
                entry["framework"] = framework
                save_progress(p)
                result["progress"] = p
            elif not result.get("error"):
                p = load_progress()
                entry = p.setdefault(task_id, {"solved": False, "attempts": 0})
                entry["attempts"] += 1
                save_progress(p)
                result["progress"] = p
            return self._json(result)

        if path == "/api/reset":
            save_progress({})
            return self._json({"progress": {}})

        return self._json({"error": "not found"}, 404)


def main():
    n_np = sum(1 for t in tasks.TASKS if "numpy" in t["frameworks"])
    n_ex = len(load_examples())
    print(f"  {len(tasks.TASKS)} problems · {len(tasks.CHAPTERS)} chapters · "
          f"{len(tasks.BOOKS)} books ({n_np} accept NumPy, {n_ex} with examples)")
    try:
        import torch
        print(f"  torch {torch.__version__} · python {sys.version.split()[0]}")
    except ImportError:
        print("  WARNING: torch is not importable by this interpreter.")
        print("  Run with ./.venv/bin/python server.py, or pip install torch numpy.")
    url = f"http://{HOST}:{PORT}"
    print(f"  serving {url}   (ctrl-C to stop)\n")
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    if "--no-open" not in sys.argv:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")


if __name__ == "__main__":
    main()
