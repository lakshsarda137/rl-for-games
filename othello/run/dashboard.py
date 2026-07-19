"""Build a standalone training dashboard from data/metrics.jsonl.

The training loop writes one JSON line per iteration to `data/metrics.jsonl`
(loss, self-play speed, buffer size, win rates, max_depth_beaten). This script
turns that log into a single self-contained HTML file with annotated charts —
each one explaining, in plain English, what it measures and what "improving"
looks like.

The output embeds the data inline, so it needs no server and works offline —
open it straight from disk, or download `metrics.jsonl` off a Kaggle run and
build the page locally:

    python run/dashboard.py                 # -> data/dashboard.html, opens it
    python run/dashboard.py --no-open       # just write the file
    python run/dashboard.py --metrics path/to/metrics.jsonl --out my.html

For a LIVE view that refreshes while training runs, start the web app instead
(`python serve/backend.py`) and open http://127.0.0.1:8000/dashboard — it reads
the same jsonl through /api/metrics. This script and that route share one HTML
template (serve/frontend/dashboard.html).
"""

import argparse
import json
import os
import sys
import webbrowser

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "data"))
TEMPLATE = os.path.normpath(os.path.join(_HERE, "..", "serve", "frontend", "dashboard.html"))
PLACEHOLDER = "<!--INJECT_METRICS-->"


def read_rows(path):
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    rows.sort(key=lambda r: r.get("iter", 0))
    return rows


def build_html(rows):
    with open(TEMPLATE) as f:
        html = f.read()
    # Inline the data so the page is fully self-contained (no server / fetch).
    inject = ("<script>window.__METRICS__ = "
              + json.dumps(rows) + ";</script>")
    if PLACEHOLDER not in html:
        raise RuntimeError(f"template missing {PLACEHOLDER!r}: {TEMPLATE}")
    return html.replace(PLACEHOLDER, inject)


def main():
    ap = argparse.ArgumentParser(description="Build the Othello training dashboard.")
    ap.add_argument("--metrics", default=os.path.join(DATA_DIR, "metrics.jsonl"),
                    help="path to metrics.jsonl (default: data/metrics.jsonl)")
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "dashboard.html"),
                    help="output HTML path (default: data/dashboard.html)")
    ap.add_argument("--no-open", action="store_true", help="don't open the file in a browser")
    args = ap.parse_args()

    rows = read_rows(args.metrics)
    if not rows:
        print(f"[dashboard] no metrics found at {args.metrics} — the page will show the "
              "'no data yet' state. Run `python run/train_loop.py --tiny` first.")
    html = build_html(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"[dashboard] wrote {args.out} ({len(rows)} iteration"
          f"{'' if len(rows) == 1 else 's'})")
    if not args.no_open:
        webbrowser.open("file://" + os.path.abspath(args.out))


if __name__ == "__main__":
    main()
