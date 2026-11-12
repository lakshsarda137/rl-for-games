"""Writes progress.html: charts of every strength check so far. Open it in a browser;
it reloads itself every 60 seconds while training runs."""

import html
import json


def _chart(title, explain, points, series, fmt="{:.2f}", zero_line=False):
    w, h, pad_l, pad_b, pad_t = 640, 220, 56, 28, 12
    xs = [p["games"] for p in points]
    vals = [p[key] for p in points for key, _, _ in series if p.get(key) is not None]
    if not xs or not vals:
        return f"<section><h2>{html.escape(title)}</h2><p>{html.escape(explain)}</p><p>No data yet.</p></section>"
    lo, hi = min(vals + ([0] if zero_line else [])), max(vals + ([0] if zero_line else []))
    if hi - lo < 1e-9:
        lo, hi = lo - 1, hi + 1
    x0, x1 = min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1

    def sx(x):
        return pad_l + (x - x0) / (x1 - x0) * (w - pad_l - 10)

    def sy(y):
        return pad_t + (hi - y) / (hi - lo) * (h - pad_t - pad_b)

    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(title)}">']
    for frac in (0, 0.5, 1):
        y = lo + frac * (hi - lo)
        parts.append(f'<line x1="{pad_l}" x2="{w-10}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" class="grid"/>')
        parts.append(f'<text x="{pad_l-6}" y="{sy(y)+4:.1f}" text-anchor="end">{fmt.format(y)}</text>')
    if zero_line and lo < 0 < hi:
        parts.append(f'<line x1="{pad_l}" x2="{w-10}" y1="{sy(0):.1f}" y2="{sy(0):.1f}" class="zero"/>')
    parts.append(f'<text x="{pad_l}" y="{h-6}">{x0:,} games</text>')
    parts.append(f'<text x="{w-10}" y="{h-6}" text-anchor="end">{x1:,} games</text>')
    legend = []
    for key, label, cls in series:
        pts = [(sx(p["games"]), sy(p[key])) for p in points if p.get(key) is not None]
        if pts:
            path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            parts.append(f'<polyline points="{path}" class="{cls}"/>')
            parts.extend(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" class="{cls}"/>' for x, y in pts)
            last = [p[key] for p in points if p.get(key) is not None][-1]
            legend.append(f'<span class="key {cls}"></span>{html.escape(label)}: {fmt.format(last)}')
    parts.append("</svg>")
    return (f"<section><h2>{html.escape(title)}</h2><p>{html.escape(explain)}</p>"
            f"<p class='legend'>{'<br>'.join(legend)}</p>{''.join(parts)}</section>")


def write(path, metrics, status, sample_text):
    evals = [m for m in metrics if m.get("kind") == "eval"]
    trains = [m for m in metrics if m.get("kind") == "train"]
    charts = [
        _chart("Points per game against the simple bot",
               "Both players' average points in the same games. The AI is better once its line is above the simple bot's line.",
               evals, [("ai_points", "AI", "a"), ("bot_points", "Simple bot", "b")], zero_line=True),
        _chart("Games won, in percent",
               "Out of all test games against the simple bot. The rest are losses or draws.",
               evals, [("win_pct", "AI wins", "a"), ("loss_pct", "AI losses", "b")], fmt="{:.0f}"),
        _chart("Guesses needed when winning",
               "Average number of guesses in the games each player won. Lower is faster.",
               evals, [("ai_win_guesses", "AI", "a"), ("bot_win_guesses", "Simple bot", "b")]),
        _chart("Prediction error",
               "How far the network's predicted points are from the real final points (lower is better).",
               trains, [("loss", "Error", "a")], fmt="{:.3f}"),
    ]
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="60"><title>Wordle Training</title>
<style>
:root {{ --bg:#fafaf7; --fg:#1d1d1b; --muted:#66655f; --grid:#dddcd5; --a:#2f6fb3; --b:#c0632b; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#18181a; --fg:#ececea; --muted:#a3a29c; --grid:#38383b; --a:#6aa6e8; --b:#e8925c; }} }}
body {{ background:var(--bg); color:var(--fg); font:16px/1.5 system-ui, sans-serif; margin:0 auto; max-width:720px; padding:16px; }}
h1 {{ font-size:1.5rem; margin:0 0 4px; }} h2 {{ font-size:1.1rem; margin:28px 0 2px; }}
p {{ margin:4px 0; color:var(--muted); }} .status {{ color:var(--fg); }}
svg {{ width:100%; height:auto; }} svg text {{ fill:var(--muted); font-size:12px; }}
.grid {{ stroke:var(--grid); }} .zero {{ stroke:var(--muted); stroke-dasharray:4 4; }}
polyline {{ fill:none; stroke-width:2; }} polyline.a {{ stroke:var(--a); }} polyline.b {{ stroke:var(--b); }}
circle.a {{ fill:var(--a); }} circle.b {{ fill:var(--b); }}
.legend {{ color:var(--fg); }} .key {{ display:inline-block; width:12px; height:12px; border-radius:2px; margin-right:6px; vertical-align:-1px; }}
.key.a {{ background:var(--a); }} .key.b {{ background:var(--b); }}
pre {{ white-space:pre-wrap; font-size:13px; background:color-mix(in srgb, var(--grid) 35%, transparent); padding:12px; border-radius:6px; }}
</style></head><body>
<h1>Wordle AI training</h1>
<p class="status">{html.escape(status)}</p>
<p>This page reloads every minute.</p>
{''.join(charts)}
<h2>Latest test game</h2>
<p>One game against the simple bot, with the AI's top choices and their expected points.</p>
<pre>{html.escape(sample_text or 'No test game yet.')}</pre>
</body></html>"""
    with open(path, "w") as f:
        f.write(page)


def read_metrics(path):
    try:
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []
