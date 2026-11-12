"""Local dashboard: load a trained Wordle AI and watch it play.

    python serve/app.py        # then open http://127.0.0.1:8002

Two modes:
  1. Play against the AI (a duel, RULES.md sections 4 and 5). You type guesses.
     The AI's guesses stay hidden until the game ends (W4.3).
  2. Watch the AI play the simple bot (best_split), one game guess by guess,
     or many games at once with a summary.

Model files (.pt checkpoints) are found in wordle/models/ and wordle/runs/.
Files added through the page are saved in wordle/models/.

API:
  GET  /api/models                     list of model files
  POST /api/models/upload?name=x.pt    body = the file's bytes
  POST /api/human/new                  {model}: start a duel against the AI
  POST /api/human/{id}/guess           {word}: your guess (the AI answers right after)
  POST /api/human/{id}/next            next game of the same match
  POST /api/watch/game                 {model, secret?, ai_first}: one full AI vs bot game
  POST /api/watch/many                 {model, words}: many games, both seats, summary
"""

import os
import random
import sys
import threading
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from engine import bots, words
from engine.game import Game
from train.lookahead import PositionCache, guess_values
from train.network import ValueNet
from train.selfplay import play_batch

MODELS_DIR = os.path.join(ROOT, "models")
RUNS_DIR = os.path.join(ROOT, "runs")
NO_CACHE = {"Cache-Control": "no-cache, no-store, must-revalidate"}
UNTRAINED = "untrained"

app = FastAPI(title="Wordle AI")
lock = threading.Lock()
cache = PositionCache()
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


# ---------- models ----------

def model_files():
    found = []
    for base in (MODELS_DIR, RUNS_DIR):
        for dirpath, _, files in os.walk(base):
            for f in files:
                if f.endswith(".pt"):
                    found.append(os.path.join(dirpath, f))
    return sorted(found, key=os.path.getmtime, reverse=True)


_loaded = {}


def load_model(model_id):
    """model_id is a path relative to wordle/ (from /api/models), or 'untrained'."""
    if model_id == UNTRAINED:
        if UNTRAINED not in _loaded:
            torch.manual_seed(0)
            net = ValueNet(64)
            net.train(False)
            _loaded[UNTRAINED] = (None, net, {"iteration": 0, "games": 0})
        return _loaded[UNTRAINED][1], _loaded[UNTRAINED][2]
    path = os.path.abspath(os.path.join(ROOT, model_id))
    if not (path.startswith(MODELS_DIR + os.sep) or path.startswith(RUNS_DIR + os.sep)) or not os.path.exists(path):
        raise HTTPException(404, "That model file was not found. Pick another one from the list.")
    mtime = os.path.getmtime(path)
    hit = _loaded.get(path)
    if hit and hit[0] == mtime:
        return hit[1], hit[2]
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        net = ValueNet(ck["hidden"])
        net.load_state_dict(ck["net"])
    except Exception:
        raise HTTPException(400, "That file is not a Wordle AI model.")
    net.train(False)
    _loaded[path] = (mtime, net, ck.get("state", {}))
    return net, ck.get("state", {})


def describe(state):
    games = state.get("games", 0)
    if not games:
        return "Not trained"
    return f"Trained on {games:,} games"


@app.get("/api/models")
def list_models():
    out = [{"id": UNTRAINED, "label": "Untrained AI (for comparison)", "detail": "Not trained"}]
    for path in model_files():
        rel = os.path.relpath(path, ROOT)
        try:
            _, state = load_model(rel)
            detail = describe(state)
        except HTTPException as e:
            detail = e.detail
        when = time.strftime("%b %d, %H:%M", time.localtime(os.path.getmtime(path)))
        out.append({"id": rel, "label": f"{rel} (saved {when})", "detail": detail})
    return out


@app.post("/api/models/upload")
async def upload_model(request: Request, name: str):
    base = os.path.basename(name)
    if not base.endswith(".pt"):
        raise HTTPException(400, "Model files end in .pt")
    data = await request.body()
    if len(data) > 200 * 1024 * 1024:
        raise HTTPException(400, "That file is too large to be a Wordle AI model.")
    os.makedirs(MODELS_DIR, exist_ok=True)
    stem, n = base[:-3], 1
    path = os.path.join(MODELS_DIR, base)
    while os.path.exists(path):
        n += 1
        path = os.path.join(MODELS_DIR, f"{stem}-{n}.pt")
    with open(path, "wb") as f:
        f.write(data)
    rel = os.path.relpath(path, ROOT)
    try:
        load_model(rel)
    except HTTPException:
        os.remove(path)
        raise
    return {"id": rel}


# ---------- the AI's choice, with its reasons ----------

def ai_choose(net, game, top_k=3):
    """Pick the AI's guess for the player whose turn it is; also return its top choices."""
    player = game.to_play
    pos = cache.get(np.asarray(game.candidates))
    q = guess_values(net, "cpu", [pos], game.guesses_made(player), player == 0)[0]
    best = np.argsort(-q)[:top_k]
    word = words.GUESSES[int(pos.guesses[best[0]])]
    options = [{"word": words.GUESSES[int(pos.guesses[i])], "points": round(float(q[i]), 2)} for i in best]
    return word, {"left_before": int(pos.n), "options": options}


def row(word, pattern, extra=None):
    r = {"word": word, "colors": words.pattern_to_colors(pattern)}
    if extra:
        r.update(extra)
    return r


# ---------- mode 1: you against the AI ----------

matches = {}


class NewHuman(BaseModel):
    model: str


class Guess(BaseModel):
    word: str


def human_view(m):
    g, you = m["game"], m["you"]
    ai = 1 - you
    view = {
        "id": m["id"],
        "model": m["model"],
        "you_first": you == 0,
        "your_rows": [row(w, p) for w, p in g.own_history(you)],
        "ai_guess_count": g.guesses_made(ai),
        "your_answers_left": int(len(g.candidates_of(you))),
        "over": g.over,
        "match": m["totals"],
        "games_played": m["games"],
        "letters": letter_status(g.own_history(you)),
    }
    if g.over:
        scores = g.duel_scores()
        view.update(
            secret=g.secret,
            winner="you" if g.winner == you else "ai" if g.winner == ai else "nobody",
            points={"you": scores[you], "ai": scores[ai]},
            ai_rows=[row(w, p, m["notes"][i]) for i, (w, p) in enumerate(g.own_history(ai))],
        )
    return view


def letter_status(history):
    """Best color seen so far for each letter, for the on-screen keyboard."""
    rank = {"gray": 0, "yellow": 1, "green": 2}
    out = {}
    for w, p in history:
        for ch, c in zip(w, words.pattern_to_colors(p)):
            if ch not in out or rank[c] > rank[out[ch]]:
                out[ch] = c
    return out


def ai_turns(m):
    """Let the AI play while it is the AI's turn."""
    g = m["game"]
    while not g.over and g.to_play != m["you"]:
        word, note = ai_choose(m["net"], g)
        m["notes"].append(note)
        g.guess(word)


def finish_if_over(m):
    g = m["game"]
    if g.over and not m["counted"]:
        scores = g.duel_scores()
        m["totals"]["you"] += scores[m["you"]]
        m["totals"]["ai"] += scores[1 - m["you"]]
        m["games"] += 1
        m["counted"] = True


def start_game(m):
    m["game"] = Game(players=2)
    m["notes"] = []
    m["counted"] = False
    ai_turns(m)


@app.post("/api/human/new")
def human_new(req: NewHuman):
    with lock:
        net, _ = load_model(req.model)
        mid = uuid.uuid4().hex[:12]
        m = {"id": mid, "model": req.model, "net": net, "you": random.randrange(2),
             "totals": {"you": 0, "ai": 0}, "games": 0}
        start_game(m)
        matches[mid] = m
        return human_view(m)


def get_match(mid):
    m = matches.get(mid)
    if m is None:
        raise HTTPException(404, "This game has ended on the server. Start a new one.")
    return m


@app.post("/api/human/{mid}/guess")
def human_guess(mid: str, req: Guess):
    with lock:
        m = get_match(mid)
        g = m["game"]
        if g.over:
            raise HTTPException(400, "This game is over. Start the next game.")
        word = req.word.strip().lower()
        if len(word) != 5 or not word.isalpha():
            raise HTTPException(400, "Type a five-letter word.")
        if word not in words.GUESS_INDEX:
            raise HTTPException(400, f"{word.upper()} is not in the word list.")
        g.guess(word)
        ai_turns(m)
        finish_if_over(m)
        return human_view(m)


@app.post("/api/human/{mid}/next")
def human_next(mid: str):
    with lock:
        m = get_match(mid)
        m["you"] = 1 - m["you"]  # W4.6: the first player alternates
        start_game(m)
        finish_if_over(m)
        return human_view(m)


# ---------- mode 2: the AI against the simple bot ----------

class WatchGame(BaseModel):
    model: str
    secret: str | None = None
    ai_first: bool = True


class WatchMany(BaseModel):
    model: str
    words: int = 100


@app.post("/api/watch/game")
def watch_game(req: WatchGame):
    with lock:
        net, _ = load_model(req.model)
        secret = (req.secret or "").strip().lower() or random.choice(words.ANSWERS)
        if secret not in words.ANSWERS:
            raise HTTPException(400, f"{secret.upper()} is not one of the 2,315 possible secret words.")
        g = Game(secret=secret, players=2)
        ai = 0 if req.ai_first else 1
        turns = []
        while not g.over:
            seat = g.to_play
            before = int(len(g.candidates))
            if seat == ai:
                word, note = ai_choose(net, g)
            else:
                word, note = bots.best_split(g), {"left_before": before}
            p = g.guess(word)
            note["left_after"] = int(len(g.candidates_of(seat)))
            turns.append({"player": "ai" if seat == ai else "bot", **row(word, p, note)})
        scores = g.duel_scores()
        return {
            "secret": secret,
            "ai_first": req.ai_first,
            "turns": turns,
            "winner": "ai" if g.winner == ai else "bot" if g.winner == 1 - ai else "nobody",
            "points": {"ai": scores[ai], "bot": scores[1 - ai]},
        }


@app.post("/api/watch/many")
def watch_many(req: WatchMany):
    n = max(1, min(int(req.words), 2315))
    with lock:
        net, _ = load_model(req.model)
        t0 = time.time()
        secrets = np.random.default_rng().choice(len(words.ANSWERS), size=n, replace=False)
        rng = np.random.default_rng(0)
        a = play_batch(("ai", "split"), secrets, cache, net, "cpu", rng)
        b = play_batch(("split", "ai"), secrets, cache, net, "cpu", rng)
    ai = np.concatenate([a.scores[:, 0], b.scores[:, 1]])
    bot = np.concatenate([a.scores[:, 1], b.scores[:, 0]])
    ai_won = np.concatenate([a.winner == 0, b.winner == 1])
    bot_won = np.concatenate([a.winner == 1, b.winner == 0])
    guesses = np.concatenate([a.win_guesses, b.win_guesses])

    def mean(x):
        return round(float(x.mean()), 2) if len(x) else None

    return {
        "games": int(2 * n),
        "seconds": round(time.time() - t0, 1),
        "ai": {"points": mean(ai), "win_pct": round(100 * float(ai_won.mean())),
               "guesses_per_win": mean(guesses[ai_won])},
        "bot": {"points": mean(bot), "win_pct": round(100 * float(bot_won.mean())),
                "guesses_per_win": mean(guesses[bot_won])},
        "draw_pct": round(100 * float((~ai_won & ~bot_won).mean())),
    }


# ---------- page ----------

@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "frontend", "index.html"), headers=NO_CACHE)


if __name__ == "__main__":
    import uvicorn

    words.table()
    print("Open http://127.0.0.1:8002 in your browser.")
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="warning")
