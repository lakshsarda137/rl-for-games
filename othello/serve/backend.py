"""FastAPI backend for the Othello web app — play any bot, or watch two bots.

A thin, single-user local server. It holds games in memory and exposes a tiny
JSON API; the browser (serve/frontend/index.html) renders the board and drives
the flow. The engine + opponents modules are the source of truth for rules and
moves — this file only wires them to HTTP.

Endpoints:
  GET  /                     -> the single-page frontend
  GET  /dashboard            -> the training-metrics dashboard (reads data/metrics.jsonl)
  GET  /api/config           -> capabilities (is Edax installed, limits)
  GET  /api/metrics          -> training metrics rows (one per iteration) for the dashboard
  POST /api/new {black,white}-> start a game, return its id + state
  POST /api/move {id, move}  -> apply a HUMAN move (a square 0-63, or 64=pass)
  POST /api/bot_move {id}    -> let the side-to-move BOT choose + play one move
  POST /api/arena {...}      -> start an N-game AZ-vs-opponent match (parallel, spectate-able)
  GET  /api/arena/{id}       -> poll a match: tally + per-game live boards
  POST /api/arena/{id}/control {action} -> pause | resume | stop the match
  POST /api/checkpoints/delete {label}  -> soft-delete a checkpoint (move to _trash)
  GET  /tournament                      -> round-robin tournament page
  POST /api/tournament {players,...}    -> start a round-robin (>=2 bots), points table
  GET  /api/tournament/{id}             -> poll: standings + live matches/games
  POST /api/tournament/{id}/control {action} -> pause | resume | stop

Player specs (strings): "human", "random", "greedy", "minimax:D", "edax:L", and the trained
net "az" / "az:<sims>" / "az@<ckpt>" / "az:<sims>@<ckpt>" (ckpt = "latest" or "iterNNNN").

Bot endpoints are plain `def` (not async), so Starlette runs them in a
threadpool — a slow deep-minimax or an Edax subprocess never blocks the event
loop or other requests.
"""

import json
import os
import re
import shutil
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "engine"))
sys.path.insert(0, os.path.join(_HERE, "..", "opponents"))
sys.path.insert(0, os.path.join(_HERE, "..", "az"))

from board_numpy import (
    BLACK,
    PASS,
    WHITE,
    apply_move,
    count_discs,
    initial_board,
    is_terminal,
    legal_moves,
    winner,
)
import edax as edax_mod
from minimax import minimax_player
from simple import _random_opening, greedy_move, random_move

FRONTEND_DIR = os.path.join(_HERE, "frontend")
DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "data"))
METRICS_PATH = os.path.join(DATA_DIR, "metrics.jsonl")
CKPT_DIR = os.path.join(DATA_DIR, "checkpoints")
EXT_DIR = os.path.join(DATA_DIR, "external_models")   # external RL opponents (*.pth.tar/*.pt)
MAX_MINIMAX_DEPTH = 8
MAX_EDAX_LEVEL = 30
DEFAULT_AZ_SIMS = 120         # MCTS sims/move for the trained net (strength vs. speed)
MIN_AZ_SIMS, MAX_AZ_SIMS = 10, 1000


def latest_checkpoint():
    """Path to the newest model checkpoint (prefer the rolling latest.pt), or None."""
    latest = os.path.join(CKPT_DIR, "latest.pt")
    if os.path.isfile(latest):
        return latest
    if not os.path.isdir(CKPT_DIR):
        return None
    best, best_it = None, -1
    for name in os.listdir(CKPT_DIR):
        if name.startswith("iter") and name.endswith(".pt"):
            try:
                it = int(name[4:-3])
            except ValueError:
                continue
            if it > best_it:
                best, best_it = os.path.join(CKPT_DIR, name), it
    return best


# A checkpoint label is a .pt filename stem: 'latest', 'iter0007', or an archived
# pull like 'run2-iter04'. Restrict to a safe charset (no separators / traversal)
# so a label can never escape CKPT_DIR.
_CKPT_LABEL_RE = re.compile(r"[a-z0-9._-]+")


def checkpoint_path(label):
    """Resolve a checkpoint label to a file path inside CKPT_DIR.

    'latest' (or empty) -> the rolling latest.pt (or newest iter file). Any other
    label -> `<label>.pt` if it exists (e.g. 'iter0007', 'run2-iter04'). Raises
    ValueError on a bad/missing label so a stale picker selection is a clean 400,
    not a torch stack trace. Path-traversal is blocked (charset + dirname check).
    """
    label = (label or "").strip().lower()
    if label in ("", "latest"):
        p = latest_checkpoint()
        if p is None:
            raise ValueError("no trained model available yet")
        return p
    if ".." in label or not _CKPT_LABEL_RE.fullmatch(label):
        raise ValueError(f"bad checkpoint label {label!r}")
    p = os.path.join(CKPT_DIR, label + ".pt")
    if os.path.dirname(os.path.abspath(p)) != os.path.abspath(CKPT_DIR) or not os.path.isfile(p):
        raise ValueError(f"checkpoint {label} not found")
    return p


def list_checkpoints():
    """All loadable checkpoints, newest first, for the web UI's checkpoint picker.

    latest.pt is listed first (its iteration read from the file, cached by mtime);
    every other `*.pt` in CKPT_DIR follows — the per-iter iterNNNN.pt files AND any
    archived pulls (run2-iter04.pt). Iteration is parsed from the name (`iter<N>`
    anywhere in it), so listing stays cheap no matter how many have accumulated.
    Files under the _trash subdir are ignored (that's where deletes go).
    """
    out = []
    latest = os.path.join(CKPT_DIR, "latest.pt")
    if os.path.isfile(latest):
        it = None
        try:
            _, it = load_az_evaluator(latest)   # cached by mtime
        except Exception:
            it = None
        out.append({"label": "latest", "iteration": it, "is_latest": True})
    others = []
    if os.path.isdir(CKPT_DIR):
        for name in sorted(os.listdir(CKPT_DIR)):
            if not name.endswith(".pt") or name == "latest.pt":
                continue
            label = name[:-3]
            m = re.search(r"iter0*(\d+)", label)
            others.append({"label": label, "iteration": int(m.group(1)) if m else None,
                           "is_latest": False})
    others.sort(key=lambda d: (d["iteration"] if d["iteration"] is not None else -1, d["label"]),
                reverse=True)
    out.extend(others)
    return out


def _parse_az_spec(spec):
    """Split an AZ spec into (sims, checkpoint_label).

    Accepted forms (checkpoint optional, sims optional):
      az                 -> (default sims, 'latest')
      az:80              -> (80,           'latest')
      az@iter0007        -> (default sims, 'iter0007')
      az:80@iter0007     -> (80,           'iter0007')
    """
    spec = (spec or "").strip().lower()
    body = spec[2:] if spec.startswith("az") else spec   # drop the leading 'az'
    ckpt = "latest"
    if "@" in body:
        body, ckpt = body.split("@", 1)
    sims = DEFAULT_AZ_SIMS
    if body.startswith(":") and body[1:]:
        try:
            sims = int(body[1:])
        except ValueError:
            raise ValueError("az sims must be an integer, e.g. 'az:200'")
    sims = max(MIN_AZ_SIMS, min(MAX_AZ_SIMS, sims))
    return sims, (ckpt or "latest")


def _is_az_spec(spec):
    spec = (spec or "").strip().lower()
    return spec == "az" or spec.startswith("az:") or spec.startswith("az@")


# --- external RL opponents (alpha-zero-general net; see az/external_bot.py) ----
_EXT_LABEL_RE = re.compile(r"[a-z0-9._-]+")
_EXT_CACHE = {}   # path -> (mtime, net); loaded once, shared across games (read-only)


def list_external_models():
    """External RL opponents installed in data/external_models (*.pth.tar / *.pt).

    These are other AlphaZero-family nets (e.g. alpha-zero-general's pretrained
    Othello) you can benchmark against — an RL peer, unlike search-based Edax."""
    out = []
    if os.path.isdir(EXT_DIR):
        for name in sorted(os.listdir(EXT_DIR)):
            stem = name[:-8] if name.endswith(".pth.tar") else (
                name[:-3] if name.endswith(".pt") else None)
            if stem:
                out.append({"label": stem, "stem": stem})
    return out


def external_model_path(stem):
    """Resolve an external-model stem to a file in EXT_DIR (traversal-guarded)."""
    stem = (stem or "").strip().lower()
    if not stem or ".." in stem or not _EXT_LABEL_RE.fullmatch(stem):
        raise ValueError(f"bad external model {stem!r}")
    for ext in (".pth.tar", ".pt"):
        p = os.path.join(EXT_DIR, stem + ext)
        if os.path.dirname(os.path.abspath(p)) == os.path.abspath(EXT_DIR) and os.path.isfile(p):
            return p
    raise ValueError(f"external model {stem} not found")


def _parse_ext_spec(spec):
    """(sims, stem) from 'extbot', 'extbot:100', 'extbot:100@azg_8x8'. No stem ->
    the first installed external model."""
    spec = (spec or "").strip().lower()
    body = spec[len("extbot"):] if spec.startswith("extbot") else spec
    stem = None
    if "@" in body:
        body, stem = body.split("@", 1)
    sims = DEFAULT_AZ_SIMS
    if body.startswith(":") and body[1:]:
        try:
            sims = int(body[1:])
        except ValueError:
            raise ValueError("extbot sims must be an integer, e.g. 'extbot:200'")
    sims = max(MIN_AZ_SIMS, min(MAX_AZ_SIMS, sims))
    if not stem:
        models = list_external_models()
        if not models:
            raise ValueError("no external RL model installed (see data/external_models/)")
        stem = models[0]["stem"]
    return sims, stem


def _is_ext_spec(spec):
    spec = (spec or "").strip().lower()
    return spec == "extbot" or spec.startswith("extbot:") or spec.startswith("extbot@")


def _load_azg_cached(path):
    """Load an external net once, cache by (path, mtime) — shared across games."""
    from external_bot import load_azg_net
    mtime = os.path.getmtime(path)
    cached = _EXT_CACHE.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    net = load_azg_net(path, "cpu")
    _EXT_CACHE[path] = (mtime, net)
    return net


def _build_ext_player(spec):
    """Turn an 'extbot' spec into a move fn: the external net driven by OUR MCTS."""
    from external_bot import azg_player
    sims, stem = _parse_ext_spec(spec)
    path = external_model_path(stem)
    net = _load_azg_cached(path)
    return azg_player(path, sims, net=net)


# Loaded Evaluators cached by (path -> (mtime, evaluator, iteration)). Keying on
# mtime means that while training is running and overwrites latest.pt, the next
# game automatically picks up the fresher weights.
_AZ_CACHE = {}


def load_az_evaluator(path):
    """Build (and cache) an Evaluator from a checkpoint. Lazy-imports torch so the
    play server still runs for the non-AZ bots even without torch present."""
    import torch
    from network import Evaluator, OthelloNet

    mtime = os.path.getmtime(path)
    cached = _AZ_CACHE.get(path)
    if cached and cached[0] == mtime:
        return cached[1], cached[2]
    ckpt = torch.load(path, map_location="cpu")
    saved = ckpt.get("config", {})
    net = OthelloNet(saved.get("num_blocks", 5), saved.get("channels", 64))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    ev = Evaluator(net, "cpu")
    it = ckpt.get("iteration")
    _AZ_CACHE[path] = (mtime, ev, it)
    return ev, it


def _build_az_player(spec):
    """Turn an AZ spec into a move fn. 'az', 'az:<sims>', 'az@<ckpt>', 'az:<sims>@<ckpt>'.

    The checkpoint selector (@iterNNNN) lets you load a SPECIFIC past agent as a
    player, not just the rolling latest — so old iterations can be pitted against
    each other or against Edax.
    """
    sims, ckpt = _parse_az_spec(spec)
    path = checkpoint_path(ckpt)               # raises ValueError if missing/bad
    evaluator, _ = load_az_evaluator(path)
    from evaluate import az_player
    return az_player(evaluator, sims)


# --- Arena: run N games AZ-vs-opponent, in PARALLEL, spectate-able ------------
# In-training eval is off by default, so this is the on-demand strength check:
# a match of N colour-alternated games returns win/draw/loss + win rate, far more
# meaningful than a single browser game. The games run CONCURRENTLY in a thread
# pool and each publishes its live board into the job dict, so the UI can (a) show
# a running tally, (b) let you watch any single game move-by-move, and (c)
# pause/resume/stop the whole match. Threads (not processes) because the point is
# concurrency + shared live state for spectating; the net eval releases the GIL so
# there's some real speedup, and every game visibly advances together regardless.
#
# `_ARENAS[job_id]`   -> the JSON-serialisable job dict the API returns (tally +
#                        per-game live boards + control flags).
# `_ARENA_PRIV[job_id]`-> non-serialisable sidecar (player factories, per-game
#                        openings/seeds, the aggregate lock). Kept apart so the job
#                        dict can be handed straight to FastAPI as JSON.
_ARENAS = {}
_ARENA_PRIV = {}
MAX_ARENA_GAMES = 200
MAX_ARENA_WORKERS = 8


def _make_factory(spec):
    """(factory, label) for a player spec, loading any heavy state (the net) ONCE.

    `factory(rng) -> move_fn` is called once per game so each concurrent game gets
    its OWN search state (a fresh MCTS / rng) — sharing one MCTS across threads
    would race. The Evaluator (the loaded net) IS shared: torch inference in eval
    mode is read-only and thread-safe, and loading it per game would be wasteful.
    """
    spec = (spec or "").strip().lower()
    if _is_az_spec(spec):
        sims, ckpt = _parse_az_spec(spec)
        path = checkpoint_path(ckpt)                    # raises on missing/bad
        evaluator, it = load_az_evaluator(path)
        from evaluate import az_player
        label = f"AZ {ckpt}" + (f"·it{it}" if it is not None and ckpt == "latest" else "") + f" · {sims} sims"
        return (lambda rng: az_player(evaluator, sims, rng=rng)), label
    if _is_ext_spec(spec):
        from external_bot import azg_player
        sims, stem = _parse_ext_spec(spec)
        path = external_model_path(stem)
        net = _load_azg_cached(path)                    # loaded once, shared across games
        return (lambda rng: azg_player(path, sims, net=net, rng=rng)), f"RL {stem} · {sims} sims"
    if spec == "random":
        return (lambda rng: (lambda b, p: random_move(b, p, rng))), "random"
    # greedy / minimax:D / edax:L are stateless move fns — build once, share safely.
    fn = build_player(spec)                              # validates; raises on bad spec
    return (lambda rng: fn), spec


def _publish_slot(slot, board, next_player, last_move):
    """Record a game's current board into its (JSON-safe) live slot after a move."""
    black_ct, white_ct = count_discs(board)
    slot["board"] = [int(x) for x in board.reshape(-1)]
    slot["to_move"] = int(next_player)
    slot["last_move"] = int(last_move)
    slot["ply"] = slot["ply"] + 1
    slot["black_count"] = int(black_ct)
    slot["white_count"] = int(white_ct)


def _arena_play_one(job, priv, idx):
    """Play one arena game move-by-move, publishing live state and honouring pause/stop."""
    cfg = priv["configs"][idx]
    az_is, opening = cfg["az_is"], list(cfg["opening"])
    slot = job["games"][idx]
    az_fn = priv["az_make"](np.random.default_rng(cfg["seed"]))
    opp_fn = priv["opp_make"](np.random.default_rng(cfg["seed"] + 1))

    board = initial_board()
    player = BLACK
    while not is_terminal(board):
        if job["cancel"]:
            slot["aborted"] = True
            slot["done"] = True
            return
        while job["paused"] and not job["cancel"]:
            time.sleep(0.05)
        if opening:
            move = opening.pop(0)
        elif not legal_moves(board, player):
            move = PASS
        else:
            move = (az_fn if player == az_is else opp_fn)(board, player)
        board = apply_move(board, player, move)
        player = -player
        _publish_slot(slot, board, player, move)

    result = int(winner(board))
    with priv["lock"]:
        slot["done"] = True
        slot["result"] = result
        if result == 0:
            slot["az_outcome"] = "D"; job["draws"] += 1
        elif result == az_is:
            slot["az_outcome"] = "W"; job["wins"] += 1
        else:
            slot["az_outcome"] = "L"; job["losses"] += 1
        job["played"] = job["wins"] + job["draws"] + job["losses"]
        n = job["played"]
        job["win_rate"] = (job["wins"] + 0.5 * job["draws"]) / n if n else 0.0


def _run_arena(job_id, az_spec, opp_spec, n_games, workers, seed):
    job = _ARENAS[job_id]
    priv = _ARENA_PRIV[job_id]
    try:
        priv["az_make"], job["az_label"] = _make_factory(az_spec)
        priv["opp_make"], job["opp_label"] = _make_factory(opp_spec)
    except Exception as exc:                    # bad spec, edax missing, unreadable ckpt
        job.update(error=str(exc), done=True)
        return

    rng = np.random.default_rng(seed)
    start = [int(x) for x in initial_board().reshape(-1)]
    for g in range(n_games):
        az_is = BLACK if g % 2 == 0 else WHITE   # alternate colours for fairness
        priv["configs"].append({"opening": _random_opening(rng, 4),
                                 "az_is": int(az_is),
                                 "seed": int(rng.integers(1 << 30))})
        job["games"].append({"idx": g, "az_is": int(az_is), "board": list(start),
                             "to_move": int(BLACK), "last_move": None, "ply": 0,
                             "black_count": 2, "white_count": 2, "done": False,
                             "aborted": False, "result": None, "az_outcome": None})

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_arena_play_one, job, priv, g) for g in range(n_games)]
            for f in as_completed(futs):
                exc = f.exception()
                if exc and not job.get("error"):
                    job["error"] = f"game error: {exc}"
    finally:
        job["done"] = True


def _trim_arenas():
    """Keep the registry small — drop the oldest finished jobs beyond 20."""
    finished = [k for k, v in _ARENAS.items() if v.get("done")]
    for k in finished[:-20]:
        _ARENAS.pop(k, None)
        _ARENA_PRIV.pop(k, None)


# --- Round-robin tournament: N participants, every pair plays a match ---------
# A "match" is games_per_match colour-alternated games between two participants;
# a "game" is one Othello game. Match scoring: more game-wins => match win (3 pts),
# a match drawn on game-wins => 1.5 each, loss => 0. Standings tiebreaker = total
# game-wins across the whole tournament. All games (across all matches) share ONE
# ThreadPoolExecutor of `concurrency` slots, so several matches can be live at once
# — the UI watches whichever matches currently have games in flight. Same live-slot
# + pause/stop machinery as the Arena; _TOURNEY_PRIV holds the non-JSON sidecar.
_TOURNEYS = {}
_TOURNEY_PRIV = {}
MAX_TOURNEY_PLAYERS = 8
MAX_TOURNEY_GAMES = 100        # games per match
MAX_TOURNEY_CONC = 8


def _pretty_label(spec):
    """Short human label for a bot spec, for the standings table + match cards."""
    s = (spec or "").strip().lower()
    if s == "random":
        return "Random"
    if s == "greedy":
        return "Greedy"
    if s.startswith("minimax:"):
        return "Minimax d" + s.split(":")[1]
    if s.startswith("edax"):
        return "Edax L" + (s.split(":")[1] if ":" in s else "6")
    if _is_ext_spec(s):
        sims, stem = _parse_ext_spec(s)
        return f"RL {stem}·{sims}"
    if _is_az_spec(s):
        sims, ck = _parse_az_spec(s)
        if ck == "latest":
            return f"AZ latest·{sims}"
        r = re.match(r"(.+)-iter0*(\d+)$", ck)      # "<run>-iter<NN>" -> include the run
        if r:
            return f"AZ {r.group(1)} iter{r.group(2)}·{sims}"
        m = re.search(r"iter0*(\d+)", ck)
        return (f"AZ iter{m.group(1)}" if m else f"AZ {ck}") + f"·{sims}"
    return spec


def _recompute_standings(job):
    """Rebuild the points table from the matches (called as games/matches finish)."""
    n = len(job["participants"])
    st = [{"i": i, "points": 0.0, "mp": 0, "mw": 0, "md": 0, "ml": 0,
           "gw": 0, "gd": 0, "gl": 0} for i in range(n)]
    for m in job["matches"]:
        a, b = m["a"], m["b"]
        st[a]["gw"] += m["a_wins"]; st[a]["gl"] += m["b_wins"]; st[a]["gd"] += m["draws"]
        st[b]["gw"] += m["b_wins"]; st[b]["gl"] += m["a_wins"]; st[b]["gd"] += m["draws"]
        if m["done"]:
            st[a]["mp"] += 1; st[b]["mp"] += 1
            if m["result_winner"] == a:
                st[a]["points"] += 3; st[a]["mw"] += 1; st[b]["ml"] += 1
            elif m["result_winner"] == b:
                st[b]["points"] += 3; st[b]["mw"] += 1; st[a]["ml"] += 1
            else:
                st[a]["points"] += 1.5; st[b]["points"] += 1.5
                st[a]["md"] += 1; st[b]["md"] += 1
    st.sort(key=lambda r: (r["points"], r["gw"]), reverse=True)   # tiebreak on game-wins
    job["standings"] = st


def _play_tourney_game(job, priv, mi, gi):
    """Play one tournament game move-by-move, publishing live board + honouring pause/stop."""
    m = job["matches"][mi]
    slot = m["games"][gi]
    if job["cancel"]:
        slot["aborted"] = True; slot["done"] = True; return
    cfg = priv["matches"][mi]["games"][gi]
    black_fn = cfg["black_make"](np.random.default_rng(cfg["seed"]))
    white_fn = cfg["white_make"](np.random.default_rng(cfg["seed"] + 1))
    board = initial_board()
    player = BLACK
    opening = list(cfg["opening"])
    while not is_terminal(board):
        if job["cancel"]:
            slot["aborted"] = True; slot["done"] = True; return
        while job["paused"] and not job["cancel"]:
            time.sleep(0.05)
        if opening:
            move = opening.pop(0)
        elif not legal_moves(board, player):
            move = PASS
        else:
            move = (black_fn if player == BLACK else white_fn)(board, player)
        board = apply_move(board, player, move)
        player = -player
        _publish_slot(slot, board, player, move)

    result = int(winner(board))
    with priv["lock"]:
        slot["done"] = True
        slot["result"] = result
        a, b = m["a"], m["b"]
        if result == 0:
            m["draws"] += 1; slot["winner"] = None
        else:
            a_won = (result == BLACK) == cfg["a_is_black"]
            if a_won:
                m["a_wins"] += 1; slot["winner"] = a
            else:
                m["b_wins"] += 1; slot["winner"] = b
        m["played"] = m["a_wins"] + m["b_wins"] + m["draws"]
        job["played_games"] += 1
        if m["played"] >= m["total"] and not m["done"]:
            m["done"] = True
            m["result_winner"] = (a if m["a_wins"] > m["b_wins"]
                                  else b if m["b_wins"] > m["a_wins"] else None)
        _recompute_standings(job)


def _run_tourney(job_id, specs, games_per_match, concurrency, seed):
    job = _TOURNEYS[job_id]
    priv = _TOURNEY_PRIV[job_id]
    try:
        makes = [_make_factory(s)[0] for s in specs]        # loads each net once, validates
    except Exception as exc:
        job.update(error=str(exc), done=True)
        return
    n = len(specs)
    job["participants"] = [{"i": i, "spec": specs[i], "label": _pretty_label(specs[i])} for i in range(n)]
    rng = np.random.default_rng(seed)
    start = [int(x) for x in initial_board().reshape(-1)]
    tasks = []
    for a in range(n):
        for b in range(a + 1, n):
            mi = len(job["matches"])
            pub_games, priv_games = [], []
            for gi in range(games_per_match):
                a_is_black = (gi % 2 == 0)                  # alternate colours across the match
                pub_games.append({"idx": gi, "a_is_black": a_is_black, "board": list(start),
                                  "to_move": int(BLACK), "last_move": None, "ply": 0,
                                  "black_count": 2, "white_count": 2, "done": False,
                                  "aborted": False, "result": None, "winner": None})
                priv_games.append({"black_make": makes[a] if a_is_black else makes[b],
                                   "white_make": makes[b] if a_is_black else makes[a],
                                   "opening": _random_opening(rng, 4),
                                   "seed": int(rng.integers(1 << 30)), "a_is_black": a_is_black})
                tasks.append((mi, gi))
            job["matches"].append({"i": mi, "a": a, "b": b,
                                   "label_a": job["participants"][a]["label"],
                                   "label_b": job["participants"][b]["label"],
                                   "games": pub_games, "a_wins": 0, "b_wins": 0, "draws": 0,
                                   "played": 0, "total": games_per_match, "done": False,
                                   "result_winner": None})
            priv["matches"].append({"games": priv_games})
    job["total_games"] = len(tasks)
    _recompute_standings(job)
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(_play_tourney_game, job, priv, mi, gi) for (mi, gi) in tasks]
            for f in as_completed(futs):
                exc = f.exception()
                if exc and not job.get("error"):
                    job["error"] = f"game error: {exc}"
    finally:
        job["done"] = True


def _trim_tourneys():
    finished = [k for k, v in _TOURNEYS.items() if v.get("done")]
    for k in finished[:-10]:
        _TOURNEYS.pop(k, None)
        _TOURNEY_PRIV.pop(k, None)


def read_metrics(path=METRICS_PATH):
    """Parse data/metrics.jsonl into a list of per-iteration dicts (skip bad lines)."""
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a half-written last line during a live run — ignore
    rows.sort(key=lambda r: r.get("iter", 0))
    return rows

app = FastAPI(title="Othello")
_GAMES = {}


# --- player construction -----------------------------------------------------
def build_player(spec):
    """Spec string -> `(board, player) -> move` function, or None for a human."""
    spec = (spec or "").strip().lower()
    if spec == "human":
        return None
    if spec == "random":
        rng = np.random.default_rng()
        return lambda b, p: random_move(b, p, rng)
    if spec == "greedy":
        return greedy_move
    if spec.startswith("minimax:"):
        depth = int(spec.split(":", 1)[1])
        if not 1 <= depth <= MAX_MINIMAX_DEPTH:
            raise ValueError(f"minimax depth must be 1-{MAX_MINIMAX_DEPTH}")
        return minimax_player(depth)
    if spec == "edax" or spec.startswith("edax:"):
        level = int(spec.split(":", 1)[1]) if ":" in spec else 6
        if not 0 <= level <= MAX_EDAX_LEVEL:
            raise ValueError(f"edax level must be 0-{MAX_EDAX_LEVEL}")
        return edax_mod.edax_player(level)  # raises EdaxNotInstalled if absent
    if _is_az_spec(spec):
        return _build_az_player(spec)       # the trained AlphaZero net (latest or a chosen checkpoint)
    if _is_ext_spec(spec):
        return _build_ext_player(spec)      # an external RL net (alpha-zero-general) via our MCTS
    raise ValueError(f"unknown player: {spec!r}")


class Game:
    def __init__(self, black_spec, white_spec):
        self.black_spec, self.white_spec = black_spec, white_spec
        self.fns = {BLACK: build_player(black_spec), WHITE: build_player(white_spec)}
        self.board = initial_board()
        self.player = BLACK
        self.last_move = None

    def is_human_turn(self):
        return self.fns[self.player] is None

    def apply(self, move):
        self.board = apply_move(self.board, self.player, move)
        self.last_move = int(move)
        self.player = -self.player

    def state(self):
        done = is_terminal(self.board)
        legal = [] if done else legal_moves(self.board, self.player)
        black_ct, white_ct = count_discs(self.board)
        return {
            "board": [int(x) for x in self.board.reshape(-1)],
            "player": int(self.player),
            "current_name": "Black" if self.player == BLACK else "White",
            "legal": legal,
            "must_pass": (not done) and len(legal) == 0,
            "done": done,
            "result": (int(winner(self.board)) if done else None),
            "black_count": black_ct,
            "white_count": white_ct,
            "black_spec": self.black_spec,
            "white_spec": self.white_spec,
            "last_move": self.last_move,
            "current_is_human": (not done) and self.is_human_turn(),
        }


def _get(game_id):
    game = _GAMES.get(game_id)
    if game is None:
        raise HTTPException(404, "unknown game id (start a new game)")
    return game


# --- request models ----------------------------------------------------------
class NewGame(BaseModel):
    black: str = "human"
    white: str = "minimax:4"


class MoveReq(BaseModel):
    game_id: str
    move: int


class GameReq(BaseModel):
    game_id: str


class ArenaReq(BaseModel):
    opponent: str = "minimax:2"   # any bot spec (not human); may be "az:80@iter0003"
    games: int = 20
    az_sims: int = DEFAULT_AZ_SIMS
    az_ckpt: str = "latest"       # which checkpoint the AZ champion plays
    workers: int = 6              # how many games run concurrently


class ArenaControl(BaseModel):
    action: str = "pause"         # "pause" | "resume" | "stop"


class DeleteCkptReq(BaseModel):
    label: str                    # checkpoint stem to remove, e.g. "run2-iter04" or "latest"


class TourneyReq(BaseModel):
    players: list = []            # >=2 bot specs (no human)
    games_per_match: int = 4
    concurrency: int = 4


# --- routes ------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/dashboard")
def dashboard():
    return FileResponse(os.path.join(FRONTEND_DIR, "dashboard.html"))


@app.get("/tournament")
def tournament_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "tournament.html"))


@app.get("/api/metrics")
def metrics():
    return {"rows": read_metrics()}


@app.post("/api/arena")
def arena_start(req: ArenaReq):
    if latest_checkpoint() is None:
        raise HTTPException(400, "no trained model available yet — train first")
    if (req.opponent or "").strip().lower() == "human":
        raise HTTPException(400, "opponent must be a bot, not human")
    n = max(1, min(MAX_ARENA_GAMES, int(req.games)))
    sims = max(MIN_AZ_SIMS, min(MAX_AZ_SIMS, int(req.az_sims)))
    workers = max(1, min(MAX_ARENA_WORKERS, int(req.workers)))
    az_ckpt = (req.az_ckpt or "latest").strip().lower()
    az_spec = f"az:{sims}@{az_ckpt}"
    # Validate both specs synchronously so a bad checkpoint / missing Edax is a
    # clean 400 right now, not a background error you'd only see by polling.
    try:
        _make_factory(az_spec)
        _make_factory(req.opponent)
    except Exception as exc:
        raise HTTPException(400, str(exc))
    job_id = uuid.uuid4().hex[:12]
    _ARENAS[job_id] = {"job_id": job_id, "total": n, "played": 0, "wins": 0,
                       "draws": 0, "losses": 0, "win_rate": 0.0, "done": False,
                       "paused": False, "cancel": False, "error": None,
                       "opponent": req.opponent, "az_ckpt": az_ckpt, "az_sims": sims,
                       "workers": workers, "az_label": None, "opp_label": None,
                       "games": []}
    _ARENA_PRIV[job_id] = {"configs": [], "lock": threading.Lock(),
                           "az_make": None, "opp_make": None}
    _trim_arenas()
    t = threading.Thread(
        target=_run_arena,
        args=(job_id, az_spec, req.opponent, n, workers,
              int.from_bytes(os.urandom(4), "little")),
        daemon=True)
    t.start()
    return _ARENAS[job_id]


@app.get("/api/arena/{job_id}")
def arena_status(job_id: str):
    job = _ARENAS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown arena job (it may have expired)")
    return job


@app.post("/api/arena/{job_id}/control")
def arena_control(job_id: str, req: ArenaControl):
    """Pause / resume / stop a running match. Games check these flags between moves."""
    job = _ARENAS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown arena job (it may have expired)")
    action = (req.action or "").strip().lower()
    if action == "pause":
        job["paused"] = True
    elif action == "resume":
        job["paused"] = False
    elif action == "stop":
        job["cancel"] = True
        job["paused"] = False
    else:
        raise HTTPException(400, "action must be pause, resume, or stop")
    return job


@app.get("/api/config")
def config():
    az_path = latest_checkpoint()
    az_iteration = None
    if az_path is not None:
        try:
            _, az_iteration = load_az_evaluator(az_path)  # warms the cache too
        except Exception:
            az_path = None                                 # unreadable/torch missing
    return {
        "edax_available": edax_mod.is_available(),
        "max_minimax_depth": MAX_MINIMAX_DEPTH,
        "max_edax_level": MAX_EDAX_LEVEL,
        "az_available": az_path is not None,
        "az_iteration": az_iteration,
        "default_az_sims": DEFAULT_AZ_SIMS,
        "min_az_sims": MIN_AZ_SIMS,
        "max_az_sims": MAX_AZ_SIMS,
        "checkpoints": list_checkpoints() if az_path is not None else [],
        "external_models": list_external_models(),
        "max_arena_workers": MAX_ARENA_WORKERS,
    }


@app.post("/api/tournament")
def tournament_start(req: TourneyReq):
    specs = [s for s in (req.players or []) if (s or "").strip()]
    if len(specs) < 2:
        raise HTTPException(400, "a tournament needs at least 2 participants")
    if len(specs) > MAX_TOURNEY_PLAYERS:
        raise HTTPException(400, f"at most {MAX_TOURNEY_PLAYERS} participants")
    if any((s or "").strip().lower() == "human" for s in specs):
        raise HTTPException(400, "participants must be bots, not human")
    gpm = max(1, min(MAX_TOURNEY_GAMES, int(req.games_per_match)))
    conc = max(1, min(MAX_TOURNEY_CONC, int(req.concurrency)))
    try:                                   # validate every spec up front -> clean 400
        for s in specs:
            _make_factory(s)
    except Exception as exc:
        raise HTTPException(400, str(exc))
    job_id = uuid.uuid4().hex[:12]
    n_matches = len(specs) * (len(specs) - 1) // 2
    _TOURNEYS[job_id] = {"job_id": job_id, "players": specs, "games_per_match": gpm,
                         "concurrency": conc, "n_matches": n_matches, "participants": [],
                         "matches": [], "standings": [], "total_games": n_matches * gpm,
                         "played_games": 0, "done": False, "paused": False,
                         "cancel": False, "error": None}
    _TOURNEY_PRIV[job_id] = {"matches": [], "lock": threading.Lock()}
    _trim_tourneys()
    t = threading.Thread(target=_run_tourney,
                         args=(job_id, specs, gpm, conc, int.from_bytes(os.urandom(4), "little")),
                         daemon=True)
    t.start()
    return _TOURNEYS[job_id]


@app.get("/api/tournament/{job_id}")
def tournament_status(job_id: str):
    job = _TOURNEYS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown tournament (it may have expired)")
    return job


@app.post("/api/tournament/{job_id}/control")
def tournament_control(job_id: str, req: ArenaControl):
    job = _TOURNEYS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown tournament (it may have expired)")
    action = (req.action or "").strip().lower()
    if action == "pause":
        job["paused"] = True
    elif action == "resume":
        job["paused"] = False
    elif action == "stop":
        job["cancel"] = True; job["paused"] = False
    else:
        raise HTTPException(400, "action must be pause, resume, or stop")
    return job


@app.post("/api/checkpoints/delete")
def delete_checkpoint(req: DeleteCkptReq):
    """Remove a checkpoint the user picked — SOFT delete: the .pt is MOVED to
    CKPT_DIR/_trash (recoverable), never hard-deleted. Returns the fresh list."""
    try:
        path = checkpoint_path(req.label)          # validates existence + path safety
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    trash = os.path.join(CKPT_DIR, "_trash")
    os.makedirs(trash, exist_ok=True)
    dest = os.path.join(trash, os.path.basename(path))
    if os.path.exists(dest):                        # don't clobber an earlier trashed copy
        stem, i = os.path.basename(path)[:-3], 1
        while os.path.exists(os.path.join(trash, f"{stem}.{i}.pt")):
            i += 1
        dest = os.path.join(trash, f"{stem}.{i}.pt")
    shutil.move(path, dest)
    _AZ_CACHE.pop(path, None)                        # drop any cached evaluator for it
    return {"deleted": req.label, "trash": os.path.relpath(dest, DATA_DIR),
            "checkpoints": list_checkpoints()}


@app.post("/api/new")
def new_game(req: NewGame):
    try:
        game = Game(req.black, req.white)
    except edax_mod.EdaxNotInstalled as exc:
        raise HTTPException(400, f"Edax not installed: {exc}")
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    game_id = uuid.uuid4().hex[:12]
    _GAMES[game_id] = game
    return {"game_id": game_id, "state": game.state()}


@app.post("/api/move")
def human_move(req: MoveReq):
    game = _get(req.game_id)
    state = game.state()
    if state["done"]:
        raise HTTPException(400, "game is over")
    if not game.is_human_turn():
        raise HTTPException(400, "it is not the human's turn")
    if state["must_pass"]:
        if req.move != PASS:
            raise HTTPException(400, "you must pass (no legal move)")
    elif req.move not in state["legal"]:
        raise HTTPException(400, f"illegal move {req.move}")
    game.apply(req.move)
    return {"state": game.state(), "move": req.move}


@app.post("/api/bot_move")
def bot_move(req: GameReq):
    game = _get(req.game_id)
    if is_terminal(game.board):
        raise HTTPException(400, "game is over")
    if game.is_human_turn():
        raise HTTPException(400, "it is the human's turn")
    move = game.fns[game.player](game.board, game.player)
    game.apply(move)
    return {"state": game.state(), "move": int(move)}


if __name__ == "__main__":
    import uvicorn

    print("Othello web app -> http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
