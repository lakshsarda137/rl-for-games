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

Player specs (strings): "human", "random", "greedy", "minimax:D", "edax:L", and the trained
net "az" / "az:<sims>" / "az@<ckpt>" / "az:<sims>@<ckpt>" (ckpt = "latest" or "iterNNNN").

Bot endpoints are plain `def` (not async), so Starlette runs them in a
threadpool — a slow deep-minimax or an Edax subprocess never blocks the event
loop or other requests.
"""

import json
import os
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


def checkpoint_path(label):
    """Resolve a checkpoint label ('latest' | 'iterNNNN') to a file path.

    'latest' (or empty) -> the rolling latest.pt (or newest iter file). A specific
    label like 'iter0007' -> that exact file. Raises ValueError if it doesn't exist,
    so a stale picker selection surfaces a clean 400 instead of a torch stack trace.
    """
    label = (label or "").strip().lower()
    if label in ("", "latest"):
        p = latest_checkpoint()
        if p is None:
            raise ValueError("no trained model available yet")
        return p
    if not (label.startswith("iter") and label[4:].isdigit()):
        raise ValueError(f"bad checkpoint label {label!r} (want 'latest' or 'iterNNNN')")
    p = os.path.join(CKPT_DIR, label + ".pt")
    if not os.path.isfile(p):
        raise ValueError(f"checkpoint {label} not found")
    return p


def list_checkpoints():
    """All loadable checkpoints, newest first, for the web UI's checkpoint picker.

    latest.pt is listed first (its iteration read from the file); the per-iteration
    iterNNNN.pt files follow, sorted by iteration descending. Only latest.pt is
    opened (to report its iteration) — the iter files' numbers come from the name,
    so listing stays cheap no matter how many have accumulated.
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
    iters = []
    if os.path.isdir(CKPT_DIR):
        for name in os.listdir(CKPT_DIR):
            if name.startswith("iter") and name.endswith(".pt"):
                try:
                    it = int(name[4:-3])
                except ValueError:
                    continue
                iters.append({"label": name[:-3], "iteration": it, "is_latest": False})
    iters.sort(key=lambda d: d["iteration"], reverse=True)
    out.extend(iters)
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


# --- routes ------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/dashboard")
def dashboard():
    return FileResponse(os.path.join(FRONTEND_DIR, "dashboard.html"))


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
        "max_arena_workers": MAX_ARENA_WORKERS,
    }


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
