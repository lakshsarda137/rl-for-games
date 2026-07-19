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

Player specs (strings): "human", "random", "greedy", "minimax:D", "edax:L".

Bot endpoints are plain `def` (not async), so Starlette runs them in a
threadpool — a slow deep-minimax or an Edax subprocess never blocks the event
loop or other requests.
"""

import json
import os
import sys
import threading
import uuid

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
from simple import greedy_move, random_move

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
    """Turn 'az' or 'az:<sims>' into a move fn driven by the latest trained net."""
    path = latest_checkpoint()
    if path is None:
        raise ValueError("no trained model available yet — run training first "
                         "(python run/train_loop.py --tiny) to create data/checkpoints/")
    sims = DEFAULT_AZ_SIMS
    if ":" in spec:
        try:
            sims = int(spec.split(":", 1)[1])
        except ValueError:
            raise ValueError("az sims must be an integer, e.g. 'az:200'")
        sims = max(MIN_AZ_SIMS, min(MAX_AZ_SIMS, sims))
    evaluator, _ = load_az_evaluator(path)
    from evaluate import az_player
    return az_player(evaluator, sims)


# --- Arena: run N games AZ-vs-opponent for AGGREGATE results ------------------
# In-training eval is off by default, so this is the on-demand strength check:
# a match of N colour-alternated games returns win/draw/loss + win rate, which is
# far more meaningful than a single browser game. Runs in a background thread so
# the UI can poll progress (a many-game match can take minutes).
_ARENAS = {}                       # job_id -> progress/result dict
_ARENA_LOCK = threading.Lock()
MAX_ARENA_GAMES = 200


def _run_arena(job_id, az_spec, opp_spec, n_games, seed):
    job = _ARENAS[job_id]
    try:
        az = build_player(az_spec)            # az / az:<sims>
        opponent = build_player(opp_spec)     # the chosen bot
    except Exception as exc:                   # bad spec, edax missing, etc.
        job.update(error=str(exc), done=True)
        return
    from simple import _random_opening, play_game
    rng = np.random.default_rng(seed)
    wins = draws = losses = 0
    for g in range(n_games):
        if job.get("cancel"):
            break
        opening = _random_opening(rng, 4)      # varied opening, both colours
        if g % 2 == 0:
            result, _, _ = play_game(az, opponent, opening)   # AZ is Black
            az_is = BLACK
        else:
            result, _, _ = play_game(opponent, az, opening)   # AZ is White
            az_is = WHITE
        if result == 0:
            draws += 1
        elif result == az_is:
            wins += 1
        else:
            losses += 1
        job.update(played=g + 1, wins=wins, draws=draws, losses=losses,
                   win_rate=(wins + 0.5 * draws) / (g + 1))
    job["done"] = True


def _trim_arenas():
    """Keep the registry small — drop the oldest finished jobs beyond 20."""
    finished = [k for k, v in _ARENAS.items() if v.get("done")]
    for k in finished[:-20]:
        _ARENAS.pop(k, None)


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
    if spec == "az" or spec.startswith("az:"):
        return _build_az_player(spec)       # the trained AlphaZero net (latest checkpoint)
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
    opponent: str = "minimax:2"   # any bot spec (not human)
    games: int = 20
    az_sims: int = DEFAULT_AZ_SIMS


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
    job_id = uuid.uuid4().hex[:12]
    _ARENAS[job_id] = {"job_id": job_id, "total": n, "played": 0, "wins": 0,
                       "draws": 0, "losses": 0, "win_rate": 0.0, "done": False,
                       "error": None, "opponent": req.opponent, "az_sims": sims}
    _trim_arenas()
    t = threading.Thread(target=_run_arena,
                         args=(job_id, f"az:{sims}", req.opponent, n, len(_ARENAS)),
                         daemon=True)
    t.start()
    return _ARENAS[job_id]


@app.get("/api/arena/{job_id}")
def arena_status(job_id: str):
    job = _ARENAS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown arena job (it may have expired)")
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
