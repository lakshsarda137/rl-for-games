"""FastAPI backend for the Othello web app — play any bot, or watch two bots.

A thin, single-user local server. It holds games in memory and exposes a tiny
JSON API; the browser (serve/frontend/index.html) renders the board and drives
the flow. The engine + opponents modules are the source of truth for rules and
moves — this file only wires them to HTTP.

Endpoints:
  GET  /                     -> the single-page frontend
  GET  /api/config           -> capabilities (is Edax installed, limits)
  POST /api/new {black,white}-> start a game, return its id + state
  POST /api/move {id, move}  -> apply a HUMAN move (a square 0-63, or 64=pass)
  POST /api/bot_move {id}    -> let the side-to-move BOT choose + play one move

Player specs (strings): "human", "random", "greedy", "minimax:D", "edax:L".

Bot endpoints are plain `def` (not async), so Starlette runs them in a
threadpool — a slow deep-minimax or an Edax subprocess never blocks the event
loop or other requests.
"""

import os
import sys
import uuid

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "engine"))
sys.path.insert(0, os.path.join(_HERE, "..", "opponents"))

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
MAX_MINIMAX_DEPTH = 8
MAX_EDAX_LEVEL = 30

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


# --- routes ------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/api/config")
def config():
    return {
        "edax_available": edax_mod.is_available(),
        "max_minimax_depth": MAX_MINIMAX_DEPTH,
        "max_edax_level": MAX_EDAX_LEVEL,
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
