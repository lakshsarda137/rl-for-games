"""The 4-component linear heuristic — the fixed yardstick minimax searches with.

This is the strong-but-simple evaluation from the Sannidhanam & Annamalai
analysis (concrete formulas per Kukreja's widely-reproduced version). Each
component is normalised to [-100, 100] as `100 * (mine - theirs) / (mine + theirs)`
(0 when the denominator is 0), then combined with fixed weights. Keeping the
weights FIXED is what makes the benchmark ladder reproducible — do not tune them
per training run.

Components (all from the given player's perspective):
  1. coin parity  — disc-count difference (matters most near the end).
  2. mobility     — legal-move-count difference (more options = better).
  3. corners      — the 4 corners, which can never be flipped.
  4. stability    — discs that can never be flipped again; their analysis found
                    this the single most valuable component.

Also exposes a simpler weighted-piece-counter opponent (fixed 8x8 weight table)
for very-early sanity checks, and that table doubles as cheap move ordering for
alpha-beta (corners rank first, X/C-squares last).
"""

import os
import sys

_ENGINE = os.path.join(os.path.dirname(__file__), "..", "engine")
if _ENGINE not in sys.path:
    sys.path.insert(0, _ENGINE)

import numpy as np

from board_numpy import BLACK, EMPTY, WHITE, count_discs, legal_moves

BOARD_N = 8
_CORNERS = ((0, 0), (0, 7), (7, 0), (7, 7))
_AXES = ((0, 1), (1, 0), (1, 1), (1, -1))  # horizontal, vertical, both diagonals

# Default component weights (§6.2 of the plan). Sane starting point; frozen once
# chosen so the benchmark stays reproducible.
DEFAULT_WEIGHTS = {"parity": 25.0, "mobility": 5.0, "corners": 30.0, "stability": 25.0}

# Classic positional weight table: corners huge, corner-adjacent (X/C) squares
# strongly negative. Used by the weighted-piece opponent and for move ordering.
WEIGHT_MATRIX = np.array([
    [100, -20, 10,  5,  5, 10, -20, 100],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [ 10,  -2, -1, -1, -1, -1,  -2,  10],
    [  5,  -2, -1, -1, -1, -1,  -2,   5],
    [  5,  -2, -1, -1, -1, -1,  -2,   5],
    [ 10,  -2, -1, -1, -1, -1,  -2,  10],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [100, -20, 10,  5,  5, 10, -20, 100],
], dtype=np.float64)


def _normalized(mine, theirs):
    """100 * (mine - theirs) / (mine + theirs), or 0 when both are 0."""
    denom = mine + theirs
    if denom == 0:
        return 0.0
    return 100.0 * (mine - theirs) / denom


# --- stability ---------------------------------------------------------------
def _line_full(board, r, c, dr, dc):
    """True if the whole line through (r, c) along axis (dr, dc) has no empties."""
    for sign in (1, -1):
        nr, nc = r, c
        while 0 <= nr < BOARD_N and 0 <= nc < BOARD_N:
            if board[nr, nc] == EMPTY:
                return False
            nr += sign * dr
            nc += sign * dc
    return True


def _cell_stable(board, stable, r, c, player):
    """Sound (under-approximating) stability test for one occupied cell.

    A disc is stable iff it is stable along ALL four axes. Along an axis it is
    stable if either neighbour is off the board (it can't be the middle of a
    flip on that axis) or is a same-colour already-stable disc, OR the entire
    line on that axis is full (no empty square can ever drive a flip through it).
    Under-approximates true stability, which is exactly what a heuristic wants.
    """
    for dr, dc in _AXES:
        axis_ok = False
        for sign in (1, -1):
            nr, nc = r + sign * dr, c + sign * dc
            if not (0 <= nr < BOARD_N and 0 <= nc < BOARD_N):
                axis_ok = True  # edge on this side
                break
            if board[nr, nc] == player and stable[nr, nc]:
                axis_ok = True
                break
        if not axis_ok and _line_full(board, r, c, dr, dc):
            axis_ok = True
        if not axis_ok:
            return False
    return True


def stable_discs(board):
    """Boolean (8, 8) grid of stable discs, grown from the corners to a fixpoint."""
    occupied = board != EMPTY
    stable = np.zeros((BOARD_N, BOARD_N), dtype=bool)
    changed = True
    while changed:
        changed = False
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                if not occupied[r, c] or stable[r, c]:
                    continue
                if _cell_stable(board, stable, r, c, board[r, c]):
                    stable[r, c] = True
                    changed = True
    return stable


# --- individual components (each returns a value in [-100, 100]) --------------
def coin_parity(board, player):
    black, white = count_discs(board)
    mine, theirs = (black, white) if player == BLACK else (white, black)
    return _normalized(mine, theirs)


def mobility(board, player):
    return _normalized(len(legal_moves(board, player)), len(legal_moves(board, -player)))


def corner_capture(board, player):
    mine = sum(1 for r, c in _CORNERS if board[r, c] == player)
    theirs = sum(1 for r, c in _CORNERS if board[r, c] == -player)
    return _normalized(mine, theirs)


def stability(board, player):
    stable = stable_discs(board)
    mine = int(np.sum(stable & (board == player)))
    theirs = int(np.sum(stable & (board == -player)))
    return _normalized(mine, theirs)


# --- combined heuristic ------------------------------------------------------
def heuristic_score(board, player, weights=None):
    """Weighted sum of the 4 components, from `player`'s perspective."""
    w = DEFAULT_WEIGHTS if weights is None else weights
    return (
        w["parity"] * coin_parity(board, player)
        + w["mobility"] * mobility(board, player)
        + w["corners"] * corner_capture(board, player)
        + w["stability"] * stability(board, player)
    )


def weighted_piece_score(board, player):
    """Simpler/faster opponent: sum of the positional weight table over discs.

    From `player`'s perspective (own squares add, opponent squares subtract).
    """
    return float(np.sum(WEIGHT_MATRIX * (board == player)) - np.sum(WEIGHT_MATRIX * (board == -player)))
