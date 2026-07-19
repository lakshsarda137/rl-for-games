"""Tunable-depth alpha-beta minimax — the whole benchmark ladder in one file.

`depth` is the single difficulty knob (§2 of the plan): the ladder from "depth 2"
to "depth 8" is this exact code with one argument changed, so the heuristic stays
fixed and the difficulty stays reproducible.

Implemented as negamax (one player's perspective, sign-flipped each ply) with
alpha-beta pruning and move ordering. Pruning efficiency depends heavily on the
ordering, so candidate moves are sorted by the positional weight table — corners
first, X/C-squares last — which is cheap and effective.

Leaf value:
  * at a true terminal, a large ± value proportional to the final disc margin, so
    real wins/losses dominate any heuristic noise;
  * otherwise the 4-component heuristic (§6.2).
"""

import math
import os
import sys

_HERE = os.path.dirname(__file__)
_ENGINE = os.path.join(_HERE, "..", "engine")
for _p in (_ENGINE, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from board_numpy import (
    PASS,
    apply_move,
    disc_diff,
    is_terminal,
    legal_moves,
    move_to_rc,
)
from encode import PASS as ENCODE_PASS  # keep the two PASS constants in agreement
from heuristic import WEIGHT_MATRIX, heuristic_score

assert PASS == ENCODE_PASS, "engine and encode disagree on the PASS index"

# Terminal payoff base: larger than the maximum reachable heuristic magnitude
# (sum of weights * 100 ~= 8500), so any decided game outranks any heuristic score.
_WIN_BASE = 100_000.0


def _terminal_value(board, player):
    """Signed terminal payoff from `player`'s view: sign(margin) * (base + |margin|)."""
    margin = disc_diff(board, player)
    if margin == 0:
        return 0.0
    return math.copysign(_WIN_BASE + abs(margin), margin)


def _order_moves(moves):
    """Order squares by the positional weight table, descending (corners first)."""
    return sorted(moves, key=lambda m: -WEIGHT_MATRIX[move_to_rc(m)])


def _negamax(board, player, depth, alpha, beta, heuristic):
    if is_terminal(board):
        return _terminal_value(board, player)
    if depth == 0:
        return heuristic(board, player)

    moves = legal_moves(board, player)
    if not moves:
        # Forced pass: hand the turn over (not terminal, since is_terminal was False).
        return -_negamax(board, -player, depth - 1, -beta, -alpha, heuristic)

    value = -math.inf
    for move in _order_moves(moves):
        child = apply_move(board, player, move)
        score = -_negamax(child, -player, depth - 1, -beta, -alpha, heuristic)
        if score > value:
            value = score
        if value > alpha:
            alpha = value
        if alpha >= beta:
            break  # this branch can't improve the caller's result
    return value


def minimax_move(board, player, depth, heuristic=heuristic_score):
    """Best move for `player` at `depth` (a square 0..63, or PASS if stuck).

    `depth` is the difficulty dial; `heuristic` is fixed (defaults to the
    4-component evaluation). Deterministic given the inputs.
    """
    moves = legal_moves(board, player)
    if not moves:
        return PASS

    best_move, best_value = None, -math.inf
    alpha, beta = -math.inf, math.inf
    for move in _order_moves(moves):
        child = apply_move(board, player, move)
        value = -_negamax(child, -player, depth - 1, -beta, -alpha, heuristic)
        if value > best_value:
            best_value, best_move = value, move
        if value > alpha:
            alpha = value
    return best_move


def minimax_player(depth, heuristic=heuristic_score):
    """Convenience: bind depth/heuristic into a `(board, player) -> move` function.

    Handy for the match/eval harness and the terminal viewer, which take players
    as uniform move-functions.
    """
    return lambda board, player: minimax_move(board, player, depth, heuristic)
