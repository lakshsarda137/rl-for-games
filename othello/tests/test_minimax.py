"""Heuristic + minimax tests. Run: python tests/test_minimax.py

Fast unit checks pin the heuristic components and stability; the slower strength
matches confirm the difficulty knob actually works (deeper minimax is stronger,
and minimax beats the rung-0 baselines). Strength thresholds sit well below the
empirically measured win rates (~0.75-0.79) so the tests are not flaky.
"""

import os
import sys

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "engine"))
sys.path.insert(0, os.path.join(_HERE, "..", "opponents"))

import numpy as np

from board_numpy import BLACK, EMPTY, WHITE, initial_board, legal_moves
from encode import PASS
from heuristic import (
    coin_parity,
    corner_capture,
    heuristic_score,
    mobility,
    stability,
    stable_discs,
    weighted_piece_score,
)
from minimax import _terminal_value, minimax_move, minimax_player
from simple import greedy_move, play_match, random_move

from harness import check, run


# --- heuristic components (instant) ------------------------------------------
def test_components():
    board = initial_board()
    check("mobility is 0 at the symmetric start", mobility(board, BLACK) == 0)
    check("parity is 0 at the symmetric start", coin_parity(board, BLACK) == 0)
    check("no corners owned at the start", corner_capture(board, BLACK) == 0)

    # More discs -> positive parity, and it is exactly antisymmetric in colour.
    heavy = np.zeros((8, 8), dtype=np.int8)
    heavy[0, :3] = BLACK
    heavy[7, 0] = WHITE
    check("parity favours the side with more discs", coin_parity(heavy, BLACK) > 0)
    check("parity is antisymmetric in perspective",
          coin_parity(heavy, BLACK) == -coin_parity(heavy, WHITE))

    # Owning a corner -> positive corner component and positive weighted score.
    corner = np.zeros((8, 8), dtype=np.int8)
    corner[0, 0] = BLACK
    check("owning a corner scores positive corner capture", corner_capture(corner, BLACK) == 100.0)
    check("weighted-piece score rewards a corner", weighted_piece_score(corner, BLACK) == 100.0)


def _only(r, c, color=BLACK):
    board = np.zeros((8, 8), dtype=np.int8)
    board[r, c] = color
    return board


def test_stability():
    check("no disc is stable at the start", not stable_discs(initial_board()).any())

    corner_stable = stable_discs(_only(0, 0))
    check("an isolated corner disc is stable", corner_stable[0, 0])
    check("only that corner is stable (nothing else)", corner_stable.sum() == 1)

    full_black = np.full((8, 8), BLACK, dtype=np.int8)
    check("a completely full board is entirely stable", stable_discs(full_black).all())
    check("owning stable corners gives positive stability", stability(_only(0, 0), BLACK) == 100.0)


# --- leaf value ordering -----------------------------------------------------
def test_terminal_dominates_heuristic():
    # all-black-but-corner is terminal with Black ~63 ahead; its payoff must
    # outrank any reachable heuristic magnitude (~8500 = sum of weights * 100).
    board = np.full((8, 8), BLACK, dtype=np.int8)
    board[0, 0] = EMPTY
    check("a won terminal outranks any heuristic score", _terminal_value(board, BLACK) > 8500)
    check("terminal payoff is sign-correct for the loser", _terminal_value(board, WHITE) < -8500)


# --- move selection ----------------------------------------------------------
def test_move_selection():
    board = initial_board()
    move = minimax_move(board, BLACK, depth=2)
    check("minimax returns a legal move", move in legal_moves(board, BLACK))

    stuck = np.full((8, 8), BLACK, dtype=np.int8)
    stuck[0, 0] = EMPTY  # White has no move here
    check("minimax passes when it has no legal move", minimax_move(stuck, WHITE, depth=2) == PASS)


# --- the difficulty knob (slower; thresholds below measured ~0.75-0.79) ------
def test_minimax_beats_random():
    r = play_match(minimax_player(1), random_move, n_games=12, opening_plies=2, seed=1)
    check(f"minimax d1 beats random (win_rate={r['win_rate']:.2f})", r["win_rate"] >= 0.65)


def test_deeper_beats_shallower():
    # The core knob claim: same code, +1 depth => stronger play.
    r = play_match(minimax_player(2), minimax_player(1), n_games=10, opening_plies=2, seed=2)
    check(f"minimax d2 beats d1 (win_rate={r['win_rate']:.2f})", r["win_rate"] >= 0.6)


# Fast: instant unit checks + one light strength match (d1 vs random, ~3s).
# Slow: the deeper strength match (d2 vs d1, ~15s of pure-Python search).
FAST = [test_components, test_stability, test_terminal_dominates_heuristic,
        test_move_selection, test_minimax_beats_random]
SLOW = [test_deeper_beats_shallower]

if __name__ == "__main__":
    run(FAST, SLOW, "minimax")
