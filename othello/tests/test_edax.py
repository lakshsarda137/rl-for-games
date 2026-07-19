"""Edax wrapper tests. Run: python tests/test_edax.py

Edax is an OPTIONAL external engine, so this suite skips cleanly when it isn't
installed (no binary / no eval.dat) instead of failing. When it is present, the
key check is orientation: every move Edax returns must be legal in OUR board
across many random positions — that's what catches a coordinate transpose bug
the symmetric opening can't reveal.
"""

import os
import sys
import time

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "engine"))
sys.path.insert(0, os.path.join(_HERE, "..", "opponents"))

import numpy as np

from board_numpy import BLACK, PASS, apply_move, initial_board, is_terminal, legal_moves
from edax import EdaxEngine, edax_player, is_available
from minimax import minimax_player
from simple import play_match

from harness import check, run

AVAILABLE = is_available()


def _skip(name):
    print(f"SKIP  {name} (Edax not installed — optional component)")


def test_opening_move_legal():
    if not AVAILABLE:
        return _skip("test_opening_move_legal")
    mv = EdaxEngine(level=4).move(initial_board(), BLACK)
    check("Edax returns a legal opening move", mv in legal_moves(initial_board(), BLACK))


def test_orientation_legality():
    """Every Edax move must be legal in our board across random midgame positions."""
    if not AVAILABLE:
        return _skip("test_orientation_legality")
    eng = EdaxEngine(level=2)
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(15):
        board, p = initial_board(), BLACK
        for _ in range(int(rng.integers(4, 40))):
            if is_terminal(board):
                break
            lm = legal_moves(board, p)
            if not lm:
                p = -p
                continue
            board = apply_move(board, p, int(rng.choice(lm)))
            p = -p
        if is_terminal(board):
            continue
        mv, lm = eng.move(board, p), legal_moves(board, p)
        assert (mv == PASS and not lm) or (mv in lm), f"Edax move {mv} not legal in {lm}"
        checked += 1
    check(f"Edax returned a legal move in all {checked} random positions", checked > 0)


def test_edax_beats_our_minimax():
    if not AVAILABLE:
        return _skip("test_edax_beats_our_minimax")
    # Even a low Edax level should beat our shallow minimax (it's the ceiling).
    r = play_match(edax_player(3), minimax_player(1), n_games=4, opening_plies=2, seed=7)
    check(f"Edax L3 beats minimax d1 (win_rate={r['win_rate']:.2f})", r["win_rate"] >= 0.6)


FAST = [test_opening_move_legal, test_orientation_legality]
SLOW = [test_edax_beats_our_minimax]

if __name__ == "__main__":
    if not AVAILABLE:
        print("Edax not installed — all Edax tests skipped (this is fine; Edax is optional).")
    run(FAST, SLOW, "edax")
