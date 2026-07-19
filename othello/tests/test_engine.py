"""Correctness tests for the reference engine. Run: python tests/test_engine.py

Two anchors pin the rules:
  * perft — count leaf nodes of the game tree to fixed depths and match the
    canonical published Othello values (4, 12, 56, 244, 1396, 8200, 55092,
    390216). Depths 1-8 are convention-independent: the shortest possible
    Othello game is 9 plies, so no pass or early terminal can occur within 8
    plies from the start, and the counts don't depend on how passes are handled.
    Source: the widely-reproduced Othello perft table (matches independent
    engines). Depths 1-6 are the FAST tier; depths 7-8 (~24s) are SLOW (full
    mode only).
  * seeded random self-play — exercises passing and termination on real
    reachable positions, asserting the invariants that are hard to hand-build.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "engine"))

import numpy as np

from board_numpy import (
    BLACK,
    EMPTY,
    PASS,
    WHITE,
    apply_move,
    count_discs,
    disc_diff,
    initial_board,
    is_terminal,
    legal_moves,
    must_pass,
    rc_to_move,
    winner,
)

from harness import check, run

# --- perft -------------------------------------------------------------------
KNOWN_PERFT = {1: 4, 2: 12, 3: 56, 4: 244, 5: 1396, 6: 8200, 7: 55092, 8: 390216}


def perft(board, player, depth):
    if depth == 0:
        return 1
    moves = legal_moves(board, player)
    if not moves:
        # Within depth<=8 from the start this never fires; handled for generality.
        if is_terminal(board):
            return 1
        return perft(board, -player, depth)  # forced pass, side to move switches
    return sum(perft(apply_move(board, player, m), -player, depth - 1) for m in moves)


def test_perft_fast():
    # Depths 1-6 (~0.7s) — convention-independent (no pass within 8 plies).
    board = initial_board()
    for depth in range(1, 7):
        check(f"perft({depth}) == {KNOWN_PERFT[depth]}",
              perft(board, BLACK, depth) == KNOWN_PERFT[depth])


def test_perft_deep():
    # Depths 7-8 (~24s) — the heavy end of the perft anchor. Full mode only.
    board = initial_board()
    for depth in (7, 8):
        check(f"perft({depth}) == {KNOWN_PERFT[depth]}",
              perft(board, BLACK, depth) == KNOWN_PERFT[depth])


# --- opening position --------------------------------------------------------
def test_initial_position():
    board = initial_board()
    black, white = count_discs(board)
    check("initial position has 2 discs each", (black, white) == (2, 2))
    # Black's four opening moves, by classic Othello geometry.
    expected = sorted([rc_to_move(2, 3), rc_to_move(3, 2), rc_to_move(4, 5), rc_to_move(5, 4)])
    check("Black has exactly the 4 known opening moves",
          legal_moves(board, BLACK) == expected)  # [19, 26, 37, 44]
    check("initial position is not terminal", not is_terminal(board))
    check("disc_diff is 0 at the start", disc_diff(board, BLACK) == 0)


def test_flip_is_correct():
    board = initial_board()
    # Black plays (2,3): flips the White disc at (3,3), which is flanked by the
    # existing Black disc at (4,3)... via the vertical run down column 3.
    after = apply_move(board, BLACK, rc_to_move(2, 3))
    check("placed disc is Black", after[2, 3] == BLACK)
    check("flanked White disc at (3,3) flipped to Black", after[3, 3] == BLACK)
    check("un-flanked White disc at (4,4) is untouched", after[4, 4] == WHITE)
    black, white = count_discs(after)
    check("after one flip: Black 4, White 1", (black, white) == (4, 1))
    check("apply_move did not mutate the input board", count_discs(board) == (2, 2))


# --- the terminal rule: "neither can move", NOT "board is full" ---------------
def test_terminal_is_not_board_full():
    # Every non-empty square is Black, with (0,0) left empty. No White disc
    # exists, so neither colour can ever flank anything => no legal moves for
    # either side => terminal, even though the board is not full.
    board = np.full((8, 8), BLACK, dtype=np.int8)
    board[0, 0] = EMPTY
    check("no Black moves on all-black-but-corner", legal_moves(board, BLACK) == [])
    check("no White moves on all-black-but-corner", legal_moves(board, WHITE) == [])
    check("position is terminal despite an empty square", is_terminal(board))
    check("board genuinely still has an empty square", np.any(board == EMPTY))
    check("winner of all-black board is Black", winner(board) == BLACK)


# --- passing and termination over real games ---------------------------------
def test_random_games_invariants():
    rng = np.random.default_rng(20260719)
    n_games = 150  # enough to exercise passing while staying in the fast tier
    pass_events = 0
    finished = 0
    for g in range(n_games):
        board = initial_board()
        player = BLACK
        for _ply in range(200):  # generous cap; real games are <= 60 placements
            if is_terminal(board):
                # Terminal must mean BOTH sides are stuck.
                assert must_pass(board, BLACK) and must_pass(board, WHITE)
                finished += 1
                break
            moves = legal_moves(board, player)
            if not moves:
                # Forced pass: opponent must have a move (else it'd be terminal).
                pass_events += 1
                assert not must_pass(board, -player)
                passed = apply_move(board, player, PASS)
                assert np.array_equal(passed, board), "pass must not change the board"
                board, player = passed, -player
                continue
            board = apply_move(board, player, int(rng.choice(moves)))
            player = -player
        else:
            raise AssertionError(f"game {g} did not terminate within 200 plies")

    check(f"all {n_games} random games terminated", finished == n_games)
    # The suite would be vacuous on passing if no pass ever occurred; confirm it did.
    check(f"random games exercised passing (saw {pass_events} pass events)",
          pass_events > 0)


FAST = [test_perft_fast, test_initial_position, test_flip_is_correct,
        test_terminal_is_not_board_full, test_random_games_invariants]
SLOW = [test_perft_deep]

if __name__ == "__main__":
    run(FAST, SLOW, "engine")
