"""Encoding + perspective-convention tests. Run: python tests/test_encode.py

Guards the two conventions encode.py owns: the 65-wide action space and the
"always the side-to-move's point of view" canonicalisation. If these drift,
value-target sign errors appear far away and are painful to trace — so they are
pinned right at the source.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "engine"))

import numpy as np

from board_numpy import (
    BLACK,
    EMPTY,
    WHITE,
    apply_move,
    initial_board,
    is_terminal,
    legal_move_mask,
    legal_moves,
)
from encode import (
    NUM_PLANES,
    PASS,
    POLICY_SIZE,
    decode,
    encode,
    legal_action_mask,
)

from harness import check, run


def _sample_boards(n=40):
    """A spread of reachable positions from seeded random self-play."""
    rng = np.random.default_rng(7)
    boards = [initial_board()]
    board, player = initial_board(), BLACK
    for _ in range(n * 4):
        if is_terminal(board):
            board, player = initial_board(), BLACK
            continue
        moves = legal_moves(board, player)
        if not moves:
            player = -player
            continue
        board = apply_move(board, player, int(rng.choice(moves)))
        player = -player
        boards.append(board)
        if len(boards) >= n:
            break
    return boards


def test_round_trip():
    for board in _sample_boards():
        for player in (BLACK, WHITE):
            planes = encode(board, player)
            check_shape = planes.shape == (NUM_PLANES, 8, 8) and planes.dtype == np.float32
            assert check_shape, f"bad planes shape/dtype: {planes.shape} {planes.dtype}"
            restored = decode(planes, player)
            assert np.array_equal(restored, board), "decode(encode(board)) != board"
    check("decode(encode(board, p)) == board for both players, many positions", True)


def test_perspective_is_side_to_move():
    board = initial_board()
    # Give the position some asymmetry so the two colours' planes really differ.
    board = apply_move(board, BLACK, legal_moves(board, BLACK)[0])

    from_black = encode(board, BLACK)
    from_white = encode(board, WHITE)

    # Plane 0 is always "my discs": Black's own == White's opponent, and v.v.
    check("own/opponent planes swap when perspective flips",
          np.array_equal(from_black[0], from_white[1])
          and np.array_equal(from_black[1], from_white[0]))
    # Plane 0 (own discs) from Black's POV must be exactly Black's discs.
    check("plane 0 is the mover's own discs",
          np.array_equal(from_black[0], (board == BLACK).astype(np.float32)))
    # Plane 2 is the mover's legal-move mask.
    check("plane 2 is the mover's legal-move mask",
          np.array_equal(from_black[2], legal_move_mask(board, BLACK).astype(np.float32)))
    # Planes are strictly {0, 1}.
    check("planes are binary", set(np.unique(from_black)).issubset({0.0, 1.0}))


def test_legal_action_mask():
    board = initial_board()
    mask = legal_action_mask(board, BLACK)
    check("action mask has length 65", mask.shape == (POLICY_SIZE,))
    legal = set(legal_moves(board, BLACK))
    ok = all((mask[a] == 1.0) == (a in legal) for a in range(64))
    check("action mask marks exactly the legal squares", ok)
    check("PASS is not legal when placements exist", mask[PASS] == 0.0)

    # A must-pass position: all Black but one empty corner -> Black cannot place.
    stuck = np.full((8, 8), BLACK, dtype=np.int8)
    stuck[0, 0] = EMPTY
    stuck_mask = legal_action_mask(stuck, WHITE)  # White also has no placement
    check("PASS is the only legal action when a player is stuck",
          stuck_mask[PASS] == 1.0 and stuck_mask[:PASS].sum() == 0.0)


FAST = [test_round_trip, test_perspective_is_side_to_move, test_legal_action_mask]
SLOW = []

if __name__ == "__main__":
    run(FAST, SLOW, "encode")
