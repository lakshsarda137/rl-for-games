"""Dihedral symmetry tests. Run: python tests/test_symmetry.py

Two things must hold before MCTS / augmentation rely on this:
  * apply-then-invert is the identity, for boards AND 65-length policies, with a
    NONZERO pass slot (index 64 must survive untouched — the classic off-by-one).
  * a board and its policy transform *together*: the disc at square s and the
    move-probability on square s land on the same new square.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "engine"))

import numpy as np

from encode import PASS, POLICY_SIZE
from symmetry import (
    NUM_SYMMETRIES,
    inverse_index,
    transform_board,
    transform_policy,
)

from harness import check, run


def test_board_round_trip():
    rng = np.random.default_rng(1)
    board = rng.integers(-1, 2, size=(8, 8)).astype(np.int8)
    for i in range(NUM_SYMMETRIES):
        restored = transform_board(transform_board(board, i), inverse_index(i))
        assert np.array_equal(restored, board), f"board symmetry {i} not invertible"
    check("board: apply-then-invert == identity for all 8 symmetries", True)


def test_policy_round_trip_with_nonzero_pass():
    rng = np.random.default_rng(2)
    policy = rng.random(POLICY_SIZE).astype(np.float32)
    policy[PASS] = 0.37  # deliberately nonzero pass mass
    for i in range(NUM_SYMMETRIES):
        transformed = transform_policy(policy, i)
        check_pass = transformed[PASS] == policy[PASS]
        assert check_pass, f"symmetry {i} moved the PASS slot"
        restored = transform_policy(transformed, inverse_index(i))
        assert np.allclose(restored, policy), f"policy symmetry {i} not invertible"
    check("policy: PASS slot fixed and apply-then-invert == identity", True)


def test_board_and_policy_move_together():
    # Put a unique marker on one square of the board and all the policy mass on
    # that same square; after any symmetry, the disc's new location must equal
    # the policy's argmax square.
    for square in (0, 1, 8, 19, 63):
        board = np.zeros((8, 8), dtype=np.int8)
        board[square // 8, square % 8] = 1

        policy = np.zeros(POLICY_SIZE, dtype=np.float32)
        policy[square] = 1.0

        for i in range(NUM_SYMMETRIES):
            tb = transform_board(board, i)
            tp = transform_policy(policy, i)
            (rr, cc) = np.argwhere(tb == 1)[0]
            board_square = rr * 8 + cc
            policy_square = int(np.argmax(tp[:PASS]))
            assert board_square == policy_square, (
                f"symmetry {i}: disc at {board_square} but policy at {policy_square}")
    check("board disc and policy mass land on the same square under all symmetries", True)


def test_symmetries_are_distinct():
    # On an asymmetric board the 8 transforms should give 8 distinct results.
    board = np.arange(64, dtype=np.int8).reshape(8, 8)
    seen = {transform_board(board, i).tobytes() for i in range(NUM_SYMMETRIES)}
    check("the 8 symmetries are distinct on an asymmetric board", len(seen) == 8)


FAST = [test_board_round_trip, test_policy_round_trip_with_nonzero_pass,
        test_board_and_policy_move_together, test_symmetries_are_distinct]
SLOW = []

if __name__ == "__main__":
    run(FAST, SLOW, "symmetry")
