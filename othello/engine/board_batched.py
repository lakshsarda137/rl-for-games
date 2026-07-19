"""Batched Othello engine — the same rules as `board_numpy`, vectorised over a
leading batch axis so B games' moves are computed with one set of array ops.

`board_numpy` processes ONE 8x8 board per call; at self-play scale (hundreds of
games × ~100 sims × ~60 plies) that per-call Python + tiny-NumPy overhead is the
real throughput wall (the network is only ~13% of the time). This module does the
identical rules on a `[B, 8, 8]` int8 stack, so the per-item Python cost is paid
once for the whole batch, not once per game. `board_numpy` stays the correctness
oracle — every function here is parity-tested against it (`tests/test_batched.py`).

Representation matches `board_numpy`: absolute colours (EMPTY=0, BLACK=+1,
WHITE=-1); `boards` is int8 `[B, 8, 8]`; `players` is int8 `[B]` (+1/-1); a square
action is `row*8+col` (0..63) and PASS is 64 (the encode.py convention). Move
generation and flipping use the standard directional "flood" (shift-and-mask over
the 8 straight-line directions), which is branchless and vectorises cleanly.
"""

import numpy as np

from board_numpy import BLACK, BOARD_N, EMPTY, WHITE

PASS = BOARD_N * BOARD_N   # 64 — the encode.py action-space convention
POLICY_SIZE = PASS + 1     # 65

_DIRS = ((-1, -1), (-1, 0), (-1, 1),
         (0, -1),           (0, 1),
         (1, -1),  (1, 0),  (1, 1))

_MAX_RUN = BOARD_N - 2      # longest flippable run between placement and cap (6)


def _shift(plane, dr, dc):
    """`out[b, r, c] = plane[b, r-dr, c-dc]`, zero-filled past the board edge.

    i.e. slide the plane's contents one step in direction (dr, dc). Works on bool
    or numeric `[B, 8, 8]` planes; no wraparound (unlike np.roll)."""
    out = np.zeros_like(plane)
    rs0, rs1 = max(0, -dr), BOARD_N - max(0, dr)
    cs0, cs1 = max(0, -dc), BOARD_N - max(0, dc)
    rd0, rd1 = max(0, dr), BOARD_N - max(0, -dr)
    cd0, cd1 = max(0, dc), BOARD_N - max(0, -dc)
    out[:, rd0:rd1, cd0:cd1] = plane[:, rs0:rs1, cs0:cs1]
    return out


def initial_boards(batch):
    """`[batch, 8, 8]` int8 stack of the standard opening (Black to move)."""
    boards = np.zeros((batch, BOARD_N, BOARD_N), dtype=np.int8)
    boards[:, 3, 3] = WHITE
    boards[:, 3, 4] = BLACK
    boards[:, 4, 3] = BLACK
    boards[:, 4, 4] = WHITE
    return boards


def _own_opp_empty(boards, players):
    p = np.asarray(players, dtype=np.int8).reshape(-1, 1, 1)
    return boards == p, boards == -p, boards == EMPTY


def legal_move_masks(boards, players):
    """`[B, 8, 8]` bool — squares where each game's `player` may place a disc.

    For each direction: start from own discs, walk over a contiguous opponent run,
    and the empty square just past it is a legal placement (that direction's
    capture). Unioned over all 8 directions."""
    own, opp, empty = _own_opp_empty(boards, players)
    moves = np.zeros(boards.shape, dtype=bool)
    for dr, dc in _DIRS:
        run = _shift(own, dr, dc) & opp          # opp adjacent to own, one step in +d
        for _ in range(_MAX_RUN):
            run = run | (_shift(run, dr, dc) & opp)   # extend along the opp run
        moves |= _shift(run, dr, dc) & empty     # land on the empty just past it
    return moves


def legal_action_masks(boards, players):
    """`[B, 65]` float32 legal-action mask (squares + PASS), matching
    `encode.legal_action_mask`: PASS (64) is legal exactly when no square is."""
    square_mask = legal_move_masks(boards, players)
    b = boards.shape[0]
    out = np.zeros((b, POLICY_SIZE), dtype=np.float32)
    out[:, :PASS] = square_mask.reshape(b, -1)
    out[out[:, :PASS].sum(1) == 0, PASS] = 1.0
    return out


def apply_moves(boards, players, actions):
    """Return a NEW `[B, 8, 8]` stack after each game plays its `actions[b]`.

    `actions[b]` is a square 0..63 or PASS (64). PASS leaves that board unchanged.
    Moves are trusted legal (the self-play caller only ever passes sampled-legal
    actions), mirroring `board_numpy.apply_move` without the debug assert."""
    actions = np.asarray(actions)
    own, opp, _ = _own_opp_empty(boards, players)

    placed = np.zeros(boards.shape, dtype=bool)          # the newly-placed disc
    not_pass = np.where(actions != PASS)[0]
    placed[not_pass, actions[not_pass] // BOARD_N, actions[not_pass] % BOARD_N] = True

    flips = np.zeros(boards.shape, dtype=bool)
    for dr, dc in _DIRS:
        run = _shift(placed, dr, dc) & opp               # opp run from the placement
        frontier = run
        for _ in range(_MAX_RUN):
            frontier = _shift(frontier, dr, dc) & opp
            run = run | frontier
        capped = (_shift(run, dr, dc) & own).any(axis=(1, 2))   # [B]: own caps the run
        flips |= run & capped.reshape(-1, 1, 1)

    colour = np.asarray(players, dtype=np.int8).reshape(-1, 1, 1)
    new = np.where(flips | placed, colour, boards)
    return new.astype(np.int8)


def _has_move(boards, colour):
    return legal_move_masks(boards, np.full(boards.shape[0], colour, np.int8)) \
        .reshape(boards.shape[0], -1).any(1)


def is_terminal(boards):
    """`[B]` bool — True where NEITHER colour has a legal move (not "board full")."""
    return ~(_has_move(boards, BLACK) | _has_move(boards, WHITE))


def count_discs(boards):
    """(`[B]` black counts, `[B]` white counts)."""
    flat = boards.reshape(boards.shape[0], -1)
    return (flat == BLACK).sum(1), (flat == WHITE).sum(1)


def winner(boards):
    """`[B]` int8: BLACK (+1) / WHITE (-1) / 0 draw, by disc count."""
    black, white = count_discs(boards)
    out = np.zeros(boards.shape[0], dtype=np.int8)
    out[black > white] = BLACK
    out[white > black] = WHITE
    return out


def encode_batch(boards, players):
    """`[B, 3, 8, 8]` float32 planes in each game's side-to-move POV, matching
    `encode.encode`: plane 0 = own discs, 1 = opponent discs, 2 = own legal mask."""
    own, opp, _ = _own_opp_empty(boards, players)
    legal = legal_move_masks(boards, players)
    return np.stack([own, opp, legal], axis=1).astype(np.float32)
