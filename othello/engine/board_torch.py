"""Torch port of the batched Othello engine — identical rules to `board_batched`,
but every op runs on **torch tensors** instead of NumPy, so the whole batch of
games can live and move on the GPU (`device="cuda"`) instead of the CPU.

Why this exists (see othello/CLAUDE.md next-steps item 3): `board_batched` is
NumPy = CPU. Batching over games removed the per-game Python overhead, but the
game-rules math itself is still stuck on one CPU core (the current throughput
wall once the network is on the GPU). This module is the op-for-op re-expression
of that same math in torch, so the rules run wherever the tensors live — CPU on a
Mac (for building + correctness tests), CUDA on Kaggle (for speed). It is NOT
PyCUDA: it's plain device-agnostic torch.

The representation matches `board_batched` / `board_numpy` exactly: absolute
colours (EMPTY=0, BLACK=+1, WHITE=-1); `boards` is int8 `[B, 8, 8]`; `players` is
int8 `[B]` (+1/-1); a square action is `row*8+col` (0..63) and PASS is 64. Because
the ops are all integer/bool (compare, shift-by-slice, &/|, sums), the results are
**bit-exact** with the NumPy engine — `board_numpy` stays the correctness oracle
and every function here is parity-tested against it (`tests/test_batched.py`).

Every function is device-agnostic: it creates its scratch tensors on the *input's*
device and returns tensors on that same device, so passing CUDA boards keeps the
whole computation on the GPU with no host round-trips.
"""

import torch

from board_numpy import BLACK, BOARD_N, EMPTY, WHITE

PASS = BOARD_N * BOARD_N   # 64 — the encode.py action-space convention
POLICY_SIZE = PASS + 1     # 65

_DIRS = ((-1, -1), (-1, 0), (-1, 1),
         (0, -1),           (0, 1),
         (1, -1),  (1, 0),  (1, 1))

_MAX_RUN = BOARD_N - 2      # longest flippable run between placement and cap (6)


def _shift(plane, dr, dc):
    """`out[b, r, c] = plane[b, r-dr, c-dc]`, zero-filled past the board edge.

    Slides the plane's contents one step in direction (dr, dc). Works on bool or
    numeric `[B, 8, 8]` tensors; no wraparound. Identical slice arithmetic to the
    NumPy `board_batched._shift` (torch supports the same positive-step slicing)."""
    out = torch.zeros_like(plane)
    rs0, rs1 = max(0, -dr), BOARD_N - max(0, dr)
    cs0, cs1 = max(0, -dc), BOARD_N - max(0, dc)
    rd0, rd1 = max(0, dr), BOARD_N - max(0, -dr)
    cd0, cd1 = max(0, dc), BOARD_N - max(0, -dc)
    out[:, rd0:rd1, cd0:cd1] = plane[:, rs0:rs1, cs0:cs1]
    return out


def initial_boards(batch, device="cpu"):
    """`[batch, 8, 8]` int8 stack of the standard opening (Black to move)."""
    boards = torch.zeros((batch, BOARD_N, BOARD_N), dtype=torch.int8, device=device)
    boards[:, 3, 3] = WHITE
    boards[:, 3, 4] = BLACK
    boards[:, 4, 3] = BLACK
    boards[:, 4, 4] = WHITE
    return boards


def _own_opp_empty(boards, players):
    p = players.to(torch.int8).reshape(-1, 1, 1)
    return boards == p, boards == -p, boards == EMPTY


def legal_move_masks(boards, players):
    """`[B, 8, 8]` bool — squares where each game's `player` may place a disc.

    For each direction: start from own discs, walk over a contiguous opponent run,
    and the empty square just past it is a legal placement. Unioned over all 8
    directions (the standard directional flood)."""
    own, opp, empty = _own_opp_empty(boards, players)
    moves = torch.zeros_like(own)
    for dr, dc in _DIRS:
        run = _shift(own, dr, dc) & opp           # opp adjacent to own, one step in +d
        for _ in range(_MAX_RUN):
            run = run | (_shift(run, dr, dc) & opp)    # extend along the opp run
        moves |= _shift(run, dr, dc) & empty      # land on the empty just past it
    return moves


def legal_action_masks(boards, players):
    """`[B, 65]` float32 legal-action mask (squares + PASS), matching
    `encode.legal_action_mask`: PASS (64) is legal exactly when no square is."""
    square_mask = legal_move_masks(boards, players)
    b = boards.shape[0]
    out = torch.zeros((b, POLICY_SIZE), dtype=torch.float32, device=boards.device)
    out[:, :PASS] = square_mask.reshape(b, -1).to(torch.float32)
    out[out[:, :PASS].sum(1) == 0, PASS] = 1.0
    return out


def apply_moves(boards, players, actions):
    """Return a NEW `[B, 8, 8]` stack after each game plays its `actions[b]`.

    `actions[b]` is a square 0..63 or PASS (64). PASS leaves that board unchanged.
    Moves are trusted legal (the self-play caller only passes sampled-legal
    actions), mirroring `board_batched.apply_moves`."""
    actions = actions.to(torch.long)
    own, opp, _ = _own_opp_empty(boards, players)

    placed = torch.zeros(boards.shape, dtype=torch.bool, device=boards.device)
    not_pass = torch.nonzero(actions != PASS, as_tuple=True)[0]
    placed[not_pass, actions[not_pass] // BOARD_N, actions[not_pass] % BOARD_N] = True

    flips = torch.zeros(boards.shape, dtype=torch.bool, device=boards.device)
    for dr, dc in _DIRS:
        run = _shift(placed, dr, dc) & opp        # opp run from the placement
        frontier = run
        for _ in range(_MAX_RUN):
            frontier = _shift(frontier, dr, dc) & opp
            run = run | frontier
        capped = (_shift(run, dr, dc) & own).reshape(boards.shape[0], -1).any(1)  # [B]
        flips |= run & capped.reshape(-1, 1, 1)

    colour = players.to(torch.int8).reshape(-1, 1, 1)
    new = torch.where(flips | placed, colour, boards)
    return new.to(torch.int8)


def _has_move(boards, colour):
    players = torch.full((boards.shape[0],), colour, dtype=torch.int8, device=boards.device)
    return legal_move_masks(boards, players).reshape(boards.shape[0], -1).any(1)


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
    out = torch.zeros(boards.shape[0], dtype=torch.int8, device=boards.device)
    out[black > white] = BLACK
    out[white > black] = WHITE
    return out


def encode_batch(boards, players):
    """`[B, 3, 8, 8]` float32 planes in each game's side-to-move POV, matching
    `encode.encode`: plane 0 = own discs, 1 = opponent discs, 2 = own legal mask."""
    own, opp, _ = _own_opp_empty(boards, players)
    legal = legal_move_masks(boards, players)
    return torch.stack([own, opp, legal], dim=1).to(torch.float32)
