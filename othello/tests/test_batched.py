"""Batched engine + batched MCTS parity tests. Run: python tests/test_batched.py

These guard the array-ops self-play path (the big throughput lever). The trusted
`board_numpy` engine and the serial `mcts.py` are the oracles: the batched
versions must reproduce them exactly. FAST tier checks a few hundred real
positions cheaply; SLOW widens the sweep.
"""

import os
import sys
from types import SimpleNamespace

_HERE = os.path.dirname(__file__)
for _p in ("engine", "opponents", "az", "run"):
    sys.path.insert(0, os.path.join(_HERE, "..", _p))
sys.path.insert(0, _HERE)

import numpy as np
import torch

import board_numpy as bn
import board_batched as bb
import board_torch as bt
from encode import encode, legal_action_mask
from network import Evaluator, OthelloNet
from mcts import MCTS
from mcts_batched import run_batched
from mcts_torch import run_torch

from harness import check, run


def _positions(n_games, seed=0):
    """Play random games with the trusted engine; return [(board, player), ...]."""
    rng = np.random.default_rng(seed)
    pos = []
    for _ in range(n_games):
        board, player = bn.initial_board(), bn.BLACK
        while not bn.is_terminal(board):
            pos.append((board.copy(), player))
            moves = bn.legal_moves(board, player)
            board = bn.apply_move(board, player, bn.PASS if not moves else int(rng.choice(moves)))
            player = -player
        pos.append((board.copy(), player))
    return pos


def _engine_parity(n_games):
    pos = _positions(n_games)
    boards = np.stack([b for b, _ in pos])
    players = np.array([p for _, p in pos], dtype=np.int8)
    rng = np.random.default_rng(1)

    ok_mask = all(np.array_equal(bb.legal_action_masks(boards, players)[i],
                                 legal_action_mask(b, p)) for i, (b, p) in enumerate(pos))
    ok_enc = all(np.array_equal(bb.encode_batch(boards, players)[i], encode(b, p))
                 for i, (b, p) in enumerate(pos))
    bb_term = bb.is_terminal(boards)
    ok_term = all(bool(bb_term[i]) == bn.is_terminal(b) for i, (b, _) in enumerate(pos))
    win = bb.winner(boards)
    ok_win = all(int(win[i]) == bn.winner(b) for i, (b, _) in enumerate(pos))

    chosen = np.array([bn.PASS if not bn.legal_moves(b, p) else int(rng.choice(bn.legal_moves(b, p)))
                       for b, p in pos])
    nxt = bb.apply_moves(boards, players, chosen)
    ok_apply = all(np.array_equal(nxt[i], bn.apply_move(b, p, int(chosen[i])))
                   for i, (b, p) in enumerate(pos))
    return len(pos), (ok_mask, ok_enc, ok_term, ok_win, ok_apply)


def test_batched_engine_parity():
    n, (ok_mask, ok_enc, ok_term, ok_win, ok_apply) = _engine_parity(6)
    check(f"legal_action_masks match board_numpy ({n} positions)", ok_mask)
    check("encode_batch matches encode", ok_enc)
    check("is_terminal matches", ok_term)
    check("winner matches", ok_win)
    check("apply_moves matches apply_move", ok_apply)


def _mcts_parity(n_games, sims):
    torch.manual_seed(0)
    ev = Evaluator(OthelloNet(num_blocks=2, channels=16))
    cfg = SimpleNamespace(c_puct=1.5, dirichlet_alpha=0.3, dirichlet_eps=0.25)
    pos = [(b, p) for b, p in _positions(n_games) if not bn.is_terminal(b)]
    boards = np.stack([b for b, _ in pos])
    players = np.array([p for _, p in pos], dtype=np.int8)

    serial = [MCTS(ev, c_puct=1.5).run(b, int(p), sims, add_noise=False) for b, p in pos]

    def looped(bs, ps):                    # bit-exact single-board eval (== serial uses)
        out = [ev(np.ascontiguousarray(b, np.int8), int(p)) for b, p in zip(bs, ps)]
        return np.stack([o[0] for o in out]).astype(np.float32), \
            np.array([o[1] for o in out], np.float32)

    counts, _ = run_batched(boards, players, sims, looped, cfg, add_noise=False)
    return len(pos), all(np.array_equal(counts[i], serial[i]) for i in range(len(pos)))


def test_batched_mcts_matches_serial():
    """With a bit-exact single-board evaluator, batched MCTS reproduces serial
    MCTS visit counts EXACTLY (same PUCT, expand, backup, tie-breaks)."""
    n, ok = _mcts_parity(3, sims=16)
    check(f"batched MCTS visit counts == serial, exactly ({n} positions)", ok)


# --- torch port parity (device-agnostic; runs on CPU tensors here) -----------
# board_torch / mcts_torch are the op-for-op torch re-expression of the NumPy
# batched engine + MCTS, so the SEARCH can run on the GPU (not just the network).
# They are checked against the SAME oracles (board_numpy, serial mcts.py). The
# engine ops are integer/bool so they are bit-exact; the MCTS float32 math turns
# out bit-exact vs NumPy on CPU too (verified below), so the tie-break argmax and
# every visit count match the serial oracle exactly.

def _torch_engine_parity(n_games):
    pos = _positions(n_games)
    boards = torch.from_numpy(np.stack([b for b, _ in pos]))
    players = torch.from_numpy(np.array([p for _, p in pos], dtype=np.int8))
    rng = np.random.default_rng(1)

    tmask = bt.legal_action_masks(boards, players).cpu().numpy()
    ok_mask = all(np.array_equal(tmask[i], legal_action_mask(b, p)) for i, (b, p) in enumerate(pos))
    tenc = bt.encode_batch(boards, players).cpu().numpy()
    ok_enc = all(np.array_equal(tenc[i], encode(b, p)) for i, (b, p) in enumerate(pos))
    tterm = bt.is_terminal(boards).cpu().numpy()
    ok_term = all(bool(tterm[i]) == bn.is_terminal(b) for i, (b, _) in enumerate(pos))
    twin = bt.winner(boards).cpu().numpy()
    ok_win = all(int(twin[i]) == bn.winner(b) for i, (b, _) in enumerate(pos))
    chosen = np.array([bn.PASS if not bn.legal_moves(b, p) else int(rng.choice(bn.legal_moves(b, p)))
                       for b, p in pos])
    tnxt = bt.apply_moves(boards, players, torch.from_numpy(chosen)).cpu().numpy()
    ok_apply = all(np.array_equal(tnxt[i], bn.apply_move(b, p, int(chosen[i])))
                   for i, (b, p) in enumerate(pos))
    return len(pos), (ok_mask, ok_enc, ok_term, ok_win, ok_apply)


def _torch_mcts_parity(n_games, sims):
    torch.manual_seed(0)
    ev = Evaluator(OthelloNet(num_blocks=2, channels=16))
    cfg = SimpleNamespace(c_puct=1.5, dirichlet_alpha=0.3, dirichlet_eps=0.25)
    pos = [(b, p) for b, p in _positions(n_games) if not bn.is_terminal(b)]
    boards = torch.from_numpy(np.stack([b for b, _ in pos]))
    players = torch.from_numpy(np.array([p for _, p in pos], dtype=np.int8))

    serial = [MCTS(ev, c_puct=1.5).run(b, int(p), sims, add_noise=False) for b, p in pos]

    def looped(bs, ps):                    # bit-exact single-board eval (torch in/out)
        bs_np, ps_np = bs.cpu().numpy(), ps.cpu().numpy()
        out = [ev(np.ascontiguousarray(b, np.int8), int(p)) for b, p in zip(bs_np, ps_np)]
        return (torch.from_numpy(np.stack([o[0] for o in out]).astype(np.float32)),
                torch.from_numpy(np.array([o[1] for o in out], np.float32)))

    counts, _ = run_torch(boards, players, sims, looped, cfg, add_noise=False)
    counts = counts.cpu().numpy()
    return len(pos), all(np.array_equal(counts[i], serial[i]) for i in range(len(pos)))


def test_torch_engine_parity():
    n, oks = _torch_engine_parity(6)
    check(f"torch engine matches board_numpy exactly ({n} positions)", all(oks))


def test_torch_mcts_matches_serial():
    """The torch MCTS (which can run the whole search on the GPU) reproduces serial
    MCTS visit counts EXACTLY on CPU tensors — same guarantee as the NumPy batched
    version, so the GPU search plays the identical game."""
    n, ok = _torch_mcts_parity(2, sims=12)
    check(f"torch MCTS visit counts == serial, exactly ({n} positions)", ok)


# --- wider sweeps (slow) -----------------------------------------------------
def test_batched_engine_parity_wide():
    n, oks = _engine_parity(40)
    check(f"engine parity holds over a wide sweep ({n} positions)", all(oks))


def test_batched_mcts_matches_serial_wide():
    n, ok = _mcts_parity(8, sims=48)
    check(f"batched MCTS == serial over a wide sweep ({n} positions)", ok)


def test_torch_engine_parity_wide():
    n, oks = _torch_engine_parity(40)
    check(f"torch engine parity holds over a wide sweep ({n} positions)", all(oks))


def test_torch_mcts_matches_serial_wide():
    n, ok = _torch_mcts_parity(8, sims=48)
    check(f"torch MCTS == serial over a wide sweep ({n} positions)", ok)


FAST = [test_batched_engine_parity, test_batched_mcts_matches_serial,
        test_torch_engine_parity, test_torch_mcts_matches_serial]
SLOW = [test_batched_engine_parity_wide, test_batched_mcts_matches_serial_wide,
        test_torch_engine_parity_wide, test_torch_mcts_matches_serial_wide]

if __name__ == "__main__":
    run(FAST, SLOW, "batched")
