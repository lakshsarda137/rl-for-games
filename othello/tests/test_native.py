"""C++ native port parity tests. Run: python tests/test_native.py

`native/othello_native.cpp` re-implements the batched engine (bitboards) and the
batched PUCT search (scalar loops) in C++. These tests pin it to the NumPy
implementations it replaced — not "close", but BIT-IDENTICAL, because that is
what lets the C++ path inherit every guarantee the NumPy path already proved
(`tests/test_batched.py` pins NumPy to the single-board `board_numpy` and the
serial `mcts.py`; if C++ == NumPy exactly, C++ == those oracles exactly too).

The float32 arithmetic matches on purpose, not by luck — see the parity notes in
the C++ module header. The two reductions whose result would depend on summation
order (the Dirichlet normaliser and root_value) are deliberately left in NumPy on
both paths, so there is nothing left that could legitimately differ.

If the extension isn't built these tests SKIP rather than fail: it is an optional
accelerator, and a machine without it is a supported configuration.
"""

import contextlib
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
import native
import selfplay
from config import Config
from mcts_batched import _run_batched_numpy, run_batched
from network import Evaluator, OthelloNet

from harness import check, run

CFG = SimpleNamespace(c_puct=1.5, dirichlet_alpha=0.3, dirichlet_eps=0.25)


@contextlib.contextmanager
def _using(flag):
    """Force the C++ (True) or NumPy (False) path for the duration of the block."""
    prev = native.enabled()
    native.set_enabled(flag)
    try:
        yield
    finally:
        native.set_enabled(prev)


def _positions(n_games, seed=0):
    """Play random games with the trusted single-board engine; return states."""
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


def _stack(pos):
    boards = np.ascontiguousarray(np.stack([b for b, _ in pos]), dtype=np.int8)
    players = np.array([p for _, p in pos], dtype=np.int8)
    return boards, players


def _evaluator():
    torch.manual_seed(0)
    return Evaluator(OthelloNet(num_blocks=2, channels=16))


def _looped(ev):
    """A bit-exact single-board evaluator in the batched calling convention, so a
    search's float rounding can't depend on how leaves were batched."""
    def evaluate(boards, players):
        out = [ev(np.ascontiguousarray(b, np.int8), int(p)) for b, p in zip(boards, players)]
        return (np.stack([o[0] for o in out]).astype(np.float32),
                np.array([o[1] for o in out], np.float32))
    return evaluate


# --- engine ------------------------------------------------------------------

def _engine_parity(n_games):
    """Every exported engine function, C++ vs NumPy, over real game positions."""
    pos = _positions(n_games)
    boards, players = _stack(pos)
    rng = np.random.default_rng(1)
    actions = np.array([bn.PASS if not bn.legal_moves(b, p) else int(rng.choice(bn.legal_moves(b, p)))
                        for b, p in pos], dtype=np.int64)
    nat = native.NATIVE

    cases = {
        "legal_move_masks": (nat.legal_move_masks(boards, players),
                             bb._np_legal_move_masks(boards, players)),
        "legal_action_masks": (nat.legal_action_masks(boards, players),
                               bb._np_legal_action_masks(boards, players)),
        "apply_moves": (nat.apply_moves(boards, players, actions),
                        bb._np_apply_moves(boards, players, actions)),
        "is_terminal": (nat.is_terminal(boards), bb._np_is_terminal(boards)),
        "winner": (nat.winner(boards), bb._np_winner(boards)),
        "encode_batch": (nat.encode_batch(boards, players),
                         bb._np_encode_batch(boards, players)),
    }
    bad = [name for name, (c, n) in cases.items() if not np.array_equal(c, n)]
    # dtype drift is as breaking as value drift here — downstream does
    # torch.from_numpy() on these and int8/bool/float32 are load-bearing.
    bad += [name + ":dtype" for name, (c, n) in cases.items() if c.dtype != n.dtype]

    c_black, c_white = nat.count_discs(boards)
    n_black, n_white = bb._np_count_discs(boards)
    if not (np.array_equal(c_black, n_black) and np.array_equal(c_white, n_white)):
        bad.append("count_discs")
    if (c_black.dtype, c_white.dtype) != (n_black.dtype, n_white.dtype):
        bad.append("count_discs:dtype")
    return len(pos), bad


def test_native_engine_matches_numpy():
    n, bad = _engine_parity(6)
    check(f"C++ engine == NumPy engine, exactly, all 7 functions ({n} positions)",
          not bad)


# --- search ------------------------------------------------------------------

def _mcts_parity(n_games, sims, add_noise, seed=7, limit=None):
    """Run the same search both ways. With noise, each path gets its own freshly
    seeded RNG — identical draws only if both consume the stream the same way.

    `limit` caps how many positions are searched. The bit-exact `_looped`
    evaluator costs one net forward PER BOARD PER SIM, so the fast tier searches
    a slice: parity that holds on 32 varied positions and fails on none is the
    same signal as 180, at a sixth of the dev-loop cost. The slow tier widens it."""
    ev = _evaluator()
    evaluate = _looped(ev)
    pos = [(b, p) for b, p in _positions(n_games) if not bn.is_terminal(b)]
    if limit is not None:
        pos = pos[::max(1, len(pos) // limit)][:limit]   # spread across the game
    boards, players = _stack(pos)

    with _using(False):
        n_counts, n_root = _run_batched_numpy(
            boards, players, sims, evaluate, CFG,
            rng=np.random.default_rng(seed), add_noise=add_noise)
    with _using(True):
        c_counts, c_root = run_batched(
            boards, players, sims, evaluate, CFG,
            rng=np.random.default_rng(seed), add_noise=add_noise)

    return len(pos), (np.array_equal(c_counts, n_counts)
                      and np.array_equal(c_root, n_root)
                      and c_counts.dtype == n_counts.dtype)


def test_native_mcts_matches_numpy():
    n, ok = _mcts_parity(2, sims=12, add_noise=False, limit=32)
    check(f"C++ MCTS visit counts + root value == NumPy, exactly ({n} positions)", ok)


def test_native_mcts_matches_numpy_with_noise():
    """The noise path is where the C++ port could most easily drift: the Dirichlet
    draw sits at a different point in the C++ call order (before the root eval,
    not after). It stays identical because `evaluate` never touches the RNG —
    this test is what holds that reasoning to account."""
    n, ok = _mcts_parity(2, sims=12, add_noise=True, limit=32)
    check(f"C++ MCTS == NumPy with root Dirichlet noise ({n} positions)", ok)


def test_native_toggle_selects_the_path():
    """`--no-native` / OTHELLO_NATIVE=0 must really route back to NumPy — a switch
    that silently does nothing would make every A/B measurement a lie."""
    boards, players = _stack(_positions(1)[:4])
    with _using(False):
        off = bb.legal_action_masks is bb._np_legal_action_masks or \
            np.array_equal(bb.legal_action_masks(boards, players),
                           bb._np_legal_action_masks(boards, players))
        used_numpy = not native.enabled()
    with _using(True):
        used_native = native.enabled()
    check("native.set_enabled() flips the engine and the results still agree",
          off and used_numpy and used_native)


# --- end-to-end --------------------------------------------------------------

def _selfplay_parity(num_games, seed=3):
    """A whole self-play batch both ways: same games, same training examples."""
    cfg = Config.tiny(selfplay_arrayops=True, games_per_iter=num_games)
    ev = _evaluator()

    def play(flag):
        with _using(flag):
            return selfplay._play_batch(ev, cfg, np.random.default_rng(seed), num_games,
                                        make_records=True)

    a, b = play(False), play(True)
    if len(a) != len(b):
        return 0, False
    for (ex_a, rec_a, plies_a), (ex_b, rec_b, plies_b) in zip(a, b):
        if plies_a != plies_b or len(ex_a) != len(ex_b) or rec_a != rec_b:
            return len(a), False
        for ta, tb in zip(ex_a, ex_b):
            if not all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(ta, tb)):
                return len(a), False
    return len(a), True


def test_native_selfplay_matches_numpy():
    n, ok = _selfplay_parity(3)
    check(f"C++ self-play produces byte-identical games + examples ({n} games)", ok)


# --- wider sweeps (slow) -----------------------------------------------------

def test_native_engine_matches_numpy_wide():
    n, bad = _engine_parity(40)
    check(f"C++ engine parity holds over a wide sweep ({n} positions)", not bad)


def test_native_mcts_matches_numpy_wide():
    n, ok = _mcts_parity(8, sims=48, add_noise=True)
    check(f"C++ MCTS == NumPy over a wide noisy sweep ({n} positions)", ok)


def test_native_selfplay_matches_numpy_wide():
    n, ok = _selfplay_parity(12)
    check(f"C++ self-play parity holds over more games ({n} games)", ok)


FAST = [test_native_engine_matches_numpy, test_native_mcts_matches_numpy,
        test_native_mcts_matches_numpy_with_noise, test_native_toggle_selects_the_path,
        test_native_selfplay_matches_numpy]
SLOW = [test_native_engine_matches_numpy_wide, test_native_mcts_matches_numpy_wide,
        test_native_selfplay_matches_numpy_wide]

if __name__ == "__main__":
    if not native.available():
        # Not a failure: the extension is an optional accelerator. Say so loudly
        # enough that nobody reads a green suite as "the C++ path was tested".
        print("### native  [SKIPPED] — extension not built ###")
        print(f"  {native.status()}")
        print("  build it with:  python native/build.py")
        sys.exit(0)
    print(f"# {native.status()}")
    run(FAST, SLOW, "native")
