"""A/B the C++ port against NumPy on a real self-play batch.

    python native/bench.py                       # quick local sanity check
    python native/bench.py --net 10x128 --games 96 --sims 96 --device cuda

Plays the SAME batch of games twice — once with the C++ engine + search, once
forced onto NumPy — and reports games/sec each way. Both runs use the same seed,
so the games are identical and the only difference measured is implementation
speed. (It also re-checks that the two agree, which makes an accidental "speedup"
from doing less work impossible to report as a win.)

WHAT THIS CAN AND CANNOT TELL YOU — read before trusting a number:

  * On a CPU-only machine (a Mac) the network forward is a MUCH larger share of
    self-play than it is on a GPU, and C++ cannot speed the network up. So a
    local ratio UNDERSTATES the GPU win. Use a local run to confirm it works and
    that nothing regressed, not to size the payoff.
  * The number that decides this is measured on the T4, at the `--games` and
    `--sims` you actually train with. `run/train_loop.py --profile` breaks the
    same run into net_fwd / net_prep / tree_ops buckets and prints the Amdahl
    ceiling; this script gives you the end-to-end ratio that ceiling predicts.
  * Small batches flatter NumPy (less per-call Python overhead to amortise away)
    and small `--sims` flatters it further. Benchmark at training settings.
"""

import argparse
import os
import sys
import time
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("engine", "az", "run"):
    sys.path.insert(0, os.path.join(_HERE, "..", _p))

import numpy as np

import board_batched as bb
import native
import selfplay
from config import Config
from mcts_batched import _run_batched_numpy, run_batched
from network import Evaluator, OthelloNet


def _midgame(games, seed=0):
    """A batch of realistic mid-game positions to benchmark on (the opening is
    unrepresentative: few legal moves, tiny flips)."""
    rng = np.random.default_rng(seed)
    boards = bb.initial_boards(games)
    players = np.full(games, 1, np.int8)
    for _ in range(20):
        masks = bb.legal_action_masks(boards, players)
        actions = np.array([int(rng.choice(np.nonzero(r)[0])) for r in masks], np.int64)
        boards, players = bb.apply_moves(boards, players, actions), -players
    return boards, players


def _time(fn, repeats):
    fn()                                    # warm up
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats


def component_bench(games, sims):
    """Time ONLY the code that was migrated, with no network in the way.

    This is the honest measure of the port itself: the end-to-end ratio below is
    that number diluted by Amdahl (the GPU forward, and the NumPy postprocessing
    that was deliberately left alone). The search figure still includes the
    Python evaluate callback on BOTH sides, so it understates the tree-op win."""
    boards, players = _midgame(games)
    masks = bb.legal_action_masks(boards, players)
    actions = np.array([int(np.nonzero(r)[0][0]) for r in masks], np.int64)
    nat = native.NATIVE

    print(f"  {'migrated op':<22}{'NumPy':>11}{'C++':>11}{'speed-up':>11}")
    for name, args in (("legal_action_masks", (boards, players)),
                       ("apply_moves", (boards, players, actions)),
                       ("encode_batch", (boards, players)),
                       ("is_terminal", (boards,)),
                       ("winner", (boards,))):
        a = _time(lambda: getattr(bb, "_np_" + name)(*args), 200) * 1e6
        b = _time(lambda: getattr(nat, name)(*args), 200) * 1e6
        print(f"  {name:<22}{a:>9.1f}us{b:>9.1f}us{a / b:>10.1f}x")

    # The tree search with a free evaluator, so the number is pure select/backup.
    cfg = SimpleNamespace(c_puct=1.5, dirichlet_alpha=0.3, dirichlet_eps=0.25)

    def const_eval(bs, ps):
        k = len(bs)
        return np.full((k, 65), 1.0 / 65, np.float32), np.zeros(k, np.float32)

    def one(use_native):
        fn = run_batched if use_native else _run_batched_numpy
        with_flag = native.enabled()
        native.set_enabled(use_native)
        try:
            return _time(lambda: fn(boards, players, sims, const_eval, cfg, add_noise=False), 5)
        finally:
            native.set_enabled(with_flag)

    a, b = one(False) * 1e3, one(True) * 1e3
    print(f"  {'run_batched (search)':<22}{a:>9.1f}ms{b:>9.1f}ms{a / b:>10.1f}x")


def _play(cfg, evaluator, games, seed, use_native):
    prev = native.enabled()
    native.set_enabled(use_native)
    try:
        t0 = time.perf_counter()
        out = selfplay._play_batch(evaluator, cfg, np.random.default_rng(seed), games)
        return time.perf_counter() - t0, out
    finally:
        native.set_enabled(prev)


def _same(a, b):
    """Did both paths really produce the same games?"""
    if len(a) != len(b):
        return False
    for (ex_a, _, plies_a), (ex_b, _, plies_b) in zip(a, b):
        if plies_a != plies_b or len(ex_a) != len(ex_b):
            return False
        for ta, tb in zip(ex_a, ex_b):
            if not all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(ta, tb)):
                return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=16, help="games in the batch (default 16)")
    ap.add_argument("--sims", type=int, default=24, help="MCTS sims per move (default 24)")
    ap.add_argument("--net", default="5x64", help="BLOCKSxCHANNELS, e.g. 10x128 (default 5x64)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda (default cpu)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not native.available():
        print(native.status())
        print("nothing to compare — build it with `python native/build.py`")
        sys.exit(1)

    blocks, channels = (int(x) for x in args.net.lower().split("x"))
    cfg = Config.tiny(num_blocks=blocks, channels=channels, sims_selfplay=args.sims,
                      temp_moves=12, selfplay_arrayops=True, augment=True)
    net = OthelloNet(num_blocks=blocks, channels=channels)
    evaluator = Evaluator(net, args.device)

    print(native.status())
    print(f"net={args.net}  games={args.games}  sims={args.sims}  device={args.device}")
    if args.device == "cpu":
        print("NOTE: on CPU the network forward dominates and C++ can't touch it — "
              "this ratio is a LOWER BOUND on the GPU win, not the answer.")
    print("-" * 60)
    print("COMPONENT — just the migrated code, no network involved:")
    component_bench(args.games, args.sims)
    print("-" * 60)
    print("END-TO-END — a real self-play batch, network included:")

    # Warm up both paths (lazy torch/cuDNN init, first-touch allocations) so the
    # first path measured isn't charged for setup the second one gets for free.
    for flag in (False, True):
        _play(cfg, evaluator, 2, args.seed, flag)

    t_np, out_np = _play(cfg, evaluator, args.games, args.seed, False)
    t_cc, out_cc = _play(cfg, evaluator, args.games, args.seed, True)

    gps_np, gps_cc = args.games / t_np, args.games / t_cc
    print(f"  {'NumPy':<8}{t_np:>9.2f}s{gps_np:>10.2f} games/s")
    print(f"  {'C++':<8}{t_cc:>9.2f}s{gps_cc:>10.2f} games/s")
    print("-" * 60)
    print(f"  speed-up: {t_np / t_cc:.2f}x")
    print(f"  identical games: {'yes' if _same(out_np, out_cc) else 'NO — PARITY BROKEN'}")


if __name__ == "__main__":
    main()
