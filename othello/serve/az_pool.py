"""Run AI (AlphaZero) moves in worker PROCESSES so concurrent games use all CPU cores.

Why processes, not threads: the serving-side search (az/mcts.py) is one network call
per simulation, and for this small net the Python overhead around each call holds the
GIL for most of the wall time — measured on a 4-vCPU server, four concurrent games in
threads ran at ~1.3x, not 4x. Each worker process loads its own copy of the net once
(~300 MB), sets torch to one thread, and flushes denormals (per-process CPU flag).

Enabled by OTHELLO_AZ_PROCS=<n> (0 / unset = run in-thread, the local default).
The search is stateless between moves (a fresh tree per call), so a move is just
`(checkpoint path, sims, board, player, seed) -> move`.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("engine", "opponents", "az"):
    p = os.path.join(_HERE, "..", sub)
    if p not in sys.path:
        sys.path.insert(0, p)

_POOL = None
_LOCK = threading.Lock()


def workers():
    try:
        return max(0, int(os.environ.get("OTHELLO_AZ_PROCS", "0")))
    except ValueError:
        return 0


def enabled():
    return workers() > 0


def _init_worker():
    import torch
    torch.set_num_threads(1)
    torch.set_flush_denormal(True)


_EVALS = {}   # worker-side cache: checkpoint path -> Evaluator


def _worker_move(path, sims, board, player, seed):
    import numpy as np
    import torch
    from network import Evaluator, OthelloNet
    from evaluate import az_player

    ev = _EVALS.get(path)
    if ev is None:
        ckpt = torch.load(path, map_location="cpu")
        cfg = ckpt.get("config", {})
        net = OthelloNet(cfg.get("num_blocks", 5), cfg.get("channels", 64))
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        ev = _EVALS[path] = Evaluator(net, "cpu")
    fn = az_player(ev, sims, rng=np.random.default_rng(seed))
    return int(fn(board, player))


def pool():
    """The shared executor (created on first use; spawn so no torch state is inherited)."""
    global _POOL
    with _LOCK:
        if _POOL is None:
            _POOL = ProcessPoolExecutor(max_workers=workers(), mp_context=mp.get_context("spawn"),
                                        initializer=_init_worker)
        return _POOL


def pooled_az_player(path, sims, rng):
    """A `(board, player) -> move` fn that runs each move in the process pool."""
    import numpy as np
    rng = rng or np.random.default_rng()

    def move_fn(board, player):
        seed = int(rng.integers(1 << 31))
        return pool().submit(_worker_move, path, sims, np.array(board), int(player), seed).result()
    return move_fn
