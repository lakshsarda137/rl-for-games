"""Self-play game generation — turns the current net into training examples.

Per move: run MCTS, form the improved policy π ∝ N^(1/τ) from the root visits,
sample a move from π, and record (planes, π, legal-mask, mover). At game end,
stamp each example's z with the final result *from that state's mover's
perspective* (+1 if that mover won). Each example is then expanded into its 8
dihedral symmetries (data augmentation).

Temperature schedule: τ=1 for the first `temp_moves` plies (explore), then τ→0
(greedy on visits) to sharpen play. Root Dirichlet noise (in MCTS) keeps
self-play from collapsing onto one line.

Optionally also emits a human-readable game record (§10) for the spectator.

THROUGHPUT has two independent levers, because self-play cost splits in two:

  1. BATCHED INFERENCE (`_play_pool`). `generate_games` plays many games
     CONCURRENTLY and evaluates all their pending MCTS leaves in ONE batched
     network call per step. Each game is a coroutine (`play_game_gen`) that yields
     the leaf it needs evaluated; the pool driver gathers one request per active
     game, runs a single `evaluator.evaluate_batch`, and sends each result back.
     Play matches the serial path regardless of batch size (differences are
     float32 matmul rounding, ~1e-7, below the level that changes a move). This
     collapses ~N net calls into ~1 — but for the small 5x64 Othello net the
     network is only ~13% of self-play time, so this alone barely moves g/s.

  2. MULTIPROCESS SELF-PLAY (`_play_parallel`). The other ~87% is pure-Python
     MCTS + NumPy engine (select/backup, apply_move, legal_moves, encode), and it
     runs single-threaded — one CPU core, which is the real wall (measured: a GPU
     run sat CPU-pinned at ~0.4 games/sec with the GPU near-idle). Fix: split the
     per-game seeds across `cfg.selfplay_workers` worker PROCESSES; each runs its
     own batched `_play_pool` on its slice with the net loaded from a shared
     state_dict. This uses all cores (and, on a GPU, keeps it busier with several
     processes issuing batches). Because every game is fully determined by its own
     seed, the set of games produced is independent of how they're partitioned.

Use both together: workers give the ~cores× speedup, batching keeps each worker's
GPU round-trips cheap. `selfplay_workers=1` keeps the plain single-process path.
"""

import atexit
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "engine"))

from board_numpy import (
    BLACK,
    apply_move,
    count_discs,
    initial_board,
    is_terminal,
    winner,
)
from encode import encode, legal_action_mask
from symmetry import NUM_SYMMETRIES, transform_board, transform_policy

import board_batched as bb

from mcts import MCTS, root_value, visit_policy
from mcts_batched import make_net_evaluator, run_batched
from profiling import PROF


def _transform_planes(planes, i):
    return np.stack([transform_board(planes[c], i) for c in range(planes.shape[0])])


def _augment(planes, pi, mask, z):
    """Expand one example into its 8 dihedral symmetries (board + policy + mask)."""
    out = []
    for i in range(NUM_SYMMETRIES):
        out.append((_transform_planes(planes, i),
                    transform_policy(pi, i),
                    transform_policy(mask, i),
                    z))
    return out


def _result_str(board):
    black, white = count_discs(board)
    margin = black - white
    return f"{'+' if margin >= 0 else ''}{margin}"  # Black's disc margin


def play_game_gen(cfg, rng, iteration=0, make_record=False):
    """One self-play game as a coroutine (the single source of game logic).

    Yields `(board, player)` whenever MCTS needs a leaf evaluated and expects
    `(priors, value)` back via `.send()`; returns `(examples, record_or_None,
    plies)` on completion. `examples` are (planes, pi, mask, z) tuples, already
    augmented (×8 unless cfg.augment is False).

    Randomness (Dirichlet noise + move sampling) flows only from `rng`, and the
    net is evaluated in eval mode, so a game depends solely on (net weights, rng)
    — not on how many other games share its evaluation batch (bar float32 matmul
    rounding, ~1e-7, which noise-protected tie-breaking absorbs). Drive one with
    `_drive_single` (serial) or many together with `_play_pool` (batched).
    """
    mcts = MCTS(c_puct=cfg.c_puct, dirichlet_alpha=cfg.dirichlet_alpha,
                dirichlet_eps=cfg.dirichlet_eps, rng=rng)
    board, player, ply = initial_board(), BLACK, 0
    history = []          # (planes, pi, mask, mover) before z is known
    record_moves = []

    while not is_terminal(board):
        root = yield from mcts.run_root_gen(board, player, cfg.sims_selfplay,
                                            add_noise=True)
        temperature = 1.0 if ply < cfg.temp_moves else 0.0
        pi = visit_policy(root.N, temperature)

        history.append((encode(board, player),
                        pi,
                        legal_action_mask(board, player) > 0,
                        player))
        move = int(rng.choice(len(pi), p=pi))

        entry = None
        if make_record:
            top = sorted(((int(a), float(pi[a])) for a in np.nonzero(pi)[0]),
                         key=lambda x: -x[1])[:3]
            entry = {"n": ply + 1, "player": "B" if player == BLACK else "W",
                     "move": move, "mcts_value": round(root_value(root), 3),
                     "top_policy": [[a, round(p, 3)] for a, p in top]}

        board = apply_move(board, player, move)
        player = -player
        ply += 1
        if entry is not None:
            entry["board"] = [int(x) for x in board.reshape(-1)]
            record_moves.append(entry)

    result = winner(board)
    examples = []
    for planes, pi, mask, mover in history:
        z = 0.0 if result == 0 else (1.0 if result == mover else -1.0)
        if cfg.augment:
            examples.extend(_augment(planes, pi, mask, z))
        else:
            examples.append((planes, pi, mask, z))

    record = None
    if make_record:
        record = {"iteration": iteration, "kind": "selfplay",
                  "result": _result_str(board), "plies": ply, "moves": record_moves}
    return examples, record, ply


def _drive_single(gen, evaluate_one):
    """Run one game coroutine to completion, one board per net call."""
    try:
        request = gen.send(None)
    except StopIteration as done:
        return done.value
    while True:
        board, player = request
        result = evaluate_one(board, player)
        try:
            request = gen.send(result)
        except StopIteration as done:
            return done.value


def play_game(evaluator, cfg, rng, iteration=0, make_record=False):
    """Play one self-play game serially (one board per net call).

    Returns (examples, record_or_None). Batched generation uses `generate_games`;
    this stays for single-game/testing use and shares the same game coroutine.
    """
    examples, record, _ = _drive_single(
        play_game_gen(cfg, rng, iteration=iteration, make_record=make_record),
        evaluator)
    return examples, record


def _batch_evaluator(evaluator):
    """A `(boards, players) -> [(priors, value), ...]` fn from any evaluator."""
    if hasattr(evaluator, "evaluate_batch"):
        return evaluator.evaluate_batch
    return lambda boards, players: [evaluator(b, p) for b, p in zip(boards, players)]


def _play_pool(evaluator, cfg, seeds, iteration=0, make_records=False):
    """Play len(seeds) games concurrently, batching every step's pending MCTS
    leaves into ONE network call. Returns [(examples, record, plies), ...] in
    seed order.

    Up to `cfg.selfplay_concurrency` games are in flight at once (that many
    boards per batched eval); as each game finishes another starts, keeping the
    batch full. Each game runs `play_game_gen(seed)`, so results are independent
    of concurrency — only throughput changes.
    """
    evaluate_batch = _batch_evaluator(evaluator)
    n = len(seeds)
    concurrency = max(1, min(int(getattr(cfg, "selfplay_concurrency", 1) or 1), n or 1))
    results = [None] * n

    def start(slot):
        gen = play_game_gen(cfg, np.random.default_rng(int(seeds[slot])),
                            iteration=iteration, make_record=make_records)
        try:
            request = gen.send(None)          # advance to first leaf request
        except StopIteration as done:         # game with no eval needed (never, here)
            results[slot] = done.value
            return None
        return [slot, gen, request]

    next_slot, active = 0, []
    while next_slot < n and len(active) < concurrency:
        entry = start(next_slot); next_slot += 1
        if entry is not None:
            active.append(entry)

    while active:
        evals = evaluate_batch([e[2][0] for e in active], [e[2][1] for e in active])
        still = []
        for entry, ev in zip(active, evals):
            slot, gen, _ = entry
            try:
                entry[2] = gen.send(ev)       # resume; get next leaf request
                still.append(entry)
            except StopIteration as done:     # this game finished
                results[slot] = done.value
                while next_slot < n:          # backfill the freed slot
                    new = start(next_slot); next_slot += 1
                    if new is not None:
                        still.append(new)
                        break
        active = still
    return results


# --- multiprocess self-play (the CPU-bound-throughput lever) -----------------

def _chunk(items, n):
    """Split `items` into `n` contiguous, near-equal slices (order preserved)."""
    n = max(1, min(n, len(items) or 1))
    k, r = divmod(len(items), n)
    out, i = [], 0
    for j in range(n):
        size = k + (1 if j < r else 0)
        out.append(items[i:i + size])
        i += size
    return [c for c in out if c]


_EXECUTOR = None
_EXECUTOR_WORKERS = None


def _get_executor(workers):
    """A persistent spawn-based process pool, reused across iterations so torch /
    CUDA init happens once per worker, not once per call."""
    global _EXECUTOR, _EXECUTOR_WORKERS
    if _EXECUTOR is None or _EXECUTOR_WORKERS != workers:
        if _EXECUTOR is not None:
            _EXECUTOR.shutdown()
        # spawn (not fork): required for CUDA in child processes, and the default
        # on macOS anyway — so the local CPU test exercises the same mechanism.
        _EXECUTOR = ProcessPoolExecutor(max_workers=workers,
                                        mp_context=mp.get_context("spawn"))
        _EXECUTOR_WORKERS = workers
    return _EXECUTOR


def shutdown_selfplay_workers():
    """Tear down the worker pool (train loop calls this; also runs at exit)."""
    global _EXECUTOR, _EXECUTOR_WORKERS
    if _EXECUTOR is not None:
        _EXECUTOR.shutdown()
        _EXECUTOR, _EXECUTOR_WORKERS = None, None


atexit.register(shutdown_selfplay_workers)


def _worker_play(payload):
    """Runs in a spawned worker: rebuild the net from a shared state_dict and play
    a slice of games via the batched pool. Returns per-game (examples, record,
    plies), exactly as `_play_pool` would in-process."""
    from types import SimpleNamespace

    import torch

    from network import Evaluator, OthelloNet  # imported here: only the workers need it

    arch, state, cfg_dict, seeds, iteration, make_records, device = payload
    torch.set_num_threads(1)   # one core per worker — avoid intraop-thread thrash
    net = OthelloNet(*arch)
    net.load_state_dict(state)
    evaluator = Evaluator(net, device)
    cfg = SimpleNamespace(**cfg_dict)
    return _play_pool(evaluator, cfg, seeds, iteration=iteration,
                      make_records=make_records)


def _worker_batch(payload):
    """Runs in a spawned worker: rebuild the net and play a slice of games via the
    ARRAY-OPS path (`_play_batch`). This is how the array-ops self-play uses more
    than one CPU core — each worker vectorises its own sub-batch on its own core."""
    from types import SimpleNamespace

    import torch

    from network import Evaluator, OthelloNet

    arch, state, cfg_dict, n_games, seed, iteration, make_records, device = payload
    torch.set_num_threads(1)
    net = OthelloNet(*arch)
    net.load_state_dict(state)
    evaluator = Evaluator(net, device)
    cfg = SimpleNamespace(**cfg_dict)
    return _play_batch(evaluator, cfg, np.random.default_rng(int(seed)), n_games,
                       iteration=iteration, make_records=make_records)


def _play_parallel(evaluator, cfg, seeds, workers, iteration=0, make_records=False):
    """Play the games in `seeds` across `workers` processes; return per-game
    results in seed order (identical set of games to the single-process path)."""
    net = evaluator.net
    payload_base = ((net.num_blocks, net.channels),
                    {k: v.detach().cpu() for k, v in net.state_dict().items()},
                    dict(vars(cfg)))
    chunks = _chunk(seeds, workers)
    payloads = [(*payload_base, chunk, iteration, make_records, evaluator.device)
                for chunk in chunks]
    executor = _get_executor(len(payloads))
    results = executor.map(_worker_play, payloads)
    return [game for worker_result in results for game in worker_result]


# --- array-ops self-play (the big throughput lever) --------------------------

def _visit_policies(counts, temperature):
    """Vectorised `visit_policy` over a batch: `[k,65]` counts -> `[k,65]` π."""
    counts = counts.astype(np.float64)
    if temperature <= 1e-8:
        out = np.zeros_like(counts)
        out[np.arange(len(counts)), counts.argmax(1)] = 1.0
        return out.astype(np.float32)
    scaled = counts ** (1.0 / temperature)
    return (scaled / scaled.sum(1, keepdims=True)).astype(np.float32)


def _play_batch(evaluator, cfg, rng, num_games, iteration=0, make_records=False,
                add_noise=True):
    """Play `num_games` self-play games as ONE batch, in lockstep, with the whole
    search vectorised over games (batched engine + batched MCTS). Returns per-game
    (examples, record, plies). This is the array-ops path: the per-game Python cost
    that dominates the coroutine path is gone — every ply advances all live games
    with one batched MCTS and one batched engine step.

    `rng` drives Dirichlet noise + move sampling for the whole batch, so games are
    coupled through it (unlike the per-seed pool path) — fine for training, and
    with `add_noise=False` + greedy play it's deterministic and matches serial
    greedy self-play move-for-move (`tests/test_batched.py`)."""
    evaluate = make_net_evaluator(evaluator.net, evaluator.device)
    boards = bb.initial_boards(num_games)
    players = np.full(num_games, BLACK, np.int8)
    alive = ~bb.is_terminal(boards)

    history = [[] for _ in range(num_games)]     # (planes, pi, mask, mover) per game
    rec_moves = [[] for _ in range(num_games)] if make_records else None
    ply = 0

    while alive.any():
        idx = np.where(alive)[0]
        sub_boards, sub_players = boards[idx], players[idx].copy()
        # search_total (includes the net eval called inside run_batched); the profiler
        # subtracts net_eval to isolate the pure CPU tree-search — see az/profiling.py.
        if PROF.enabled:
            _s0 = PROF.clock()
        counts, root_vals = run_batched(sub_boards, sub_players, cfg.sims_selfplay,
                                        evaluate, cfg, rng=rng, add_noise=add_noise)
        if PROF.enabled:
            PROF.add("search_total", PROF.clock() - _s0); PROF.count("plies_searched")
        temperature = 1.0 if ply < cfg.temp_moves else 0.0
        pis = _visit_policies(counts, temperature)
        planes = bb.encode_batch(sub_boards, sub_players)
        masks = bb.legal_action_masks(sub_boards, sub_players) > 0

        if temperature <= 1e-8:
            moves = pis.argmax(1)
        else:
            u = rng.random(len(pis))
            moves = (np.cumsum(pis, axis=1) < u[:, None]).sum(1)
        moves = moves.astype(np.int64)

        for j, g in enumerate(idx):
            history[g].append((planes[j], pis[j], masks[j], int(sub_players[j])))
            if make_records:
                top = sorted(((int(a), float(pis[j, a])) for a in np.nonzero(pis[j])[0]),
                             key=lambda x: -x[1])[:3]
                rec_moves[g].append({
                    "n": ply + 1, "player": "B" if sub_players[j] == BLACK else "W",
                    "move": int(moves[j]), "mcts_value": round(float(root_vals[j]), 3),
                    "top_policy": [[a, round(p, 3)] for a, p in top]})

        boards[idx] = bb.apply_moves(sub_boards, sub_players, moves)
        players[idx] = -players[idx]
        ply += 1
        newly_terminal = bb.is_terminal(boards[idx])
        if make_records:
            for j, g in enumerate(idx):
                rec_moves[g][-1]["board"] = [int(x) for x in boards[g].reshape(-1)]
        alive[idx[newly_terminal]] = False

    # postproc: x8 dihedral augmentation + example/record building (NumPy on CPU).
    if PROF.enabled:
        _p0 = PROF.clock()
    results = bb.winner(boards)
    black_disc, white_disc = bb.count_discs(boards)
    per_game = []
    for g in range(num_games):
        result = int(results[g])
        examples = []
        for planes_g, pi_g, mask_g, mover in history[g]:
            z = 0.0 if result == 0 else (1.0 if result == mover else -1.0)
            if cfg.augment:
                examples.extend(_augment(planes_g, pi_g, mask_g, z))
            else:
                examples.append((planes_g, pi_g, mask_g, z))
        record = None
        if make_records:
            margin = int(black_disc[g] - white_disc[g])
            record = {"iteration": iteration, "kind": "selfplay",
                      "result": f"{'+' if margin >= 0 else ''}{margin}",
                      "plies": len(history[g]), "moves": rec_moves[g]}
        per_game.append((examples, record, len(history[g])))
    if PROF.enabled:
        PROF.add("postproc", PROF.clock() - _p0)
    return per_game


def _play_batch_torch(evaluator, cfg, rng, num_games, iteration=0, make_records=False,
                      add_noise=True):
    """Torch twin of `_play_batch`: play `num_games` games as ONE batch, in
    lockstep, with the whole search on TORCH tensors (`engine/board_torch.py` +
    `az/mcts_torch.py`) instead of NumPy. Because those are device-agnostic torch,
    the search runs wherever `evaluator.device` is — the GPU on Kaggle — so the
    tree-search + game-rules (the CPU wall of the NumPy array-ops path) move onto
    the device too, not just the network. Returns per-game (examples, record, plies).

    The per-ply loop is identical to `_play_batch`; only the board state + search
    moved to torch. Training examples are still built as NumPy (the replay buffer +
    `_augment` are NumPy), converted once per ply at the recording boundary. Policy
    forming + move sampling reuse `_play_batch`'s exact NumPy code driven by the
    same `rng`, so with `add_noise=False` + greedy play it matches serial greedy
    self-play move-for-move (`tests/test_az.py`)."""
    import torch

    import board_torch as bt
    from mcts_torch import make_net_evaluator_torch, run_torch

    device = evaluator.device
    evaluate = make_net_evaluator_torch(evaluator.net, device)
    boards = bt.initial_boards(num_games, device)
    players = torch.full((num_games,), BLACK, dtype=torch.int8, device=device)
    alive = ~bt.is_terminal(boards)

    history = [[] for _ in range(num_games)]     # (planes, pi, mask, mover) per game
    rec_moves = [[] for _ in range(num_games)] if make_records else None
    ply = 0

    while bool(alive.any()):
        idx = torch.nonzero(alive, as_tuple=True)[0]
        sub_boards, sub_players = boards[idx], players[idx].clone()
        counts, root_vals = run_torch(sub_boards, sub_players, cfg.sims_selfplay,
                                      evaluate, cfg, rng=rng, add_noise=add_noise)
        counts_np = counts.cpu().numpy()
        temperature = 1.0 if ply < cfg.temp_moves else 0.0
        pis = _visit_policies(counts_np, temperature)
        planes = bt.encode_batch(sub_boards, sub_players).cpu().numpy()
        masks = (bt.legal_action_masks(sub_boards, sub_players) > 0).cpu().numpy()
        sub_players_np = sub_players.cpu().numpy()

        if temperature <= 1e-8:
            moves = pis.argmax(1)
        else:
            u = rng.random(len(pis))
            moves = (np.cumsum(pis, axis=1) < u[:, None]).sum(1)
        moves = moves.astype(np.int64)

        idx_list = idx.cpu().tolist()
        root_vals_np = root_vals.cpu().numpy() if make_records else None
        for j, g in enumerate(idx_list):
            history[g].append((planes[j], pis[j], masks[j], int(sub_players_np[j])))
            if make_records:
                top = sorted(((int(a), float(pis[j, a])) for a in np.nonzero(pis[j])[0]),
                             key=lambda x: -x[1])[:3]
                rec_moves[g].append({
                    "n": ply + 1, "player": "B" if sub_players_np[j] == BLACK else "W",
                    "move": int(moves[j]), "mcts_value": round(float(root_vals_np[j]), 3),
                    "top_policy": [[a, round(p, 3)] for a, p in top]})

        boards[idx] = bt.apply_moves(sub_boards, sub_players,
                                     torch.from_numpy(moves).to(device))
        players[idx] = -players[idx]
        ply += 1
        newly_terminal = bt.is_terminal(boards[idx])
        if make_records:
            updated = boards[idx].cpu().numpy()   # only the k just-moved boards
            for j, g in enumerate(idx_list):
                rec_moves[g][-1]["board"] = [int(x) for x in updated[j].reshape(-1)]
        alive[idx[newly_terminal]] = False

    results = bt.winner(boards).cpu().numpy()
    black_disc, white_disc = bt.count_discs(boards)
    black_disc, white_disc = black_disc.cpu().numpy(), white_disc.cpu().numpy()
    per_game = []
    for g in range(num_games):
        result = int(results[g])
        examples = []
        for planes_g, pi_g, mask_g, mover in history[g]:
            z = 0.0 if result == 0 else (1.0 if result == mover else -1.0)
            if cfg.augment:
                examples.extend(_augment(planes_g, pi_g, mask_g, z))
            else:
                examples.append((planes_g, pi_g, mask_g, z))
        record = None
        if make_records:
            margin = int(black_disc[g] - white_disc[g])
            record = {"iteration": iteration, "kind": "selfplay",
                      "result": f"{'+' if margin >= 0 else ''}{margin}",
                      "plies": len(history[g]), "moves": rec_moves[g]}
        per_game.append((examples, record, len(history[g])))
    return per_game


def _play_batch_parallel(evaluator, cfg, rng, num_games, workers,
                         iteration=0, make_records=False):
    """Run the array-ops path across `workers` processes — each vectorises its own
    sub-batch of games on its own CPU core. This is the "use every core" lever
    on top of array-ops (array-ops already uses one core fully). Returns per-game
    results in worker order."""
    net = evaluator.net
    payload_base = ((net.num_blocks, net.channels),
                    {k: v.detach().cpu() for k, v in net.state_dict().items()},
                    dict(vars(cfg)))
    sizes = [len(c) for c in _chunk(list(range(num_games)), workers)]
    seeds = rng.integers(1, np.iinfo(np.int64).max, size=len(sizes))
    payloads = [(*payload_base, sizes[w], int(seeds[w]), iteration, make_records,
                 evaluator.device) for w in range(len(sizes))]
    executor = _get_executor(len(payloads))
    results = executor.map(_worker_batch, payloads)
    return [game for worker_result in results for game in worker_result]


def generate_games(evaluator, cfg, rng, num_games, iteration=0, make_records=False):
    """Play `num_games` self-play games; return (examples, records, stats).

    Path selection (see the module docstring):
      * `cfg.selfplay_arrayops` (default) — the whole batch is searched with array
        ops. `cfg.selfplay_torch` picks the TORCH engine + MCTS (`_play_batch_torch`),
        which runs the whole search on `evaluator.device` (the GPU on Kaggle);
        otherwise the NumPy array-ops path (`_play_batch`) runs on the CPU, and
        `cfg.selfplay_workers > 1` splits it across worker PROCESSES
        (`_play_batch_parallel`), each vectorising its own sub-batch on its own core.
      * else — the coroutine pool (`_play_pool`) or its multiprocess variant.
    The array-ops path couples games through `rng`; the pool paths seed each game
    independently (reproducible per seed), which the parity tests rely on.
    """
    workers = int(getattr(cfg, "selfplay_workers", 1) or 1)
    use_torch = getattr(cfg, "selfplay_torch", False)
    # selfplay_torch is itself an array-ops path (torch engine + MCTS), so it takes
    # the array-ops branch even if selfplay_arrayops (the NumPy flag) is off.
    if (getattr(cfg, "selfplay_arrayops", False) or use_torch) and num_games > 1:
        if use_torch:
            # Torch (device-agnostic) array-ops: the whole search runs on tensors,
            # so on a GPU it uses the device for the SEARCH itself, not just the net.
            # One process (on a GPU that's optimal — one process, one big batch); the
            # multi-worker split is a CPU-only lever and doesn't apply here.
            per_game = _play_batch_torch(evaluator, cfg, rng, num_games,
                                         iteration=iteration, make_records=make_records)
        elif workers > 1:
            per_game = _play_batch_parallel(evaluator, cfg, rng, num_games,
                                            min(workers, num_games),
                                            iteration=iteration, make_records=make_records)
        else:
            per_game = _play_batch(evaluator, cfg, rng, num_games, iteration=iteration,
                                   make_records=make_records)
    else:
        seeds = [int(s) for s in rng.integers(1, np.iinfo(np.int64).max, size=num_games)]
        if workers > 1 and num_games > 1:
            per_game = _play_parallel(evaluator, cfg, seeds, min(workers, num_games),
                                      iteration=iteration, make_records=make_records)
        else:
            per_game = _play_pool(evaluator, cfg, seeds, iteration=iteration,
                                  make_records=make_records)

    examples, records, lengths = [], [], []
    for ex, rec, plies in per_game:
        examples.extend(ex)
        if rec is not None:
            records.append(rec)
        lengths.append(plies)
    stats = {"num_games": num_games, "num_examples": len(examples),
             "avg_game_len": float(np.mean(lengths)) if lengths else None}
    return examples, records, stats
