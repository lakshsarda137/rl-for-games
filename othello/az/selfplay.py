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

BATCHED INFERENCE (the throughput fix). Naively, self-play evaluates ONE board
per network call, so on a GPU the CPU pins at 100% while the GPU idles — a GPU
evaluates 256 boards ≈ as fast as 1. Fix: `generate_games` plays many games
CONCURRENTLY and evaluates all their pending MCTS leaves in ONE batched network
call per step (`_play_pool`). Each game is a coroutine (`play_game_gen`) that
yields the leaf it needs evaluated; the pool driver gathers one request per
active game, runs a single `evaluator.evaluate_batch`, and sends each result
back. Within a game the search stays strictly sequential, and every game is
seeded independently, so play matches the serial path regardless of batch size
(differences are float32 rounding in the batched matmul, ~1e-7, below the level
that changes any move) — batching changes speed, not strength.
"""

import os
import sys

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

from mcts import MCTS, root_value, visit_policy


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


def generate_games(evaluator, cfg, rng, num_games, iteration=0, make_records=False):
    """Play `num_games` self-play games with batched inference; return
    (examples, records, stats).

    Games run concurrently and share one batched network call per step (see
    `_play_pool`) — the fix for GPU starvation. Each game gets its own seed
    drawn from `rng`, so play is reproducible and independent of batch size.
    """
    seeds = [int(s) for s in rng.integers(1, np.iinfo(np.int64).max, size=num_games)]
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
