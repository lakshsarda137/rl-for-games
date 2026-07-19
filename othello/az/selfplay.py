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


def play_game(evaluator, cfg, rng, iteration=0, make_record=False):
    """Play one self-play game. Returns (examples, record_or_None).

    `examples` are (planes, pi, mask, z) tuples, already augmented (×8 unless
    cfg.augment is False).
    """
    mcts = MCTS(evaluator, c_puct=cfg.c_puct, dirichlet_alpha=cfg.dirichlet_alpha,
                dirichlet_eps=cfg.dirichlet_eps, rng=rng)
    board, player, ply = initial_board(), BLACK, 0
    history = []          # (planes, pi, mask, mover) before z is known
    record_moves = []

    while not is_terminal(board):
        root = mcts.run_root(board, player, cfg.sims_selfplay, add_noise=True)
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
    return examples, record


def generate_games(evaluator, cfg, rng, num_games, iteration=0, make_records=False):
    """Play `num_games` self-play games; return (examples, records, stats)."""
    examples, records, lengths = [], [], []
    for g in range(num_games):
        ex, rec = play_game(evaluator, cfg, rng, iteration=iteration,
                            make_record=make_records)
        examples.extend(ex)
        if rec is not None:
            records.append(rec)
        lengths.append(rec["plies"] if rec else None)
    stats = {"num_games": num_games, "num_examples": len(examples),
             "avg_game_len": float(np.mean([l for l in lengths if l])) if make_records else None}
    return examples, records, stats
