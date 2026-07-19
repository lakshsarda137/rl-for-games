"""Evaluation — measure the agent on the minimax ladder (plan §2, §8.6).

The headline metric is `max_depth_beaten`: the deepest minimax the agent beats
with win rate >= the threshold (default 0.55), over `eval_games` colour-alternated
games. Plotting it over iterations gives the rising-staircase strength curve.

The agent plays greedily on MCTS visit counts (τ=0, no root noise), so eval is
reproducible given a seed.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "engine"))
sys.path.insert(0, os.path.join(_HERE, "..", "opponents"))
sys.path.insert(0, _HERE)

from minimax import minimax_player
from simple import play_match

from mcts import MCTS


def az_player(evaluator, sims, c_puct=1.5, rng=None):
    """A greedy-on-visits AlphaZero move function: `(board, player) -> move`."""
    mcts = MCTS(evaluator, c_puct=c_puct, rng=rng or np.random.default_rng())

    def move_fn(board, player):
        counts = mcts.run(board, player, sims, add_noise=False)
        return int(counts.argmax())  # τ=0: most-visited (always a legal action)

    return move_fn


def ladder_eval(evaluator, cfg, rng=None, sims=None, depths=None, n_games=None):
    """Play the agent vs minimax at each depth; return win rates + max_depth_beaten."""
    rng = rng or np.random.default_rng(cfg.seed)
    sims = sims or cfg.sims_eval
    depths = depths or cfg.eval_depths
    n_games = n_games or cfg.eval_games

    agent = az_player(evaluator, sims, c_puct=cfg.c_puct, rng=rng)
    winrates, max_beaten = {}, 0
    for depth in depths:
        # Fresh seed per depth so colour/opening variation is comparable.
        res = play_match(agent, minimax_player(depth), n_games=n_games,
                         opening_plies=2, seed=int(rng.integers(1 << 30)))
        winrates[depth] = res["win_rate"]
        if res["win_rate"] >= cfg.promote_threshold:
            max_beaten = max(max_beaten, depth)
    return {"winrate": winrates, "max_depth_beaten": max_beaten}
