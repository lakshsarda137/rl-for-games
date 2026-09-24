"""Self-play: the network plays itself, with search, to make training data.

Many games run side by side in one C++ Env. At every decision the search looks
ahead in imagined versions of the hidden cards, and we record:
  * what the player could see (planes, scalars, legal moves),
  * the search's visit counts, as the move choice the network should learn,
  * the opponent's real hand, as the answer for the hand guess,
and once the hand ends, how it ended for that player (the value target).

Early moves of each hand are picked in proportion to the visits, so games
differ; later moves take the most visited move.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "engine"))

from native import rummy_native as rn


def choose(pi, explore, rng):
    """Pick an action from a visit distribution: sample while exploring, else
    the most visited (ties broken at random)."""
    if explore:
        return int(rng.choice(len(pi), p=pi))
    best = np.flatnonzero(pi == pi.max())
    return int(rng.choice(best))


def play(evaluator, cfg, n_hands, seed):
    """Play `n_hands` hands of self-play. Returns (examples dict, stats dict)."""
    rng = np.random.default_rng(seed)
    n = min(cfg.selfplay_games, n_hands)
    env = rn.Env(n, seed=seed, turn_cap=cfg.turn_cap)
    started = n
    finished = 0
    pending = [[] for _ in range(n)]     # this hand's records, per game
    decisions = np.zeros(n, dtype=np.int64)
    out = {k: [] for k in ("planes", "scalars", "legal", "pi", "z", "hand")}
    turns, capped, pile_takes, draws = [], 0, 0, 0

    while finished < n_hands:
        active = ~env.done()
        planes, scalars = env.observe()
        legal = env.legal_mask()
        opp = env.opponent_hands()
        movers = env.to_move()
        visits, _, _ = env.search(evaluator, active, worlds=cfg.worlds, sims=cfg.sims,
                                  c_puct=cfg.c_puct, dir_alpha=cfg.dir_alpha,
                                  dir_eps=cfg.dir_eps, seed=int(rng.integers(1 << 62)))
        actions = np.full(n, -1, dtype=np.int32)
        for i in np.flatnonzero(active):
            total = visits[i].sum()
            pi = visits[i] / total if total > 0 else legal[i] / legal[i].sum()
            actions[i] = choose(pi, decisions[i] < cfg.temp_moves, rng)
            decisions[i] += 1
            pending[i].append((planes[i], scalars[i], legal[i], pi.astype(np.float32), movers[i], opp[i]))
            if scalars[i, 0] == 1:
                draws += 1
                pile_takes += actions[i] == rn.ACT_PILE

        rewards, done = env.step(actions)
        for i in np.flatnonzero(done):
            s = env.state(int(i))
            turns.append(s["turn"])
            capped += s["capped"]
            for p_, s_, l_, pi_, mover, h_ in pending[i]:
                out["planes"].append(p_)
                out["scalars"].append(s_)
                out["legal"].append(l_)
                out["pi"].append(pi_)
                out["z"].append(rewards[i, mover] / rn.VALUE_SCALE)
                out["hand"].append(h_)
            pending[i] = []
            decisions[i] = 0
            finished += 1
            if started < n_hands:
                env.reset(int(i))
                started += 1

    examples = {
        "planes": np.asarray(out["planes"], dtype=np.uint8),
        "scalars": np.asarray(out["scalars"], dtype=np.float32),
        "legal": np.asarray(out["legal"], dtype=bool),
        "pi": np.asarray(out["pi"], dtype=np.float32),
        "z": np.asarray(out["z"], dtype=np.float32),
        "hand": np.asarray(out["hand"], dtype=np.uint8),
    }
    stats = {
        "hands": finished,
        "positions": len(out["z"]),
        "avg_turns": float(np.mean(turns)) if turns else 0.0,
        "capped": int(capped),
        "pile_take_rate": pile_takes / max(draws, 1),
    }
    return examples, stats
