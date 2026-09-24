"""How strong is the network? Play it against the greedy bot on duplicate deals.

Card games are mostly luck hand to hand, so a plain win count needs thousands
of hands to mean anything. Duplicate deals cancel most of the luck: every
shuffle is played twice with the seats swapped, so the network and the greedy
bot each get to play both sets of cards.

The main number is points per hand: the network's average points gained (+)
or lost (-) per hand against greedy. It comes with a 95% range: the true
average is very likely inside it.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "engine"))

from native import rummy_native as rn


def agent_actions(env, evaluator, turn, sims, worlds, c_puct, seed):
    """The network's move in every game where `turn` is true: the most visited
    move after searching, or with sims == 0 the network's top choice alone."""
    if sims > 0:
        visits, _, _ = env.search(evaluator, turn, worlds=worlds, sims=sims, c_puct=c_puct, seed=seed)
        scores = visits
    else:
        planes, scalars = env.observe()
        scores, _, _ = evaluator(planes, scalars, env.legal_mask())
    legal = env.legal_mask()
    scores = np.where(legal, scores + 1e-6, -1.0)   # ties go to any legal move
    return np.where(turn, scores.argmax(1), -1).astype(np.int32)


def vs_greedy(evaluator, pairs, seed, sims=0, worlds=8, c_puct=1.5, turn_cap=200):
    """Play `pairs` duplicate pairs (2 * pairs hands). Returns a results dict."""
    n = 2 * pairs
    env = rn.Env(n, seed=seed, turn_cap=turn_cap)
    agent_seat = np.tile([0, 1], pairs)
    deal_rng = np.random.default_rng(seed)
    for j in range(pairs):
        deal = int(deal_rng.integers(1 << 62))
        first = j % 2
        env.reset(2 * j, first=first, seed=deal)       # network plays seat 0's cards
        env.reset(2 * j + 1, first=first, seed=deal)   # same deal, network plays seat 1's
    points = np.zeros(n)
    step_rng = np.random.default_rng(seed + 1)
    while not env.done().all():
        done = env.done()
        turn = (env.to_move() == agent_seat) & ~done
        greedy = env.greedy_actions()
        mine = agent_actions(env, evaluator, turn, sims, worlds, c_puct, int(step_rng.integers(1 << 62)))
        rewards, _ = env.step(np.where(turn, mine, greedy).astype(np.int32))
        points += rewards[np.arange(n), agent_seat]

    wins = sum(1 for i in range(n) if env.state(i)["winner"] == agent_seat[i])
    capped = sum(1 for i in range(n) if env.state(i)["capped"])
    per_pair = points.reshape(pairs, 2).sum(1) / 2          # the luck-cancelled unit
    mean = float(per_pair.mean())
    half = float(1.96 * per_pair.std(ddof=1) / np.sqrt(pairs)) if pairs > 1 else float("nan")
    return {"hands": n, "points_per_hand": mean, "range_95": half,
            "win_rate": wins / n, "capped": capped}
