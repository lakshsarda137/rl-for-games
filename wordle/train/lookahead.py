"""How the AI picks a guess: exact one-guess lookahead plus the value network.

For a position (the answers that still fit the player's colors):
1. Make a short list of guesses: the SHORT_INFO guesses that split the answers
   most evenly, plus the SHORT_CAND best-splitting guesses that could be the
   answer. The best_split bot's choice is always on the list.
2. For each guess on the list, split the answers into groups by the colors the
   guess would get. Each group has a known chance (its size divided by the
   number of answers), because every remaining answer is equally likely.
3. The all-green group is a win now, worth W5.1 points exactly. Every other
   group is scored by the value network.
4. The guess's expected score is the chance-weighted sum. Pick the best one
   (with some randomness during training).

The work in steps 1 and 2 depends only on the answer list, so it is saved in
a cache and reused whenever the same list appears again.
"""

import numpy as np

from engine import bots, words
from engine.game import win_points

from .network import extras, predict

SHORT_INFO = 24
SHORT_CAND = 16
ALL_GREEN = words.ALL_GREEN


class Position:
    """Everything about one answer list that does not depend on the network."""

    __slots__ = ("cands", "n", "split_guess", "guesses", "is_answer", "group_members", "group_slot", "group_size")

    def __init__(self, cands):
        t = words.table()
        self.cands = cands
        self.n = n = len(cands)
        if n == 1:
            self.split_guess = int(cands[0])
            self.guesses = np.array([cands[0]])
            self.is_answer = np.array([True])
            self.group_members, self.group_slot, self.group_size = [], np.zeros(0, int), np.zeros(0, int)
            return
        score = bots.split_scores(cands)
        score[cands] += 1e-6  # same tie-break as bots.best_split
        self.split_guess = int(cands[0]) if n <= 2 else int(np.argmax(score))
        k = min(SHORT_INFO, len(score))
        top = np.argpartition(-score, k - 1)[:k]
        top = top[np.argsort(-score[top], kind="stable")]
        cand_top = cands[np.argsort(-score[cands], kind="stable")[:SHORT_CAND]]
        seen, order = set(), []
        for g in [self.split_guess, *cand_top.tolist(), *top.tolist()]:
            if g not in seen:
                seen.add(g)
                order.append(int(g))
        self.guesses = np.array(order)
        cand_set = set(cands.tolist())
        self.is_answer = np.array([g in cand_set for g in order])
        members, slots, sizes = [], [], []
        for slot, g in enumerate(order):
            pats = t[g, cands]
            srt = np.argsort(pats, kind="stable")
            p_sorted = pats[srt]
            cuts = np.flatnonzero(np.diff(p_sorted)) + 1
            for grp, first_pat in zip(np.split(srt, cuts), p_sorted[np.r_[0, cuts]]):
                if first_pat != ALL_GREEN:
                    members.append(cands[grp])
                    slots.append(slot)
                    sizes.append(len(grp))
        self.group_members, self.group_slot, self.group_size = members, np.array(slots), np.array(sizes)


class PositionCache:
    def __init__(self, limit=200_000):
        self.limit = limit
        self.store = {}
        self.hits = self.misses = 0

    def get(self, cands):
        key = cands.astype(np.int16).tobytes()
        pos = self.store.get(key)
        if pos is None:
            self.misses += 1
            if len(self.store) >= self.limit:
                self.store.clear()
            pos = self.store[key] = Position(cands)
        else:
            self.hits += 1
        return pos


def guess_values(net, device, positions, used, first):
    """Expected final score of every short-list guess, for a list of positions
    that all have the same guesses used and the same seat."""
    members, rows, owners = [], [], []
    exact_draw = (not first) and used + 1 == 6  # second player's 6th guess: if it misses, the game is a draw
    for i, pos in enumerate(positions):
        if exact_draw:
            continue
        for m in pos.group_members:
            members.append(m)
            rows.append(extras(len(m), used + 1, first))
            owners.append(i)
    values = predict(net, members, rows, device) if members else np.zeros(0, np.float32)
    out, start = [], 0
    for i, pos in enumerate(positions):
        q = pos.is_answer.astype(np.float64) * win_points(used + 1) / pos.n
        g = len(pos.group_members)
        if not exact_draw and g:
            v = values[start : start + g]
            start += g
            np.add.at(q, pos.group_slot, pos.group_size * v / pos.n)
        out.append(q)
    return out


def choose(q, rng, temperature=0.0, epsilon=0.0):
    """Index of the chosen guess: best value, or a random pick while exploring."""
    if epsilon and rng.random() < epsilon:
        return int(rng.integers(len(q)))
    if temperature > 0:
        z = (q - q.max()) / temperature
        p = np.exp(z)
        return int(rng.choice(len(q), p=p / p.sum()))
    return int(np.argmax(q))
