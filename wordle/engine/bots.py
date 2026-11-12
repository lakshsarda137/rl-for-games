"""Two simple players with no training. They are the comparison points for the trained AI.

random_candidate: guesses a random word that could still be the answer.
best_split: tries every allowed guess and picks the one that splits the remaining
    answers into the most even groups of color patterns (highest "entropy", a
    measure of how much a guess is expected to narrow down the answer). If two
    guesses are equal, it prefers one that could be the answer.
"""

import random

import numpy as np

from . import words

_first_guess = None


def random_candidate(game, rng=random):
    return words.ANSWERS[rng.choice(list(game.candidates))]


def split_scores(candidates):
    """Entropy, in bits, of every allowed guess against the given candidate answers.

    entropy = log2(n) - (1/n) * sum over color groups of size*log2(size).
    Small lists use sorting (cost grows with n); large lists use counting.
    """
    n = len(candidates)
    if n <= 96:
        return _split_scores_sorted(candidates)
    return _split_scores_counted(candidates)


def _split_scores_sorted(candidates):
    t = words.table()
    n = len(candidates)
    rows = t.shape[0]
    flat = np.sort(t[:, candidates], axis=1).astype(np.int32)
    flat += (np.arange(rows, dtype=np.int32) * 256)[:, None]
    flat = flat.ravel()
    starts = np.r_[0, np.flatnonzero(flat[1:] != flat[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(flat)]).astype(np.float64)
    row_of = flat[starts] // 256
    s = np.bincount(row_of, weights=sizes * np.log2(sizes), minlength=rows)
    return np.log2(n) - s / n


def _split_scores_counted(candidates):
    t = words.table()
    n = len(candidates)
    cols = t[:, candidates].astype(np.int64)
    cols += 243 * np.arange(t.shape[0])[:, None]
    counts = np.bincount(cols.ravel(), minlength=243 * t.shape[0]).reshape(t.shape[0], 243)
    p = counts / n
    with np.errstate(divide="ignore", invalid="ignore"):
        return -(np.where(p > 0, p * np.log2(p), 0.0)).sum(axis=1)


def best_split(game):
    global _first_guess
    cands = game.candidates
    if len(cands) <= 2:
        return words.ANSWERS[cands[0]]
    if len(cands) == len(words.ANSWERS) and _first_guess is not None:
        return _first_guess
    score = split_scores(cands)
    score[cands] += 1e-6  # tie-break toward words that could be the answer
    word = words.GUESSES[int(np.argmax(score))]
    if len(cands) == len(words.ANSWERS):
        _first_guess = word
    return word
