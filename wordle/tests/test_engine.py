"""Checks for the word lists, colors, games and simple bots. --full checks every table entry."""

import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from engine import bots, words
from engine.game import Game

FULL = "--full" in sys.argv
G, Y, X = "green", "yellow", "gray"


def colors(guess, secret):
    return words.pattern_to_colors(words.feedback(guess, secret))


def check_lists():  # W1.1, W1.2
    assert len(words.ANSWERS) == 2315 and len(words.GUESSES) == 12972
    assert len(set(words.GUESSES)) == 12972
    assert all(len(w) == 5 and w.isalpha() and w.islower() for w in words.GUESSES)


def check_colors():  # W2.1-W2.5
    assert colors("geese", "those") == [X, X, X, G, G]
    assert colors("kebab", "abbey") == [X, Y, G, Y, Y]
    assert colors("bobby", "abbey") == [Y, X, G, X, G]
    assert colors("speed", "erase") == [Y, X, Y, Y, X]
    assert colors("crane", "crane") == [G] * 5 and words.feedback("crane", "crane") == 242
    assert colors("fuzzy", "crane") == [X] * 5 and words.feedback("fuzzy", "crane") == 0


def check_table():
    t = words.table()
    rng = random.Random(1)
    pairs = [(g, a) for g in range(len(words.GUESSES)) for a in range(len(words.ANSWERS))] if FULL else [
        (rng.randrange(len(words.GUESSES)), rng.randrange(len(words.ANSWERS))) for _ in range(200_000)
    ]
    for g, a in pairs:
        assert t[g, a] == words.feedback(words.GUESSES[g], words.ANSWERS[a]), (words.GUESSES[g], words.ANSWERS[a])
    return len(pairs)


def check_solo():  # W3
    g = Game(secret="crane")
    try:
        g.guess("zzzzz")
        raise AssertionError("bad word accepted")
    except ValueError:
        pass
    assert len(g.history) == 0  # W1.3
    g.guess("slate")
    assert words.ANSWERS.index("crane") in g.candidates
    g.guess("crane")
    assert g.over and g.winner == 0 and g.solo_score() == 2
    g = Game(secret="crane")
    for w in ["fuzzy", "plumb", "dight", "vowed", "jumpy", "piggy"]:
        g.guess(w)
    assert g.over and g.winner is None and g.solo_score() == 7


def check_duel():  # W4
    g = Game(secret="crane", players=2)
    assert g.to_play == 0
    g.guess("slate")
    assert g.to_play == 1
    # W4.3: player 0's colors narrow only player 0's list, not player 1's.
    assert len(g.candidates_of(0)) < len(words.ANSWERS)
    assert len(g.candidates) == len(g.candidates_of(1)) == len(words.ANSWERS)
    assert g.own_history(1) == [] and g.own_history(0)[0][0] == "slate"
    g.guess("crane")
    assert g.over and g.winner == 1
    assert g.duel_scores() == [-3, 6]  # W5.1: player 1 won with 1 guess; W5.2: player 0 had many answers left
    g = Game(secret="crane", players=2)
    for _ in range(6):
        g.guess("fuzzy")
        g.guess("plumb")
    assert g.over and g.winner is None
    assert g.duel_scores() == [0, 0]  # W5.3


def check_duel_scores():  # W5
    from engine.game import loss_points, win_points

    assert [win_points(n) for n in (1, 2, 4, 6)] == [6, 5, 3, 1]
    assert [loss_points(n) for n in (1, 2, 10, 11, 2315)] == [-1, -2, -2, -3, -3]
    # Player 0 guesses slate, then crane (wins with 2 guesses). Player 1 guesses fuzzy once.
    g = Game(secret="crane", players=2)
    g.guess("slate")
    g.guess("fuzzy")
    g.guess("crane")
    left = len(g.candidates_of(1))
    assert g.duel_scores() == [5, loss_points(left)] and left > 10


def check_bots():
    first = bots.best_split(Game(secret="crane"))
    rng = random.Random(2)
    secrets = rng.sample(words.ANSWERS, 2315 if FULL else 300)
    scores = []
    for s in secrets:
        g = Game(secret=s)
        while not g.over:
            g.guess(bots.best_split(g))
        scores.append(g.solo_score())
    return first, float(np.mean(scores)), max(scores)


if __name__ == "__main__":
    t0 = time.time()
    check_lists()
    check_colors()
    n = check_table()
    check_solo()
    check_duel()
    check_duel_scores()
    first, mean, worst = check_bots()
    assert worst <= 6, f"best_split failed a word (worst {worst})"
    assert mean < 3.7, mean
    print(f"best_split: first guess {first}, average {mean:.3f} guesses, worst {worst}")
    print(f"all checks passed ({n:,} table entries compared) in {time.time() - t0:.1f}s")
