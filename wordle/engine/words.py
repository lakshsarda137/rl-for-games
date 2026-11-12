"""Word lists and the color table (RULES.md sections 1 and 2).

The color table holds the color pattern (W2.5) for every allowed guess against
every possible answer: table[guess_index, answer_index]. It has 12,972 x 2,315
entries (30 MB), is built once in about a minute, and is saved to
data/feedback_table.npy. After that, every game step is a table lookup.
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
TABLE_PATH = os.path.join(DATA, "feedback_table.npy")

ALL_GREEN = 242


def _load(name):
    with open(os.path.join(DATA, name)) as f:
        return f.read().split()


ANSWERS = _load("answers.txt")
GUESSES = _load("guesses.txt")
assert GUESSES[: len(ANSWERS)] == ANSWERS, "W1.2: guesses.txt must start with the answers"
GUESS_INDEX = {w: i for i, w in enumerate(GUESSES)}


def feedback(guess, secret):
    """Reference colors for one guess, written to follow W2.1-W2.5 directly. Slow but simple."""
    colors = [0] * 5
    unmarked = {}
    for i in range(5):
        if guess[i] == secret[i]:
            colors[i] = 2
        else:
            unmarked[secret[i]] = unmarked.get(secret[i], 0) + 1
    for i in range(5):
        if colors[i] == 0 and unmarked.get(guess[i], 0) > 0:
            colors[i] = 1
            unmarked[guess[i]] -= 1
    return sum(c * 3**i for i, c in enumerate(colors))


def pattern_to_colors(p):
    """Turn a pattern number into a list such as ['gray', 'yellow', 'green', 'gray', 'gray']."""
    names = ("gray", "yellow", "green")
    return [names[(p // 3**i) % 3] for i in range(5)]


def _letters(words):
    return np.array([[ord(c) - 97 for c in w] for w in words], dtype=np.int8)


def build_table(guesses=GUESSES, answers=ANSWERS, chunk=500):
    """Fast version of feedback() for every pair, done on whole arrays at once."""
    g_all, a = _letters(guesses), _letters(answers)
    out = np.empty((len(guesses), len(answers)), dtype=np.uint8)
    powers = np.array([1, 3, 9, 27, 81], dtype=np.uint8)
    for start in range(0, len(guesses), chunk):
        g = g_all[start : start + chunk]
        green = g[:, None, :] == a[None, :, :]  # (G, A, 5)
        # Count each answer letter that is not green (W2.4).
        unmarked = np.zeros((len(g), len(answers), 26), dtype=np.int8)
        ai =np.broadcast_to(a[None, :, :], green.shape)
        gi_rows, ai_cols = np.indices(green.shape[:2])
        for i in range(5):
            not_green = ~green[:, :, i]
            unmarked[gi_rows[not_green], ai_cols[not_green], ai[:, :, i][not_green]] += 1
        colors = green.astype(np.uint8) * 2
        for i in range(5):  # left to right (W2.4)
            letter = np.broadcast_to(g[:, None, i], green.shape[:2])
            left = unmarked[gi_rows, ai_cols, letter]
            yellow = (~green[:, :, i]) & (left > 0)
            colors[:, :, i][yellow] = 1
            unmarked[gi_rows[yellow], ai_cols[yellow], letter[yellow]] -= 1
        out[start : start + chunk] = (colors * powers).sum(axis=2, dtype=np.uint16).astype(np.uint8)
    return out


_table = None


def table():
    """Load the color table, building and saving it the first time."""
    global _table
    if _table is None:
        if os.path.exists(TABLE_PATH):
            _table = np.load(TABLE_PATH)
        else:
            _table = build_table()
            np.save(TABLE_PATH, _table)
    return _table
