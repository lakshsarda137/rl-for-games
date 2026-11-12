"""Solo game (RULES.md section 3) and duel (section 4).

In the duel each player sees only their own guesses and colors (W4.3), so each
player has their own list of answers that still fit their own colors
("candidates"). What a player knows is: their candidates, the turn number, and
whether they went first.
"""

import random

import numpy as np

from . import words

MAX_SOLO_GUESSES = 6  # W3.1
MAX_DUEL_GUESSES = 12  # W4.5
FAIL_SCORE = 7  # W3.3


class Game:
    """One secret word and the shared history of guesses. Used for both modes."""

    def __init__(self, secret=None, players=1, rng=random):
        self.secret = secret if secret is not None else rng.choice(words.ANSWERS)  # W1.1
        self.secret_index = words.ANSWERS.index(self.secret)
        self.players = players
        self.history = []  # list of (player, guess word, pattern number); the engine sees all of it
        self._candidates = [np.arange(len(words.ANSWERS)) for _ in range(players)]
        self.winner = None  # player number, or None
        self.over = False

    @property
    def to_play(self):
        return len(self.history) % self.players  # W4.2

    @property
    def candidates(self):
        """Answers that still fit the colors seen by the player whose turn it is."""
        return self._candidates[self.to_play]

    def candidates_of(self, player):
        return self._candidates[player]

    def own_history(self, player):
        """Only this player's guesses and colors (W4.3)."""
        return [(w, p) for who, w, p in self.history if who == player]

    @property
    def max_guesses(self):
        return MAX_SOLO_GUESSES if self.players == 1 else MAX_DUEL_GUESSES

    def guess(self, word):
        """Make a guess for the player whose turn it is. Returns the pattern number."""
        if self.over:
            raise ValueError("the game is over")
        word = word.lower()
        if word not in words.GUESS_INDEX:
            raise ValueError(f"'{word}' is not in the word list")  # W1.3
        g = words.GUESS_INDEX[word]
        pattern = int(words.table()[g, self.secret_index])
        player = self.to_play
        self.history.append((player, word, pattern))
        row = words.table()[g]
        mine = self._candidates[player]
        self._candidates[player] = mine[row[mine] == pattern]
        if pattern == words.ALL_GREEN:  # W3.2, W4.4
            self.winner, self.over = player, True
        elif len(self.history) >= self.max_guesses:  # W3.2, W4.5
            self.over = True
        return pattern

    def solo_score(self):
        """W3.3: guesses used, or 7 if the word was not found."""
        return len(self.history) if self.winner is not None else FAIL_SCORE

    def guesses_made(self, player):
        return sum(1 for who, _, _ in self.history if who == player)

    def duel_scores(self):
        """W5: one score per player, available once the game is over."""
        assert self.over and self.players == 2
        if self.winner is None:
            return [0, 0]  # W5.3
        scores = [0, 0]
        scores[self.winner] = win_points(self.guesses_made(self.winner))
        loser = 1 - self.winner
        scores[loser] = loss_points(len(self._candidates[loser]))
        return scores


def win_points(guesses_used):
    """W5.1"""
    return 7 - guesses_used


def loss_points(answers_left):
    """W5.2"""
    if answers_left <= 1:
        return -1
    if answers_left <= 10:
        return -2
    return -3
