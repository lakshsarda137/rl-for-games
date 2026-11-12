"""Use a saved checkpoint to choose guesses in a normal engine Game (for play.py)."""

import numpy as np

from engine import words

from .lookahead import Position, guess_values


class AIPlayer:
    def __init__(self, checkpoint, device="cpu"):
        from .train_loop import load_net

        self.net, ck = load_net(checkpoint, device)
        self.device = device
        self.info = f"iteration {ck['state']['iteration']}, {ck['state']['games']:,} training games"
        self.notes = []  # (word, top choices) for each guess, to show after the game

    def choose(self, game, top_k=3):
        player = game.to_play
        pos = Position(np.asarray(game.candidates))
        q = guess_values(self.net, self.device, [pos], game.guesses_made(player), player == 0)[0]
        best = np.argsort(-q)[:top_k]
        word = words.GUESSES[int(pos.guesses[best[0]])]
        self.notes.append((word, pos.n, [(words.GUESSES[int(pos.guesses[i])], float(q[i])) for i in best]))
        return word
