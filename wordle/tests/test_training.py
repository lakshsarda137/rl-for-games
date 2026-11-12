"""Checks for the training code: split scores, lookahead values, self-play scoring,
the strength check, checkpoints, and a tiny end-to-end training run."""

import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch

from engine import bots, words
from engine.game import Game, loss_points, win_points
from train import lookahead
from train.lookahead import Position, PositionCache, guess_values
from train.network import ValueNet, extras, predict
from train.selfplay import play_batch


def check_split_scores():
    rng = np.random.default_rng(0)
    for n in (2, 3, 7, 30, 96, 97, 400):
        c = np.sort(rng.choice(2315, size=n, replace=False))
        a, b = bots._split_scores_sorted(c), bots._split_scores_counted(c)
        assert np.allclose(a, b), n


class ConstNet(torch.nn.Module):
    """Predicts the same value for every position."""

    def __init__(self, v):
        super().__init__()
        self.v = v

    def forward(self, idx, off, ex):
        return torch.full((len(off),), float(self.v))


def check_lookahead():
    # W5.1 exact part: with 1 answer left, guessing it is worth 7 - (used + 1).
    pos = Position(np.array([words.ANSWERS.index("crane")]))
    q = guess_values(ConstNet(0.0), "cpu", [pos], used=2, first=True)[0]
    assert q[0] == win_points(3) == 4
    # With a network that always says 0, a guess's value is P(win now) * points.
    cands = np.array(sorted(words.ANSWERS.index(w) for w in ["light", "might", "night", "sight", "fight"]))
    pos = Position(cands)
    q = guess_values(ConstNet(0.0), "cpu", [pos], used=1, first=True)[0]
    for slot, g in enumerate(pos.guesses):
        expected = (1 / 5) * win_points(2) if g in cands else 0.0
        assert abs(q[slot] - expected) < 1e-9
    # With a network that always says v, the non-winning part adds (1 - P(win now)) * v.
    q2 = guess_values(ConstNet(1.5), "cpu", [pos], used=1, first=True)[0]
    for slot, g in enumerate(pos.guesses):
        expected = (1 / 5) * win_points(2) + (4 / 5) * 1.5 if g in cands else 1.5
        assert abs(q2[slot] - expected) < 1e-9
    # Second player's 6th guess: a miss is a draw, worth exactly 0.
    q3 = guess_values(ConstNet(9.0), "cpu", [pos], used=5, first=False)[0]
    for slot, g in enumerate(pos.guesses):
        assert abs(q3[slot] - ((1 / 5) * win_points(6) if g in cands else 0.0)) < 1e-9
    # Group sizes add up: every candidate is either a win or in exactly one group.
    for slot in range(len(pos.guesses)):
        assert pos.group_size[pos.group_slot == slot].sum() + pos.is_answer[slot] == 5
    # The best_split choice is always on the short list.
    full = Position(np.arange(2315))
    assert words.GUESSES[full.split_guess] == "soare" and full.guesses[0] == full.split_guess


def check_selfplay_matches_engine():
    """Batched self-play must give the same games and scores as the plain engine."""
    cache = PositionCache()
    secrets = np.random.default_rng(3).choice(2315, size=60, replace=False)
    res = play_batch(("split", "split"), secrets, cache, trace_games=tuple(range(60)))
    for g, s in enumerate(secrets):
        game = Game(secret=words.ANSWERS[s], players=2)
        while not game.over:
            game.guess(bots.best_split(game))
        assert [w for _, w, _ in game.history] == [r["word"] for r in res.traces[g]], words.ANSWERS[s]
        assert list(res.scores[g]) == game.duel_scores()
    # Both seats see only their own colors, so with the same bot they make the same guesses.
    tr = res.traces[0]
    assert [r["word"] for r in tr if r["seat"] == 0][:len([r for r in tr if r["seat"] == 1])] == \
        [r["word"] for r in tr if r["seat"] == 1]


def check_samples_and_scores():
    cache = PositionCache()
    net = ValueNet(32)
    net.train(False)
    rng = np.random.default_rng(1)
    secrets = rng.integers(2315, size=50)
    res = play_batch(("ai", "ai"), secrets, cache, net, "cpu", rng, 0.25, 0.03, (True, True))
    for g in range(50):
        w = res.winner[g]
        if w >= 0:
            assert res.scores[g, w] == win_points(res.win_guesses[g])
            assert res.scores[g, 1 - w] in (-1, -2, -3)
        else:
            assert list(res.scores[g]) == [0, 0]
    for m, e, g, s in res.samples:
        assert len(m) >= 1 and e.shape == (11,) and 1 <= np.argmax(e[:7]) <= 6
    assert len(predict(net, [m for m, _, _, _ in res.samples[:10]], [e for _, e, _, _ in res.samples[:10]], "cpu")) == 10


def check_tiny_run():
    from train import train_loop

    out = tempfile.mkdtemp()
    try:
        train_loop.main(["--tiny", "--no-wandb", "--out", out, "--iterations", "4", "--eval-every", "2"])
        for f in ("latest.pt", "best.pt", "ckpt_00002.pt", "ckpt_00004.pt", "progress.html", "progress.txt",
                  "samples.txt", "metrics.jsonl"):
            assert os.path.exists(os.path.join(out, f)), f
        net, ck = train_loop.load_net(os.path.join(out, "latest.pt"))
        assert ck["state"]["iteration"] == 4
        # Resuming continues from the checkpoint instead of starting over.
        train_loop.main(["--tiny", "--no-wandb", "--out", out, "--iterations", "5", "--eval-every", "0",
                         "--resume", "auto"])
        _, ck = train_loop.load_net(os.path.join(out, "latest.pt"))
        assert ck["state"]["iteration"] == 5 and ck["state"]["games"] == 5 * 64
    finally:
        shutil.rmtree(out)


if __name__ == "__main__":
    t0 = time.time()
    check_split_scores()
    check_lookahead()
    check_selfplay_matches_engine()
    check_samples_and_scores()
    check_tiny_run()
    print(f"all training checks passed in {time.time() - t0:.1f}s")
