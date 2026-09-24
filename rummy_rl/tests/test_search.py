"""Tests for the AI: imagined hidden cards, search, network, training step.
Run: python tests/test_search.py
"""

import os
import sys
import tempfile
from dataclasses import replace

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "engine"))
sys.path.insert(0, os.path.join(HERE, "..", "az"))
sys.path.insert(0, os.path.join(HERE, "..", "run"))

import numpy as np
import torch

from hand import parse_card
from native import rummy_native as rn

from harness import check, run


def cards(text):
    return [parse_card(t) for t in text.split()]


def rest_of_deck(*used):
    return sorted(set(range(52)) - set().union(*map(set, used)))


def uniform_evaluator(planes, scalars, legal):
    """Knows nothing: every legal move equally likely, every result 0."""
    priors = legal / legal.sum(1, keepdims=True)
    return priors.astype(np.float32), np.zeros(len(legal), np.float32), np.ones((len(legal), 52), np.float32)


# --------------------------------------------------------------------------- #
# Imagined hidden cards
# --------------------------------------------------------------------------- #

def test_worlds_keep_what_you_can_see():
    env = rn.Env(64, seed=8, turn_cap=0)
    rng = np.random.default_rng(0)
    checked = reshuffled = 0
    for step in range(900):
        env.step(np.where(np.arange(64) % 3 == 0, env.random_actions(seed=step), env.greedy_actions()))
        for i in np.flatnonzero(env.done()):
            env.reset(int(i))
        if step % 15:
            continue
        for i in range(64):
            s = env.state(i)
            if s["phase"] == "over":
                continue
            p, o = s["to_move"], 1 - s["to_move"]
            w = env.sample_world(i, [], int(rng.integers(1 << 62)))
            placed = w["hands"][0] + w["hands"][1] + w["draw_pile"] + w["discard_pile"]
            ok = (w["hands"][p] == s["hands"][p] and w["discard_pile"] == s["discard_pile"]
                  and set(s["held_known"][o]) <= set(w["hands"][o])
                  and len(w["hands"][o]) == len(s["hands"][o]) and len(w["draw_pile"]) == len(s["draw_pile"])
                  and sorted(placed) == list(range(52)))
            if s["reshuffles"]:
                reshuffled += 1
                ok &= set(w["draw_pile"]) <= set(s["last_recycled"])
            if not ok:
                check(f"imagined world breaks what player {p} can see in game {i}", False)
            checked += 1
    check(f"{checked} imagined worlds keep your hand, the discard pile and cards seen taken", checked > 500)
    check(f"after a reshuffle the draw pile only holds reshuffled cards ({reshuffled} worlds)", reshuffled > 20)


def test_worlds_follow_the_hand_guess():
    h0 = cards("2S 4S 6S 8S 10S QS 2H 4H 6H 8H 10H QH 2D")
    h1 = cards("3S 5S 7S 9S JS KS 3H 5H 7H 9H JH KH 3D")
    pile = cards("5D")
    env = rn.Env(1, seed=0, turn_cap=0)
    env.load(0, h0, h1, rest_of_deck(h0, h1, pile), pile, to_move=0)
    likely, unlikely = parse_card("AC"), parse_card("KC")
    weights = [1.0] * 52
    weights[likely], weights[unlikely] = 1000.0, 0.0
    hits = sum(likely in env.sample_world(0, weights, s)["hands"][1] for s in range(200))
    misses = sum(unlikely in env.sample_world(0, weights, s)["hands"][1] for s in range(200))
    check(f"a card guessed very likely is usually dealt to the opponent ({hits}/200)", hits > 190)
    check(f"a card guessed unlikely rarely is ({misses}/200)", misses < 10)


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #

def test_search_counts():
    env = rn.Env(6, seed=3, turn_cap=0)
    env.step(np.full(6, rn.ACT_STOCK, np.int32))           # everyone to the throw phase
    active = np.array([True, True, True, False, True, True])
    visits, values, guess = env.search(uniform_evaluator, active, worlds=3, sims=10, seed=1)
    legal = env.legal_mask()
    check("visits only on legal moves", not (visits[~legal] > 0).any())
    check("each searched game gets worlds x sims visits", np.allclose(visits[active].sum(1), 30))
    check("games not asked for are left alone", visits[3].sum() == 0)
    check("returns the network's hand guess for each searched game", guess.shape == (6, 52) and guess[0].min() == 1)


def test_search_takes_a_winning_card():
    h0 = cards("6H 7H 8H 9H 5S 5D 5C 8C 9C JC QC KC 2S")   # needs 10C
    h1 = cards("AS 3S 4S 6S 7S 8S 9S 10S JS QS KS AH 2H")
    pile = cards("10C")
    env = rn.Env(1, seed=0, turn_cap=0)
    env.load(0, h0, h1, rest_of_deck(h0, h1, pile), pile, to_move=0)
    visits, _, _ = env.search(uniform_evaluator, np.array([True]), worlds=4, sims=16, seed=2)
    check("with no knowledge at all, search still takes the card that wins at once",
          int(visits[0].argmax()) == rn.ACT_PILE)


def test_search_is_reproducible():
    a, b = rn.Env(4, seed=9), rn.Env(4, seed=9)
    va, _, _ = a.search(uniform_evaluator, np.ones(4, bool), worlds=2, sims=8, dir_eps=0.25, seed=5)
    vb, _, _ = b.search(uniform_evaluator, np.ones(4, bool), worlds=2, sims=8, dir_eps=0.25, seed=5)
    check("same seed, same search", np.array_equal(va, vb))


# --------------------------------------------------------------------------- #
# Network and training
# --------------------------------------------------------------------------- #

def test_network_shapes():
    from network import Evaluator, RummyNet, fold_ace
    net = RummyNet(blocks=1, channels=8)
    env = rn.Env(5, seed=1)
    planes, scalars = env.observe()
    policy, value, hand = net(torch.from_numpy(planes), torch.from_numpy(scalars))
    check("network outputs 54 move scores, 1 result, 52 hand guesses",
          policy.shape == (5, 54) and value.shape == (5,) and hand.shape == (5, 52))
    grid = torch.zeros(1, 4, 14)
    grid[0, 2, 13] = 1.0     # high ace of diamonds
    check("the high-ace column counts toward the ace", fold_ace(grid)[0, 2 * 13 + 0] == 1.0)
    priors, values, guess = Evaluator(net)(planes, scalars, env.legal_mask())
    legal = env.legal_mask()
    check("evaluator gives no weight to illegal moves and sums to 1",
          priors[~legal].max() == 0 and np.allclose(priors.sum(1), 1, atol=1e-5))
    check("values are in [-1, 1] and guesses are probabilities",
          np.abs(values).max() <= 1 and 0 <= guess.min() and guess.max() <= 1)


def test_training_round_trip():
    from config import Config
    from evaluate import vs_greedy
    from network import Evaluator, RummyNet
    from selfplay import play
    from train import ReplayBuffer, train_steps
    from train_loop import load_net, save

    cfg = replace(Config.tiny(), blocks=1, channels=8, worlds=2, sims=3, selfplay_games=4,
                  turn_cap=30, batch_size=32, steps_per_iter=3, device="cpu")
    net = RummyNet(cfg.blocks, cfg.channels)
    examples, stats = play(Evaluator(net), cfg, n_hands=4, seed=0)
    n = stats["positions"]
    check(f"self-play records every decision ({n} positions from {stats['hands']} hands)",
          stats["hands"] == 4 and all(len(v) == n for v in examples.values()))
    check("move targets are probabilities over legal moves",
          np.allclose(examples["pi"].sum(1), 1) and not (examples["pi"][~examples["legal"]] > 0).any())
    check("hand answers are 13-card hands", (examples["hand"].sum(1) == 13).all())
    check("result targets are in [-1, 1]", np.abs(examples["z"]).max() <= 1)

    buf = ReplayBuffer(1000)
    buf.add(examples)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    losses = train_steps(net, buf, opt, cfg, np.random.default_rng(0))
    check("a training step runs with all three losses finite",
          losses["steps"] == 3 and all(np.isfinite(losses[k]) for k in ("policy", "value", "hand")))

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.pt")
        save(path, net, opt, cfg, 1)
        loaded, ckpt = load_net(path)
        same = all(torch.equal(a, b) for a, b in zip(net.state_dict().values(), loaded.state_dict().values()))
        check("a saved checkpoint loads back identically", same and ckpt["iteration"] == 1)

    ev = vs_greedy(Evaluator(net), pairs=3, seed=4, sims=0, turn_cap=60)
    check("the greedy check plays duplicate pairs and reports points per hand",
          ev["hands"] == 6 and np.isfinite(ev["points_per_hand"]))


if __name__ == "__main__":
    run([test_worlds_keep_what_you_can_see, test_worlds_follow_the_hand_guess, test_search_counts,
         test_search_takes_a_winning_card, test_search_is_reproducible, test_network_shapes,
         test_training_round_trip],
        [], title="AI (search, network, training)")
