"""Tests for the C++ engine (native/rummy_native.cpp). Run: python tests/test_native.py [--full]

Three anchors:
  * the C++ scorer must match engine/hand.py (the reference) on every hand;
  * one test per game rule in RULES.md, on hand-built positions (Env.load);
  * thousands of bot-vs-bot hands with every rule checked after every move.
"""

import os
import random
import sys
import time

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "engine"))

import numpy as np

from hand import best_discard, parse_card, score_hand
from native import rummy_native as rn

from harness import check, run
from test_hand import _meld_rich_hand

ALL = set(range(52))


def cards(text):
    return [parse_card(t) for t in text.split()]


def rest_of_deck(*used):
    """Every card not in `used`, as a list (used as the draw pile)."""
    taken = set().union(*map(set, used))
    return sorted(ALL - taken)


# --------------------------------------------------------------------------- #
# Scorer parity
# --------------------------------------------------------------------------- #

def _parity(n, seed):
    rng = random.Random(seed)
    bad = []
    for i in range(n):
        m = _meld_rich_hand(rng, 13) if i % 2 else sum(1 << c for c in rng.sample(range(52), 13))
        if rn.score_hand(m) != score_hand(m):
            bad.append(m)
    for _ in range(n // 4):
        m = _meld_rich_hand(rng, 14)
        if rn.best_discard(m) != best_discard(m):
            bad.append(m)
    check(f"C++ scorer matches the Python reference on {n + n // 4} hands", not bad)


def test_scorer_parity():
    _parity(8000, seed=11)


def test_scorer_rejects_bad_input():
    for name, fn, mask in [("12 cards", rn.score_hand, (1 << 12) - 1),
                           ("bits above 51", rn.score_hand, ((1 << 12) - 1) | (1 << 60)),
                           ("13 cards to best_discard", rn.best_discard, (1 << 13) - 1)]:
        try:
            fn(mask)
            ok = False
        except ValueError:
            ok = True
        check(f"rejects {name}", ok)


# --------------------------------------------------------------------------- #
# Game rules
# --------------------------------------------------------------------------- #

def test_deal():
    env = rn.Env(200, seed=5)
    firsts = set()
    for i in range(len(env)):
        s = env.state(i)
        h0, h1 = s["hands"]
        placed = h0 + h1 + s["draw_pile"] + s["discard_pile"]
        if not (len(h0) == len(h1) == 13 and len(s["discard_pile"]) == 1 and len(s["draw_pile"]) == 25
                and sorted(placed) == list(range(52)) and s["phase"] == "draw"):
            check(f"R2.1-R2.3 deal in game {i}", False)
        firsts.add(s["to_move"])
    check("R2.1-R2.3: 13 cards each, 1 face up, 25 in the draw pile, all 52 once", True)
    check("R2.4: either player can start", firsts == {0, 1})
    env.reset(0, first=1)
    check("R2.4: first player can be chosen", env.state(0)["to_move"] == 1)


def test_reproducible():
    a, b = rn.Env(8, seed=42), rn.Env(8, seed=42)
    for _ in range(60):
        acts = a.greedy_actions()
        a.step(acts)
        b.step(b.greedy_actions())
    check("same seed, same games", all(a.state(i) == b.state(i) for i in range(8)))
    c = rn.Env(8, seed=43)
    check("different seed, different deal", a.state(0)["hands"] != c.state(0)["hands"] or
          rn.Env(8, seed=42).state(0)["hands"] != c.state(0)["hands"])


def test_draw_and_throw():
    env = rn.Env(1, seed=0, turn_cap=0)
    h0 = cards("2S 4S 6S 8S 10S QS 2H 4H 6H 8H 10H QH 2D")
    h1 = cards("3S 5S 7S 9S JS KS 3H 5H 7H 9H JH KH 3D")
    pile = cards("5D")
    draw = rest_of_deck(h0, h1, pile)
    env.load(0, h0, h1, draw, pile, to_move=0)
    top_of_draw = draw[-1]
    check("draw phase: only the two draws are legal",
          list(np.flatnonzero(env.legal_mask()[0])) == [rn.ACT_STOCK, rn.ACT_PILE])
    env.step(np.array([rn.ACT_STOCK]))
    s = env.state(0)
    check("R3.1: drawing from the draw pile takes its top card", top_of_draw in s["hands"][0] and s["drawn"] == top_of_draw)
    check("discard phase: exactly the 14 cards in hand are legal",
          sorted(np.flatnonzero(env.legal_mask()[0])) == sorted(s["hands"][0]))
    try:
        env.step(np.array([h1[0]]))
        ok = False
    except ValueError:
        ok = True
    check("throwing a card you don't hold is refused", ok)
    env.step(np.array([parse_card("QS")]))
    s = env.state(0)
    check("R3.2: the thrown card goes on top of the discard pile", s["discard_pile"][-1] == parse_card("QS"))
    check("turn passes to the other player", s["to_move"] == 1 and s["phase"] == "draw" and s["turn"] == 1)
    check("everyone can see what was thrown", parse_card("QS") in s["discarded_by"][0])

    # R3.3: take the top discard and throw it straight back.
    env.step(np.array([rn.ACT_PILE]))
    s = env.state(0)
    qs = parse_card("QS")
    check("taking the discard is public knowledge", qs in s["held_known"][1] and qs in s["taken_by"][1])
    env.step(np.array([qs]))
    s = env.state(0)
    check("R3.3: the taken card can be thrown back at once",
          s["discard_pile"][-1] == qs and qs not in s["held_known"][1] and not env.validate())


def test_auto_declare():
    # Player 0 needs one card: 6-9H run, 5S 5D 5C, 8C 9C 10C, J-Q-K of clubs... all but 10C.
    h0 = cards("6H 7H 8H 9H 5S 5D 5C 8C 9C JC QC KC 2S")
    h1 = cards("AS 3S 4S 6S 7S 8S 9S 10S JS QS KS AH 2H")
    pile = cards("10C")  # the missing card, on top of the discard pile
    env = rn.Env(1, seed=0, turn_cap=0)
    env.load(0, h0, h1, rest_of_deck(h0, h1, pile), pile, to_move=0)
    rewards, done = env.step(np.array([rn.ACT_PILE]))
    s = env.state(0)
    want_points = score_hand(sum(1 << c for c in h1))[1]
    check("R3.4: a winning draw declares automatically", bool(done[0]) and s["phase"] == "over" and s["winner"] == 0)
    check("R3.4: the engine throws the spare card", s["declare_card"] == parse_card("2S") and s["discard_pile"][-1] == parse_card("2S"))
    check("R6.1/R6.2: loser scores their hand, winner 0", s["points"] == want_points)
    check("T9.2: reward is +points for the winner, -points for the loser",
          list(rewards[0]) == [want_points, -want_points])
    check("nothing is legal after the hand ends", not env.legal_mask()[0].any())


def test_dealt_win_needs_a_draw():
    # R3.5: player 0 is dealt a complete hand; it only ends after they draw.
    h0 = cards("2H 3H 4H 5H 7S 7D 7C 9S 9D 9C JS JD JC")
    h1 = cards("AS 2S 3S 4S 5S 6S 8S 10S QS KS AH 6H 8H")
    pile = cards("KC")
    env = rn.Env(1, seed=0, turn_cap=0)
    env.load(0, h0, h1, rest_of_deck(h0, h1, pile), pile, to_move=0)
    check("R3.5: a dealt valid hand has not won yet", env.state(0)["phase"] == "draw" and env.state(0)["winner"] == -1)
    env.step(np.array([rn.ACT_STOCK]))
    s = env.state(0)
    check("R3.5: it declares right after the draw", s["winner"] == 0 and s["points"] > 0)


def test_reshuffle():
    # R4: empty draw pile; 26 cards in the discard pile, top card QD.
    h0 = cards("2S 4S 6S 8S 10S QS 2H 4H 6H 8H 10H QH 2D")
    h1 = cards("3S 5S 7S 9S JS KS 3H 5H 7H 9H JH KH 3D")
    pile = rest_of_deck(h0, h1, cards("QD")) + cards("QD")
    env = rn.Env(1, seed=0, turn_cap=0)
    env.load(0, h0, h1, [], pile, to_move=0)
    env.step(np.array([rn.ACT_STOCK]))
    s = env.state(0)
    check("R4.1: an empty draw pile is rebuilt from the discards", s["reshuffles"] == 1 and len(s["draw_pile"]) == 24)
    check("R4.2: the top discard stays put", s["discard_pile"] == cards("QD"))
    check("R8.2: everyone knows which cards were shuffled back", sorted(s["recycled"]) == sorted(pile[:-1]))
    check("rules still hold after a reshuffle", not env.validate())


def test_turn_cap():
    env = rn.Env(16, seed=9, turn_cap=6)
    total = np.zeros((16, 2))
    for _ in range(12):
        r, _ = env.step(env.random_actions(seed=1))
        total += r
    states = [env.state(i) for i in range(16)]
    capped = [s for s in states if s["capped"]]
    check("T9.1: the turn cap ends hands", len(capped) > 0 and all(s["turn"] == 6 for s in capped))
    check("T9.1: a capped hand is a draw with no points",
          all(s["winner"] == -1 for s in capped) and
          all(total[i].sum() == 0 for i in range(16)))


def test_observation_hides_opponent():
    h0 = cards("2S 4S 6S 8S 10S QS 2H 4H 6H 8H 10H QH 2D")
    h1 = cards("3S 5S 7S 9S JS KS 3H 5H 7H 9H JH KH 3D")
    pile = cards("5D")
    draw = rest_of_deck(h0, h1, pile)
    env = rn.Env(2, seed=0, turn_cap=0)
    env.load(0, h0, h1, draw, pile, to_move=0)
    # Same position, but the opponent's unseen KH is swapped with a card deep in the draw pile.
    swapped_h1 = [c if c != parse_card("KH") else draw[0] for c in h1]
    swapped_draw = [parse_card("KH")] + draw[1:]
    env.load(1, h0, swapped_h1, swapped_draw, pile, to_move=0)
    planes, scalars = env.observe()
    check("R8: observation shape", planes.shape == (2, rn.N_PLANES, 52) and scalars.shape == (2, rn.N_SCALARS))
    check("R8.3: the opponent's hidden cards don't change what you see",
          np.array_equal(planes[0], planes[1]) and np.array_equal(scalars[0], scalars[1]))
    check("R8.1: plane 0 is exactly your own hand", sorted(np.flatnonzero(planes[0, 0])) == sorted(h0))
    check("top of the discard pile is shown", list(np.flatnonzero(planes[0, 1])) == pile)

    # After the opponent takes a discard, you know they hold it.
    env.step(np.array([rn.ACT_STOCK, rn.ACT_STOCK]))
    env.step(np.array([parse_card("QS"), parse_card("QS")]))
    env.step(np.array([rn.ACT_PILE, rn.ACT_PILE]))  # opponent takes QS
    env.step(np.array([parse_card("3D"), parse_card("3D")]))
    planes, _ = env.observe()
    check("R8.2: a card the opponent took from the pile is shown as theirs",
          planes[0, 3, parse_card("QS")] == 1 and planes[0, 5, parse_card("QS")] == 1)
    check("R8.2: a card the opponent threw is shown", planes[0, 4, parse_card("3D")] == 1)


def test_bots_follow_rules():
    env = rn.Env(256, seed=21, turn_cap=200)
    total = 0
    for b0, b1 in [("random", "random"), ("greedy", "random"), ("greedy", "greedy")]:
        finished, turns, w0, w1, capped, net0, violations = env.play(1500, b0, b1, check_rules=True)
        total += violations
        check(f"{b0} vs {b1}: 1500 hands, rules checked after every move ({violations} broken)",
              finished == 1500 and violations == 0)
    finished, turns, w0, w1, capped, net0, _ = env.play(2000, "greedy", "random")
    check(f"greedy beats random ({w0} of {finished} hands)", w0 >= 0.95 * finished)


def test_python_stepping_matches_rules():
    # Same stress as above but through step(), the path the RL training will use.
    env = rn.Env(64, seed=4, turn_cap=200)
    for _ in range(1500):
        acts = np.where(np.arange(64) % 2 == 0, env.greedy_actions(), env.random_actions(seed=_))
        env.step(acts)
        errs = env.validate()
        if errs:
            check(f"step() keeps the rules: {errs[:3]}", False)
        for i in np.flatnonzero(env.done()):
            env.reset(int(i))
    check("1500 batched steps of 64 games keep every rule", True)


def test_speed():
    env = rn.Env(1024, seed=2, turn_cap=200)
    for b0, b1 in [("random", "random"), ("greedy", "greedy")]:
        t = time.perf_counter()
        finished, turns, *_ = env.play(10000, b0, b1)
        dt = time.perf_counter() - t
        print(f"  {b0} vs {b1}: {finished / dt:,.0f} hands/s, {turns / dt / 1e6:.2f}M turns/s (one core)")
    check("timing measured", True)


def test_scorer_parity_large():
    _parity(200000, seed=12)


def test_bots_follow_rules_large():
    env = rn.Env(1024, seed=22, turn_cap=200)
    for b0, b1 in [("random", "random"), ("greedy", "random"), ("greedy", "greedy")]:
        finished, *_, violations = env.play(50000, b0, b1, check_rules=True)
        check(f"{b0} vs {b1}: 50000 hands, {violations} rules broken", finished == 50000 and violations == 0)


if __name__ == "__main__":
    run([test_scorer_parity, test_scorer_rejects_bad_input, test_deal, test_reproducible,
         test_draw_and_throw, test_auto_declare, test_dealt_win_needs_a_draw, test_reshuffle,
         test_turn_cap, test_observation_hides_opponent, test_bots_follow_rules,
         test_python_stepping_matches_rules, test_speed],
        [test_scorer_parity_large, test_bots_follow_rules_large],
        title="C++ engine")
