"""Tests for engine/hand.py. Run: python tests/test_hand.py [--full]

Two anchors:
  * hand-built cases, one per rule in RULES.md sections 5 and 6, with scores
    worked out by hand in the comments;
  * parity against a brute-force oracle on random and meld-rich hands. The oracle
    is written independently: it enumerates runs of EVERY length (3-13) and tracks
    the "one run of 4+" requirement inside the recursion, so it also checks the
    engine's shortcut of only enumerating runs of length 3-5.
"""

import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "engine"))

from hand import (
    VALUE,
    arrange_hand,
    best_discard,
    card,
    card_str,
    mask_cards,
    parse_card,
    score_hand,
    to_mask,
)

from harness import check, run


# --------------------------------------------------------------------------- #
# Hand-built cases
# --------------------------------------------------------------------------- #

def test_valid_shapes():
    cases = {
        "4+3+3+3": "2H 3H 4H 5H 7S 7D 7C 9S 9D 9C JS JD JC",
        "4+3+6": "6H 7H 8H 9H 5S 5D 5C 8C 9C 10C JC QC KC",
        "5+4+4 (ace low and ace high)": "AS 2S 3S 4S 5S 7H 8H 9H 10H JD QD KD AD",
        "7+3+3": "3C 4C 5C 6C 7C 8C 9C KS KH KD 2H 2D 2S",
        "10+3": "AD 2D 3D 4D 5D 6D 7D 8D 9D 10D QS QH QC",
        "13-card run": "AH 2H 3H 4H 5H 6H 7H 8H 9H 10H JH QH KH",
        "J-Q-K-A is a 4-run": "JS QS KS AS 2H 3H 4H 6D 6C 6H 8D 8C 8H",
    }
    for name, hand in cases.items():
        check(f"win: {name}", score_hand(hand) == (True, 0))


def test_no_four_run_is_full_count():
    # Every card is in a 3-run or 3-set, but nothing is 4+ long.
    # 2+3+4 + 6+7+8 + 9+9+9 + 10+10+10 + 5 = 92
    check("all 3-melds -> full count 92",
          score_hand("2H 3H 4H 6S 7S 8S 9D 9C 9H KS KH KD 5C") == (False, 92))
    # A 4-card SET is not a run. 28 + 9 + 29 + 30 = 96
    check("set of four does not count as the 4-run",
          score_hand("7S 7H 7D 7C 2H 3H 4H 9S 10S JS QD KD AD") == (False, 96))


def test_no_wraparound():
    # K-A-2-3 of spades would be a 4-run only with wrap. A-2-3 is a legal 3-run,
    # but there is still no 4-run, so full count: 25 + 18 + 27 + 30 = 100.
    check("K-A-2-3 is not a 4-run",
          score_hand("KS AS 2S 3S 5H 6H 7H 9D 9C 9H JD JC JH") == (False, 100))


def test_spec_example():
    # RULES.md R6.4 example: 6-9H run, KKK set, leftovers 2+5+9+10+3+7 = 36.
    check("RULES.md example scores 36",
          score_hand("6H 7H 8H 9H KS KH KD 2S 5C 9D QH 3S 7C") == (False, 36))


def test_optimal_not_greedy():
    # Taking the longest run 4H-8H leaves 8S 8D KS KD QC = 46. Shortening it to
    # 4H-7H frees 8H for the 8-set, leaving only KS KD QC = 30.
    check("splits a run to complete a set (30, not 46)",
          score_hand("4H 5H 6H 7H 8H 8S 8D 2C 3C 4C KS KD QC") == (False, 30))


def test_best_discard():
    win, score, d = best_discard("6H 7H 8H 9H 5S 5D 5C 8C 9C 10C JC QC KC 2S")
    check("14 cards: winning discard found", (win, score, card_str(d)) == (True, 0, "2S"))
    # Spec example + KC: KKKK set, leftovers 2S 5C 9D QH 3S 7C = 36; throw QH -> 26.
    win, score, d = best_discard("6H 7H 8H 9H KS KH KD 2S 5C 9D QH 3S 7C KC")
    check("14 cards: non-win discards the costliest leftover",
          (win, score, card_str(d)) == (False, 26, "QH"))


def test_input_validation():
    for name, bad in [("duplicate card", "2H 2H 3H 4H 5H 6H 7H 8H 9H 10H JH QH KH"),
                      ("12 cards", "2H 3H 4H 5H 6H 7H 8H 9H 10H JH QH KH"),
                      ("bad token", "1H 3H 4H 5H 6H 7H 8H 9H 10H JH QH KH AH")]:
        try:
            score_hand(bad)
            ok = False
        except ValueError:
            ok = True
        check(f"rejects {name}", ok)
    check("parse is case-insensitive", parse_card("qs") == parse_card("QS") == card(11, 0))
    check("ids, tokens and masks agree",
          score_hand([card(r, 1) for r in range(13)]) == score_hand(to_mask("AH 2H 3H 4H 5H 6H 7H 8H 9H 10H JH QH KH")))


# --------------------------------------------------------------------------- #
# Brute-force oracle
# --------------------------------------------------------------------------- #

def _oracle_melds(mask):
    """Every meld in `mask`, as (mask, value, is_4plus_run)."""
    out = []
    for s in range(4):
        for length in range(3, 14):
            for start in range(14 - length + 1):
                cards = [card(p % 13, s) for p in range(start, start + length)]
                if len(set(cards)) == length and all(mask >> c & 1 for c in cards):
                    m = sum(1 << c for c in cards)
                    out.append((m, sum(VALUE[c] for c in cards), length >= 4))
    for r in range(13):
        present = [card(r, s) for s in range(4) if mask >> card(r, s) & 1]
        n = len(present)
        for bits in range(1 << n):
            chosen = [present[i] for i in range(n) if bits >> i & 1]
            if len(chosen) >= 3:
                out.append((sum(1 << c for c in chosen), sum(VALUE[c] for c in chosen), False))
    return out


def oracle_score(mask):
    melds = _oracle_melds(mask)
    memo = {}
    neg = float("-inf")

    def f(m):
        """(best cover of m, best cover of m that uses a 4+ run)."""
        if m == 0:
            return 0, neg
        if m in memo:
            return memo[m]
        low = m & -m
        any_best, long_best = f(m ^ low)
        for g, v, is_long in melds:
            if g & low and m & g == g:
                a, l = f(m ^ g)
                any_best = max(any_best, v + a)
                long_best = max(long_best, v + (a if is_long else l))
        memo[m] = (any_best, long_best)
        return memo[m]

    total = sum(VALUE[c] for c in mask_cards(mask))
    _, long_best = f(mask)
    return total if long_best == neg else total - long_best


def _meld_rich_hand(rng, size):
    """A hand built mostly from random runs and sets, then lightly perturbed, so
    wins and near-wins are common (uniform hands are almost never close)."""
    used = set()
    while len(used) < size:
        room = size - len(used)
        cards = None
        for _ in range(50):
            if rng.random() < 0.6:
                length = rng.randint(3, 7)
                start = rng.randint(0, 14 - length)
                s = rng.randrange(4)
                cand = [card(p % 13, s) for p in range(start, start + length)]
            else:
                r = rng.randrange(13)
                cand = [card(r, s) for s in rng.sample(range(4), rng.choice((3, 4)))]
            if len(cand) <= room and not used.intersection(cand):
                cards = cand
                break
        if cards is None:
            cards = [rng.choice([c for c in range(52) if c not in used])]
        used.update(cards)
    for _ in range(rng.choice((0, 0, 1, 2))):
        used.remove(rng.choice(sorted(used)))
        used.add(rng.choice([c for c in range(52) if c not in used]))
    return sum(1 << c for c in used)


def _parity(n_each, seed):
    rng = random.Random(seed)
    wins = pure_losses = full_counts = 0
    for i in range(2 * n_each):
        if i < n_each:
            mask = sum(1 << c for c in rng.sample(range(52), 13))
        else:
            mask = _meld_rich_hand(rng, 13)
        got = score_hand(mask)
        want = oracle_score(mask)
        if got != (want == 0, want):
            check(f"parity on {' '.join(card_str(c) for c in mask_cards(mask))}: "
                  f"engine {got}, oracle {want}", False)
        total = sum(VALUE[c] for c in mask_cards(mask))
        wins += want == 0
        full_counts += want == total
        pure_losses += 0 < want < total
    check(f"{2 * n_each} hands match the oracle "
          f"({wins} wins, {pure_losses} partial melds, {full_counts} full counts)", True)
    check("meld-rich hands exercise every outcome", wins > 0 and pure_losses > 0 and full_counts > 0)

    for _ in range(n_each // 4):
        mask = _meld_rich_hand(rng, 14)
        win, score, d = best_discard(mask)
        want = min(oracle_score(mask ^ (1 << c)) for c in mask_cards(mask))
        if (win, score) != (want == 0, want) or oracle_score(mask ^ (1 << d)) != score:
            check(f"14-card parity on {' '.join(card_str(c) for c in mask_cards(mask))}", False)
    check(f"{n_each // 4} fourteen-card hands match the oracle", True)


def test_oracle_parity():
    _parity(1500, seed=0)


def _meld_kind(cards):
    """'set', 'run' (with its length) or None, checked from first principles."""
    ranks = {c % 13 for c in cards}
    suits = [c // 13 for c in cards]
    if len(cards) < 3:
        return None
    if len(ranks) == 1 and len(set(suits)) == len(cards) <= 4:
        return "set"
    if len(set(suits)) != 1 or len(ranks) != len(cards):
        return None
    for positions in ({r for r in ranks}, {13 if r == 0 else r for r in ranks}):
        if max(positions) - min(positions) == len(cards) - 1:
            return "run"
    return None


def test_arrangement():
    rng = random.Random(3)
    for i in range(2000):
        mask = sum(1 << c for c in rng.sample(range(52), 13)) if i % 4 == 0 else _meld_rich_hand(rng, 13)
        win, score, melds, leftovers = arrange_hand(mask)
        where = " ".join(card_str(c) for c in mask_cards(mask))
        used = [c for m in melds for c in m] + leftovers
        ok = (sorted(used) == mask_cards(mask)
              and (win, score) == score_hand(mask)
              and score == sum(VALUE[c] for c in leftovers)
              and all(_meld_kind(m) for m in melds)
              and (not melds or (_meld_kind(melds[0]) == "run" and len(melds[0]) >= 4)))
        if not ok:
            check(f"arrangement valid on {where}: {melds} / {leftovers}", False)
    check("2000 arrangements: valid melds, 4+ run first, cover the hand, sum to the score", True)


def test_speed():
    rng = random.Random(1)
    hands13 = [_meld_rich_hand(rng, 13) for _ in range(2000)]
    hands14 = [_meld_rich_hand(rng, 14) for _ in range(500)]
    t = time.perf_counter()
    for h in hands13:
        score_hand(h)
    t13 = (time.perf_counter() - t) / len(hands13)
    t = time.perf_counter()
    for h in hands14:
        best_discard(h)
    t14 = (time.perf_counter() - t) / len(hands14)
    print(f"  score_hand   : {t13 * 1e6:7.0f} us/hand")
    print(f"  best_discard : {t14 * 1e6:7.0f} us/hand")
    check("timing measured", True)


def test_oracle_parity_large():
    _parity(25000, seed=7)


if __name__ == "__main__":
    run([test_valid_shapes, test_no_four_run_is_full_count, test_no_wraparound,
         test_spec_example, test_optimal_not_greedy, test_best_discard,
         test_input_validation, test_oracle_parity, test_arrangement, test_speed],
        [test_oracle_parity_large],
        title="hand evaluation")
