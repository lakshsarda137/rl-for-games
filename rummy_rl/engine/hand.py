"""Hand evaluation: is a hand a valid declare, and what is its minimum score?

Implements RULES.md sections 5 and 6:

  * A meld is a run (3+ consecutive cards of one suit, ace low or high, no wrap)
    or a set (3-4 cards of one rank, all different suits).
  * A hand scores 0 and is a WIN when all 13 cards split into melds and at least
    one meld is a run of 4 or more.
  * Otherwise the hand scores the lowest total achievable: if a run of 4+ exists,
    cards packed into melds score 0 and leftovers score their value; if no run of
    4+ exists, every card scores (full count).

Card representation: an int 0..51, `suit * 13 + rank`, rank 0 = A, 1 = 2, ...,
12 = K; suits are S H D C. A hand is a 52-bit int mask with bit `card` set, so
melds are masks too and "meld fits in hand" is one AND.

Why only runs of length 3-6 are enumerated: a run of 7+ splits into a run of
4+ plus a run of 3+ (7 = 4+3, 8 = 4+4, ...), covering the same cards and still
meeting the "one run of 4+" requirement. A run of 6 does NOT reduce that way
(6 = 3+3 loses the 4+ run), so it is enumerated. The brute-force oracle in
tests/test_hand.py enumerates every run length and checks this equivalence on
random and meld-rich hands.

Search: the best cover of a mask is found by always deciding the lowest card in
it (leave it as a scoring leftover, or cover it with one of the melds containing
it), memoised on the mask. The mandatory 4+ run is placed first, then the rest is
covered freely.
"""

from itertools import combinations

SUITS = "SHDC"
RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
HAND_SIZE = 13

# A, 10, J, Q, K = 10; 2-9 = face value.
VALUE = tuple(10 if r == 0 or r >= 9 else r + 1 for _ in SUITS for r in range(13))


def card(rank, suit):
    """Card id from a rank index (0 = A .. 12 = K) and a suit index (0..3)."""
    return suit * 13 + rank


def parse_card(token):
    """'10H' / 'qs' / 'AD' -> card id. Raises ValueError on anything else."""
    t = token.strip().upper()
    rank, suit = t[:-1], t[-1:]
    if suit not in SUITS or rank not in RANKS:
        raise ValueError(f"bad card {token!r} (rank A 2-10 J Q K, suit S H D C)")
    return card(RANKS.index(rank), SUITS.index(suit))


def card_str(c):
    return RANKS[c % 13] + SUITS[c // 13]


def to_mask(cards):
    """Hand as a mask. Accepts a mask, a string '6H 7H ...', or an iterable of
    card ids / tokens. Raises ValueError on duplicate cards."""
    if isinstance(cards, int):
        return cards
    if isinstance(cards, str):
        cards = cards.replace(",", " ").split()
    mask = 0
    for c in cards:
        c = parse_card(c) if isinstance(c, str) else c
        if not 0 <= c < 52:
            raise ValueError(f"card id out of range: {c}")
        if mask >> c & 1:
            raise ValueError(f"duplicate card {card_str(c)}")
        mask |= 1 << c
    return mask


def mask_cards(mask):
    """Card ids in a mask, ascending."""
    return [c for c in range(52) if mask >> c & 1]


def mask_value(mask):
    return sum(VALUE[c] for c in range(52) if mask >> c & 1)


def _candidate_melds(hand):
    """All melds inside `hand` that the search needs: runs of length 3-6 and
    sets of 3-4. Returns (melds, pure) as lists of (mask, value); `pure` is the
    subset of runs with length 4-6."""
    melds, pure = [], []
    for s in range(4):
        # position p in 0..13 is rank p % 13 (position 13 is the high ace)
        for length in (3, 4, 5, 6):
            for start in range(14 - length + 1):
                m = 0
                for p in range(start, start + length):
                    m |= 1 << card(p % 13, s)
                if hand & m == m:
                    item = (m, mask_value(m))
                    melds.append(item)
                    if length >= 4:
                        pure.append(item)
    for r in range(13):
        present = [card(r, s) for s in range(4) if hand >> card(r, s) & 1]
        for k in (3, 4):
            for combo in combinations(present, k):
                m = sum(1 << c for c in combo)
                melds.append((m, mask_value(m)))
    return melds, pure


class _Solver:
    """Minimum-score search over subsets of one hand. Melds are enumerated once
    for the whole hand, so every sub-hand (e.g. each possible discard from 14
    cards) shares them and the memo."""

    def __init__(self, hand):
        melds, self.pure = _candidate_melds(hand)
        self.by_card = {}
        for m, v in melds:
            for c in mask_cards(m):
                self.by_card.setdefault(c, []).append((m, v))
        self.memo = {0: 0}

    def covered(self, mask):
        """Max total value of `mask` coverable by disjoint melds (no 4+ run
        requirement)."""
        hit = self.memo.get(mask)
        if hit is not None:
            return hit
        low = mask & -mask
        best = self.covered(mask ^ low)       # lowest card stays a leftover
        for m, v in self.by_card.get(low.bit_length() - 1, ()):
            if mask & m == m:
                got = v + self.covered(mask ^ m)
                if got > best:
                    best = got
        self.memo[mask] = best
        return best

    def _best_pure(self, mask):
        """(covered value, 4+ run mask) of the best cover that includes a 4+ run,
        or (-1, None) if `mask` holds no 4+ run."""
        best, best_run = -1, None
        for m, v in self.pure:
            if mask & m == m:
                got = v + self.covered(mask ^ m)
                if got > best:
                    best, best_run = got, m
        return best, best_run

    def score(self, mask):
        """Minimum score of the cards in `mask` under RULES.md R6.4."""
        best, _ = self._best_pure(mask)
        total = mask_value(mask)
        return total if best < 0 else total - best

    def arrange(self, mask):
        """One minimum-score arrangement of `mask`: (melds, leftover_mask), the
        4+ run first. Walks back through the memo, so it costs no extra search.
        With no 4+ run nothing may be melded (full count): ([], mask)."""
        _, run = self._best_pure(mask)
        if run is None:
            return [], mask
        melds, leftovers, rest = [run], 0, mask ^ run
        while rest:
            low = rest & -rest
            target = self.covered(rest)
            if self.covered(rest ^ low) == target:
                leftovers |= low
                rest ^= low
                continue
            for m, v in self.by_card[low.bit_length() - 1]:
                if rest & m == m and v + self.covered(rest ^ m) == target:
                    melds.append(m)
                    rest ^= m
                    break
        return melds, leftovers


def score_hand(cards):
    """Evaluate a 13-card hand.

    Returns (is_win, min_score). is_win is True exactly when min_score is 0:
    every card value is at least 2, so a 0 score means every card is melded
    with a 4+ run present, which is the definition of a valid hand (R5.5)."""
    mask = to_mask(cards)
    n = bin(mask).count("1")
    if n != HAND_SIZE:
        raise ValueError(f"expected {HAND_SIZE} cards, got {n}")
    score = _Solver(mask).score(mask)
    return score == 0, score


def _run_positions(cards):
    """Positions 0..13 of a run's cards, ace high (13) when the run has a king."""
    high = any(c % 13 == 12 for c in cards)
    return sorted(13 if c % 13 == 0 and high else c % 13 for c in cards)


def _merge_runs(melds):
    """The search splits long runs into pieces of 3-6 cards. Join same-suit runs
    that continue each other (never across K-A-2) so a 7-card run reads as one
    meld, not 4+3. The 4+ run stays first."""
    melds = [list(m) for m in melds]
    is_run = lambda m: len({c % 13 for c in m}) > 1
    merged = True
    while merged:
        merged = False
        for i, a in enumerate(melds):
            for j, b in enumerate(melds):
                if (i != j and is_run(a) and is_run(b) and a[0] // 13 == b[0] // 13
                        and _run_positions(a)[-1] + 1 == _run_positions(b)[0]):
                    melds[min(i, j)] = sorted(a + b)
                    del melds[max(i, j)]
                    merged = True
                    break
            if merged:
                break
    return melds


def arrange_hand(cards):
    """Like score_hand, but also returns one arrangement that achieves the score.

    Returns (is_win, min_score, melds, leftovers): melds is a list of card-id
    lists (the 4+ run first, adjoining runs joined), leftovers the card ids that
    score."""
    mask = to_mask(cards)
    n = bin(mask).count("1")
    if n != HAND_SIZE:
        raise ValueError(f"expected {HAND_SIZE} cards, got {n}")
    melds, leftovers = _Solver(mask).arrange(mask)
    score = mask_value(leftovers)
    return score == 0, score, _merge_runs(mask_cards(m) for m in melds), mask_cards(leftovers)


def best_discard(cards):
    """Evaluate a 14-card hand (after drawing): which discard leaves the best
    13 cards.

    Returns (is_win, min_score, discard_card_id). If is_win, discarding that card
    and declaring is legal. Ties on score keep the highest-value discard."""
    mask = to_mask(cards)
    n = bin(mask).count("1")
    if n != HAND_SIZE + 1:
        raise ValueError(f"expected {HAND_SIZE + 1} cards, got {n}")
    solver = _Solver(mask)
    best = None
    for c in mask_cards(mask):
        score = solver.score(mask ^ (1 << c))
        if best is None or score < best[0] or (score == best[0] and VALUE[c] > VALUE[best[1]]):
            best = (score, c)
    return best[0] == 0, best[0], best[1]
