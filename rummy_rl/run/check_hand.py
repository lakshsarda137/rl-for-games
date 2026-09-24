"""Type a hand, see what the engine makes of it.

    python run/check_hand.py                          # interactive: one hand per line
    python run/check_hand.py 6H 7H 8H 9H KS KH ...    # one-shot

13 cards: is it a valid declare, and what is the minimum score (with the melds
and the cards that score)? 14 cards (after a draw): which card to discard, then
the same report for the 13 that are kept.

Cards are rank + suit: ranks A 2-10 J Q K (T also works for 10), suits S H D C.
Case, commas and spacing don't matter.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "engine"))

from hand import (
    HAND_SIZE,
    RANKS,
    VALUE,
    arrange_hand,
    best_discard,
    parse_card,
    to_mask,
)

SYMBOL = "♠♥♦♣"
COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
RED, BOLD, DIM, GREEN, RESET = ("\033[31m", "\033[1m", "\033[2m", "\033[32m", "\033[0m") if COLOR else ("",) * 5


def show(c):
    s = RANKS[c % 13] + SYMBOL[c // 13]
    return f"{RED}{s}{RESET}" if c // 13 in (1, 2) else s


def show_all(cards):
    return " ".join(show(c) for c in cards)


def in_order(cards, run=False):
    """Display order: by suit then rank, ace low; in a run that has a king, ace high."""
    high = run and any(c % 13 == 12 for c in cards)
    return sorted(cards, key=lambda c: (c // 13, 13 if high and c % 13 == 0 else c % 13))


def parse_line(line):
    tokens = line.replace(",", " ").split()
    cards = []
    for t in tokens:
        t = t.strip().upper()
        if t[:1] == "T":
            t = "10" + t[1:]
        cards.append(parse_card(t))
    to_mask(cards)  # raises on duplicates
    return cards


def report(cards):
    lines = []
    if len(cards) == HAND_SIZE + 1:
        _, _, discard = best_discard(cards)
        cards = [c for c in cards if c != discard]
        lines.append(f"Discard   : {show(discard)}")
    elif len(cards) != HAND_SIZE:
        raise ValueError(f"enter {HAND_SIZE} or {HAND_SIZE + 1} cards (got {len(cards)})")

    win, score, melds, leftovers = arrange_hand(cards)
    lines.insert(0, f"Hand      : {show_all(in_order(cards))}")
    if win:
        lines.append(f"{GREEN}{BOLD}WIN{RESET} - valid declare, score 0")
    else:
        lines.append(f"{BOLD}NOT A WIN{RESET} - minimum score {BOLD}{score}{RESET}")
    if not melds:
        lines.append(f"{DIM}No run of 4 or more, so nothing melds: every card scores (full count).{RESET}")
    for i, m in enumerate(melds):
        kind = "set" if len({c % 13 for c in m}) == 1 else "run"
        tag = "  <- the 4+ run" if i == 0 else ""
        lines.append(f"  {kind:<4}{len(m):>2}  {show_all(in_order(m, run=kind == 'run'))}{DIM}{tag}{RESET}")
    if leftovers:
        parts = " + ".join(str(VALUE[c]) for c in in_order(leftovers))
        lines.append(f"Scoring   : {show_all(in_order(leftovers))}   ({parts} = {score})")
    return "\n".join(lines)


def main():
    if len(sys.argv) > 1:
        try:
            print(report(parse_line(" ".join(sys.argv[1:]))))
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        return

    print("Enter 13 cards (a hand) or 14 (after a draw), e.g. 6H 7H 8H 9H KS KH KD 2S 5C 9D QH 3S 7C")
    print("Ranks A 2-10 J Q K (T = 10), suits S H D C. Blank line or q to quit.")
    while True:
        try:
            line = input("\ncards> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if line.lower() in ("", "q", "quit", "exit"):
            return
        try:
            print(report(parse_line(line)))
        except ValueError as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
