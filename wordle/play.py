"""Play Wordle in the terminal.

python play.py                          solo game
python play.py duel                     duel against the best_split bot (RULES.md section 4)
python play.py duel --ai runs/main/latest.pt   duel against a trained AI checkpoint
"""

import argparse
import random

from engine import bots, words
from engine.game import Game

SQUARES = {"green": "G", "yellow": "Y", "gray": "-"}


def show(player, word, pattern):
    marks = "".join(SQUARES[c] for c in words.pattern_to_colors(pattern))
    print(f"  {player:>4}: {word.upper()}  {marks}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", nargs="?", default="solo", choices=["solo", "duel"])
    p.add_argument("--ai", help="checkpoint file of a trained AI to play against")
    a = p.parse_args()
    duel = a.mode == "duel"
    ai = None
    if a.ai:
        from train.ai_player import AIPlayer

        ai = AIPlayer(a.ai)
        print(f"Playing against the trained AI ({ai.info}).")
    name = "AI" if ai else "Bot"
    game = Game(players=2 if duel else 1)
    you = random.randrange(2) if duel else 0
    print("Colors: G = right letter, right place.  Y = in the word, wrong place.  - = not in the word.")
    if duel:
        print("You go first." if you == 0 else f"The {name.lower() if name == 'Bot' else name} goes first.")
    while not game.over:
        if game.to_play == you:
            try:
                word = input("Your guess: ").strip()
                p = game.guess(word)
            except ValueError as e:
                print(f"  {e}")
                continue
            show("You", word, p)
        else:
            game.guess(ai.choose(game) if ai else bots.best_split(game))
            if not game.over:
                print(f"  {name:>4}: made a guess (hidden from you)")
    print(f"The word was {game.secret.upper()}.")
    if duel:
        print(f"The {name}'s guesses:")
        for i, (word, pat) in enumerate(game.own_history(1 - you)):
            show(name, word, pat)
            if ai:
                _, n, opts = ai.notes[i]
                print(f"        ({n} possible answer{'s' if n != 1 else ''}; top choices by expected points: "
                      + ", ".join(f"{w.upper()} {v:+.2f}" for w, v in opts) + ")")
        if game.winner is None:
            print("Nobody found it. Draw: 0 points each.")
        else:
            scores = game.duel_scores()
            print(("You win." if game.winner == you else f"The {name} wins.")
                  + f" Points: you {scores[you]:+d}, {name} {scores[1 - you]:+d}.")
    elif game.winner is None:
        print("Not solved in 6 guesses.")
    else:
        print(f"Solved in {len(game.history)} guesses.")


if __name__ == "__main__":
    main()
