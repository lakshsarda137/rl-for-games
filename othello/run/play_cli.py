"""Terminal Othello: watch two bots play, or play one yourself. No browser needed.

Each side is one of:
    human            you type moves
    random           uniform random legal move
    greedy           maximise immediate disc count
    minimax:D        alpha-beta minimax at depth D (e.g. minimax:4)
    edax:L           (not wired yet — placeholder for the strong external engine)

Examples:
    # Watch depth-4 minimax (Black) take on depth-2 minimax (White):
    python run/play_cli.py --black minimax:4 --white minimax:2 --delay 0.4

    # Play as Black against depth-3 minimax, with your legal moves marked '*':
    python run/play_cli.py --black human --white minimax:3

    # Watch minimax dismantle a random bot:
    python run/play_cli.py --black minimax:3 --white random --delay 0.3
"""

import argparse
import os
import sys
import time

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "engine"))
sys.path.insert(0, os.path.join(_HERE, "..", "opponents"))

import numpy as np

from board_numpy import (
    BLACK,
    PASS,
    WHITE,
    apply_move,
    count_discs,
    initial_board,
    is_terminal,
    legal_moves,
    move_to_rc,
    rc_to_move,
    render,
    winner,
)
from minimax import minimax_player
from simple import greedy_move, random_move

_NAME = {BLACK: "Black(X)", WHITE: "White(O)"}


def _human_player(board, player):
    """Prompt the user for a legal move; PASS is automatic when stuck."""
    moves = legal_moves(board, player)
    if not moves:
        print(f"  {_NAME[player]} has no legal move — passing.")
        return PASS
    legal_rc = {move_to_rc(m) for m in moves}
    while True:
        raw = input(f"  {_NAME[player]} move as 'row col' (0-7), or 'q' to quit: ").strip()
        if raw.lower() in ("q", "quit", "exit"):
            print("Bye.")
            sys.exit(0)
        try:
            r, c = (int(x) for x in raw.replace(",", " ").split())
        except ValueError:
            print("  ! enter two numbers, e.g. '2 3'")
            continue
        if (r, c) in legal_rc:
            return rc_to_move(r, c)
        print(f"  ! ({r},{c}) is not legal. Legal: {sorted(legal_rc)}")


def make_player(spec, rng):
    """Turn a spec string into a `(board, player) -> move` function."""
    spec = spec.strip().lower()
    if spec == "human":
        return _human_player
    if spec == "random":
        return lambda b, p: random_move(b, p, rng)
    if spec == "greedy":
        return greedy_move
    if spec.startswith("minimax:"):
        return minimax_player(int(spec.split(":", 1)[1]))
    if spec == "edax" or spec.startswith("edax:"):
        from edax import EdaxNotInstalled, edax_player
        level = int(spec.split(":", 1)[1]) if ":" in spec else 6
        try:
            return edax_player(level)
        except EdaxNotInstalled as exc:
            print(f"Edax not available: {exc}")
            sys.exit(2)
    raise SystemExit(f"unknown player spec: {spec!r} "
                     "(use human | random | greedy | minimax:D)")


def _describe_move(player, move):
    if move == PASS:
        return f"{_NAME[player]} passes"
    r, c = move_to_rc(move)
    return f"{_NAME[player]} plays ({r},{c})"


def play(black_spec, white_spec, delay, seed, show_marks):
    rng = np.random.default_rng(seed)
    players = {BLACK: make_player(black_spec, rng), WHITE: make_player(white_spec, rng)}
    specs = {BLACK: black_spec, WHITE: white_spec}

    board = initial_board()
    player = BLACK
    ply = 0
    print(f"\nBlack(X) = {black_spec}    White(O) = {white_spec}\n")
    print(render(board, player if show_marks else None))

    while not is_terminal(board):
        move = players[player](board, player)
        board = apply_move(board, player, move)
        ply += 1
        player = -player
        black, white = count_discs(board)
        print(f"\nmove {ply}: {_describe_move(-player, move)}   [X {black} : {white} O]")
        # Only auto-delay between bot moves; human turns pace themselves.
        marks_for = player if (show_marks and specs[player] == "human") else (player if show_marks else None)
        print(render(board, marks_for))
        if delay and specs[player] != "human":
            time.sleep(delay)

    result = winner(board)
    black, white = count_discs(board)
    outcome = "draw" if result == 0 else f"{_NAME[result]} wins"
    print(f"\n=== game over: {outcome}  (X {black} : {white} O) ===")
    return result


def main():
    ap = argparse.ArgumentParser(description="Watch or play terminal Othello.")
    ap.add_argument("--black", default="human", help="Black player spec")
    ap.add_argument("--white", default="minimax:3", help="White player spec")
    ap.add_argument("--delay", type=float, default=0.3, help="seconds between bot moves")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (random players)")
    ap.add_argument("--no-marks", action="store_true", help="don't mark legal moves")
    args = ap.parse_args()
    play(args.black, args.white, args.delay, args.seed, show_marks=not args.no_marks)


if __name__ == "__main__":
    main()
