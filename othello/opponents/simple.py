"""Throwaway opponents for the bottom of the ladder (rung 0) and a match runner.

`random_move` / `greedy_move` are the very-early sanity opponents. `play_match`
is the neutral game loop every player plugs into — a player is just a
`(board, player) -> move` function, so minimax, random, greedy, a human, or
(later) Edax all share one harness.
"""

import os
import sys

_ENGINE = os.path.join(os.path.dirname(__file__), "..", "engine")
if _ENGINE not in sys.path:
    sys.path.insert(0, _ENGINE)

import numpy as np

from board_numpy import (
    BLACK,
    PASS,
    WHITE,
    apply_move,
    count_discs,
    disc_diff,
    initial_board,
    is_terminal,
    legal_moves,
    winner,
)


def random_move(board, player, rng=None):
    """Uniformly random legal move (PASS if stuck)."""
    moves = legal_moves(board, player)
    if not moves:
        return PASS
    rng = np.random.default_rng() if rng is None else rng
    return int(rng.choice(moves))


def greedy_move(board, player):
    """Maximise immediate disc count — a weak but non-trivial baseline."""
    moves = legal_moves(board, player)
    if not moves:
        return PASS
    return max(moves, key=lambda m: disc_diff(apply_move(board, player, m), player))


def play_game(black_player, white_player, opening_moves=None):
    """Play one game to the end; return (winner, final_board, move_history).

    `black_player` / `white_player` are `(board, player) -> move` functions.
    `opening_moves`, if given, is a list of forced initial moves (used to seed
    varied openings so deterministic bots don't replay one identical game).
    """
    board = initial_board()
    player = BLACK
    history = []
    players = {BLACK: black_player, WHITE: white_player}

    forced = list(opening_moves or [])
    while not is_terminal(board):
        if forced:
            move = forced.pop(0)
        elif not legal_moves(board, player):
            move = PASS
        else:
            move = players[player](board, player)
        board = apply_move(board, player, move)
        history.append((player, move))
        player = -player

    return winner(board), board, history


def _random_opening(rng, n_plies):
    """A short list of random legal moves from the start, to diversify openings."""
    board = initial_board()
    player = BLACK
    moves = []
    for _ in range(n_plies):
        if is_terminal(board):
            break
        legal = legal_moves(board, player)
        move = PASS if not legal else int(rng.choice(legal))
        board = apply_move(board, player, move)
        moves.append(move)
        player = -player
    return moves


def play_match(player_a, player_b, n_games=20, opening_plies=2, seed=0):
    """Play `player_a` vs `player_b` over `n_games`, colours alternated evenly.

    Returns a dict with player_a's win/draw/loss counts and win rate (draws
    count as half). Each pair of games shares one random opening (played by both
    colour assignments) so the comparison is fair and varied.
    """
    rng = np.random.default_rng(seed)
    wins = draws = losses = 0
    for g in range(n_games):
        opening = _random_opening(rng, opening_plies)
        # Alternate which colour player_a takes.
        if g % 2 == 0:
            result, _, _ = play_game(player_a, player_b, opening)
            a_is = BLACK
        else:
            result, _, _ = play_game(player_b, player_a, opening)
            a_is = WHITE
        if result == 0:
            draws += 1
        elif result == a_is:
            wins += 1
        else:
            losses += 1
    win_rate = (wins + 0.5 * draws) / n_games if n_games else 0.0
    return {"wins": wins, "draws": draws, "losses": losses, "win_rate": win_rate,
            "n_games": n_games}
