"""Play many duels at the same time (RULES.md sections 4 and 5).

All games in a batch take turns together: turn 0 is player 0's first guess in
every game, turn 1 is player 1's first guess, and so on. This lets the network
score the positions of all games in one large batch.

A seat is played by one of:
- "ai":    the lookahead in lookahead.py with the value network
- "split": the best_split bot (no training)
"""

from dataclasses import dataclass, field

import numpy as np

from engine import words
from engine.game import MAX_DUEL_GUESSES, loss_points, win_points

from .lookahead import choose, guess_values
from .network import extras

ALL_GREEN = words.ALL_GREEN


@dataclass
class BatchResult:
    scores: np.ndarray  # (games, 2) final score of each seat (W5)
    winner: np.ndarray  # (games,) 0, 1, or -1 for a draw
    win_guesses: np.ndarray  # (games,) guesses the winner used, 0 for a draw
    samples: list = field(default_factory=list)  # (members, extras row, game, seat) for training
    traces: dict = field(default_factory=dict)  # game -> list of turn records, for inspection


def play_batch(seats, secrets, cache, net=None, device="cpu", rng=None, temperature=0.0, epsilon=0.0,
               record=(False, False), trace_games=(), top_k=3):
    """seats: ("ai" or "split", "ai" or "split"). secrets: answer indices, one per game."""
    rng = rng or np.random.default_rng()
    table = words.table()
    n_games = len(secrets)
    full = np.arange(len(words.ANSWERS))
    cands = [[full, full] for _ in range(n_games)]
    over = np.zeros(n_games, dtype=bool)
    winner = np.full(n_games, -1)
    win_guesses = np.zeros(n_games, dtype=int)
    samples = []
    traces = {g: [] for g in trace_games}
    trace_set = set(trace_games)

    for turn in range(MAX_DUEL_GUESSES):
        seat, used = turn % 2, turn // 2
        first = seat == 0
        active = np.flatnonzero(~over)
        if len(active) == 0:
            break
        positions = [cache.get(cands[g][seat]) for g in active]
        if seats[seat] == "ai":
            unique, slot_of, unique_pos = {}, [], []
            for pos in positions:
                if id(pos) not in unique:
                    unique[id(pos)] = len(unique_pos)
                    unique_pos.append(pos)
                slot_of.append(unique[id(pos)])
            q_unique = guess_values(net, device, unique_pos, used, first)
        for j, g in enumerate(active):
            pos = positions[j]
            if seats[seat] == "ai":
                q = q_unique[slot_of[j]]
                k = choose(q, rng, temperature, epsilon)
                guess = int(pos.guesses[k])
            else:
                q = None
                if epsilon and rng.random() < epsilon:
                    guess = int(pos.guesses[rng.integers(len(pos.guesses))])
                else:
                    guess = pos.split_guess
            pattern = int(table[guess, secrets[g]])
            mine = cands[g][seat]
            new = mine[table[guess, mine] == pattern]
            cands[g][seat] = new
            if g in trace_set:
                rec = {"seat": seat, "word": words.GUESSES[guess], "pattern": pattern,
                       "before": pos.n, "after": len(new)}
                if q is not None:
                    best = np.argsort(-q)[:top_k]
                    rec["options"] = [(words.GUESSES[int(pos.guesses[i])], float(q[i])) for i in best]
                traces[g].append(rec)
            if pattern == ALL_GREEN:
                over[g] = True
                winner[g] = seat
                win_guesses[g] = used + 1
            elif record[seat]:
                samples.append((new, extras(len(new), used + 1, first), g, seat))

    scores = np.zeros((n_games, 2))
    for g in range(n_games):
        w = winner[g]
        if w >= 0:
            scores[g, w] = win_points(win_guesses[g])
            scores[g, 1 - w] = loss_points(len(cands[g][1 - w]))
    return BatchResult(scores, winner, win_guesses, samples, traces)


def format_trace(trace, secret, names=("Player 1", "Player 2")):
    """A readable account of one game, including the AI's top choices."""
    squares = {"gray": "-", "yellow": "Y", "green": "G"}
    lines = [f"Secret word: {words.ANSWERS[secret].upper()}"]
    for rec in trace:
        marks = "".join(squares[c] for c in words.pattern_to_colors(rec["pattern"]))
        line = (f"  {names[rec['seat']]:<10} {rec['word'].upper()}  {marks}   "
                f"possible answers: {rec['before']} before, {rec['after']} after")
        lines.append(line)
        if "options" in rec:
            opts = ", ".join(f"{w.upper()} {v:+.2f}" for w, v in rec["options"])
            lines.append(f"             expected points of its top choices: {opts}")
    return "\n".join(lines)
