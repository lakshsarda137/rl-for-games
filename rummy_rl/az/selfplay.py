"""Self-play: the network plays itself, with search, to make training data.

Many games run side by side in one C++ Env. At every decision the search looks
ahead in imagined versions of the hidden cards, and we record:
  * what the player could see (planes, scalars, legal moves),
  * the search's visit counts, as the move choice the network should learn,
  * the opponent's real hand, as the answer for the hand guess,
and once the hand ends, how it ended for that player (the value target).

Early moves of each hand are picked in proportion to the visits, so games
differ; later moves take the most visited move.
"""

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "engine"))

from native import rummy_native as rn


def choose(pi, explore, rng):
    """Pick an action from a visit distribution: sample while exploring, else
    the most visited (ties broken at random)."""
    if explore:
        return int(rng.choice(len(pi), p=pi))
    best = np.flatnonzero(pi == pi.max())
    return int(rng.choice(best))


def progress_line(fraction, moves_per_s, seconds):
    return (f"  self-play: {100 * min(fraction, 1):.0f}% done, {moves_per_s:,.0f} moves/s, "
            f"{seconds:.0f}s so far")


def play(evaluator, cfg, n_hands, seed, progress=None, status=None, tag=""):
    """Play `n_hands` hands of self-play. Returns (examples dict, stats dict).

    progress(text), if given, is called about every 30 seconds with a line
    saying how far self-play has got, so a long iteration never looks stuck.
    status, if given, is a dict shared with the main process: status[tag] is
    kept up to date as (hands' worth of work done, moves made), so the main
    process can report one combined line for several GPUs."""
    rng = np.random.default_rng(seed)
    cap = cfg.turn_cap or 200
    n = min(cfg.selfplay_games, n_hands)
    env = rn.Env(n, seed=seed, turn_cap=cfg.turn_cap)
    started = n
    finished = 0
    pending = [[] for _ in range(n)]     # this hand's records, per game
    decisions = np.zeros(n, dtype=np.int64)
    out = {k: [] for k in ("planes", "scalars", "legal", "pi", "z", "hand")}
    turns, capped, pile_takes, draws = [], 0, 0, 0
    t_start = t_report = time.time()
    moves = 0

    while finished < n_hands:
        active = ~env.done()
        # Work done, in hands: finished hands count 1, a hand in play counts the
        # share of the turn limit it has used (most early hands run to the limit).
        done_work = finished + np.minimum(decisions[active] / 2 / cap, 1.0).sum()
        if status is not None:
            status[tag] = (float(done_work), moves)
        if progress and time.time() - t_report > 30:
            t_report = time.time()
            progress(progress_line(done_work / n_hands, moves / (t_report - t_start), t_report - t_start))
        planes, scalars = env.observe()
        legal = env.legal_mask()
        opp = env.opponent_hands()
        movers = env.to_move()
        visits, _, _ = env.search(evaluator, active, worlds=cfg.worlds, sims=cfg.sims,
                                  c_puct=cfg.c_puct, dir_alpha=cfg.dir_alpha,
                                  dir_eps=cfg.dir_eps, seed=int(rng.integers(1 << 62)))
        actions = np.full(n, -1, dtype=np.int32)
        for i in np.flatnonzero(active):
            total = visits[i].sum()
            pi = visits[i] / total if total > 0 else legal[i] / legal[i].sum()
            actions[i] = choose(pi, decisions[i] < cfg.temp_moves, rng)
            decisions[i] += 1
            moves += 1
            pending[i].append((planes[i], scalars[i], legal[i], pi.astype(np.float32), movers[i], opp[i]))
            if scalars[i, 0] == 1:
                draws += 1
                pile_takes += actions[i] == rn.ACT_PILE

        rewards, done = env.step(actions)
        for i in np.flatnonzero(done):
            s = env.state(int(i))
            turns.append(s["turn"])
            capped += s["capped"]
            for p_, s_, l_, pi_, mover, h_ in pending[i]:
                out["planes"].append(p_)
                out["scalars"].append(s_)
                out["legal"].append(l_)
                out["pi"].append(pi_)
                out["z"].append(rewards[i, mover] / rn.VALUE_SCALE)
                out["hand"].append(h_)
            pending[i] = []
            decisions[i] = 0
            finished += 1
            if started < n_hands:
                env.reset(int(i))
                started += 1

    examples = {
        "planes": np.asarray(out["planes"], dtype=np.uint8),
        "scalars": np.asarray(out["scalars"], dtype=np.float32),
        "legal": np.asarray(out["legal"], dtype=bool),
        "pi": np.asarray(out["pi"], dtype=np.float32),
        "z": np.asarray(out["z"], dtype=np.float32),
        "hand": np.asarray(out["hand"], dtype=np.uint8),
    }
    stats = {
        "hands": finished,
        "positions": len(out["z"]),
        "avg_turns": float(np.mean(turns)) if turns else 0.0,
        "capped": int(capped),
        "pile_take_rate": pile_takes / max(draws, 1),
        "draws": draws,
    }
    return examples, stats


# --------------------------------------------------------------------------- #
# Several GPUs: each one plays its share of the hands in its own process
# --------------------------------------------------------------------------- #

def _worker(state_dict, net_size, cfg, n_hands, seed, device, use_hand_guess, tag, status):
    """Runs in a separate process: rebuild the network on `device` and play."""
    import torch
    from network import Evaluator, RummyNet

    torch.set_num_threads(1)
    net = RummyNet(*net_size)
    net.load_state_dict(state_dict)
    return play(Evaluator(net, device, use_hand_guess), cfg, n_hands, seed, status=status, tag=tag)


def merge(results):
    """Combine (examples, stats) from several workers into one."""
    examples = {k: np.concatenate([r[0][k] for r in results]) for k in results[0][0]}
    hands = sum(r[1]["hands"] for r in results)
    draws = sum(r[1]["draws"] for r in results)
    stats = {
        "hands": hands,
        "positions": sum(r[1]["positions"] for r in results),
        "avg_turns": sum(r[1]["avg_turns"] * r[1]["hands"] for r in results) / max(hands, 1),
        "capped": sum(r[1]["capped"] for r in results),
        "pile_take_rate": sum(r[1]["pile_take_rate"] * r[1]["draws"] for r in results) / max(draws, 1),
        "draws": draws,
    }
    return examples, stats


def play_on_gpus(pool, status, net, devices, cfg, n_hands, seed, use_hand_guess=True, progress=None):
    """Split `n_hands` across `devices` (one worker process each) and merge.
    `status` is a shared dict (multiprocessing Manager) the workers report into;
    every 30 seconds progress() gets one line for all GPUs together."""
    from concurrent.futures import wait

    state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    share = [n_hands // len(devices) + (k < n_hands % len(devices)) for k in range(len(devices))]
    status.clear()
    t_start = time.time()
    jobs = [pool.submit(_worker, state, (net.blocks_n, net.channels), cfg, share[k], seed + 7919 * k,
                        dev, use_hand_guess, f"gpu{k}", status)
            for k, dev in enumerate(devices) if share[k] > 0]
    while True:
        finished, _ = wait(jobs, timeout=30)
        if len(finished) == len(jobs):
            break
        if progress:
            parts = list(status.values())
            work = sum(p[0] for p in parts)
            moves = sum(p[1] for p in parts)
            elapsed = time.time() - t_start
            progress(progress_line(work / n_hands, moves / elapsed, elapsed))
    return merge([j.result() for j in jobs])
