"""Train the Wordle duel AI.

    python train/train_loop.py --tiny                 laptop check, about 2 minutes
    python train/train_loop.py --hours 2 --out runs/main
    python train/train_loop.py --kaggle --hours 10 --resume auto --out /kaggle/working/main

Each iteration:
1. Self-play: the AI plays many duels against itself and saves each position
   it reached together with its final score (RULES.md W5).
   The first --warmup-iters iterations use the best_split bot instead, so the
   network starts out predicting how that bot's games end (copy first, then improve).
2. Training: the network is adjusted to predict those final scores.
Every --eval-every iterations, a strength check plays test duels against the
best_split bot (the same secret words every time, each word played from both
seats) and saves a checkpoint.

What you can watch while it runs (all in the --out folder):
    progress.html   charts of every strength check plus the latest test game
    progress.txt    the newest status line
    samples.txt     every test game shown so far, with the AI's top choices
    metrics.jsonl   all numbers, one line per event
    latest.pt       newest checkpoint; play it with: python play.py --ai <folder>/latest.pt
"""

import argparse
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import numpy as np
import torch
import torch.nn.functional as F

from engine import words
from train import report
from train.lookahead import PositionCache
from train.network import ValueNet, batch_tensors
from train.selfplay import format_trace, play_batch

PRESETS = {
    "tiny": dict(games=64, iterations=6, eval_every=2, eval_words=40, warmup_iters=2, hidden=64, ckpt_every=2),
    "laptop": dict(games=256, iterations=200, eval_every=5, eval_words=150, warmup_iters=3, hidden=256, ckpt_every=25),
    "kaggle": dict(games=1024, iterations=100000, eval_every=5, eval_words=400, warmup_iters=3, hidden=256, ckpt_every=50),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tiny", action="store_true", help="small fast run to check that everything works")
    p.add_argument("--kaggle", action="store_true", help="settings for a Kaggle GPU")
    p.add_argument("--out", default=None, help="folder for checkpoints and progress files")
    p.add_argument("--hours", type=float, default=None, help="stop after this many hours")
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--games", type=int, default=None, help="self-play games per iteration")
    p.add_argument("--eval-every", type=int, default=None, help="iterations between strength checks (0 = never)")
    p.add_argument("--eval-words", type=int, default=None, help="secret words per strength check (each played twice)")
    p.add_argument("--ckpt-every", type=int, default=None, help="iterations between numbered checkpoints")
    p.add_argument("--warmup-iters", type=int, default=None, help="iterations that use best_split games")
    p.add_argument("--hidden", type=int, default=None, help="network size")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=512, help="positions per training step")
    p.add_argument("--reuse", type=float, default=4.0, help="how many times each new position is trained on")
    p.add_argument("--buffer", type=int, default=300_000, help="positions kept for training")
    p.add_argument("--temperature", type=float, default=0.25, help="randomness of self-play choices, in points")
    p.add_argument("--epsilon", type=float, default=0.03, help="chance of a fully random short-list guess")
    p.add_argument("--resume", default=None, help="checkpoint path, or 'auto' for <out>/latest.pt")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-wandb", action="store_true")
    a = p.parse_args(argv)
    preset = PRESETS["tiny" if a.tiny else "kaggle" if a.kaggle else "laptop"]
    for k, v in preset.items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    if a.out is None:
        a.out = os.path.join(HERE, "..", "runs", "tiny" if a.tiny else "main")
    if a.device is None:
        a.device = "cuda" if torch.cuda.is_available() else "cpu"
    return a


class Log:
    def __init__(self, out, use_wandb, config):
        self.out = out
        self.path = os.path.join(out, "metrics.jsonl")
        self.wandb = None
        if use_wandb and os.environ.get("WANDB_API_KEY"):
            try:
                import wandb

                self.wandb = wandb.init(project="wordle-duel", config=config, resume="allow")
            except Exception as e:  # training must not stop because of logging
                print(f"W&B not used: {e}")

    def write(self, record):
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
        if self.wandb:
            self.wandb.log({f"{record['kind']}/{k}": v for k, v in record.items() if isinstance(v, (int, float))},
                           step=record["games"])


def train_steps(net, opt, buffer, n_steps, batch, device, rng):
    if len(buffer) < batch or n_steps == 0:
        return float("nan")
    net.train()
    total = 0.0
    for _ in range(n_steps):
        pick = rng.integers(len(buffer), size=batch)
        members = [buffer[i][0] for i in pick]
        rows = [buffer[i][1] for i in pick]
        target = torch.tensor([buffer[i][2] for i in pick], dtype=torch.float32, device=device)
        idx, off, ex = batch_tensors(members, rows, device)
        loss = F.smooth_l1_loss(net(idx, off, ex), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += loss.item()
    net.eval()
    return total / n_steps


def strength_check(net, device, cache, eval_secrets, trace=True):
    """Duels against best_split on fixed secret words, each word from both seats."""
    rng = np.random.default_rng(0)
    a = play_batch(("ai", "split"), eval_secrets, cache, net, device, rng, trace_games=(0,) if trace else ())
    b = play_batch(("split", "ai"), eval_secrets, cache, net, device, rng)
    ai = np.concatenate([a.scores[:, 0], b.scores[:, 1]])
    bot = np.concatenate([a.scores[:, 1], b.scores[:, 0]])
    ai_won = np.concatenate([a.winner == 0, b.winner == 1])
    bot_won = np.concatenate([a.winner == 1, b.winner == 0])
    guesses = np.concatenate([a.win_guesses, b.win_guesses])
    diff = ai - bot
    sample = format_trace(a.traces[0], eval_secrets[0], ("AI", "Simple bot")) if trace else ""
    return {
        "ai_points": float(ai.mean()),
        "bot_points": float(bot.mean()),
        "margin": float(diff.mean()),
        "margin_err": float(2 * diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else 0.0,
        "win_pct": float(100 * ai_won.mean()),
        "loss_pct": float(100 * bot_won.mean()),
        "draw_pct": float(100 * (~ai_won & ~bot_won).mean()),
        "ai_win_guesses": float(guesses[ai_won].mean()) if ai_won.any() else None,
        "bot_win_guesses": float(guesses[bot_won].mean()) if bot_won.any() else None,
        "test_games": int(len(diff)),
    }, sample


def save(path, net, opt, state, args):
    tmp = path + ".tmp"
    """opt=None saves only the network (small file, for playing, not for resuming)."""
    torch.save({"net": net.state_dict(), "opt": opt.state_dict() if opt else None, "hidden": net.hidden,
                "state": state, "args": vars(args)}, tmp)
    os.replace(tmp, path)


def load_net(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = ValueNet(ck["hidden"]).to(device)
    net.load_state_dict(ck["net"])
    net.eval()
    return net, ck


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    words.table()
    net = ValueNet(args.hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    state = {"iteration": 0, "games": 0, "seconds": 0.0, "best_margin": None, "reference": None}
    resume = os.path.join(args.out, "latest.pt") if args.resume == "auto" else args.resume
    if resume and os.path.exists(resume):
        ck = torch.load(resume, map_location=device, weights_only=False)
        net = ValueNet(ck["hidden"]).to(device)
        net.load_state_dict(ck["net"])
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        if ck["opt"]:
            opt.load_state_dict(ck["opt"])
        state = ck["state"]
        print(f"Resumed from {resume}: iteration {state['iteration']}, {state['games']:,} games.")
    else:
        if args.resume == "auto":
            print("No checkpoint found; starting fresh.")
        for name in ("metrics.jsonl", "samples.txt"):  # a fresh run must not mix with an old run's numbers
            if os.path.exists(os.path.join(args.out, name)):
                os.remove(os.path.join(args.out, name))
    net.eval()

    log = Log(args.out, not args.no_wandb, vars(args))
    cache = PositionCache()
    eval_secrets = np.random.default_rng(12345).choice(len(words.ANSWERS), size=args.eval_words, replace=False)
    buffer = []
    print(f"Device: {device}. Network size {net.hidden}. {args.games} games per iteration. Output: {args.out}")

    if state["reference"] is None and args.eval_every:
        rb = play_batch(("split", "split"), eval_secrets, cache)
        state["reference"] = {"points_first": float(rb.scores[:, 0].mean()),
                              "points_second": float(rb.scores[:, 1].mean()),
                              "win_guesses": float(rb.win_guesses[rb.winner >= 0].mean())}
        r = state["reference"]
        print(f"Reference (simple bot against itself): first player {r['points_first']:+.2f} points, "
              f"second player {r['points_second']:+.2f} points, {r['win_guesses']:.2f} guesses per win.")

    start_time, start_seconds, start_iter = time.time(), state["seconds"], state["iteration"]
    sample_text = ""
    while True:
        elapsed_h = (time.time() - start_time) / 3600  # --hours counts this session only
        if state["iteration"] >= args.iterations or (args.hours and elapsed_h >= args.hours):
            break
        it = state["iteration"] + 1
        warm = it <= args.warmup_iters

        t0 = time.time()
        secrets = rng.integers(len(words.ANSWERS), size=args.games)
        if warm:
            res = play_batch(("split", "split"), secrets, cache, rng=rng, epsilon=0.15, record=(True, True))
        else:
            res = play_batch(("ai", "ai"), secrets, cache, net, device, rng, temperature=args.temperature,
                             epsilon=args.epsilon, record=(True, True))
        new = [(m, e, float(res.scores[g, s])) for m, e, g, s in res.samples]
        buffer.extend(new)
        if len(buffer) > args.buffer:
            del buffer[: len(buffer) - args.buffer]
        t1 = time.time()
        steps = math.ceil(len(new) * args.reuse / args.batch)
        loss = train_steps(net, opt, buffer, steps, args.batch, device, rng)
        t2 = time.time()

        state["iteration"] = it
        state["games"] += args.games
        state["seconds"] = start_seconds + time.time() - start_time
        if args.hours:
            pct = 100 * min(1.0, (time.time() - start_time) / 3600 / args.hours)
        else:
            pct = 100 * (it - start_iter) / max(1, args.iterations - start_iter)
        hit = cache.hits / max(1, cache.hits + cache.misses)
        status = (f"{pct:5.1f}% done. Iteration {it}{' (warm-up, simple bot games)' if warm else ''}: "
                  f"{state['games']:,} games so far, "
                  + (f"prediction error {loss:.3f}. " if loss == loss else "not enough positions to train on yet. ") +
                  f"Self-play {t1-t0:.1f}s, training {t2-t1:.1f}s ({steps} steps). "
                  f"Cache reuse {100*hit:.0f}%.")
        print(status, flush=True)
        with open(os.path.join(args.out, "progress.txt"), "w") as f:
            f.write(status + "\n")
        log.write({"kind": "train", "iteration": it, "games": state["games"], "loss": loss if loss == loss else None,
                   "selfplay_s": t1 - t0, "train_s": t2 - t1, "positions": len(new),
                   "selfplay_win_guesses": float(res.win_guesses[res.winner >= 0].mean()) if (res.winner >= 0).any() else None})

        last = state["iteration"] >= args.iterations
        if args.eval_every and (it % args.eval_every == 0 or last):
            t3 = time.time()
            m, sample_text = strength_check(net, device, cache, eval_secrets)
            m.update(kind="eval", iteration=it, games=state["games"], eval_s=time.time() - t3)
            log.write(m)
            better = m["margin"] - m["margin_err"] > 0
            print(f"  Strength check ({m['test_games']} games against the simple bot, {m['eval_s']:.0f}s): "
                  f"AI {m['ai_points']:+.2f} points per game, bot {m['bot_points']:+.2f}. "
                  f"Difference {m['margin']:+.2f} (plus or minus {m['margin_err']:.2f}). "
                  f"AI won {m['win_pct']:.0f}%, lost {m['loss_pct']:.0f}%, drew {m['draw_pct']:.0f}%. "
                  f"Guesses per win: AI {m['ai_win_guesses'] or float('nan'):.2f}, "
                  f"bot {m['bot_win_guesses'] or float('nan'):.2f}."
                  + (" The AI is clearly stronger." if better else ""), flush=True)
            print("  " + sample_text.replace("\n", "\n  "), flush=True)
            with open(os.path.join(args.out, "samples.txt"), "a") as f:
                f.write(f"=== iteration {it}, {state['games']:,} games ===\n{sample_text}\n\n")
            if it % args.ckpt_every == 0 or last:
                save(os.path.join(args.out, f"ckpt_{it:05d}.pt"), net, None, state, args)
            if state["best_margin"] is None or m["margin"] > state["best_margin"]:
                state["best_margin"] = m["margin"]
                save(os.path.join(args.out, "best.pt"), net, None, state, args)
        save(os.path.join(args.out, "latest.pt"), net, opt, state, args)
        report.write(os.path.join(args.out, "progress.html"), report.read_metrics(log.path), status, sample_text)

    print(f"Finished: {state['iteration']} iterations, {state['games']:,} games, "
          f"{state['seconds']/3600:.2f} hours. Files are in {args.out}")


if __name__ == "__main__":
    main()
