"""Top-level AlphaZero training loop — the Kaggle/local entrypoint (plan §12).

One iteration = self-play (generate examples) -> train (minibatch updates on the
replay buffer) -> occasionally evaluate on the minimax ladder -> checkpoint +
emit spectator records + metric logs. Everything is config-driven (run/config.py).

    python run/train_loop.py --tiny            # seconds-scale smoke run on CPU
    python run/train_loop.py --iterations 40   # full-ish local run

Artifacts land under data/ (gitignored): checkpoints/, game_records/, tb_logs/.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("engine", "opponents", "az"):
    sys.path.insert(0, os.path.join(_HERE, "..", _p))
sys.path.insert(0, _HERE)

from config import Config
from evaluate import ladder_eval
from network import Evaluator, OthelloNet
from replay_buffer import ReplayBuffer
from selfplay import generate_games
from train import make_optimizer, train_steps

DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "data"))


def flat_metrics(loss, buffer_len, games_per_sec, avg_len, evals):
    """One flat {name: number} dict per iteration, for jsonl + TensorBoard."""
    m = {
        "loss/total": loss["total"], "loss/policy": loss["policy"],
        "loss/value": loss["value"], "buffer_size": buffer_len,
        "selfplay_games_per_sec": games_per_sec,
    }
    if avg_len:
        m["game_len_avg"] = avg_len
    if evals:
        m["max_depth_beaten"] = evals["max_depth_beaten"]
        for d, wr in evals["winrate"].items():
            m[f"winrate/minimax_d{d}"] = wr
    return m


class MetricsLogger:
    """Robust metrics sink: always writes metrics.jsonl; TensorBoard is optional.

    TensorBoard/protobuf versions clash in some environments, so tb is off by
    default and fully guarded — if it can't initialise, we note it once and keep
    going on the jsonl log (which is always the source of truth).
    """

    def __init__(self, out_dir, use_tb=False):
        os.makedirs(out_dir, exist_ok=True)
        self.jsonl = open(os.path.join(out_dir, "metrics.jsonl"), "a")
        self.tb = None
        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(os.path.join(out_dir, "tb_logs"))
                self.tb.add_scalar("_init", 0.0, 0)
                self.tb.flush()
            except Exception as exc:
                print(f"[metrics] TensorBoard disabled ({type(exc).__name__}); "
                      "using metrics.jsonl. Pin a compatible protobuf to enable tb.")
                self.tb = None

    def log(self, iteration, metrics):
        self.jsonl.write(json.dumps({"iter": iteration, **metrics}) + "\n")
        self.jsonl.flush()
        if self.tb:
            try:
                for name, value in metrics.items():
                    self.tb.add_scalar(name, value, iteration)
            except Exception:
                self.tb = None  # give up on tb; jsonl keeps going

    def close(self):
        self.jsonl.close()
        if self.tb:
            self.tb.close()


def save_checkpoint(net, cfg, iteration, metrics, path):
    torch.save({"state_dict": net.state_dict(), "config": vars(cfg),
                "iteration": iteration, "metrics": metrics}, path)


def _write_records(records, out_dir, iteration):
    for g, rec in enumerate(records):
        path = os.path.join(out_dir, f"iter{iteration:04d}_g{g:03d}.json")
        with open(path, "w") as f:
            json.dump(rec, f)


def train(cfg, out_dir=DATA_DIR, eval_every=1, log=True, use_tb=False, verbose=True):
    """Run the full loop for cfg.iterations; return (net, buffer, history)."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    rec_dir = os.path.join(out_dir, "game_records")
    for d in (ckpt_dir, rec_dir):
        os.makedirs(d, exist_ok=True)

    logger = MetricsLogger(out_dir, use_tb=use_tb) if log else None

    net = OthelloNet(cfg.num_blocks, cfg.channels).to(cfg.device)
    optimizer = make_optimizer(net, cfg)
    buffer = ReplayBuffer(cfg.buffer_size)
    history = []

    for it in range(1, cfg.iterations + 1):
        # --- self-play ---
        net.eval()
        evaluator = Evaluator(net, cfg.device)
        t0 = time.time()
        examples, records, sp = generate_games(
            evaluator, cfg, rng, cfg.games_per_iter, iteration=it, make_records=True)
        sp_time = time.time() - t0
        buffer.extend(examples)
        _write_records(records, rec_dir, it)

        # --- train ---
        loss = train_steps(net, buffer, optimizer, cfg)

        # --- evaluate (periodic; the expensive part) ---
        evals = {}
        if eval_every and it % eval_every == 0:
            net.eval()
            evals = ladder_eval(Evaluator(net, cfg.device), cfg, rng)

        # --- log + checkpoint ---
        games_per_sec = cfg.games_per_iter / sp_time if sp_time else 0.0
        row = {"iter": it, "loss": loss, "buffer": len(buffer),
               "avg_game_len": sp["avg_game_len"], "games_per_sec": games_per_sec,
               **evals}
        history.append(row)
        if logger:
            logger.log(it, flat_metrics(loss, len(buffer), games_per_sec,
                                        sp["avg_game_len"], evals))
        save_checkpoint(net, cfg, it, row, os.path.join(ckpt_dir, f"iter{it:04d}.pt"))

        if verbose:
            wr = (" | " + " ".join(f"d{d}:{w:.0%}" for d, w in evals["winrate"].items())
                  + f" maxbeat:{evals['max_depth_beaten']}") if evals else ""
            print(f"iter {it:3d}  loss {loss['total']:.3f} "
                  f"(p {loss['policy']:.3f} v {loss['value']:.3f})  "
                  f"buf {len(buffer):6d}  {games_per_sec:.1f} g/s{wr}")

    if logger:
        logger.close()
    return net, buffer, history


def main():
    ap = argparse.ArgumentParser(description="AlphaZero Othello training loop.")
    ap.add_argument("--tiny", action="store_true", help="tiny CPU config (smoke run)")
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--tensorboard", action="store_true",
                    help="also log to TensorBoard (needs a compatible protobuf)")
    ap.add_argument("--out", default=DATA_DIR)
    args = ap.parse_args()

    cfg = Config.tiny() if args.tiny else Config()
    if args.iterations is not None:
        cfg.iterations = args.iterations
    print(f"Training: {'tiny' if args.tiny else 'full'} config, "
          f"{cfg.iterations} iterations, device={cfg.device}")
    train(cfg, out_dir=args.out, eval_every=args.eval_every, use_tb=args.tensorboard)


if __name__ == "__main__":
    main()
