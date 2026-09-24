"""Train the rummy network: self-play with search, learn, check against greedy, repeat.

    python run/train_loop.py --tiny                 # few-minute check on a laptop
    python run/train_loop.py --kaggle --resume auto # the real run on a GPU
    python run/train_loop.py --tiny --no-hand-guess # comparison run: hand guessing off

Everything trains together from the first iteration: the move choice, the
result estimate and the hand guess are three outputs of one network, and the
search uses the hand guess while it plays.

Each run writes to data/<run name>/: checkpoints (latest.pt and every eval),
and metrics.jsonl with one line per iteration.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
sys.path.insert(0, os.path.join(_ROOT, "az"))
sys.path.insert(0, _HERE)

from config import Config
from evaluate import vs_greedy
from network import Evaluator, RummyNet, pick_device
from selfplay import play
from train import ReplayBuffer, lr_at_iteration, train_steps


def save(path, net, optimizer, cfg, iteration):
    tmp = path + ".tmp"
    torch.save({"net": net.state_dict(), "optimizer": optimizer.state_dict(),
                "config": asdict(cfg), "iteration": iteration}, tmp)
    os.replace(tmp, path)   # never leave a half-written checkpoint behind


def load_net(path, device="cpu"):
    """A trained network from a checkpoint, ready to play (used by the website too)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    c = ckpt["config"]
    net = RummyNet(c["blocks"], c["channels"])
    net.load_state_dict(ckpt["net"])
    return net.to(device).eval(), ckpt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiny", action="store_true", help="small, fast settings for a laptop check")
    ap.add_argument("--kaggle", action="store_true", help="settings for a Kaggle GPU")
    ap.add_argument("--iterations", type=int, default=None, help="stop after this many (default: 6 tiny, 200 otherwise)")
    ap.add_argument("--run", default=None, help="run name, the folder under data/ (default from the settings)")
    ap.add_argument("--resume", default=None, help="'auto' to continue data/<run>/latest.pt, or a checkpoint path")
    ap.add_argument("--no-hand-guess", action="store_true", help="comparison run: no hand-guess loss, search imagines cards uniformly")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None, help="folder for this run's files (default data/<run>); on Kaggle use /kaggle/working/...")
    ap.add_argument("--eval-every", type=int, default=None, help="check against greedy every N iterations (0 = never)")
    ap.add_argument("--wandb", action="store_true", help="stream metrics to Weights & Biases (needs WANDB_API_KEY)")
    args = ap.parse_args(argv)

    cfg = Config.tiny() if args.tiny else Config.kaggle() if args.kaggle else Config()
    cfg = replace(cfg, device=pick_device(args.device))
    if args.no_hand_guess:
        cfg = replace(cfg, hand_weight=0.0, use_hand_guess=False)
    if args.eval_every is not None:
        cfg = replace(cfg, eval_every=args.eval_every)
    iterations = args.iterations or (6 if args.tiny else 200)
    run = args.run or ("tiny" if args.tiny else "main") + ("-nohand" if args.no_hand_guess else "")
    out_dir = args.out or os.path.join(_ROOT, "data", run)
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    net = RummyNet(cfg.blocks, cfg.channels).to(cfg.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    start = 1
    resume = os.path.join(out_dir, "latest.pt") if args.resume == "auto" else args.resume
    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=cfg.device, weights_only=False)
        net.load_state_dict(ckpt["net"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start = ckpt["iteration"] + 1
        rng = np.random.default_rng(args.seed + start)
        print(f"Resuming {resume} at iteration {start}")
    buffer = ReplayBuffer(cfg.buffer_size)
    params = sum(p.numel() for p in net.parameters())
    print(f"Run '{run}' on {cfg.device}: {params:,} weights, {cfg.worlds} worlds x {cfg.sims} sims, "
          f"{cfg.hands_per_iter} hands per iteration, hand guessing {'on' if cfg.use_hand_guess else 'off'}")
    wandb = None
    if args.wandb:
        import wandb
        wandb.init(project="rummy-rl", name=run, config=asdict(cfg), resume="allow")

    for it in range(start, start + iterations):
        t0 = time.time()
        evaluator = Evaluator(net, cfg.device, use_hand_guess=cfg.use_hand_guess)
        examples, sp = play(evaluator, cfg, cfg.hands_per_iter, seed=int(rng.integers(1 << 62)))
        buffer.add(examples)
        t_play = time.time() - t0

        for group in optimizer.param_groups:
            group["lr"] = lr_at_iteration(cfg, it)
        losses = train_steps(net, buffer, optimizer, cfg, rng)
        record = {"iteration": it, "lr": optimizer.param_groups[0]["lr"], **losses, **sp,
                  "buffer": len(buffer), "selfplay_s": round(t_play, 1), "total_s": 0}

        line = (f"iter {it:4d}  loss: move {losses['policy']:.3f}  result {losses['value']:.4f}  "
                f"hand {losses['hand']:.3f}  |  {sp['hands']} hands, {sp['avg_turns']:.0f} turns avg, "
                f"takes discard {sp['pile_take_rate']:.0%}  |  {t_play:.0f}s play")
        if cfg.eval_every and it % cfg.eval_every == 0:
            ev = vs_greedy(Evaluator(net, cfg.device, cfg.use_hand_guess), cfg.eval_pairs,
                           seed=10_000 + it, sims=cfg.eval_sims, worlds=cfg.eval_worlds,
                           c_puct=cfg.c_puct, turn_cap=cfg.turn_cap)
            record["eval"] = ev
            line += (f"  |  vs greedy: {ev['points_per_hand']:+.1f} +/- {ev['range_95']:.1f} points/hand, "
                     f"wins {ev['win_rate']:.0%}")
            save(os.path.join(out_dir, f"iter{it:04d}.pt"), net, optimizer, cfg, it)
        save(os.path.join(out_dir, "latest.pt"), net, optimizer, cfg, it)
        record["total_s"] = round(time.time() - t0, 1)
        with open(os.path.join(out_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")
        if wandb:
            flat = {k: v for k, v in record.items() if k != "eval"}
            flat.update({f"eval/{k}": v for k, v in record.get("eval", {}).items()})
            wandb.log(flat, step=it)
        print(line + f"  ({record['total_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
