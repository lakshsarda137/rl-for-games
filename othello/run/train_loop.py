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
from selfplay import generate_games, shutdown_selfplay_workers
from train import make_optimizer, train_steps

DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "data"))


def resolve_device(requested):
    """Turn 'cuda'/'auto'/'cpu' into a device that actually exists here."""
    if requested in ("cuda", "auto") and torch.cuda.is_available():
        return "cuda"
    if requested == "cuda":
        print("[device] CUDA not available -> falling back to CPU")
        return "cpu"
    if requested == "auto":
        return "cpu"
    return requested


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
    """Robust metrics sink: always writes metrics.jsonl; TensorBoard + W&B optional.

    metrics.jsonl is the source of truth and never fails. TensorBoard and Weights
    & Biases are both fully guarded — if either can't initialise (not installed,
    not logged in, protobuf clash), we note it once and keep going on the jsonl.

    Weights & Biases is the LIVE remote view: the training process pushes each
    iteration to W&B's servers over the internet, so you watch a real-time
    dashboard at wandb.ai from anywhere (e.g. your laptop while training runs on
    Kaggle) — no waiting for a run to finish and be pulled. Needs `pip install
    wandb` and a WANDB_API_KEY (see run/KAGGLE.md).
    """

    def __init__(self, out_dir, use_tb=False, append=False,
                 use_wandb=False, wandb_project=None, wandb_run=None, config=None):
        # A fresh run truncates metrics.jsonl (start a clean timeline); a resume
        # appends, so the log continues the earlier run's iterations in one file.
        os.makedirs(out_dir, exist_ok=True)
        self.jsonl = open(os.path.join(out_dir, "metrics.jsonl"), "a" if append else "w")
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

        self.wandb = None
        if use_wandb:
            try:
                import wandb
                # A named run uses id=name + resume="allow" so the SAME run (and one
                # continuous curve) resumes across sessions. No name -> let W&B
                # auto-generate a fresh run (don't pass resume without an id).
                init_kwargs = {"project": wandb_project or "othello-alphazero",
                               "config": config}
                if wandb_run:
                    init_kwargs.update(id=wandb_run, name=wandb_run, resume="allow")
                self.wandb = wandb.init(**init_kwargs)
                print(f"[metrics] Weights & Biases live dashboard: {self.wandb.url}")
            except Exception as exc:
                print(f"[metrics] wandb disabled ({type(exc).__name__}: {exc}); "
                      "using metrics.jsonl. Check `pip install wandb` + WANDB_API_KEY.")
                self.wandb = None
        # Artifact name for uploaded checkpoints; equals the run name so a local
        # `pull_wandb.py --run NAME` can find it (falls back to "checkpoint").
        self._ckpt_name = wandb_run or "checkpoint"

    def log(self, iteration, metrics):
        self.jsonl.write(json.dumps({"iter": iteration, **metrics}) + "\n")
        self.jsonl.flush()
        if self.tb:
            try:
                for name, value in metrics.items():
                    self.tb.add_scalar(name, value, iteration)
            except Exception:
                self.tb = None  # give up on tb; jsonl keeps going
        if self.wandb:
            try:
                self.wandb.log(metrics, step=iteration)
            except Exception:
                self.wandb = None  # give up on wandb; jsonl keeps going

    def log_checkpoint(self, path, iteration):
        """Upload a checkpoint to W&B as a versioned artifact (alias 'latest').

        This is what lets you PLAY the bot mid-training: the running process pushes
        the current weights to W&B, and `run/pull_wandb.py` (or the W&B UI) fetches
        them from your laptop at any time — no waiting for the run to finish, no
        second Kaggle cell. No-op unless W&B is active; never breaks training.
        """
        if not self.wandb or not os.path.isfile(path):
            return
        try:
            import wandb
            art = wandb.Artifact(self._ckpt_name, type="model",
                                 metadata={"iteration": iteration})
            art.add_file(path, name="latest.pt")
            self.wandb.log_artifact(art, aliases=["latest", f"iter{iteration}"])
        except Exception as exc:
            print(f"[metrics] wandb checkpoint upload skipped "
                  f"({type(exc).__name__}: {exc})")

    def close(self):
        self.jsonl.close()
        if self.tb:
            self.tb.close()
        if self.wandb:
            try:
                self.wandb.finish()
            except Exception:
                pass


def save_checkpoint(net, cfg, iteration, metrics, path, optimizer=None, rng=None):
    """Write a resumable checkpoint.

    Stores everything needed to pick training back up exactly where it stopped:
    the network weights, the full config (so we know the architecture), the
    iteration counter, the latest metrics, and — for a faithful resume — the
    optimizer state (Adam's per-parameter moments) plus the RNG states. The
    replay buffer is deliberately NOT stored (it refills over ~1-2 iterations).
    """
    payload = {"state_dict": net.state_dict(), "config": vars(cfg),
               "iteration": iteration, "metrics": metrics}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if rng is not None:
        payload["numpy_rng_state"] = rng.bit_generator.state
        payload["torch_rng_state"] = torch.get_rng_state()
    torch.save(payload, path)


def _find_latest_checkpoint(ckpt_dir):
    """Newest iterNNNN.pt in ckpt_dir (highest iteration), or None if there are none."""
    if not os.path.isdir(ckpt_dir):
        return None
    best, best_it = None, -1
    for name in os.listdir(ckpt_dir):
        if name.startswith("iter") and name.endswith(".pt"):
            try:
                it = int(name[4:-3])
            except ValueError:
                continue
            if it > best_it:
                best, best_it = os.path.join(ckpt_dir, name), it
    return best


def load_for_resume(path, cfg, ckpt_dir):
    """Resolve a --resume value to a checkpoint and rebuild net + optimizer from it.

    `path` may be an explicit .pt file, or "auto"/"latest" to grab the newest
    checkpoint in `ckpt_dir`. The network is rebuilt with the *checkpoint's*
    architecture (num_blocks/channels) so the weights load cleanly; the rest of
    the hyperparameters come from the current `cfg`, so you can change lr, sims,
    games_per_iter, etc. across sessions. Returns (net, optimizer, start_iter,
    ckpt) — ckpt is the raw loaded dict so the caller can restore RNG state
    without re-reading the file.
    """
    if path in ("auto", "latest"):
        resolved = _find_latest_checkpoint(ckpt_dir)
        if resolved is None:
            raise FileNotFoundError(
                f"--resume {path}: no iter*.pt checkpoints found in {ckpt_dir}")
        path = resolved
    if not os.path.isfile(path):
        raise FileNotFoundError(f"--resume: checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=cfg.device)
    saved = ckpt.get("config", {})
    nb = saved.get("num_blocks", cfg.num_blocks)
    ch = saved.get("channels", cfg.channels)
    if (nb, ch) != (cfg.num_blocks, cfg.channels):
        print(f"[resume] checkpoint architecture is {nb}x{ch}; using it "
              f"(config asked for {cfg.num_blocks}x{cfg.channels}).")

    net = OthelloNet(nb, ch).to(cfg.device)
    net.load_state_dict(ckpt["state_dict"])
    optimizer = make_optimizer(net, cfg)
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        # Honour any lr/weight_decay the user changed for this session.
        for group in optimizer.param_groups:
            group["lr"] = cfg.lr
            group["weight_decay"] = cfg.weight_decay
    start_iter = int(ckpt.get("iteration", 0))
    print(f"[resume] loaded {os.path.basename(path)} (iteration {start_iter}); "
          f"continuing at iteration {start_iter + 1}. "
          f"{'optimizer restored' if 'optimizer' in ckpt else 'fresh optimizer'}; "
          "replay buffer starts empty and refills.")
    return net, optimizer, start_iter, ckpt


def _write_records(records, out_dir, iteration):
    for g, rec in enumerate(records):
        path = os.path.join(out_dir, f"iter{iteration:04d}_g{g:03d}.json")
        with open(path, "w") as f:
            json.dump(rec, f)


def train(cfg, out_dir=DATA_DIR, eval_every=None, log=True, use_tb=False,
          verbose=True, resume=None, use_wandb=False, wandb_project=None,
          wandb_run=None, wandb_ckpt_every=2):
    """Run the loop for cfg.iterations more iterations; return (net, buffer, history).

    `resume` (a checkpoint path, or "auto"/"latest") continues an earlier run:
    the net + optimizer are loaded and iteration numbering picks up where the
    checkpoint left off, so metrics.jsonl and checkpoints keep a single timeline
    across sessions. `cfg.iterations` is always "how many MORE iterations now".
    """
    cfg.device = resolve_device(cfg.device)
    if eval_every is None:                     # default: honour the config
        eval_every = getattr(cfg, "eval_every", 1)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    rec_dir = os.path.join(out_dir, "game_records")
    for d in (ckpt_dir, rec_dir):
        os.makedirs(d, exist_ok=True)

    logger = MetricsLogger(out_dir, use_tb=use_tb, append=bool(resume),
                           use_wandb=use_wandb, wandb_project=wandb_project,
                           wandb_run=wandb_run, config=vars(cfg)) if log else None

    if resume:
        net, optimizer, start_iter, ckpt = load_for_resume(resume, cfg, ckpt_dir)
        if "numpy_rng_state" in ckpt:            # deterministic continuation
            rng.bit_generator.state = ckpt["numpy_rng_state"]
        if "torch_rng_state" in ckpt:
            torch.set_rng_state(ckpt["torch_rng_state"].to("cpu"))
    else:
        net = OthelloNet(cfg.num_blocks, cfg.channels).to(cfg.device)
        optimizer = make_optimizer(net, cfg)
        start_iter = 0
    buffer = ReplayBuffer(cfg.buffer_size)
    history = []

    try:
        for it in range(start_iter + 1, start_iter + 1 + cfg.iterations):
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
            save_checkpoint(net, cfg, it, row, os.path.join(ckpt_dir, f"iter{it:04d}.pt"),
                            optimizer=optimizer, rng=rng)
            # A stable name for the newest checkpoint, so `--resume` and
            # load-to-play don't need to know the iteration number.
            latest_path = os.path.join(ckpt_dir, "latest.pt")
            save_checkpoint(net, cfg, it, row, latest_path, optimizer=optimizer, rng=rng)
            # Push the current weights to W&B every N iters so you can pull + play
            # the bot mid-training (no-op unless --wandb). See run/pull_wandb.py.
            if logger and wandb_ckpt_every and it % wandb_ckpt_every == 0:
                logger.log_checkpoint(latest_path, it)

            if verbose:
                wr = (" | " + " ".join(f"d{d}:{w:.0%}" for d, w in evals["winrate"].items())
                      + f" maxbeat:{evals['max_depth_beaten']}") if evals else ""
                print(f"iter {it:3d}  loss {loss['total']:.3f} "
                      f"(p {loss['policy']:.3f} v {loss['value']:.3f})  "
                      f"buf {len(buffer):6d}  {games_per_sec:.1f} g/s{wr}")
    finally:
        shutdown_selfplay_workers()   # reap self-play worker processes
        if logger:
            logger.close()
    return net, buffer, history


def main():
    ap = argparse.ArgumentParser(description="AlphaZero Othello training loop.")
    ap.add_argument("--tiny", action="store_true", help="tiny CPU config (smoke run)")
    ap.add_argument("--kaggle", action="store_true", help="modest GPU config (first Kaggle run)")
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None,
                    help="run the minimax-ladder eval every N iterations (0 = never). "
                         "Default: from the config (kaggle=0/off, others=1). Eval is "
                         "inspection-only — it never affects what the net learns.")
    ap.add_argument("--sims", type=int, default=None, metavar="N",
                    help="MCTS simulations per move in SELF-PLAY (overrides cfg.sims_selfplay). "
                         "Higher = stronger training targets but fewer games/sec. Vary it per "
                         "resumed session for a low->high ramp (e.g. 96 then 200).")
    ap.add_argument("--sims-eval", type=int, default=None, metavar="N",
                    help="MCTS simulations per move in EVALUATION (overrides cfg.sims_eval).")
    ap.add_argument("--workers", type=int, default=None,
                    help="self-play worker processes (default from config; set to #CPU cores)")
    ap.add_argument("--net", default=None, metavar="BxC",
                    help="override network size as BLOCKSxCHANNELS (e.g. 10x128). Bigger = "
                         "stronger ceiling but needs more iterations. On resume the checkpoint's "
                         "own architecture wins (weights must match).")
    ap.add_argument("--games", type=int, default=None, metavar="N",
                    help="self-play games per iteration (overrides cfg.games_per_iter). The "
                         "torch search path (--selfplay-torch) wants a big batch here.")
    ap.add_argument("--selfplay-torch", action="store_true",
                    help="run array-ops self-play on the TORCH engine + MCTS so the whole search "
                         "runs on the GPU (not just the net). Device-agnostic; pair with a big "
                         "--games batch. Opt-in — measure the g/s before relying on it.")
    ap.add_argument("--device", default=None, help="override device (cuda/cpu/auto)")
    ap.add_argument("--tensorboard", action="store_true",
                    help="also log to TensorBoard (needs a compatible protobuf)")
    ap.add_argument("--wandb", action="store_true",
                    help="also stream metrics live to Weights & Biases (needs "
                         "`pip install wandb` + WANDB_API_KEY). Watch at wandb.ai.")
    ap.add_argument("--wandb-project", default="othello-alphazero",
                    help="W&B project name (default: othello-alphazero)")
    ap.add_argument("--wandb-run", default=None, metavar="NAME",
                    help="W&B run name/id. Reuse the SAME name with --resume to "
                         "continue one live curve across sessions.")
    ap.add_argument("--wandb-ckpt-every", type=int, default=2, metavar="N",
                    help="with --wandb, upload the checkpoint to W&B every N iters so "
                         "you can pull + play the bot mid-training (0 = never).")
    ap.add_argument("--resume", default=None, metavar="CKPT",
                    help="continue from a checkpoint: a path, or 'auto'/'latest' for "
                         "the newest in <out>/checkpoints. --iterations is then how "
                         "many MORE iterations to run.")
    ap.add_argument("--out", default=DATA_DIR)
    args = ap.parse_args()

    if args.kaggle:
        cfg, name = Config.kaggle(), "kaggle"
    elif args.tiny:
        cfg, name = Config.tiny(), "tiny"
    else:
        cfg, name = Config(), "full"
    if args.iterations is not None:
        cfg.iterations = args.iterations
    if args.net is not None:
        try:
            nb, ch = args.net.lower().split("x")
            cfg.num_blocks, cfg.channels = int(nb), int(ch)
        except ValueError:
            ap.error(f"--net expects BLOCKSxCHANNELS (e.g. 10x128), got {args.net!r}")
    if args.games is not None:
        cfg.games_per_iter = args.games
    if args.selfplay_torch:
        cfg.selfplay_torch = True
    if args.sims is not None:
        cfg.sims_selfplay = args.sims
    if args.sims_eval is not None:
        cfg.sims_eval = args.sims_eval
    if args.workers is not None:
        cfg.selfplay_workers = args.workers
    if args.device:
        cfg.device = args.device
    if args.eval_every is not None:
        cfg.eval_every = args.eval_every
    verb = "more iterations (resume)" if args.resume else "iterations"
    eval_note = "eval off" if cfg.eval_every == 0 else f"eval every {cfg.eval_every}"
    sp_backend = "torch-search" if getattr(cfg, "selfplay_torch", False) else (
        "arrayops" if cfg.selfplay_arrayops else "pool")
    print(f"Training: {name} config, {cfg.iterations} {verb}, "
          f"net {cfg.num_blocks}x{cfg.channels}, device={resolve_device(cfg.device)}, "
          f"{cfg.games_per_iter} games/iter, {cfg.sims_selfplay} self-play sims "
          f"({sp_backend}), {eval_note}")
    train(cfg, out_dir=args.out, use_tb=args.tensorboard, resume=args.resume,
          use_wandb=args.wandb, wandb_project=args.wandb_project, wandb_run=args.wandb_run,
          wandb_ckpt_every=args.wandb_ckpt_every)


if __name__ == "__main__":
    main()
