"""Pull the CURRENT training checkpoint from Weights & Biases into local data/.

This is how you play the bot **mid-training**. While training runs on Kaggle with
`--wandb`, it uploads the checkpoint to W&B every few iterations as a versioned
model artifact (alias `latest`). This script downloads that `latest` artifact and
drops it at `data/checkpoints/latest.pt`, so the local web app plays the freshest
weights — no waiting for the run to finish, no second Kaggle cell.

    # one-shot: grab the current weights
    python run/pull_wandb.py --run run1

    # keep pulling the newest every 2 min while you play across the day
    python run/pull_wandb.py --run run1 --watch 120

`--run` is the W&B run name you trained with (`--wandb-run run1`); it doubles as
the artifact name. `--project` defaults to othello-alphazero; `--entity` defaults
to your logged-in W&B account.

PREREQUISITES (local, one-time):
  1. pip install wandb
  2. wandb login        (paste your key from https://wandb.ai/authorize — stored in
                         ~/.netrc; or set WANDB_API_KEY in your shell)

No W&B account interaction is destructive here; it only overwrites
data/checkpoints/latest.pt with the downloaded version.

(Alternative with zero local setup: open the run on wandb.ai -> Artifacts ->
the model -> Download, and drop latest.pt into data/checkpoints/ yourself.)
"""

import argparse
import os
import shutil
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "data"))


def _artifact_ref(entity, project, name, alias):
    prefix = f"{entity}/" if entity else ""
    return f"{prefix}{project}/{name}:{alias}"


def pull_once(entity, project, name, alias, out):
    """Download the checkpoint artifact and install it at out/checkpoints/latest.pt."""
    try:
        import wandb
    except ImportError:
        sys.exit("`wandb` not installed. Run: pip install wandb  (then `wandb login`).")

    ref = _artifact_ref(entity, project, name, alias)
    try:
        art = wandb.Api().artifact(ref, type="model")
    except Exception as exc:
        print(f"[pull-wandb] could not fetch {ref} ({type(exc).__name__}: {exc}). "
              "Has training uploaded a checkpoint yet, and is --run/--project correct?")
        return False

    tmp = art.download()                       # W&B cache dir with latest.pt
    src = os.path.join(tmp, "latest.pt")
    if not os.path.isfile(src):                # fall back to any .pt in the artifact
        pts = [f for f in os.listdir(tmp) if f.endswith(".pt")]
        if not pts:
            print(f"[pull-wandb] artifact {ref} has no .pt file.")
            return False
        src = os.path.join(tmp, pts[0])

    dst_dir = os.path.join(out, "checkpoints")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "latest.pt")
    shutil.copy2(src, dst)
    it = (art.metadata or {}).get("iteration", "?")
    print(f"[pull-wandb] installed {ref} (iteration {it}) -> {os.path.relpath(dst)}")
    print("[pull-wandb] start/refresh serve/backend.py; the web app uses the new "
          "weights on the next New Game.")
    return True


def main():
    ap = argparse.ArgumentParser(description="Pull the current W&B training checkpoint into data/.")
    ap.add_argument("--run", default="run1",
                    help="W&B run name = artifact name (your --wandb-run value; default run1)")
    ap.add_argument("--project", default="othello-alphazero", help="W&B project")
    ap.add_argument("--entity", default=None, help="W&B entity (default: your logged-in account)")
    ap.add_argument("--alias", default="latest", help="artifact alias (default: latest)")
    ap.add_argument("--out", default=DATA_DIR, help="local data dir (default: data/)")
    ap.add_argument("--watch", type=int, default=0, metavar="SECS",
                    help="re-pull every SECS seconds instead of once")
    args = ap.parse_args()

    if not args.watch:
        ok = pull_once(args.entity, args.project, args.run, args.alias, args.out)
        sys.exit(0 if ok else 1)

    print(f"[pull-wandb] watching every {args.watch}s — Ctrl-C to stop.")
    try:
        while True:
            pull_once(args.entity, args.project, args.run, args.alias, args.out)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\n[pull-wandb] stopped.")


if __name__ == "__main__":
    main()
