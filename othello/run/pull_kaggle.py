"""Pull the latest checkpoint + metrics from a Kaggle run into local data/.

This is the bridge that lets your LOCAL web app (http://127.0.0.1:8000) render a
model + dashboard that was actually trained on Kaggle's GPU. It uses the official
`kaggle` CLI to download a committed notebook's output (or a dataset), finds the
newest checkpoint and `metrics.jsonl` in it, and drops them into `data/` where the
server and `run/dashboard.py` look.

    # one-shot pull from a committed notebook's output
    python run/pull_kaggle.py --kernel <username>/<kernel-slug>

    # or from a dataset you version from the notebook
    python run/pull_kaggle.py --dataset <username>/<dataset-slug>

    # keep polling every 5 min so the local dashboard tracks a live Kaggle run
    python run/pull_kaggle.py --kernel <username>/<kernel-slug> --watch 300

PREREQUISITES (one-time):
  1. pip install kaggle
  2. Kaggle -> Account -> "Create New API Token" -> save kaggle.json to
     ~/.kaggle/kaggle.json  (chmod 600).
  3. On Kaggle, your training notebook must PERSIST /kaggle/working — use
     "Save & Run All (Commit)". Its output is then fetchable with
     `kaggle kernels output`. (Or `kaggle datasets version` a folder and use
     --dataset here.) The download step can't be tested without your credentials.

Nothing here is destructive beyond overwriting data/checkpoints/latest.pt and
data/metrics.jsonl with the freshly downloaded versions.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "data"))


def _find_checkpoint(root):
    """Newest checkpoint under `root`: prefer latest.pt, else highest iterNNNN.pt."""
    latest, best, best_it = None, None, -1
    for dirpath, _, files in os.walk(root):
        for name in files:
            if name == "latest.pt":
                latest = os.path.join(dirpath, name)
            elif name.startswith("iter") and name.endswith(".pt"):
                try:
                    it = int(name[4:-3])
                except ValueError:
                    continue
                if it > best_it:
                    best, best_it = os.path.join(dirpath, name), it
    return latest or best


def _find_metrics(root):
    """The metrics.jsonl under `root` (first one found), or None."""
    for dirpath, _, files in os.walk(root):
        if "metrics.jsonl" in files:
            return os.path.join(dirpath, "metrics.jsonl")
    return None


def install_from_dir(src, out):
    """Copy the checkpoint + metrics found under `src` into the local data layout.

    Returns a short human summary of what was installed (raises if nothing found).
    """
    ckpt = _find_checkpoint(src)
    metrics = _find_metrics(src)
    if not ckpt and not metrics:
        raise FileNotFoundError(
            f"no checkpoint (*.pt) or metrics.jsonl found in the download ({src}). "
            "Did the notebook write to --out and get committed?")
    done = []
    if ckpt:
        dst_dir = os.path.join(out, "checkpoints")
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, "latest.pt")
        shutil.copy2(ckpt, dst)
        done.append(f"checkpoint {os.path.basename(ckpt)} -> {os.path.relpath(dst)}")
    if metrics:
        os.makedirs(out, exist_ok=True)
        dst = os.path.join(out, "metrics.jsonl")
        shutil.copy2(metrics, dst)
        n = sum(1 for _ in open(metrics))
        done.append(f"metrics.jsonl ({n} iterations) -> {os.path.relpath(dst)}")
    return "; ".join(done)


def _kaggle_cli():
    exe = shutil.which("kaggle")
    if not exe:
        sys.exit("`kaggle` CLI not found. Install it (pip install kaggle) and add an "
                 "API token at ~/.kaggle/kaggle.json (Kaggle -> Account -> Create New "
                 "API Token). See the module docstring.")
    return exe


def _download(kaggle, kernel, dataset, dest):
    """Run the kaggle CLI to download into `dest`. Returns True on success."""
    if kernel:
        cmd = [kaggle, "kernels", "output", kernel, "-p", dest]
    else:
        cmd = [kaggle, "datasets", "download", "-d", dataset, "-p", dest, "--unzip"]
    print("[pull] $", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout, res.stderr, sep="\n")
        return False
    return True


def pull_once(kernel, dataset, out):
    kaggle = _kaggle_cli()
    with tempfile.TemporaryDirectory(prefix="kaggle_pull_") as tmp:
        if not _download(kaggle, kernel, dataset, tmp):
            print("[pull] download failed (see output above) — skipping this cycle.")
            return False
        summary = install_from_dir(tmp, out)
    print(f"[pull] installed: {summary}")
    print("[pull] refresh http://127.0.0.1:8000/dashboard (or rebuild with "
          "run/dashboard.py); the web app will use the new weights on the next New Game.")
    return True


def main():
    ap = argparse.ArgumentParser(description="Pull a Kaggle run's checkpoint + metrics into local data/.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--kernel", metavar="USER/SLUG", help="committed notebook to pull output from")
    src.add_argument("--dataset", metavar="USER/SLUG", help="dataset to download")
    ap.add_argument("--out", default=DATA_DIR, help="local data dir (default: data/)")
    ap.add_argument("--watch", type=int, default=0, metavar="SECS",
                    help="poll every SECS seconds instead of pulling once")
    args = ap.parse_args()

    if not args.watch:
        ok = pull_once(args.kernel, args.dataset, args.out)
        sys.exit(0 if ok else 1)

    print(f"[pull] watching every {args.watch}s — Ctrl-C to stop.")
    try:
        while True:
            pull_once(args.kernel, args.dataset, args.out)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\n[pull] stopped.")


if __name__ == "__main__":
    main()
