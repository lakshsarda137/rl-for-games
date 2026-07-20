# Training on Kaggle (free GPU)

`train_loop.py` is device-agnostic — it auto-detects CUDA. You clone this repo
into a Kaggle notebook, turn on the GPU, and run. Two ways to see how training is
going:

- **Live, while it runs** → Weights & Biases (`--wandb`), watched from your laptop.
  This is the only way to see *in-progress* remote training.
- **After a committed run** → pull the checkpoint + `metrics.jsonl` down and use the
  local web app / dashboard (`run/pull_kaggle.py`). This is for *playing* the finished
  model, not live monitoring.

The good settings are baked into the **`--kaggle`** config: the 10×128 net, array-ops
self-play in a **single process** (fastest on a shared GPU), eval off, 30 iterations.
(A bigger net raises the strength ceiling but needs more iterations to fill — so plan
a longer, multi-session run, or drop back to `--net 5x64` for the old quick config.)

## One-time setup

**A. The notebook** — <https://www.kaggle.com> → **Create → New Notebook**. Right
sidebar: **Accelerator → GPU T4 x2** (or P100), and **Internet → On** (needed to
`git clone` and to reach W&B).

**B. Weights & Biases** (to watch training live) — make a free account at
<https://wandb.ai>, copy your key from <https://wandb.ai/authorize>, then in the
notebook add it as a **Secret**: *Add-ons → Secrets → Add secret*, name it exactly
`WANDB_API_KEY`. **Never paste the key into a code cell.**

**C. Kaggle API token** (only needed *locally*, to pull results to your Mac) —
`pip install kaggle`, then a token from <https://www.kaggle.com/settings> → **API →
Create New Token** → save to `~/.kaggle/kaggle.json` (or `export KAGGLE_API_TOKEN=…`
in your shell profile). See "Pull results to your machine" below.

## The notebook cells

**Cell 1 — get the code** (re-clone every session; Kaggle wipes the disk):
```python
!git clone --depth 1 https://github.com/lakshsarda137/rl-for-games.git
%cd rl-for-games/othello
```

**Cell 2 — confirm the GPU:**
```python
import torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
```

**Cell 3 — smoke run** (2 iters, eval off, weights thrown away — just checks speed):
```python
!python -u run/train_loop.py --kaggle --iterations 2 --eval-every 0 --out /kaggle/working/az_smoke
```
Look at the per-iteration `X.X g/s`. Baseline was ~0.4; array-ops hits ~2.1 on a T4.
(`nvidia-smi` shows modest GPU + a busy CPU — the search math is NumPy on the CPU,
only the network is on the GPU; that's the known ceiling.)

**Cell 4 — the real run, with the live W&B dashboard:**
```python
!pip install -q wandb                       # if the Kaggle image doesn't have it
from kaggle_secrets import UserSecretsClient
import os
os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")

!python run/train_loop.py --kaggle --wandb --wandb-run run1 --out /kaggle/working/az_data
```
It prints a **wandb.ai URL** — open it on your Mac and the loss/speed curves update
**every iteration, live**. When it's done, **Save Version → "Save & Run All
(Commit)"** so the checkpoint + metrics become downloadable (that's what makes the
local pull work).

*No W&B?* Drop the three wandb lines and the `--wandb --wandb-run run1` flags —
training is identical, you just lose the live view.

## Command reference (run / resume / options)

Run from the `othello/` dir. `--out /kaggle/working/az_data` keeps outputs together.

| Goal | Command |
|---|---|
| Smoke test (speed) | `python run/train_loop.py --kaggle --iterations 2 --eval-every 0 --out /kaggle/working/az_data` |
| Real run (live W&B) | `python run/train_loop.py --kaggle --wandb --wandb-run run1 --out /kaggle/working/az_data` |
| Resume next session | `python run/train_loop.py --kaggle --resume auto --wandb --wandb-run run1 --out /kaggle/working/az_data` |
| + strength curves | add `--eval-every 5` (win-rate / `max_depth_beaten` every 5 iters) |
| Run more/fewer iters | add `--iterations N` (when resuming, N = N *more* iterations) |
| Stronger training targets | add `--sims N` (MCTS sims/move in self-play; default 96). Higher = better targets, fewer games/sec |
| Bigger/smaller net | add `--net BxC` (default `10x128`; `--net 5x64` = the old quick net) |
| More games per iter | add `--games N` (self-play batch size; default 96) |
| GPU search (experimental) | add `--selfplay-torch --games N` (runs the SEARCH on the GPU, not just the net; use a BIG N and compare g/s to the default) |

The flags that matter:
- **`--kaggle`** — the GPU config: 10×128 net, array-ops self-play, `workers=1`, eval off, 30 iters, `device=cuda`.
- **`--net BxC`** — override the network size (e.g. `--net 5x64` for the old net, `--net 10x128` = default).
  Bigger = stronger ceiling but slower to fill; on **resume** the checkpoint's own architecture wins.
- **`--games N`** — self-play games per iteration (default 96). Pair a big N with `--selfplay-torch`.
- **`--selfplay-torch`** — run array-ops self-play on the **Torch** engine + MCTS so the whole *search*
  runs on the GPU (the NumPy default keeps the search on the CPU; only the net is on-device). Correctness
  is proven identical to the NumPy path; the **speed is experimental** — use a big `--games` batch (the
  per-step cost is fixed regardless of batch, so it only amortises when many games run at once) and
  compare the `g/s` line to a plain `--kaggle` run before trusting it.
- **`--wandb [--wandb-run NAME]`** — stream metrics live to wandb.ai. Reuse the **same** NAME with `--resume` to continue **one** live curve across sessions.
- **`--resume auto`** — continue the newest checkpoint (see next section). `--iterations N` then means N *more* iterations.
- **`--eval-every N`** — minimax-ladder eval every N iters (**0 = off, the default**). It's inspection-only (never affects learning) and the slowest part of an iteration, so it's off by default; measure strength on demand with the web **Arena** instead, or turn it on here for live strength curves.
- **`--sims N` / `--sims-eval N`** — MCTS simulations per move in self-play / eval (override `cfg.sims_selfplay` / `cfg.sims_eval`; defaults 96 / 128). More self-play sims = a stronger "teacher" → better training targets, at fewer games/sec. To ramp low→high, just pass a bigger `--sims` on later resumed sessions (e.g. session 1 `--sims 96`, session 2 `--resume auto --sims 200`).

## Resuming across sessions

A Kaggle session is wiped when it ends, so a multi-session run carries the
checkpoint across. The `.pt` holds weights + optimizer + RNG + config + iteration,
so resume is seamless.

1. **End each session** with **Save Version → "Save & Run All (Commit)"** so
   `/kaggle/working` is persisted.
2. **Next session**, bring the previous checkpoint back — simplest is **Add Input →
   Notebook Output → your notebook**, which mounts it under `/kaggle/input/…`. Then:
   ```python
   import glob, os, shutil
   os.makedirs("/kaggle/working/az_data/checkpoints", exist_ok=True)
   src = sorted(glob.glob("/kaggle/input/**/latest.pt", recursive=True))
   assert src, "Attach your previous notebook output as an Input first."
   shutil.copy(src[0], "/kaggle/working/az_data/checkpoints/latest.pt")
   print("resuming from", src[0])
   !python run/train_loop.py --kaggle --resume auto --wandb --wandb-run run1 --out /kaggle/working/az_data
   ```
`--resume auto` picks the newest checkpoint; iteration numbering + `metrics.jsonl`
continue on one timeline; reusing `--wandb-run run1` continues the same live curve.
(The replay buffer isn't saved — it refills over the first 1–2 iters.) A big run is
inherently several sessions: ~45 min per 30 iters, and Kaggle gives roughly ~30h of
GPU per week with ~9–12h per session — so a ~1000-iteration run is many resumed
sessions, watched live on W&B each time.

## Play the bot MID-training (no waiting, no second cell)

You do **not** have to wait for a run to finish to play the current bot — and you
**can't** do it from a second Kaggle cell (the notebook runs one cell at a time, so
a new cell just queues behind the training cell). The trick is the same push model
as W&B metrics: when you train with `--wandb`, the process **uploads the checkpoint
to W&B every few iterations** (`--wandb-ckpt-every N`, default 5) as a model
artifact aliased `latest`. You then fetch it to your Mac **any time**, decoupled
from the training cell:

```bash
# local, one-time: pip install wandb ; wandb login  (key from wandb.ai/authorize)
cd othello
python run/pull_wandb.py --run run1            # grab the current weights now
python run/pull_wandb.py --run run1 --watch 120   # keep the local copy fresh
python serve/backend.py                        # play "AZ net" at http://127.0.0.1:8000
```
`--run run1` matches your `--wandb-run run1`. (Zero-setup alternative: open the run
on wandb.ai → **Artifacts** → the model → **Download**, and drop `latest.pt` into
`data/checkpoints/`.) So you can play the bot at iteration 12, 40, 200 — whatever it
has reached — while it keeps training on Kaggle.

## Pull the FINAL results to your machine (committed run)

`pull_wandb.py` (above) is the live path. `pull_kaggle.py` is the after-the-fact
path: it fetches a **committed** run's full output (checkpoint + `metrics.jsonl` +
game records). Needs the Kaggle API token (setup **C**) and a committed run.

```bash
cd othello
python run/pull_kaggle.py --kernel lakshsarda/othello-rl-training            # one-shot
python run/pull_kaggle.py --kernel lakshsarda/othello-rl-training --watch 300   # re-pull each new commit
```
It installs the newest checkpoint → `data/checkpoints/latest.pt` and metrics →
`data/metrics.jsonl`. Then:
```bash
python serve/backend.py     # http://127.0.0.1:8000        : play "AZ net", or Arena for aggregate win rates
                            # http://127.0.0.1:8000/dashboard : the metric curves
```
`pull_kaggle` only sees a **committed** run's output (not an in-progress one) — that
is exactly why W&B is the live view. (`--dataset <user>/<slug>` instead of `--kernel`
if you version outputs as a Kaggle Dataset.)

Everything committed lands under `/kaggle/working/az_data/`: `checkpoints/iter####.pt`
+ `latest.pt`, `metrics.jsonl`, `game_records/*.json`. The **Output** tab also lists
them for manual download if you'd rather not use the API.

## Notes
- **Deps:** Kaggle images ship torch + numpy (no install for training). `--wandb`
  needs `pip install wandb` (Cell 4 does it). Edax / FastAPI are only for the local app.
- **On a GPU keep `workers=1`.** `--workers >1` makes processes contend for the one
  shared GPU and is *slower* (measured 2.1→1.3 g/s at 4 workers on a T4); it only
  helps a `--device cpu` run. `--kaggle` already sets `workers=1`.
- **Throughput:** the default lever is NumPy array-ops self-play (~2.1 g/s on a T4, ~5×
  the 0.4 baseline), whose tree-search/rules math is NumPy-on-CPU (only the network is on
  the GPU — the known ceiling). The **`--selfplay-torch`** path moves that search onto the
  GPU too; it's correctness-proven but its speed is still to be measured (see CLAUDE.md
  next-steps item 1) — that's the point of smoking it with a big `--games` and reading g/s.
- **TensorBoard** stays off (protobuf clash in some images); `metrics.jsonl` + W&B
  are the metric sinks. `metrics.jsonl` is always the source of truth.
