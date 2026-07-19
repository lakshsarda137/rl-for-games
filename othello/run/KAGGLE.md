# Training on Kaggle (free GPU)

The code is device-agnostic — `train_loop.py` auto-detects CUDA. On Kaggle you
clone this repo, enable the GPU, and run one command. Checkpoints, game records,
and `metrics.jsonl` land in `/kaggle/working/` so you can download them after.

## One-time steps
1. Go to <https://www.kaggle.com> → **Create → New Notebook**.
2. In the right sidebar: **Session options → Accelerator → GPU T4 x2** (or P100).
   Also turn **Internet → On** (needed to `git clone`).
3. Paste the three cells below and **Run All**.

## The cells

**Cell 1 — get the code**
```python
!git clone --depth 1 https://github.com/lakshsarda137/rl-for-games.git
%cd rl-for-games/othello
```

**Cell 2 — confirm the GPU is visible**
```python
import torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
```

**Cell 3a — SMOKE RUN first** (2 iterations, eval skipped so you get the
self-play speed number fast; weights thrown away on purpose). The `--kaggle`
config runs **array-ops** self-play (batched engine + MCTS, games searched in
lockstep) in a **single process** — on a GPU that's the fastest, because multiple
workers would all share the one GPU and contend for it.
```python
!python -u run/train_loop.py --kaggle --iterations 2 --eval-every 0 --out /kaggle/working/az_smoke
```
**What to look for:** the per-iteration `X.X g/s`. Baseline was **~0.4 g/s**;
array-ops hits **~2.1 g/s** on a T4. (`nvidia-smi` will still show modest GPU use
and a busy CPU — the search math is NumPy on the CPU; the network is what's on the
GPU. That's the known ceiling; the only thing that lifts it is the future
Torch-CUDA port of the search, not more CPU workers.)
> DON'T pass `--workers >1` on a GPU: they share one device and contend, which is
> *slower* (measured 2.1→1.3 g/s at 4 workers on a T4). Workers help only a
> `--device cpu` run.

**Cell 3b — the real run** (5x64 net, ~30 iterations; finishes within a session
and should climb the minimax ladder toward depth-4). Run this only after the
smoke number looks good.
```python
!python run/train_loop.py --kaggle --out /kaggle/working/az_data
```
To run *longer than one session*, resume from the previous session's checkpoint
(see "Resuming across sessions" below):
```python
!python run/train_loop.py --kaggle --resume auto --out /kaggle/working/az_data
```

Progress prints per iteration: total/policy/value loss, buffer size, games/sec.

**Evaluation is OFF by default** in the `--kaggle` config (`eval_every=0`). The
minimax-ladder eval is inspection-only — it never affects what the net learns —
and it's the slowest part of an iteration, so training runs faster without it.
Measure strength on demand instead (the web **Arena**, below). To turn it back on
for a strength curve, add `--eval-every 5` (every 5th iteration) or `--eval-every 1`.

## Getting your results back
Everything is written under `/kaggle/working/az_data/`:
- `checkpoints/iter####.pt` + `latest.pt` — the model (weights + optimizer + config)
- `metrics.jsonl` — the training curves (one JSON line per iteration)
- `game_records/*.json` — self-play games (for the web spectator later)

After the run, the **Output** tab lists these for download.

### Straight into your local web app (the pipeline)
To play/inspect the Kaggle-trained model in your **local** app at
<http://127.0.0.1:8000>, pull its `latest.pt` + `metrics.jsonl` into local `data/`
with the Kaggle API (needs `pip install kaggle` + an API token, and the notebook
**committed** via *Save & Run All* so its output is fetchable):

```bash
python run/pull_kaggle.py --kernel <username>/<kernel-slug>          # one-shot
python run/pull_kaggle.py --kernel <username>/<kernel-slug> --watch 300   # poll every 5 min
```

It drops the newest checkpoint at `data/checkpoints/latest.pt` and metrics at
`data/metrics.jsonl`. Then start `python serve/backend.py` — the play UI picks up
the new weights on the next New Game, and `/dashboard` shows the curves. (If you
version outputs as a Kaggle Dataset instead of a committed kernel, use
`--dataset <username>/<slug>`.)

## Resuming across sessions

A Kaggle session is wiped when it ends, so a multi-session run has to carry the
checkpoint across. The `.pt` file holds everything needed — weights, optimizer
state, RNG state, config, and the iteration number — so resuming is seamless.

1. **End of session:** download `checkpoints/latest.pt` from the **Output** tab
   (or add `/kaggle/working/az_data` as a notebook output / a Kaggle Dataset).
2. **Next session:** make that file available again — the simplest is to attach
   it as an **input dataset** (right sidebar → *Add Input*), which mounts it under
   `/kaggle/input/<your-dataset>/`. Copy it into place and resume:
   ```python
   !mkdir -p /kaggle/working/az_data/checkpoints
   !cp /kaggle/input/<your-dataset>/latest.pt /kaggle/working/az_data/checkpoints/
   !python run/train_loop.py --kaggle --resume auto --out /kaggle/working/az_data
   ```
   `--resume auto` picks the newest checkpoint in that dir; or point at it
   explicitly with `--resume /kaggle/working/az_data/checkpoints/latest.pt`.

`--iterations N` means **N more** iterations when resuming, and the iteration
counter + `metrics.jsonl` continue on one timeline. (The replay buffer isn't
saved — it refills over the first 1–2 iterations, which is fine.) So each session
adds ~30 iterations on top of the last, and the strength curve keeps climbing.

## Notes
- **No `pip install` needed** — Kaggle images ship torch + numpy. (Edax, FastAPI,
  and TensorBoard are not used by training.)
- **TensorBoard** stays off by default (a protobuf clash spams errors in some
  images); `metrics.jsonl` is the source of truth. Add `--tensorboard` only if
  your image's protobuf is compatible.
- **Long runs:** enable **Save & Run All (Commit)** for background execution so
  training survives you closing the tab (Kaggle allows ~9–12h sessions).
- **Scaling up:** two throughput levers are in place — batched inference (many
  games share one net call) and **multiprocess self-play** (`--workers N` splits
  games over N processes). The workers are the real win here, because self-play is
  CPU-bound single-threaded Python; set `--workers` to the session's vCPU count.
  Watch `selfplay_games_per_sec`. The next, bigger lever (moving the MCTS/engine
  onto the GPU via a CUDA kernel) needs an NVIDIA GPU to build — it's future work.
