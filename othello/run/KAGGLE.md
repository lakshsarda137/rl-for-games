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

**Cell 3a — SMOKE RUN first** (validate batched self-play on the GPU: just 2
iterations, evaluation skipped so you get the self-play speed number fast). This
throws away its weights on purpose — the point is to confirm the batched
architecture feeds the GPU before committing a real run.
```python
!python run/train_loop.py --kaggle --iterations 2 --eval-every 0 --out /kaggle/working/az_smoke
```
**What to look for:** the per-iteration `X.X g/s` (games/sec). The pre-batching
baseline was ~0.5 g/s with the GPU idle at ~20%. Batched inference should be
several times higher, and `nvidia-smi` (run `!nvidia-smi` in a cell) should show
much higher GPU utilization during self-play. That jump *is* the validation.

**Cell 3b — the real run** (5x64 net, ~30 iterations; finishes within a session
and should climb the minimax ladder toward depth-4). Run this only after the
smoke number looks good — and note that without `--resume` a session timeout
loses progress, so a long multi-session run should wait until resume is wired.
```python
!python run/train_loop.py --kaggle --out /kaggle/working/az_data
```

Progress prints per iteration: total/policy/value loss, buffer size, games/sec,
and (when eval runs) win rates vs minimax + `max_depth_beaten` — the headline
strength number.

## Getting your results back
Everything is written under `/kaggle/working/az_data/`:
- `checkpoints/iter####.pt` — the model each iteration (weights + config + metrics)
- `metrics.jsonl` — the training/strength curves (one JSON line per iteration)
- `game_records/*.json` — self-play games (for the web spectator later)

After the run, the **Output** tab lists these for download. To resume next
session: download the latest checkpoint. A `--resume <ckpt>` loader is **not
wired yet** — until it is, a new session starts a fresh net, so don't rely on a
long run surviving a timeout. (This is the next thing to build before the real
multi-session run.)

## Notes
- **No `pip install` needed** — Kaggle images ship torch + numpy. (Edax, FastAPI,
  and TensorBoard are not used by training.)
- **TensorBoard** stays off by default (a protobuf clash spams errors in some
  images); `metrics.jsonl` is the source of truth. Add `--tensorboard` only if
  your image's protobuf is compatible.
- **Long runs:** enable **Save & Run All (Commit)** for background execution so
  training survives you closing the tab (Kaggle allows ~9–12h sessions).
- **Scaling up:** batched self-play inference is now in place — self-play plays
  `selfplay_concurrency` games at once and evaluates all their pending MCTS leaves
  in one GPU call per step (`--kaggle` uses 96 games / 96 sims as one wave). To
  push the GPU harder, raise `games_per_iter` **and** `selfplay_concurrency`
  together (bigger eval batch) via a config edit; watch `selfplay_games_per_sec`.
