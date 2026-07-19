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

**Cell 3 — train** (modest first run: 5x64 net, ~30 iterations; finishes well
within a session and should climb the minimax ladder toward depth-4)
```python
!python run/train_loop.py --kaggle --out /kaggle/working/az_data
```

Progress prints per iteration: total/policy/value loss, buffer size, games/sec,
and (every iteration) win rates vs minimax + `max_depth_beaten` — the headline
strength number.

## Getting your results back
Everything is written under `/kaggle/working/az_data/`:
- `checkpoints/iter####.pt` — the model each iteration (weights + config + metrics)
- `metrics.jsonl` — the training/strength curves (one JSON line per iteration)
- `game_records/*.json` — self-play games (for the web spectator later)

After the run, the **Output** tab lists these for download. To resume next
session: download the latest checkpoint, and we'll add a `--resume <ckpt>` load
(not wired yet — say the word).

## Notes
- **No `pip install` needed** — Kaggle images ship torch + numpy. (Edax, FastAPI,
  and TensorBoard are not used by training.)
- **TensorBoard** stays off by default (a protobuf clash spams errors in some
  images); `metrics.jsonl` is the source of truth. Add `--tensorboard` only if
  your image's protobuf is compatible.
- **Long runs:** enable **Save & Run All (Commit)** for background execution so
  training survives you closing the tab (Kaggle allows ~9–12h sessions).
- **Scaling up:** the current bottleneck is unbatched self-play inference. The
  next optimization is batched leaf evaluation (evaluate many games' positions in
  one GPU call), which unlocks the full 160-sim / 200-game config.
