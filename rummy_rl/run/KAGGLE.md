# Training the rummy AI on Kaggle (free GPU)

Same setup as the Othello project. Clone the repo into a Kaggle notebook, turn
on the GPU, build the C++ engine, and run the training loop.

## One-time setup

1. <https://www.kaggle.com> → **Create → New Notebook**. In the right sidebar set
   **Accelerator → GPU T4 x2** (or P100) and **Internet → On**.
2. (Optional, to watch training live) add your Weights & Biases key as a notebook
   secret: *Add-ons → Secrets → Add secret*, named exactly `WANDB_API_KEY`.

## Notebook cells

**1. Get the code** (every session; Kaggle wipes the disk):
```python
!git clone --depth 1 https://github.com/lakshsarda137/rl-for-games.git
%cd rl-for-games/rummy_rl
```

**2. Build the C++ engine** (about 10 seconds) and run the tests:
```python
!python native/build.py
!python run_tests.py
```

**3. Quick check** (2 iterations, no strength check). Look at the seconds per
iteration to plan how long the real run will take:
```python
!python -u run/train_loop.py --kaggle --iterations 2 --eval-every 0 --out /kaggle/working/smoke
```

**4. The real run** (writes checkpoints to /kaggle/working/main):
```python
from kaggle_secrets import UserSecretsClient
import os
os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")   # skip without W&B

!python -u run/train_loop.py --kaggle --wandb --resume auto --out /kaggle/working/main
```
Every 5 iterations it plays 400 hands against the greedy bot on duplicate deals and
prints `vs greedy: +X.X +/- Y.Y points/hand`. The AI is better than greedy once that
range sits above zero.

**Continuing in a new session:** Kaggle sessions stop after a few hours. Save the
notebook's output (`/kaggle/working/main`), attach it to the next session as a
dataset, copy it back to `/kaggle/working/main`, and run the same cell 4 again:
`--resume auto` continues from `latest.pt`.

**Comparison run** (only if you want the "hand guessing helps" result): run this in
a second notebook at the same time.
```python
!python -u run/train_loop.py --kaggle --no-hand-guess --resume auto --out /kaggle/working/nohand
```

## Bringing a model home

Download `latest.pt` and `metrics.jsonl` from the notebook's output and put them in
`rummy_rl/data/main/` on your machine (`data/` is not committed to git).
