# Training the Wordle duel AI on Kaggle (free GPU)

## One-time setup

1. <https://www.kaggle.com>, then **Create**, then **New Notebook**.
2. In the right sidebar, under **Session options**: set **Accelerator** to **GPU T4 x2** and turn **Internet** on.
   (Kaggle asks for a phone number once before it allows GPU and internet.)
3. Optional, to watch charts from your phone or laptop while training runs: add your
   Weights & Biases key under **Add-ons**, then **Secrets**, named exactly `WANDB_API_KEY`.

## Notebook cells

**1. Get the code** (every session; Kaggle clears the disk between sessions):
```python
!git clone --depth 1 -b wordle https://github.com/lakshsarda137/rl-for-games.git
%cd rl-for-games/wordle
```

**2. Run the tests** (about 30 seconds; builds the color table first):
```python
!python run_tests.py
```

**3. Quick check** (about 1 minute). Shows the speed per iteration and one strength check:
```python
!python -u train/train_loop.py --kaggle --iterations 6 --eval-every 3 --eval-words 100 --no-wandb --out /kaggle/working/smoke
```

**4. The real run** (stops after 10 hours; saves everything to /kaggle/working/main):
```python
import os
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
except Exception:
    print("No W&B key; progress is still printed below and saved in /kaggle/working/main")
!python -u train/train_loop.py --kaggle --hours 10 --resume auto --out /kaggle/working/main
```

## Watching it while it runs

- Every iteration prints one line: percent done, games played, prediction error, and seconds spent.
- Every 5 iterations (about 1 to 2 minutes) it plays 800 test games against the simple bot and
  prints points per game for both players, win percentage, guesses per win, and one full test
  game with the AI's top choices.
- In the notebook's file panel (`/kaggle/working/main`), `progress.html` has charts of every
  strength check. Download it and open it in a browser. `samples.txt` has every test game.
- To play against the newest checkpoint, download `latest.pt` and run on your laptop:
  `python play.py duel --ai latest.pt`

## Continuing in a new session

Kaggle sessions stop after at most 12 hours. Save the notebook's output folder, attach it to the
next session as a dataset, copy it back to `/kaggle/working/main`, and run cell 4 again.
`--resume auto` continues from `latest.pt`.
