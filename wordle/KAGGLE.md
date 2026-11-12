# Training the Wordle duel AI on Kaggle (free GPU)

The notebook is already set up: <https://www.kaggle.com/code/lakshsarda/wordle-duel-training>
(GPU T4 x2, internet on). Its cells are the same as `kaggle_notebook.ipynb` in this folder.
To make a new one: **Create**, then **New Notebook**, then **File**, then **Import Notebook**, and
pick `kaggle_notebook.ipynb`. Then set **Settings**, **Accelerator** to **GPU T4 x2**.

Kaggle gives 30 GPU hours per week. A session stops after at most 12 hours.

## The cells

1. **Get the code.** Downloads the `wordle` branch from GitHub (or updates it) and shows the GPU.
2. **Tests** (about 1 minute on Kaggle).
3. **Quick check** (about 2 minutes): 6 short training rounds and 2 strength checks.
   The next cell shows its charts.
4. **Training** (3 hours per run). Continues from an earlier run's output if it is attached as input.
5. **Charts for the real run.** Run it after cell 4 stops.

## Watching it while it runs

- Every training round (about 10 seconds for 1,024 games) prints one line: percent done,
  games played, prediction error, and seconds spent.
- Every 5 rounds it plays 800 test games against the simple bot (the same 400 secret words
  each time, each word from both seats). It prints points per game for both players, win
  percentage, guesses per win, and one full test game with the AI's top choices.
- The files in `/kaggle/working/main` (right sidebar, **Output**) update during the run:
  `progress.html` (charts), `samples.txt` (every test game), `progress.txt` (newest line),
  `metrics.jsonl` (all numbers), and checkpoints `latest.pt`, `best.pt`, `ckpt_XXXXX.pt`.
- To play against a checkpoint, download it and run on your laptop, inside `wordle/`:
  `python play.py duel --ai latest.pt`

## Continuing in a new session

Save the notebook's output folder, attach it to the next session as a dataset, copy it back to
`/kaggle/working/main`, and run cells 1 and 4 again. `--resume auto` continues from `latest.pt`.
