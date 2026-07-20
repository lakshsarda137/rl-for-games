# Othello Self-Play RL

A scaled-down **AlphaZero** that learns Othello (Reversi) purely from self-play:
a policy+value neural network guided by Monte-Carlo Tree Search, trained against
its own games, and benchmarked on a ladder of minimax opponents (plus the strong
external engine **Edax**). Comes with a terminal viewer and a local web app to
watch bots play or play them yourself.

Full design rationale lives in [`othello_alphazero_implementation_plan.md`](othello_alphazero_implementation_plan.md).
Agent-facing status and non-obvious decisions live in [`CLAUDE.md`](CLAUDE.md).

## Quick start

```bash
# Run the tests (fast tier, ~10s wall — suites run in parallel)
python run_tests.py                 # add --full for the heavy tier (~35s wall); --serial to serialize

# Play in the terminal (you are Black, vs depth-3 minimax)
python run/play_cli.py --black human --white minimax:3
python run/play_cli.py --black minimax:4 --white edax:4 --delay 0.4   # watch two bots

# Web app (board in the browser) -> http://127.0.0.1:8000
python serve/backend.py            # play the trained "AZ net" vs any bot / yourself;
                                   # also serves the live dashboard at /dashboard

# Train the AlphaZero agent (tiny CPU smoke run)
python run/train_loop.py --tiny
python run/train_loop.py --kaggle --resume auto   # continue the newest checkpoint

# See how training is going (annotated charts from data/metrics.jsonl)
python run/dashboard.py            # -> data/dashboard.html, opens in your browser
```

Training on a free Kaggle GPU: see [`run/KAGGLE.md`](run/KAGGLE.md).

## Playing / measuring the trained net

Start the web app (`python serve/backend.py`) and pick **"AZ net"** for either
side to play the latest trained checkpoint against yourself or any bot. The
**Arena** panel there runs an N-game, colour-alternated match of the net vs a bot
of your choice and reports aggregate win/draw/loss + win rate — a far more reliable
strength read than a single game. (In-training minimax eval is inspection-only and
**off by default** in the Kaggle config; the Arena is the on-demand replacement.)

To play a model trained on Kaggle locally, pull it down first:
- **Mid-training (any time, no waiting)** — `python run/pull_wandb.py --run run1`
  grabs the current checkpoint that training uploads to W&B every few iterations.
- **After a committed run** — `python run/pull_kaggle.py --kernel <user>/<slug>`.

See `run/KAGGLE.md`.

## Watching training progress

Every training iteration appends one JSON line to `data/metrics.jsonl` (loss,
self-play speed, replay-buffer size, win rates, `max_depth_beaten`). Two ways to
read it, both drawing the same annotated charts — each one explains in plain
English what it measures and what "improving" looks like:

- **Static** — `python run/dashboard.py` builds a self-contained `data/dashboard.html`
  (data embedded, no server). Ideal for a downloaded Kaggle run: grab `metrics.jsonl`
  and build the page locally.
- **Live** — start the web app (`python serve/backend.py`) and open
  <http://127.0.0.1:8000/dashboard>; it reads the jsonl through `/api/metrics` and
  auto-refreshes every 15s while training runs.
- **Live from a remote GPU (Kaggle)** — add `--wandb` to stream each iteration to
  Weights & Biases and watch a real-time dashboard at wandb.ai from your laptop.
  This is the only way to see *in-progress* remote training (the local dashboard
  needs the jsonl pulled down first). See `run/KAGGLE.md`.

## Resuming training across sessions

Checkpoints (`data/checkpoints/iterNNNN.pt`, plus a rolling `latest.pt`) store the
weights, optimizer state, and RNG state. Continue an interrupted run with:

```bash
python run/train_loop.py --kaggle --resume auto             # newest checkpoint
python run/train_loop.py --kaggle --resume data/checkpoints/iter0030.pt
# on Kaggle, add the live dashboard and keep one continuous curve across sessions:
python run/train_loop.py --kaggle --resume auto --wandb --wandb-run run1
```

Iteration numbering and `metrics.jsonl` continue on one timeline, so `--iterations N`
means "run N *more* iterations". (The replay buffer isn't checkpointed — it refills
over the first 1-2 iterations.) This is what makes multi-session Kaggle runs work:
resume the latest checkpoint each session and, with the same `--wandb-run`, W&B keeps
one live curve. Full Kaggle runbook: [`run/KAGGLE.md`](run/KAGGLE.md).

## Repository layout

```
othello/
├── engine/                 # the game — the correctness oracle for everything else
│   ├── board_numpy.py      #   NumPy Othello rules (moves, flips, passing, terminal, scoring) — the oracle
│   ├── board_batched.py    #   same rules vectorised over B boards [B,8,8] (for fast self-play)
│   ├── encode.py           #   board <-> NN input planes; the 65-action space; PERSPECTIVE convention
│   └── symmetry.py          #   8-fold dihedral transforms (board + 65-policy, jointly)
├── opponents/              # the yardsticks the agent is measured against
│   ├── heuristic.py        #   4-component eval (parity, mobility, corners, stability) + weight table
│   ├── minimax.py          #   alpha-beta minimax; depth = the difficulty dial
│   ├── simple.py           #   random / greedy players + the match runner (play_match)
│   ├── edax.py             #   wrapper driving the external Edax engine as a subprocess
│   └── EDAX_SETUP.md       #   how to build/install Edax (it lives in gitignored third_party/)
├── az/                     # the AlphaZero learner
│   ├── network.py          #   policy+value ResNet + Evaluator (net -> priors, value; batched)
│   ├── mcts.py             #   PUCT search (a coroutine), Dirichlet root noise, visit_policy
│   ├── mcts_batched.py     #   B MCTS trees searched in lockstep as flat arrays (fast self-play)
│   ├── selfplay.py         #   game generation -> training examples + records (array-ops / pool / multi-core)
│   ├── replay_buffer.py    #   rolling FIFO of (planes, pi, mask, z)
│   ├── train.py            #   loss = value MSE + masked policy cross-entropy + L2
│   └── evaluate.py         #   ladder eval vs minimax; max_depth_beaten
├── run/                    # orchestration + entrypoints
│   ├── config.py           #   all hyperparameters: Config (full) / Config.tiny() / Config.kaggle()
│   ├── play_cli.py         #   terminal viewer: watch bots or play one yourself
│   ├── train_loop.py       #   top-level loop; device auto-detect; --tiny/--kaggle; --resume
│   ├── dashboard.py        #   build a standalone HTML dashboard from data/metrics.jsonl
│   ├── pull_kaggle.py      #   pull a committed Kaggle run's checkpoint + metrics into data/
│   ├── pull_wandb.py       #   pull the CURRENT checkpoint from W&B (play the bot mid-training)
│   └── KAGGLE.md           #   how to train on a free Kaggle GPU
├── serve/                  # the web app (local, single-user)
│   ├── backend.py          #   FastAPI: /api/new, /api/move, /api/bot_move; /dashboard, /api/metrics
│   └── frontend/
│       ├── index.html      #   self-contained board UI (play any bot / watch bots)
│       └── dashboard.html  #   annotated training-metrics charts (live or static)
├── tests/                  # tiered test suites (fast default, --full heavy)
│   ├── harness.py          #   check() + the fast/full runner
│   └── test_*.py           #   engine parity/perft, encode, symmetry, minimax, edax, az pipeline
├── run_tests.py            # runs all suites in parallel; warns if the fast tier exceeds 20s wall
├── data/                   # (gitignored) checkpoints, game_records, metrics.jsonl
└── third_party/edax/       # (gitignored) the built Edax binary + eval weights
```

## How the pieces fit

1. **Engine** defines the rules and is the single source of truth. Everything
   else calls it; `board_numpy.py` is the reference oracle, `board_batched.py` is a
   vectorised copy for fast self-play (a Torch/CUDA version is a future optimization).
2. **Encoding** turns a board into what the network sees — always from the
   *side-to-move's* perspective (see `encode.py`, this is the key convention).
3. **Network** maps a position to (move priors, value). **MCTS** uses it to look
   ahead and produce a stronger move distribution than the raw network.
4. **Self-play** plays games with MCTS, recording each position, the search's
   move preferences, and (at game end) who won. **Training** nudges the network
   toward those targets. Repeat → the agent improves.
5. **Evaluation** measures strength on the minimax ladder; the headline metric is
   `max_depth_beaten` — the deepest minimax the agent beats ≥55% of the time.

## Difficulty dials

- **Minimax:** `minimax:D` where `D` is search depth (1–8). Deeper = stronger
  (and slower in pure Python: d5 ≈ 4s/move, d6 ≈ 9s/move).
- **Edax:** `edax:L` where `L` is the level (0–30). Even low levels are very
  strong. Requires a local Edax build (see `opponents/EDAX_SETUP.md`); optional.

## Testing

Two tiers (see `tests/harness.py`), and suites run in parallel (`run_tests.py`):
- **FAST** (default, ~10s wall) — runs after every change: correctness, encoding,
  symmetry, batched-engine/MCTS parity, overfit-tiny, a light strength match.
- **FULL** (`--full`, ~35s wall) — adds deep perft, strength matches, multi-process
  self-play, the end-to-end training loop. Run when a change warrants the heavy checks.
