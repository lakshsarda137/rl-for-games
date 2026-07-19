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
# Run the tests (fast tier, ~13s)
python run_tests.py                 # add --full for the heavy tier (~55s)

# Play in the terminal (you are Black, vs depth-3 minimax)
python run/play_cli.py --black human --white minimax:3
python run/play_cli.py --black minimax:4 --white edax:4 --delay 0.4   # watch two bots

# Web app (board in the browser) -> http://127.0.0.1:8000
python serve/backend.py

# Train the AlphaZero agent (tiny CPU smoke run)
python run/train_loop.py --tiny
```

Training on a free Kaggle GPU: see [`run/KAGGLE.md`](run/KAGGLE.md).

## Repository layout

```
othello/
├── engine/                 # the game — the correctness oracle for everything else
│   ├── board_numpy.py      #   NumPy Othello rules (moves, flips, passing, terminal, scoring)
│   ├── encode.py           #   board <-> NN input planes; the 65-action space; PERSPECTIVE convention
│   └── symmetry.py          #   8-fold dihedral transforms (board + 65-policy, jointly)
├── opponents/              # the yardsticks the agent is measured against
│   ├── heuristic.py        #   4-component eval (parity, mobility, corners, stability) + weight table
│   ├── minimax.py          #   alpha-beta minimax; depth = the difficulty dial
│   ├── simple.py           #   random / greedy players + the match runner (play_match)
│   ├── edax.py             #   wrapper driving the external Edax engine as a subprocess
│   └── EDAX_SETUP.md       #   how to build/install Edax (it lives in gitignored third_party/)
├── az/                     # the AlphaZero learner
│   ├── network.py          #   policy+value ResNet + Evaluator (net -> priors, value)
│   ├── mcts.py             #   PUCT search, Dirichlet root noise, visit_policy
│   ├── selfplay.py         #   MCTS-driven game generation -> training examples + spectator records
│   ├── replay_buffer.py    #   rolling FIFO of (planes, pi, mask, z)
│   ├── train.py            #   loss = value MSE + masked policy cross-entropy + L2
│   └── evaluate.py         #   ladder eval vs minimax; max_depth_beaten
├── run/                    # orchestration + entrypoints
│   ├── config.py           #   all hyperparameters: Config (full) / Config.tiny() / Config.kaggle()
│   ├── play_cli.py         #   terminal viewer: watch bots or play one yourself
│   ├── train_loop.py       #   top-level loop; device auto-detect; --tiny/--kaggle
│   └── KAGGLE.md           #   how to train on a free Kaggle GPU
├── serve/                  # the web app (local, single-user)
│   ├── backend.py          #   FastAPI: /api/new, /api/move, /api/bot_move
│   └── frontend/index.html #   self-contained board UI (play any bot / watch bots)
├── tests/                  # tiered test suites (fast default, --full heavy)
│   ├── harness.py          #   check() + the fast/full runner
│   └── test_*.py           #   engine parity/perft, encode, symmetry, minimax, edax, az pipeline
├── run_tests.py            # runs all suites; warns if the fast tier exceeds 20s
├── data/                   # (gitignored) checkpoints, game_records, metrics.jsonl
└── third_party/edax/       # (gitignored) the built Edax binary + eval weights
```

## How the pieces fit

1. **Engine** defines the rules and is the single source of truth. Everything
   else calls it; the NumPy version is the reference (a batched CUDA version is a
   future optimization).
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

Two tiers (see `tests/harness.py`):
- **FAST** (default, ~13s) — runs after every change: correctness, encoding,
  symmetry, overfit-tiny, a light strength match.
- **FULL** (`--full`, ~55s) — adds deep perft, strength matches, the end-to-end
  training loop. Run when a change warrants the heavy checks.
