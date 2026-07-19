# Othello Self-Play RL — Implementation Plan

**Project:** A scaled-down AlphaZero that learns Othello (Reversi) purely from self-play, with (A) a live spectator, (B) a PyCUDA-accelerated game engine, (C) a metrics dashboard, and (D) a web app to play the trained bot.

**Audience:** A coding agent executing the build, plus a human (RL beginner) supervising it.

**Guiding principle:** Get a *correct* end-to-end pipeline running at tiny scale first, then scale up and optimize. Never block the whole project on the hardest component (the CUDA kernel or Edax). Every phase must produce something runnable and testable.

---

## 1. Goals and non-goals

### Goals
1. Train an Othello agent by AlphaZero-style self-play (neural network + MCTS).
2. Beat a tunable minimax opponent up to a strong depth (primary target: **clearly beat minimax depth-6, ideally depth-8**, using the strong 4-component heuristic in §6).
3. As a stretch external benchmark, beat **Edax** (the strongest open-source Othello engine) at a low, empirically-calibrated level.
4. Spectate self-play games (via near-live replay of recorded games).
5. Play the trained bot yourself in a browser.
6. Log and graph training metrics with clear definitions.
7. Use **PyCUDA** for a batched, bitboard-based Othello engine (the resume-worthy custom-kernel component). PyTorch handles the neural network.

### Non-goals
- State-of-the-art strength. We are not trying to beat Edax at its top (superhuman) levels.
- Distributed / multi-GPU training. Single free GPU (Kaggle) is the target.
- A production web service. The web app is a local, single-user tool.

---

## 2. Success criteria (the benchmark ladder)

We do **not** use a single fixed opponent. Strength is measured on a **ladder**, and the headline metric is *the deepest minimax the agent can beat*.

| Rung | Opponent | Role |
|------|----------|------|
| 0 | Random / greedy (max-disc) | Throwaway sanity check, very early training only |
| 1 | Minimax depth 2 (strong heuristic) | Early progress signal |
| 2 | Minimax depth 4 (strong heuristic) | "Not embarrassing ourselves" checkpoint |
| 3 | **Minimax depth 6** (strong heuristic) | **Primary success target** |
| 4 | **Minimax depth 8** (strong heuristic) | Stretch success target |
| 5 | Edax at level N (calibrated) | External stretch ceiling |

- **Minimax depth is the difficulty knob.** It is the *same* alpha-beta code with one parameter changed, so the whole ladder costs almost nothing to implement.
- **Headline metric:** `max_depth_beaten` = the largest depth `d` at which the agent wins ≥ 55% of a fixed eval match (colors alternated). Plot this over training iterations — a rising staircase.
- **Definition of "beat":** win rate ≥ 55% over an eval match of `EVAL_GAMES` games (default 100), with colors alternated evenly, and ideally averaged over a few random-seeded openings to reduce variance.
- **Edax:** treat as an optional top rung. Even moderate Edax levels are very strong; realistic target is a *low* level. Calibrate empirically (see §6.3).

**Why a ladder, not one benchmark:** any single opponent only discriminates for one phase of training (a weak one saturates at ~100% win, a strong one at ~0%). The ladder always has a rung near 50%, so the signal never goes flat.

---

## 3. System architecture

Four components. Note the two UI requirements (watch + play) share one frontend.

```
                    ┌───────────────────────────────────────────┐
                    │            KAGGLE (free GPU: T4/P100)       │
                    │                                             │
  ┌──────────────┐  │   ┌────────────────┐    ┌───────────────┐  │
  │ PyCUDA engine│◄─┼──►│  Self-play loop │───►│ Replay buffer │  │
  │ (batched     │  │   │  (MCTS + net)   │    └──────┬────────┘  │
  │  bitboards)  │  │   └───────┬────────┘           │           │
  └──────────────┘  │           │            ┌───────▼────────┐  │
                    │           │            │ Training step  │  │
  ┌──────────────┐  │   ┌───────▼────────┐   │ (PyTorch/CUDA) │  │
  │ Minimax /    │◄─┼──►│  Evaluation    │   └───────┬────────┘  │
  │ Edax         │  │   │  harness       │           │           │
  └──────────────┘  │   └───────┬────────┘   ┌───────▼────────┐  │
                    │           │            │ Checkpoints +  │  │
                    │           └───────────►│ game records + │  │
                    │                        │ metric logs    │  │
                    └────────────────────────┴───────┬────────┘  │
                                                     │ download   │
                    ┌────────────────────────────────▼──────────┐
                    │                 LOCAL MACHINE               │
                    │  ┌───────────────┐    ┌──────────────────┐ │
                    │  │ FastAPI backend│──►│ Web app (browser)│ │
                    │  │  + WebSockets  │    │  • Watch mode    │ │
                    │  └───────────────┘    │  • Play mode     │ │
                    │  ┌───────────────┐    └──────────────────┘ │
                    │  │ TensorBoard    │                         │
                    │  └───────────────┘                         │
                    └─────────────────────────────────────────────┘
```

**Data flow:** Kaggle trains and emits three artifact types — model **checkpoints**, self-play **game records**, and **metric logs**. These are downloaded (manually or via the Kaggle API). The local machine runs the web app (play + watch-via-replay) and TensorBoard.

---

## 4. Environment, stack, and constraints

### Compute
- **Training:** Kaggle Notebooks — ~30 GPU-hrs/week, T4 (16GB) or P100 (16GB), sessions up to ~9h, background execution supported. No local NVIDIA GPU available.
- **Local (play + watch + dashboards):** CPU is fine. The web app runs inference on CPU for human play (few MCTS sims needed) and replays recorded games.

### Key libraries
- `torch` — neural network, training, CUDA-backed NN math.
- `pycuda` — custom batched bitboard engine kernels. **Fallback:** a `numba`-JIT or NumPy engine with identical semantics, so the pipeline runs even where PyCUDA is unavailable/awkward (e.g., local CPU).
- `numpy` — reference engine, data handling.
- `fastapi` + `uvicorn` + `websockets` — local backend and live streaming.
- `tensorboard` (primary) and optionally `wandb` — metrics.
- `edax` — external binary (compiled from source), driven as a subprocess (stretch).

### Hard constraint to respect
The trainer lives in a sandboxed Kaggle session that does **not** natively expose a public web server. Therefore requirement A ("spectate self-play") is satisfied by **recording games during training and replaying them locally as near-live**, not by streaming the exact in-progress game. (Optional upgrade paths: a `cloudflared`/`ngrok` tunnel from Kaggle, or generating fresh self-play locally on CPU for genuine liveness — see §9.3.)

---

## 5. Repository layout

```
othello-az/
├── engine/
│   ├── board_numpy.py        # Reference engine (correctness + local play)
│   ├── board_cuda.py         # PyCUDA batched bitboard engine
│   ├── bitboard_kernels.cu   # CUDA C: legal moves, apply move, terminal
│   ├── symmetry.py           # 8-fold dihedral transforms (board + policy)
│   └── encode.py             # board <-> NN input planes; move indexing
├── opponents/
│   ├── minimax.py            # tunable-depth alpha-beta + heuristic
│   ├── heuristic.py          # 4-component evaluation (§6)
│   └── edax.py               # subprocess wrapper (stretch)
├── az/
│   ├── network.py            # policy+value ResNet
│   ├── mcts.py               # PUCT search
│   ├── selfplay.py           # game generation -> training examples + records
│   ├── replay_buffer.py      # rolling example store
│   ├── train.py              # training step + loss
│   └── evaluate.py           # ladder eval, Elo, win rates
├── serve/
│   ├── backend.py            # FastAPI: play endpoints + spectate WebSocket
│   └── frontend/             # board UI (HTML+Canvas or React), 2 modes
├── run/
│   ├── train_loop.py         # top-level orchestration (Kaggle entrypoint)
│   └── config.py             # all hyperparameters (§11)
├── data/                     # checkpoints, game_records, tb_logs (gitignored)
└── tests/                    # parity, perft, mcts sanity, overfit-tiny
```

---

## 6. Component spec — opponents (build these EARLY; they're the yardstick)

### 6.1 Tunable minimax (`opponents/minimax.py`)
- **Algorithm:** minimax with **alpha-beta pruning** and **move ordering** (order candidate moves by heuristic value, corners first — pruning efficiency depends heavily on this).
- **Signature:** `minimax_move(board, player, depth, heuristic) -> move`. `depth` is the difficulty knob (used for the entire ladder in §2).
- **Leaf evaluation:** at `depth == 0` or terminal, return the heuristic score (§6.2). At true terminal, return a large ± value proportional to final disc margin so wins/losses dominate heuristic noise.
- **Performance note:** deep minimax in pure Python is slow, and eval needs many games. Two mitigations: (1) alpha-beta + ordering, (2) evaluate opponents using the fast batched engine (§7) so depth-6/8 matches are affordable. Depth is the clean, reproducible dial; keep the heuristic fixed.

### 6.2 Strong heuristic (`opponents/heuristic.py`)
Use the well-documented **4-component linear heuristic** (Sannidhanam & Annamalai analysis; concrete formulas per Kukreja's widely-reproduced version). Each component is normalized to [-100, 100] as `100 * (max - min) / (max + min)` (return 0 when denominator is 0), then combined with weights.

Components:
1. **Coin parity** — disc-count difference. `100 * (my_discs - opp_discs) / (my_discs + opp_discs)`.
2. **Mobility** — legal-move-count difference (same normalized form). More options = better.
3. **Corner capture** — the 4 corners; same normalized form. Corners can never be flipped.
4. **Stability** — discs that can never be flipped again (corners, then filled edges, then squares enclosed by stable discs). Their analysis found this the single most valuable component.

**Default weights** (tune later; these are a sane start): `stability: 25, corners: 30, mobility: 5, parity: 25`. Optionally schedule weights by game phase (parity matters more near the end), but a fixed set is fine for a benchmark opponent. Keep weights **fixed** once chosen so the benchmark is reproducible.

Also expose a **static weight-matrix** opponent (weighted piece counter: fixed 8×8 weight table, corners strongly positive, corner-adjacent squares strongly negative) as an optional simpler/faster opponent for very-early sanity checks.

### 6.3 Edax integration (`opponents/edax.py`) — STRETCH
- Compile Edax from source; invoke as a subprocess.
- Drive it by feeding a board position + side-to-move and reading its chosen move, with its **level** set to a fixed value (level is the tunable strength knob; higher = deeper/stronger).
- **Calibration:** run the trained agent vs Edax across levels to find the highest level it can beat. Report that level. Realistic target is a **low** level; top levels are essentially superhuman and out of scope.
- Keep Edax fully optional and behind a flag — it must never block the core loop.

---

## 7. Component spec — Othello engine

Two implementations with **identical semantics**, verified against each other (§13).

### 7.1 Reference engine (`engine/board_numpy.py`)
- Clear, obviously-correct NumPy/Python implementation. Used for: local human play, the web app, generating ground truth for tests, and as the PyCUDA fallback.
- API: `initial_board()`, `legal_moves(board, player)`, `apply_move(board, player, move)`, `is_terminal(board)`, `winner(board)`, `must_pass(board, player)`.
- **Passing:** if a player has no legal move, they pass; if both pass, game ends. Represent pass explicitly (move index 64).

### 7.2 PyCUDA batched engine (`engine/board_cuda.py`, `bitboard_kernels.cu`) — the CUDA showcase
- **Representation:** each board = two 64-bit integers (`own`, `opp` bitboards). One GPU thread simulates one game/board.
- **Kernels:**
  - `legal_moves_kernel` — bitwise shift/mask in 8 directions to compute the legal-move bitmask.
  - `apply_move_kernel` — compute flipped discs via directional scans, update both bitboards.
  - `terminal_kernel` — detect no-moves-for-both and count discs.
- **Purpose:** run **thousands of self-play games / MCTS leaf evaluations in parallel**, which is the main throughput lever in AlphaZero. Benchmark games/sec before vs after — that number is a metric (§8) and the resume story.
- **Fallback flag:** `ENGINE=cuda|numba|numpy`. `numba` gives a fast CPU path for local use.

### 7.3 Encoding (`engine/encode.py`)
- **NN input planes** (all from the perspective of the player to move): plane 0 = current player's discs, plane 1 = opponent's discs, plane 2 = legal-move mask (optional but helpful), plane 3 = all-ones side-to-move indicator (optional). Shape `[C, 8, 8]`.
- **Move indexing:** 0–63 = squares (row-major), 64 = pass. Policy vector length = **65**.

### 7.4 Symmetry (`engine/symmetry.py`)
- The board has 8-fold dihedral symmetry (4 rotations × 2 reflections). Provide functions to transform a board and its 65-length policy consistently. Used for **8× data augmentation** (§8.2) and optionally for averaging network evaluations.

---

## 8. Component spec — the learning system (AlphaZero core)

### 8.1 Network (`az/network.py`)
- **Architecture:** small ResNet. Input planes → conv stem → `NUM_BLOCKS` residual blocks (`CHANNELS` filters) → two heads.
  - **Policy head:** conv → flatten → linear → 65 logits (softmax over legal moves; mask illegal moves to -inf before softmax).
  - **Value head:** conv → flatten → linear → 1 → `tanh` (range [-1, 1], from the to-move player's perspective).
- **Start small:** `NUM_BLOCKS=5`, `CHANNELS=64`. Othello is tiny; scale up only if needed.

### 8.2 Self-play (`az/selfplay.py`)
For each move in each self-play game:
1. Run MCTS (§8.3) from the current position with `SIMS` simulations.
2. Form the **improved policy** `π(a) ∝ N(a)^(1/τ)` from root visit counts (`τ` = temperature).
3. Sample a move from `π` (exploration) and play it.
4. Record a training example `(state s, policy π, to_move)`; fill in `z` (final result from that player's perspective) once the game ends.

**Temperature schedule:** `τ = 1.0` for the first `TEMP_MOVES` (≈ 15–20) moves, then `τ → 0` (greedy on visits) for the rest. This explores early, sharpens late.

**Root exploration noise:** at the root, mix Dirichlet noise into priors: `P = (1-ε)·P + ε·Dir(α)`, with `ε = 0.25`, `α ≈ 0.3`. Ensures self-play doesn't collapse onto one line.

**Augmentation:** expand each example into its 8 dihedral symmetries before storing.

**Records for spectating:** in parallel, write a human-readable **game record** (§10) with per-move board state, chosen move, MCTS root value, and top policy moves.

### 8.3 MCTS (`az/mcts.py`)
This is the policy-improvement engine. Per edge store `{P, N, W, Q=W/N}`. Each of `SIMS` simulations:
1. **Select** from root, descending by **PUCT**:
   `PUCT(s,a) = Q(s,a) + c_puct · P(s,a) · sqrt(Σ_b N(s,b)) / (1 + N(s,a))`
   (prior guides where to look; `Q` from look-ahead takes over as visits grow; the `N` denominator discourages over-visiting).
2. **Expand + evaluate** at a leaf: call the network **once** to get value `v` and child priors `P`. **No random rollouts** — the value head is the score.
3. **Back up** `v` along the path (negate per ply for alternating players), updating `N`, `W`, `Q`.

Output = root **visit distribution**, which is the improved policy. It beats the raw network policy because it integrates value estimates of many deeper positions.

Params: `SIMS` (100–200 self-play, more for eval), `c_puct ≈ 1.5`.

### 8.4 Replay buffer (`az/replay_buffer.py`)
- Rolling FIFO of the most recent `BUFFER_SIZE` examples (≈ last 20–50 iterations' worth).
- Each entry: `(planes, π (65,), z scalar)`.

### 8.5 Training (`az/train.py`)
- Sample random minibatches from the buffer. Loss per example:
  `L = (z - v)²  −  πᵀ log(p)  +  c·‖θ‖²`
  (value MSE + policy cross-entropy + L2). Illegal moves masked out of `p`.
- Optimizer: Adam (`lr=1e-3`, cosine or step decay) or SGD+momentum; `weight_decay=1e-4`. `BATCH=512`, `STEPS_PER_ITER` ≈ a few hundred.

### 8.6 Evaluation (`az/evaluate.py`)
- **Ladder eval:** candidate vs minimax at each depth in `{2,4,6,8}`; `EVAL_GAMES` games, colors alternated. Compute win rate per depth and `max_depth_beaten`.
- **Relative eval:** candidate vs previous best; update an **Elo** rating from the outcome.
- **Gatekeeping:** promote candidate to "best" (the self-play generator) only if it beats current best by a margin (e.g., ≥ 55%).
- **Edax eval:** optional, behind a flag (§6.3).

---

## 9. Observability

### 9.1 Metrics and definitions (log all to TensorBoard)

| Metric | Definition | What it tells you |
|--------|-----------|-------------------|
| `loss/policy` | Cross-entropy between net policy `p` and MCTS visits `π` | How well the net predicts good moves |
| `loss/value` | MSE between predicted `v` and outcome `z` | How well it predicts the winner |
| `loss/total` | policy + value + L2 | Headline training curve |
| `policy_entropy` | Entropy of the net's move distribution | High→low = growing decisiveness (sanity check) |
| `winrate/minimax_d{2,4,6,8}` | Win rate vs each minimax depth | Absolute strength per rung |
| **`max_depth_beaten`** | Deepest minimax with ≥55% win | **Primary headline metric (rising staircase)** |
| `elo` | Elo vs previous checkpoints | Relative improvement each iteration |
| `game_len_avg` | Mean moves per self-play game | Strategy maturation signal |
| `selfplay_games_per_sec` | Self-play throughput | CUDA speedup story (before/after kernel) |
| `buffer_size` | Examples currently in buffer | Pipeline health |
| `mcts_root_value_avg` | Mean root value estimate in self-play | Confidence / balance check |

### 9.2 Live spectator (requirement A)
- Training writes **game records** (§10) continuously into `data/game_records/`.
- These are bundled into the downloadable Kaggle artifact.
- Locally, the backend streams a chosen game's moves over a **WebSocket** with a configurable per-move delay, so the browser renders it as a "live" game.

### 9.3 Optional genuine liveness
If truly-live spectating is wanted despite training on Kaggle:
- **Option 1:** run a handful of self-play games *locally on CPU* using the latest downloaded checkpoint (slow but genuinely live).
- **Option 2:** expose the Kaggle dashboard via a `cloudflared`/`ngrok` tunnel (works, but flaky and dies on session timeout).
Primary design = replay of downloaded records (§9.2).

---

## 10. Data formats

**Training example** (stored in buffer / npz shards):
```
planes: float32 [C,8,8]   policy: float32 [65]   value: float32 scalar   to_move: int
```

**Game record** (JSON, for spectating):
```json
{
  "iteration": 42, "kind": "selfplay", "result": "+6",
  "moves": [
    {"n": 1, "player": "B", "move": 19, "board": "....",
     "mcts_value": 0.12, "top_policy": [[19,0.44],[26,0.31]]}
  ]
}
```

**Checkpoint:** `torch` `state_dict` + metadata `{iteration, hyperparams, metrics_snapshot, elo}`.

---

## 11. Hyperparameters (starting config, `run/config.py`)

| Group | Param | Default |
|-------|-------|---------|
| Network | `NUM_BLOCKS` / `CHANNELS` | 5 / 64 |
| MCTS | `SIMS` (selfplay/eval) | 160 / 400 |
| MCTS | `c_puct` | 1.5 |
| MCTS | Dirichlet `α` / `ε` | 0.3 / 0.25 |
| Self-play | `GAMES_PER_ITER` | 200 |
| Self-play | `TEMP_MOVES` / `τ` | 16 / 1.0→0 |
| Buffer | `BUFFER_SIZE` | 200,000 |
| Train | `BATCH` / `STEPS_PER_ITER` | 512 / 400 |
| Train | `lr` / `weight_decay` | 1e-3 / 1e-4 |
| Eval | `EVAL_GAMES` / depths | 100 / {2,4,6,8} |
| Eval | promote threshold | 55% |
| Augment | dihedral symmetries | 8 |

All values are starting points; expect to tune `SIMS`, `GAMES_PER_ITER`, and network size against available GPU-hours.

---

## 12. Build phases (each ends with a runnable, tested artifact)

**Phase 0 — Engine + board UI.** NumPy engine + a browser board where you play a random bot. *Acceptance:* legal moves render, flips are correct, games end correctly, human can complete a game.

**Phase 1 — Tunable minimax + heuristic.** Alpha-beta with the 4-component heuristic, depth as a parameter. Play it in the UI at several depths. *Acceptance:* depth-4 plays visibly sensible Othello (takes corners, avoids X-squares); `max_depth_beaten` harness runs.

**Phase 2 — AlphaZero pipeline at tiny scale (CPU).** Network + MCTS + self-play + buffer + training + TensorBoard, tiny sizes. *Acceptance:* loss decreases; overfit-tiny test passes; the loop runs end to end without errors.

**Phase 3 — Scale on Kaggle GPU + ladder eval.** Move training to Kaggle, scale network/sims/games, wire in ladder + Elo, emit checkpoints/records/logs. *Acceptance:* `max_depth_beaten` climbs past depth-4 over iterations; artifacts download cleanly.

**Phase 4 — PyCUDA batched engine.** Bitboard kernels; verify parity with NumPy engine; route self-play/eval through it. *Acceptance:* engine-parity test passes on thousands of random games; `selfplay_games_per_sec` measurably higher; benchmark recorded.

**Phase 5 — Web app: watch + play, integrated.** FastAPI backend, WebSocket spectator replaying records, human-vs-checkpoint play mode, TensorBoard alongside. *Acceptance:* you can watch a recorded self-play game render move-by-move and play a full game against a chosen checkpoint.

**Phase 6 (stretch) — Edax.** Compile, wrap, calibrate the level the agent beats. *Acceptance:* reproducible win rate vs Edax at level N reported.

---

## 13. Testing strategy (`tests/`)

- **Engine parity:** run thousands of random games through both NumPy and PyCUDA engines; every legal-move set, flip, and terminal result must match exactly.
- **Perft / move-count:** count legal moves / reachable states to fixed shallow depths against known values.
- **MCTS sanity:** more simulations ⇒ stronger play (higher win rate vs a fixed shallow minimax).
- **Overfit-tiny:** the network can drive loss near zero on a handful of fixed positions (confirms the training path works).
- **Encoding round-trip:** `decode(encode(board)) == board`; symmetry transforms are self-consistent (apply then invert = identity) for board and policy jointly.
- **Eval determinism:** with fixed seeds and `τ=0`, eval results are reproducible.

---

## 14. Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| PyCUDA setup is fiddly / unavailable locally | `numba`/`numpy` fallback with identical semantics; CUDA is an optimization, not a dependency of correctness |
| Kaggle session timeouts / 30h/week cap | Checkpoint every iteration; resume from latest; use background execution; keep iterations short |
| No live web server on Kaggle | Record-and-replay design (§9.2); tunnel or local self-play as optional upgrades |
| Deep minimax eval too slow | Alpha-beta + move ordering; run eval through the batched engine; cap eval game counts |
| Self-play collapses / no exploration | Dirichlet root noise, temperature schedule, promotion gating |
| Training unstable | Small net first, L2, lr decay, watch value/policy loss separately |

---

## 15. Appendix — CUDA / PyCUDA primer (for the learning/resume angle)

**CUDA** is NVIDIA's platform for running general-purpose code on GPUs. A GPU has thousands of small cores and excels at doing the *same operation across huge amounts of data at once*. You write a small function (a **kernel**) that the GPU runs in parallel across thousands of data elements.

**PyTorch already uses CUDA** under the hood for all neural-network math — you do **not** write kernels for the network. **PyCUDA** is a separate, lower-level tool: it lets you write your own CUDA C kernels, compile them at runtime, move data to/from the GPU, and launch them from Python.

**Where PyCUDA earns its place here:** the batched **bitboard Othello engine** (§7.2). Othello move-generation and disc-flipping are pure bit operations, so one GPU thread can simulate one game and thousands run at once. Since generating self-play data is AlphaZero's main bottleneck, this is a legitimate speedup *and* a genuine custom-kernel artifact — not a contrived use.

**Free GPU access:** Kaggle Notebooks (~30 GPU-hrs/week, T4/P100, up to ~9h sessions, background execution) is the primary target; Google Colab's free tier (T4, ~12h sessions but disconnects after inactivity, throttled by compute-unit demand) is a secondary option. No credit card required for either; both support installing PyCUDA and PyTorch.
