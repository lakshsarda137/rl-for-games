# CLAUDE.md — agent orientation for the Othello project

Read this first. It captures where things stand, what's next, and the decisions
and preferences that are **not** obvious from reading the code.

## What this is
A from-scratch AlphaZero for Othello, built phase by phase against
`othello_alphazero_implementation_plan.md`. Each phase ends in a tested,
runnable artifact. See `README.md` for structure.

## Current status (as of 2026-07-19)
- **Phase 0 (engine + encoding)** ✅ — `engine/`. Perft 1–8 match canonical
  Othello values; encode round-trips; symmetry invertible.
- **Phase 1 (minimax + heuristic)** ✅ — `opponents/`. Depth is the difficulty
  dial; d2 beats random 20/20, d2 beats d1 ~75%.
- **Edax integration** ✅ (pulled forward from stretch Phase 6) — `opponents/edax.py`.
  Built for arm64 in gitignored `third_party/edax/` (recipe: `opponents/EDAX_SETUP.md`).
  Edax L3 beats our minimax d4 ~83%.
- **Phase 2 (AlphaZero pipeline)** ✅ — `az/`. Overfit-tiny loss → 0.0002;
  end-to-end loop runs; an 8-iter tiny CPU run reached `max_depth_beaten=1`
  (beat minimax d1 88%). The pipeline *learns*, not just runs.
- **Web app (Phase 5, play mode)** ✅ — `serve/`. Play any bot / watch bots.
  Self-play *replay* in the UI is not built (records exist in `data/game_records/`).
- **Kaggle setup** ✅ — code pushed to GitHub (github.com/lakshsarda137/rl-for-games);
  `Config.kaggle()` + `run/KAGGLE.md`.
- **First GPU run VERIFIED on Kaggle (2026-07-19)** — on a T4: `CUDA: True`, and both
  `--tiny --device cuda` and `--kaggle` run end to end on GPU (loss prints, eval works,
  checkpoints written).
- **Batched self-play inference ✅ (2026-07-19)** — plays `selfplay_concurrency` games
  concurrently and evaluates ALL their pending MCTS leaves in ONE network call per step
  (~16× fewer forward calls). Byte-identical play vs serial per seed
  (`test_batched_selfplay_matches_serial`). **BUT it barely moved throughput**: a live
  Kaggle T4 run still sat at **~0.4 games/sec with the GPU near-idle and CPU pinned**.
  Diagnosis (correcting the plan's premise): for the small 5×64 net the network is only
  ~13% of self-play time — the real wall is single-threaded pure-Python MCTS + NumPy
  engine. Batching was necessary but not sufficient.
- **Multiprocess self-play ✅ (2026-07-19)** — a fallback throughput lever, now superseded
  by array-ops for the real config. `cfg.selfplay_workers` spawn processes each run the
  batched pool on a slice of the per-game seeds (`az/selfplay.py::_play_parallel`). Measured
  **3.5× at 4 workers on a 10-core Mac**; byte-identical to the in-process pools
  (`test_parallel_selfplay_matches_inprocess`). Used only when `selfplay_arrayops=False`.
- **Array-ops self-play ✅ built + parity-verified (2026-07-19)** — the real rewrite (Phase 4
  in spirit). `engine/board_batched.py` (batched Othello engine over `[B,8,8]`) +
  `az/mcts_batched.py` (B trees in lockstep as flat arrays) + `az/selfplay.py::_play_batch`
  (whole game batch searched with array ops). Kills the per-game Python overhead that was
  ~87% of self-play. **Correctness is exhaustively verified vs the serial oracles** (engine
  exact on 3633 positions, MCTS visit counts exact on 434, self-play move-for-move greedy;
  `tests/test_batched.py`). `cfg.selfplay_arrayops=True` is the default; `kaggle` uses it.
  **SPEED IS NOT YET SETTLED**: CPU-to-CPU it's ~2.2× the pool, but that's a floor, not the
  verdict — the tree/engine math is NumPy on the CPU (only the net is on device), so the GPU
  number is genuinely unknown until smoked. If the GPU smoke is CPU-capped, the last lever is
  porting the tree ops to Torch-CUDA tensors (needs an NVIDIA GPU to build). **No long GPU
  run; resume/load-to-play still unwired.**

## Next steps (in likely order)
1. **Self-play throughput — built, GPU speed unsettled.** Three levers exist (see the
   `az/selfplay.py` module docstring): batched inference (coroutine `_play_pool`),
   multiprocess (`_play_parallel`, fallback), and **array-ops** (`_play_batch` + batched
   engine/MCTS, the default). All are correctness-verified vs serial. **Immediate open
   item: GPU-smoke the array-ops path** (`--kaggle`, `run/KAGGLE.md` cell 3a) to get the
   real g/s — it's the honest unknown. If it's CPU-capped (tree/engine are NumPy on CPU),
   port those to Torch-CUDA tensors (item 3; needs an NVIDIA GPU to build).
2. **Checkpoint resume + load-to-play** (user asked to do this next, alongside a metrics
   dashboard). `train_loop.train` starts fresh; add `--resume <ckpt>` (load `state_dict`)
   so training continues across Kaggle sessions / weekly quota — without it, each new
   session restarts from scratch, so a real multi-session run isn't safe yet. Also wire
   "load checkpoint → az_player" into `play_cli`/`backend`. **Also wanted: a metrics
   dashboard** that plots `data/metrics.jsonl` directly (the plan's pillar C; TensorBoard
   is the intended-but-broken viewer here, so read the jsonl instead). Both were paused in
   favour of the throughput fix.
3. **Phase 4 — PyCUDA batched bitboard engine** (`engine/board_cuda.py`). This is the
   BIG throughput lever the multiprocessing work only approximates: move the pure-Python
   MCTS/engine grind (the actual ~87% bottleneck) onto the GPU. NOTE: needs an NVIDIA GPU
   even to develop — **can't be built or verified on the user's Mac** (this is why
   multiprocessing was done first). Verify parity vs the NumPy engine.
4. **Wire self-play records into the web UI** (watch/replay mode; records already exist
   in `data/game_records/`).
5. **Elo + promotion gating** in evaluation (currently just win-rate ladder).
6. **Polish the web UI** — the user finds it visually rough; a design pass is wanted.

## Non-obvious architectural decisions
These will bite you if you change code without knowing them:

- **Perspective convention (most important).** The engine stores ABSOLUTE colours
  (`BLACK=+1`, `WHITE=-1`, `EMPTY=0`). `encode(board, player)` ALWAYS canonicalises
  to the side-to-move's POV (plane 0 = my discs, plane 1 = opponent's). The value
  head is "good for the side to move," so **MCTS negates value per ply** and
  self-play stamps `z` **per state's mover**. The authoritative statement is the
  `encode.py` module docstring. Don't introduce absolute-colour value anywhere.
- **`NUM_PLANES = 3`, not 4.** Because the board is canonicalised, the plan's
  all-ones "side-to-move" plane is redundant, so it's omitted. `encode.NUM_PLANES`
  is the single source of truth for the net's input channels.
- **MCTS backup is player-compared, not alternation-assumed.** In `mcts._simulate`,
  a leaf value `v` is added to edge `(node, a)` as `+v` if `node.player == leaf_player`
  else `-v`. This is what makes **passes** correct: a forced pass is a node whose
  only legal action is PASS (index 64), and playing it flips the mover like any
  move. Never assume strict color alternation.
- **Terminal = neither side can move**, NOT "board is full" (`is_terminal`). A
  side can be wiped out or both stuck early. Keep this when porting to CUDA.
- **PASS is symmetry-invariant.** In `symmetry.transform_policy`, indices 0–63
  permute but index 64 (pass) is a fixed point. Never reshape a 65-vector to a grid.
- **Edax quirks** (`opponents/edax.py`): must send `mode 3` before `go`; must NOT
  send `quit` (Edax block-buffers stdout on a pipe and a quit-exit drops the move —
  close stdin/EOF instead). Edax's square index == our move index. It's launched
  per-move (~70ms); fine for play, not for bulk.
- **Test tiering.** `tests/harness.py` splits each suite into FAST/SLOW. FAST must
  stay under 20s total (`run_tests.py` warns otherwise). Put anything with real
  search depth, many-game matches, or deep perft in SLOW.
- **Metrics: `data/metrics.jsonl` is the source of truth.** TensorBoard is broken
  in this environment (protobuf/tensorboard `GetPrototype` clash) so it's opt-in
  (`--tensorboard`) and fully guarded. Don't rely on tb.
- **Deep minimax is slow** (pure Python; d6 ≈ 9s/move). Interactive but not snappy;
  the batched engine is the eventual fix. Bot moves in the web backend run in a
  threadpool so they don't block the server.

## Key user preferences (picked up from conversation)
- **RL/DL background is beginner-level.** Explain concepts simply, with concrete
  analogies (they appreciated cooking/reading analogies for depth vs width). When
  introducing something new, say what it is in plain terms before diving in.
- **Fast dev loop is important to them.** They explicitly asked for a <20s default
  test tier with heavy tests gated behind a flag. Keep new tests tiered.
- **Wants to see and play the bots** — watching Edax vs minimax and playing as a
  beginner motivated pulling Edax forward. Keep things runnable/observable.
- **Web UI works but they find it visually rough** ("kind of disgusting"). A polish
  pass is wanted eventually; functionality-first was accepted for now.
- **Prefers decisive progress over excessive questions** — make sensible defaults
  and proceed, surfacing decisions rather than blocking. But they DO ask lots of
  good "why" questions; be ready to justify design choices.
- **Compute-budget conscious.** They asked about multi-accounting Kaggle to double
  the free tier — that violates Kaggle ToS (do NOT help with it). Legit paths:
  resume weekly on one account (quota resets), and Colab as a separate free provider.
- Works through the plan **phase by phase** and likes knowing what phase they're in
  and what's next.

## Gotchas for running things
- Modules use `sys.path` inserts to import siblings (engine/opponents/az), not a
  package. Run scripts from the `othello/` directory (or via the given entrypoints).
- Static linters flag the `sys.path` imports in tests as unresolved — false
  positives; the tests pass.
- `data/` and `third_party/` are gitignored (checkpoints, records, Edax binary +
  14MB eval weights). Rummy (a sibling repo dir) is intentionally NOT pushed.
