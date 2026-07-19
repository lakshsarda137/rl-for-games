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
- **Array-ops self-play ✅ — the real throughput rewrite (2026-07-19).**
  `engine/board_batched.py` (batched Othello engine over `[B,8,8]`, directional flood) +
  `az/mcts_batched.py` (B MCTS trees in lockstep as flat `[B,max_nodes,65]` arrays, one batched
  net call per sim step) + `az/selfplay.py::_play_batch` (whole game batch searched with array
  ops). Kills the per-game Python overhead that was ~87% of self-play. **Correctness exhaustively
  verified vs the serial oracles** (engine exact on 3633 positions, MCTS visit counts exact on 434,
  self-play move-for-move greedy; `tests/test_batched.py`). Default (`cfg.selfplay_arrayops=True`).
  **GPU smoke DONE: 2.1 games/sec on a Kaggle T4 (~5× the 0.4 baseline)** — good outcome, the GPU
  lifted the network floor. BUT the tree/engine math is still **NumPy on the CPU** (only the net is
  on-device), so the smoke showed one CPU core pegged and the GPU ~27% idle → now CPU-bound.
- **Multiprocess self-play ✅ (2026-07-19)** — `cfg.selfplay_workers` spawn PROCESSES each play a
  slice of the games on their own CPU core. Works over the coroutine pool (`_play_parallel`) AND,
  via **Option B**, over array-ops (`_play_batch_parallel`/`_worker_batch`). Measured 3.5× (pool)
  and **3.4× (array-ops) at 4 workers** on a 10-core Mac; byte-identical to the in-process split
  (`test_parallel_selfplay_matches_inprocess`, `test_arrayops_parallel_matches_inprocess`).
  **CAUTION — workers>1 helps a CPU-ONLY run, but is SLOWER on a GPU** (measured **2.1 g/s at
  workers=1 vs 1.3 at workers=4 on a T4**). Not a capacity problem — the GPU was ~27% util/near-empty
  memory. It's serialization+latency: without MPS a single GPU time-slices between processes'
  CUDA contexts (one at a time + switch cost), and self-play is latency-bound (each sim step the CPU
  waits on a small net eval), so 4 processes just queue at the one GPU and every round-trip slows. The
  local 3.4× was a `device=cpu` benchmark and did NOT transfer. **On GPU: one process, big batch** —
  let the GPU's own cores parallelise inside the batch. `kaggle` config = array-ops, `selfplay_workers=1`.
  **No long GPU run yet; resume/load-to-play still unwired.**

## Next steps (in likely order)

> **▶ CURRENT PRIORITY (decided by the user 2026-07-19): item 2 — resume + dashboard + a real
> training run.** Throughput is "good enough" (2.1 g/s); the goal now is an actual trained bot to
> watch/play. Do NOT start the GPU port (item 3) — the user explicitly deferred it.
>
> **UPDATE (2026-07-19): resume ✅, dashboard ✅, web load-to-play ✅, on-demand Arena eval ✅, and
> a Kaggle→local pull pipeline ✅ are BUILT** (see item 2). **In-training eval is now OFF by default**
> (kaggle `eval_every=0`) — it was inspection-only; strength is measured on demand via the web Arena.
> The only thing still open in item 2 is **the real multi-session Kaggle run** (the payoff).

1. **Self-play throughput — DONE.** Best GPU config is **single-process array-ops, ~2.1 g/s on a T4**
   (~5× the 0.4 baseline). `kaggle` config = array-ops, `selfplay_workers=1`. **On a GPU keep
   workers=1** — >1 is slower (shared-GPU serialization, see the status bullet). A 30-iter run is
   ~45 min. The only bigger lever left is item 3 (deferred).
2. **Checkpoint resume + load-to-play + metrics dashboard + a real run.** Status of each hook:
   - **`--resume` ✅ DONE.** `train_loop.py`: `save_checkpoint` now also stores `optimizer.state_dict()`
     + numpy/torch RNG state; `load_for_resume(path, cfg, ckpt_dir)` accepts a path OR `"auto"`/`"latest"`
     (newest `iterNNNN.pt`), rebuilds `OthelloNet` from the *checkpoint's* architecture, restores
     weights + optimizer + RNG, and `train()` continues numbering from `iteration+1` (so `--iterations N`
     = N *more* iters; metrics.jsonl + checkpoints stay on one timeline). A rolling `latest.pt` is written
     each iter for convenience. Replay buffer is NOT checkpointed (refills over 1–2 iters, as planned).
     Covered by `test_resume_continues_training` (SLOW). **This unblocks multi-session Kaggle runs.**
   - **Metrics dashboard ✅ DONE.** ONE template `serve/frontend/dashboard.html` (theme-aware, vanilla
     inline-SVG line charts, no external libs), driven two ways: **static** via `run/dashboard.py`
     (embeds the jsonl into a standalone `data/dashboard.html` — works offline, ideal for a downloaded
     Kaggle run) and **live** via `serve/backend.py` routes `/dashboard` + `/api/metrics` (auto-refresh
     15s). Charts, each with a plain-English "what it measures / what good looks like": `max_depth_beaten`
     (headline staircase), win-rate vs each minimax depth (with the 55% promotion line), loss
     (total/policy/value), self-play g/s, buffer size, avg game length + an accessible data table.
     Palette is the validated dataviz default. **TensorBoard stays skipped (protobuf-broken here).**
   - **Load-to-play (web UI) ✅ DONE.** `serve/backend.py` loads the latest checkpoint
     (`latest_checkpoint()` → prefer `latest.pt`, else newest `iterNNNN.pt`; `load_az_evaluator`
     builds the net from the checkpoint's own num_blocks/channels, cached by file mtime so a
     running server auto-picks-up fresher weights during training) and exposes it as the **`az`**
     / **`az:<sims>`** player spec (via `az_player` from `az/evaluate.py`). `/api/config` reports
     `az_available` + `az_iteration`; the play UI (`serve/frontend/index.html`) shows "AZ net (iter N)"
     as a pickable player for either side (param = MCTS sims/move), so you can pair the trained net
     vs human/greedy/minimax/edax. Play ↔ dashboard are cross-linked. **Still optional:** a
     `run/play_cli.py --black az:<ckpt>` terminal option (same `az_player`; not requested yet).
   - **On-demand Arena eval ✅ DONE.** In-training minimax eval is now OFF by default
     (`Config.eval_every`, kaggle=0; `--eval-every N` overrides; train() reads cfg when the arg is
     None). Replaced by the web **Arena**: `POST /api/arena {opponent, games, az_sims}` runs an
     N-game colour-alternated match (AZ vs the chosen bot) in a **background thread**, `GET
     /api/arena/{id}` polls progress; the play UI shows a live tally + final win rate. Reuses
     `simple.play_game` + `_random_opening`. This is the accurate aggregate strength read the user
     wanted (vs one-off browser games), on demand instead of every iteration.
   - **Kaggle→local pull pipeline ✅ DONE.** `run/pull_kaggle.py` shells out to the `kaggle` CLI
     (`kernels output` for a committed notebook, or `datasets download`), finds the newest
     checkpoint (`latest.pt`/highest `iterNNNN.pt`) + `metrics.jsonl` in the download, and installs
     them into local `data/` so the port-8000 app renders the Kaggle-trained model + dashboard.
     `--watch SECS` polls. The file-locate/install logic is unit-tested; the CLI download step needs
     the user's Kaggle token (can't be tested here). See `run/KAGGLE.md`.
   - **Then the real run — STILL OPEN.** `python run/train_loop.py --kaggle` on Kaggle (workers=1);
     download the latest `data/checkpoints/latest.pt` (or `iter####.pt`) each session and
     `--resume auto` it next session; watch `max_depth_beaten` climb toward depth-4 on the dashboard.
     This is the payoff of all the throughput work.
3. **(Optional, big) Port the batched engine + MCTS to Torch-CUDA tensors.** The array-ops search is
   NumPy = CPU; moving it to Torch tensors on `cuda` is the only path to use the idle GPU (~27%) and
   scale with batch size. **What is/isn't the bottleneck (know this):** the *game play* (self-play) is
   the bottleneck, and within it the **tree-search + game-rules** (`board_batched`/`mcts_batched`, NumPy)
   is what's stuck on the CPU. The *network* work — both position eval during search AND the weight
   updates in `train_steps` — is already on the GPU and is NOT the problem. So this port is specifically
   about moving the search/rules to the GPU.
   **The port is welded to running FAR more games at once** (hundreds–thousands, not 96): the per-op
   launch overhead is fixed *per step* regardless of batch (step count = sims×moves×depth, independent of
   games), so you want each step doing many more games. Bonus: on the GPU big batches are ~free, and more
   games/iter is also a training-data plus (GPU mem is wide open, ~177MB/15GB). But **big batch is
   NECESSARY, maybe not SUFFICIENT** — the ~1M tiny op-launches/round set a time floor that could dominate
   even at large batch; only op fusion (`torch.compile`/Triton, hard with MCTS's dynamic control flow) or
   a GPU experiment settles it. On the CURRENT CPU path, raising `games_per_iter` just costs proportionally
   more time — big batches only pay off once the search is on the GPU.
   **CORRECTION to earlier notes: this is device-agnostic Torch, buildable + correctness-testable on the
   Mac (CPU tensors) — only its GPU *speed* needs Kaggle. NOT PyCUDA** (PyCUDA custom kernels are the
   NVIDIA-only, resume-flex variant, a further step). The proven NumPy `board_batched`/`mcts_batched` are
   the exact blueprint. Pursue only if chasing max speed.
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
  (`--tensorboard`) and fully guarded. Don't rely on tb. The **dashboard**
  (`run/dashboard.py` static, or `/dashboard` live in `serve/backend.py`) reads
  this jsonl — both share ONE template `serve/frontend/dashboard.html`. If you add
  a metric, add it in `flat_metrics` (train_loop.py) AND as a chart/table column in
  that template (the JS keys off the exact `loss/*`, `winrate/minimax_d*`, etc. names).
- **Deep minimax is slow** (pure Python; d6 ≈ 9s/move). Interactive but not snappy;
  the batched engine is the eventual fix. Bot moves in the web backend run in a
  threadpool so they don't block the server.
- **Self-play has THREE interchangeable paths, all parity-verified against the serial
  `mcts.py` oracle** (`cfg` selects; see `az/selfplay.py` module docstring):
  (1) coroutine pool `_play_pool` — per-game seeds, reproducible; the original
  batched-inference path. (2) **array-ops `_play_batch` (DEFAULT)** — batched engine
  (`engine/board_batched.py`) + batched MCTS (`az/mcts_batched.py`), all games searched in
  lockstep as arrays; the fast one. (3) `cfg.selfplay_workers > 1` runs *either* across
  spawn processes (one CPU core each). Keep `tests/test_batched.py` + the `test_az.py`
  array-ops/parallel tests green — they pin every path to the serial oracle.
- **GOLDEN RULE for self-play changes: "vectorised" ≠ "on the GPU".** `board_batched` /
  `mcts_batched` are **NumPy = CPU**; only the network (`Evaluator`) runs on-device. Batching
  over games removes per-game Python overhead but does NOT move the search to the GPU. Getting
  the search onto the GPU is a separate, unbuilt Torch-CUDA port (next-steps item 3).
- **`mcts.py` is a COROUTINE.** `run_root_gen` *yields* each leaf's `(board, player)` and gets
  `(priors, value)` back via `.send()`; serial `run`/`run_root` drive it with the single-board
  evaluator (so `az_player`/eval are unchanged), and the coroutine pool drives many at once. One
  search implementation, three drivers.
- **Batched inference is batch-size-independent by design.** `Evaluator` runs the net in eval
  mode, so BatchNorm uses fixed running stats and a board's output doesn't depend on its batch
  (matches single-board eval to ~1e-7 float rounding). This is *why* batched/array-ops play
  matches serial; root Dirichlet noise keeps the tiny rounding from ever flipping a move.
- **`run_tests.py` runs suites in PARALLEL** (ThreadPoolExecutor over subprocesses); FAST budget
  is wall-clock (~8–12s), not the sum. `--serial` forces one-at-a-time. Prefer parallelism /
  SLOW-tiering over shrinking a real test to fit the budget.

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
- **Measure honestly; never dress a partial result as a verdict.** This user is sharp and
  WILL catch over-promising and inconsistency (they did, repeatedly and correctly). Rules that
  came out of it: don't present a CPU-relative speedup as the GPU verdict; **state what a
  measurement can and cannot show BEFORE running it**; give ranges with caveats, not hype. The
  real throughput arc was batching ~1×, multiprocess 3.5×, array-ops CPU ~2.2× / GPU ~5× — all
  well below the "10–50×" floated early. Under-promise.
- **"DO NOT CODE" / "stop" means stop instantly and REVERT.** When the user halts you, stop
  mid-task and `git`-revert anything added since their last instruction (this happened — a
  half-built Torch port was deleted on request). Don't argue or "just finish".
- **Wants max optimisation but is pragmatic about risk.** Asked to "optimise to the fullest,"
  then, once effort/risk/payoff were laid out in *very simple* terms, chose the cheap safe CPU-core
  bump over the risky GPU-tensor port. Present options plainly (what it means, effort, risk,
  payoff) and let them choose rather than unilaterally chasing the flashy path.
- **Compute is real to them** — a full `--kaggle` run is now feasible (~45 min at current speed).
  Don't burn Kaggle sessions on unvalidated code; correctness-test on the Mac first, then smoke.

## Gotchas for running things
- Modules use `sys.path` inserts to import siblings (engine/opponents/az), not a
  package. Run scripts from the `othello/` directory (or via the given entrypoints).
- Static linters flag the `sys.path` imports in tests as unresolved — false
  positives; the tests pass.
- `data/` and `third_party/` are gitignored (checkpoints, records, Edax binary +
  14MB eval weights). Rummy (a sibling repo dir) is intentionally NOT pushed.
