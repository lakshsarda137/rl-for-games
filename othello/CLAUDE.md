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
  `Config.kaggle()` + `run/KAGGLE.md`. **First real GPU run has not been done yet.**

## Next steps (in likely order)
1. **Run the first Kaggle GPU training** (`run/KAGGLE.md`, `--kaggle`). Confirm it
   trains on GPU and `max_depth_beaten` climbs past depth-2 toward depth-4.
2. **Batched self-play inference** — THE key performance lever. MCTS currently
   evaluates ONE board per network call (see `az/mcts.py` `_expand` → `Evaluator.__call__`).
   Unbatched single-position eval barely uses the GPU. Batching leaf evals across
   many concurrent games unlocks the full 160-sim / 200-game config. This is the
   real Phase 3 work and interweaves with Phase 4.
3. **Checkpoint resume** — `train_loop.train` starts fresh each run. Add
   `--resume <ckpt>` that loads `state_dict` (needed to continue across Kaggle
   sessions / weekly quota).
4. **Phase 4 — PyCUDA batched bitboard engine** (`engine/board_cuda.py`, the
   resume-worthy custom-kernel piece). Verify parity vs NumPy engine.
5. **Wire self-play records into the web UI** (watch/replay mode).
6. **Elo + promotion gating** in evaluation (currently just win-rate ladder).

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
