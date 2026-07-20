# CLAUDE.md — agent orientation for the Othello project

Read this first. It captures where things stand, what's next, and the decisions
and preferences that are **not** obvious from reading the code.

## What this is
A from-scratch AlphaZero for Othello, built phase by phase against
`othello_alphazero_implementation_plan.md`. Each phase ends in a tested,
runnable artifact. See `README.md` for structure.

## Current status (as of 2026-07-20)
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
- **Resume + dashboard + web load-to-play + Arena + W&B ✅ (2026-07-19/20)** — see next-steps item 2
  (all built). `--resume auto`, checkpoint/optimizer/RNG saved, `latest.pt`; local metrics dashboard
  (`run/dashboard.py` + `/dashboard`); play the trained net in the web UI as the `az`/`az:<sims>` player;
  on-demand **Arena** for aggregate strength; **eval OFF by default** (`eval_every=0`).
- **W&B live monitoring + mid-training play ✅ (2026-07-20)** — `--wandb [--wandb-run NAME]` streams
  metrics live to wandb.ai AND uploads the checkpoint as a `latest`-aliased model artifact every
  `--wandb-ckpt-every N` iters; `run/pull_wandb.py --run NAME` fetches the current weights to local
  `data/` at any time (play the bot mid-training — Kaggle can't serve intermediate weights; push-from-
  inside is the only way). Verified end-to-end on Kaggle (graphs render, artifact uploads). `--sims N` /
  `--sims-eval N` override MCTS sims/move per session (for a low→high ramp across resumes).
- **Most recent REAL run (2026-07-20): the user ran 25 iterations, `--sims 110`, `--wandb-run run1`,
  eval off, on a Kaggle T4.** So the pipeline is proven at real scale end-to-end (train → live W&B →
  pull → play). This was a SMALL-scale net (5×64) trained briefly, so it's a modest bot, not strong.
- **Scaling levers BUILT: 10×128 net + device-agnostic Torch search port ✅ (2026-07-20).** The two
  "scaling for strength" items from the prior priority. **(a) Net bumped to 10×128** in
  `Config.kaggle()` (was 5×64); `--net BxC` overrides per session (`--net 5x64` reproduces the old
  run), `--games N` overrides `games_per_iter`. Net compute is on the idle GPU so it's ~free; a bigger
  net raises the strength CEILING but needs more iterations to fill it — train longer to match.
  **(b) `engine/board_torch.py` + `az/mcts_torch.py`** are the op-for-op Torch re-expression of the
  NumPy batched engine + MCTS, so the SEARCH itself (not just the net) runs on `device`.
  Device-agnostic torch, NOT PyCUDA — built + correctness-tested on the Mac with CPU tensors.
  **Correctness EXHAUSTIVELY verified vs the same oracles:** torch engine bit-exact vs `board_numpy`
  (2407 positions); torch MCTS visit counts bit-exact vs the serial `mcts.py` oracle (434 positions,
  48 sims — the torch/numpy float32 math matches to the bit on CPU); torch self-play matches serial
  greedy move-for-move AND is byte-identical to the NumPy array-ops path with noise+temperature
  (`tests/test_batched.py`, `tests/test_az.py`, all green, FAST budget still <20s). Wired **OPT-IN**
  via `--selfplay-torch` / `cfg.selfplay_torch` (`_play_batch_torch`); the NumPy array-ops path stays
  the default. **GPU SPEED NOW MEASURED on a T4 (see next bullet)** — it scales with batch but tops out
  ~2 g/s and is parked as opt-in; NumPy stays the training path.
- **Torch GPU-search port MEASURED on a T4 (2026-07-20) — parked as opt-in; NumPy is the training path.**
  Smoke sweep on a Kaggle T4 (10×128 net, 96 sims, eval off), Torch self-play **g/s vs batch: 0.5 @96
  games / 1.5 @512 / 2.0 @1024; GPU util 41% / 62% / 76%**. So it DOES scale with batch (amortises the
  fixed per-step op-launch cost) and does move the search onto the GPU — BUT: (a) at the batch training
  actually wants (~96 games) it's **3× SLOWER** than NumPy (0.5 vs 1.5 g/s), and (b) even at 1024 games
  it only reaches ~2.0 g/s ≈ the OLD 5×64 NumPy ceiling. The T4 tops out ~2–2.5 g/s because the search
  is **op-launch/sync-bound** (CPU pegged, GPU mem ~5% used, util capped ~76%), not compute-bound. Its
  fast regime (huge batch) is the opposite of training's (many iterations at moderate batch), so on a T4
  it isn't worth using. **DECISION (user, 2026-07-20): train on the NumPy array-ops path; keep
  `--selfplay-torch` as a verified, parked opt-in for future better hardware (A100) or an op-fusion
  (`torch.compile`/CUDA-graphs/Triton) follow-up.** The port is correct, not wasted — it just doesn't
  win on this GPU. Honest arc, measured not guessed.
- **First real 10×128 run (2026-07-20): from scratch, `--kaggle --sims 110 --iterations 30 --wandb
  --wandb-run run2`, 96 games/iter, eval off, NumPy array-ops on a T4.** The proper strength run:
  bigger net + more sims + live W&B. Watch on wandb.ai; `pull_wandb.py --run run2` to play mid-training.
- **Web-app polish DONE (2026-07-20), visually verified before run2 — see next-steps 4–5.** Three asks,
  all in `serve/`: **(a) board aspect-ratio bug FIXED** — `.board` grid had `grid-template-columns` but
  no `grid-template-rows`, so cells were content-height rectangles; added `grid-template-rows:repeat(8,1fr)`
  (+ `aspect-ratio:1` on `.cell`). **GOTCHA hit + fixed:** the per-side checkpoint `.row.ckpt-row`s stayed
  visible when hidden because author `.row{display:flex}` beats the UA `[hidden]{display:none}` — added a
  global `[hidden]{display:none!important}`. **(b) Arena overhauled** — was N games sequential in one thread
  with only a tally; now runs games **concurrently in a `ThreadPoolExecutor`** (threads, not procs: the point
  is shared live state for spectating + the net eval frees the GIL), each game **publishes its live board**
  into the job dict so the UI shows a **grid of mini-boards + a focus board you click to watch any one**, plus
  **pause/resume/stop** (`POST /api/arena/{id}/control`; games check flags between moves). **(c) Checkpoint
  picker** — `/api/config` now returns `checkpoints:[{label,iteration,is_latest}]`; AZ player spec extended
  to **`az:<sims>@<ckpt>`** (`az`, `az:80`, `az@iter0007`, `az:80@iter0007`; parsed by `_parse_az_spec`,
  resolved by `checkpoint_path`), so you can load a SPECIFIC past iter as a player and Arena iterN-vs-iterM /
  vs-edax. Backend end-to-end tested (parallel/spectate/pause/stop, bad-ckpt→400); board+picker+spectate
  screenshotted. **Backend must be launched from `othello/`** (sys.path sibling imports) — `python serve/backend.py`.
- **Full UI rebuild (2026-07-20, after user pushback: "looks horrible", leaky/overflowing controls).**
  `serve/frontend/index.html` rewritten from scratch with a proper design system (CSS custom-prop tokens for
  color/spacing/radius; one `.card`/`.field`/`.btn` component set instead of nested `<fieldset>`s). Fixes the
  concrete blunders: **controls can't overflow** (custom `appearance:none` selects with an inline-SVG chevron +
  `min-width:0` on flex children), **starting position now renders on load** (client-side `previewState`, no
  more blank green rectangle), a real **scoreboard** (two team chips, active side highlighted only DURING a
  game), thinking-dots status, and the Arena spectate view restyled (state pill, progress bar, tally CHIPS,
  focus board + outcome-tinted mini-board grid). Same functionality + same API contract; `az:<sims>@<ckpt>`
  spec, checkpoint subrows, arena controls all preserved. Verified with before/after headless screenshots.
  **Checkpoint upload cadence default changed 5→2** (`--wandb-ckpt-every`, `run/train_loop.py` both the
  argparse default and the `train()` kwarg) so mid-training pulls are fresher; docs updated to match.
- **Checkpoint persistence + delete-from-UI (2026-07-20, user hit data loss).** `latest.pt` is a single
  ROLLING slot, so `pull_wandb.py` overwriting it clobbered the previous model (user pulled run2-iter4 over
  run1-iter25 and lost iter25 locally — still on W&B run1). Fixes: **(a)** `pull_wandb.py` now ALSO writes a
  stable archival copy `data/checkpoints/<run>-iter<NN>.pt` on every pull, so pulls are non-destructive and
  each stays selectable. **(b)** `backend.list_checkpoints()`/`checkpoint_path()` generalised to ANY `*.pt`
  stem (not just `latest`/`iterNNNN`) — parses iteration from an `iter<N>` substring; path-traversal blocked
  (charset regex + dirname==CKPT_DIR check); `_trash/` subdir excluded. So archived pulls appear in the picker
  and play via `az:<sims>@<run>-iter<NN>`. **(c)** New `POST /api/checkpoints/delete {label}` — **SOFT delete**
  (moves the `.pt` to `data/checkpoints/_trash/`, never `os.remove`), surfaced as a **Models card** in the web
  UI (each checkpoint + a Delete button; confirm dialog; refreshes the list + all ckpt selects). Tested
  end-to-end (archived-name listing/parse/play, soft-delete→_trash, traversal/missing→404); real models
  untouched. **To recover the lost iter25: `python run/pull_wandb.py --run run1`** (now archives it as
  `run1-iter25.pt`).
- **Checkpoint picker UX fix (2026-07-20, user: "which iter is the opponent?").** The old two-control
  design (player type = "AZ net" + a SEPARATE checkpoint dropdown) was ambiguous — the type label showed the
  *latest* iter while the real choice was in the second dropdown. Rebuilt so **every checkpoint is its own
  option in ONE dropdown** (value `az@<ckpt>`, grouped under an "AlphaZero net" `<optgroup>`), for the Black/
  White player selects and the Arena opponent; the secondary checkpoint dropdowns are gone. `specOf` maps the
  `az@<ckpt>` option value → `az:<sims>@<ckpt>`. The Arena AZ *champion* stays a checkpoint-only dropdown
  (it's always AZ). Also added **per-colour identity in gameplay**: the scoreboard shows each side's player
  under its name (`prettySpec`, e.g. "AZ iter 25", "Minimax d4"), and the Arena focus label names which
  version plays which colour. Verified: DOM dump shows one option per checkpoint; `az@<ckpt>` specs create
  games end-to-end.
- **Round-robin tournament (2026-07-20, user asked).** New page `serve/frontend/tournament.html` (route
  `GET /tournament`; linked from the play topbar). Add ≥2 bots (any checkpoint/minimax/edax/random/greedy,
  each its own participant), set games/match + concurrency, and every pair plays a match. **Scoring:** more
  game-wins → match win = **3 pts**, match drawn on game-wins = **1.5** each, loss = **0**; standings
  **tiebreaker = total game-wins** across the tournament. Backend: `_run_tourney` builds all `nC2` matches,
  submits every game to ONE `ThreadPoolExecutor(concurrency)` so **several matches run live at once**;
  `_play_tourney_game` reuses the Arena's live-slot/publish/pause-stop machinery; `_recompute_standings`
  rebuilds the points table as games finish. `_TOURNEYS`/`_TOURNEY_PRIV` mirror the arena split. Endpoints:
  `POST /api/tournament {players,games_per_match,concurrency}`, `GET /api/tournament/{id}`,
  `POST .../control {pause|resume|stop}`. Frontend: live **standings table** (medals, pts, W·D·L, gold
  game-wins), a **matches grid** (pending/live/done + score + winner), and a **spectate panel** reusing the
  game-viewer (focus board + clickable game tiles) for whichever match you click. Backend tested end-to-end
  (round-robin count, concurrent live matches, 3/1.5/0 scoring, tiebreak); UI screenshot-verified.

## Next steps (in likely order)

> **▶ CURRENT PRIORITY (updated 2026-07-20): a real 10×128 training run is IN PROGRESS on the NumPy
> path; next work is WEB-APP / BENCHMARK polish while it trains.** The scaling exploration is settled:
> net bumped to 10×128, Torch search port built + measured on a T4 and PARKED (op-launch-bound, ~2 g/s
> ceiling — see the status bullets; decision: train on NumPy). The first real run is
> `--kaggle --sims 110 --iterations 30 --wandb --wandb-run run2` (96 games/iter, from scratch). Open work:
>
> 1. **Let the run train; watch strength.** ~2 g/s on a T4 → a 30-iter run is well under an hour.
>    Watch loss/g/s live on wandb.ai; pull to play mid-training. Resume across sessions with the SAME
>    `--wandb-run run2` + `--resume auto` for a longer run. Strength = net size × iterations.
> 2. **Web app + benchmark polish (user asked, 2026-07-20) — DONE 2026-07-20, verified before run2.** All
>    three asks shipped (see the "Web-app polish DONE" status bullet + next-steps 4–5): board aspect-ratio
>    FIXED; Arena now parallel + spectate-any-game + pause/stop; checkpoint picker (`az:<sims>@<ckpt>`) on the
>    play UI and both Arena sides. **Still open:** spectate/replay of `data/game_records/*.json` (item 4) and
>    Elo instead of the raw win-rate ladder (item 5); a general design pass (item 6).
>
> **Bottleneck cheat-sheet (settled):** more **sims / more games → CPU** on the NumPy path (search+rules
> are NumPy, ~87% of self-play). Bigger **net → GPU** (cheap). The Torch port moved the search onto the
> GPU but the T4 is op-launch-bound there (~2 g/s), so it doesn't win — NumPy is the training path. A
> real search speedup would need op-fusion (`torch.compile`/CUDA-graphs/Triton, hard with MCTS's dynamic
> control flow) or better hardware; not worth it for this project now.

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
3. **Port the batched engine + MCTS to Torch tensors — ✅ BUILT + CPU-VERIFIED + T4-MEASURED (2026-07-20);
   PARKED (op-launch-bound on a T4, ~2 g/s, doesn't beat NumPy — see status bullets).**
   `engine/board_torch.py` + `az/mcts_torch.py` are the op-for-op Torch
   re-expression of `board_batched`/`mcts_batched`, so the tree-search + game-rules (the NumPy=CPU wall)
   run on `device`. Self-play driver: `selfplay._play_batch_torch` (torch twin of `_play_batch`), opt-in
   via `cfg.selfplay_torch` / `--selfplay-torch`; `make_net_evaluator_torch` keeps the net eval on-device
   (no NumPy round-trip inside the search). **What is/isn't the bottleneck (still true):** the *search +
   rules* were the CPU wall; the *network* (search eval + `train_steps` updates) was already on the GPU.
   This port moves the search/rules onto the GPU too. **Device-agnostic Torch, NOT PyCUDA** — built and
   correctness-tested on the Mac with CPU tensors (PyCUDA custom kernels would be the NVIDIA-only,
   further step). Parity proven bit-for-bit vs the NumPy/serial oracles (engine on 2407 positions, MCTS
   on 434 positions, self-play byte-identical to array-ops), `tests/test_batched.py` + `tests/test_az.py`.
   **What's LEFT:** the GPU throughput measurement (priority item 1). The port is **welded to running FAR
   more games at once** — the per-op launch overhead is fixed *per step* regardless of batch (step count
   = sims×moves×depth, independent of games), so amortise it with a big `--games`. **Big batch is
   NECESSARY, maybe not SUFFICIENT**: the ~1M tiny op-launches/round set a floor only a GPU run (or op
   fusion via `torch.compile`/Triton, hard with dynamic MCTS control flow) settles. On the CPU path a
   bigger batch just costs proportionally more, so the payoff is a GPU-only question — go measure it.
4. **Web app — the user's concrete asks (2026-07-20). Board + Arena ✅ DONE 2026-07-20; records-replay OPEN.**
   - **Board aspect-ratio bug ✅ DONE** — `.board` grid had no `grid-template-rows`, so cells were
     content-height rectangles; added `grid-template-rows:repeat(8,1fr)` + `aspect-ratio:1` on `.cell`, and a
     global `[hidden]{display:none!important}` (author `.row{display:flex}` was overriding the UA hidden rule).
   - **Arena controls ✅ DONE** — games now run **concurrently** (`ThreadPoolExecutor`, `--workers`/`max 8`),
     each **publishes its live board** into the job dict (`job["games"][i]`), and the UI shows a **mini-board
     grid + clickable focus board** to watch any one, plus **pause/resume/stop** via `POST /api/arena/{id}/control`
     (games poll `job["cancel"]`/`job["paused"]` between moves). Threads not procs — the point is shared live
     state + the net eval frees the GIL. `_ARENA_PRIV` holds the non-JSON sidecar (factories/lock/openings).
   - **Spectate self-play records — STILL OPEN** — watch/replay mode from `data/game_records/*.json` (records
     exist, UI not built). The Arena mini-board renderer in `index.html` is the reusable piece to build on.
5. **Benchmark past checkpoints ✅ (picker DONE 2026-07-20); Elo/promotion gating STILL OPEN.** The picker
   shipped: `/api/config` returns `checkpoints[]`, the AZ spec is now `az:<sims>@<ckpt>` (e.g. `az:80@iter0007`,
   `az@iter0003`; `_parse_az_spec`/`checkpoint_path`), the play UI has a per-side ckpt dropdown, and the Arena
   lets BOTH sides pick a checkpoint — so you can Arena iterN vs iterM vs edax now. **Left:** an Elo rating
   from those head-to-heads instead of the raw win-rate ladder, and a strength-curve chart.
6. **General web-UI polish / design pass** — the user finds it visually rough beyond the board bug.

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
  (`--tensorboard`) and fully guarded. Don't rely on tb. **Weights & Biases** is the
  LIVE remote view (`--wandb [--wandb-run NAME]`, same guarded pattern as tb): the
  process pushes each iteration to wandb.ai over the internet, which is the ONLY way
  to watch in-progress Kaggle training (pull/`kaggle kernels output` only publishes a
  COMMITTED run's output at completion — push-from-inside vs pull-artifact). Reuse the
  same `--wandb-run` with `--resume` to continue one live curve. **To PLAY the bot
  mid-training** (Kaggle can't serve intermediate weights — one kernel, cells run
  serially, and `kernels output` only publishes a COMMITTED run): with `--wandb`,
  training uploads `latest.pt` to W&B as a `latest`-aliased artifact every
  `--wandb-ckpt-every N` iters (default 2); `run/pull_wandb.py --run NAME` fetches it
  locally any time. `pull_kaggle.py` is the after-a-commit path; `pull_wandb.py` is
  the live path. The **dashboard**
  (`run/dashboard.py` static, or `/dashboard` live in `serve/backend.py`) reads
  this jsonl — both share ONE template `serve/frontend/dashboard.html`. If you add
  a metric, add it in `flat_metrics` (train_loop.py) AND as a chart/table column in
  that template (the JS keys off the exact `loss/*`, `winrate/minimax_d*`, etc. names).
- **Deep minimax is slow** (pure Python; d6 ≈ 9s/move). Interactive but not snappy;
  the batched engine is the eventual fix. Bot moves in the web backend run in a
  threadpool so they don't block the server.
- **Self-play has FOUR interchangeable paths, all parity-verified against the serial
  `mcts.py` oracle** (`cfg` selects; see `az/selfplay.py` module docstring):
  (1) coroutine pool `_play_pool` — per-game seeds, reproducible; the original
  batched-inference path. (2) **NumPy array-ops `_play_batch` (DEFAULT)** — batched engine
  (`engine/board_batched.py`) + batched MCTS (`az/mcts_batched.py`), all games searched in
  lockstep as arrays; the fast CPU one. (3) `cfg.selfplay_workers > 1` runs (1) or (2) across
  spawn processes (one CPU core each). (4) **Torch array-ops `_play_batch_torch`** (opt-in,
  `cfg.selfplay_torch`) — the same array-ops search but on torch tensors (`engine/board_torch.py`
  + `az/mcts_torch.py`), so it runs on `device` (GPU). Keep `tests/test_batched.py` + the
  `test_az.py` array-ops/parallel/torch tests green — they pin every path to the serial oracle.
- **GOLDEN RULE for self-play changes: "vectorised" ≠ "on the GPU".** `board_batched` /
  `mcts_batched` are **NumPy = CPU** (still the DEFAULT); only the network (`Evaluator`) runs
  on-device there. Batching over games removes per-game Python overhead but does NOT move the
  search to the GPU. The Torch twins that DO put the search on-device — `board_torch` /
  `mcts_torch`, opt-in via `--selfplay-torch` — now exist (built + CPU-verified 2026-07-20); their
  GPU speed is still unmeasured. So: NumPy path = CPU search; Torch path = on-device search.
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
