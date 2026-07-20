# Training graphs — what every metric means

This is a plain-English guide to every chart the training loop produces. The same
metrics show up in three places:

- **Live on Weights & Biases** (`wandb.ai/<you>/othello-alphazero`) while training runs — add `--wandb` to the run.
- **The local dashboard** — `http://127.0.0.1:8000/dashboard` (or `python run/dashboard.py`), which reads `data/metrics.jsonl`.
- **`data/metrics.jsonl`** — one JSON line per iteration; the raw source of truth.

**The 10-second mental model.** Each *iteration* the agent (1) plays a batch of
games against itself, (2) trains its neural network on those games, and
(3) optionally tests itself against fixed minimax opponents. The graphs below
track that loop. They fall into three groups:

- **Strength** — is it actually getting better at Othello? *(the payoff)*
- **Learning** — is the network fitting its training data? *(the engine)*
- **Health** — is the loop running sanely? *(the plumbing)*

A note before you read: loss going down is **not** the goal in itself — *strength*
going up is. Loss is noisy here because the network trains on games made by an
ever-changing version of itself (a moving target), so judge it by its long-run
drift, not iteration-to-iteration wiggles.

---

## Strength metrics (the payoff)

> These come from **evaluation**, which is **OFF by default** (`eval_every=0`). They
> only appear if you train with `--eval-every N`. With eval off, measure strength
> on demand instead with the web **Arena** (play the net vs a bot over N games).

### `max_depth_beaten` — the headline number
- **Layman:** "How tough an opponent can the bot reliably beat?" We keep a ladder of
  minimax opponents that think 1, 2, 4, … moves ahead (deeper = harder). This is the
  *deepest* one the bot wins against at least 55% of the time.
- **What it means:** the single best summary of the agent's playing strength over time.
- **Good behavior:** a **rising staircase** — it sits at a level for a while, then
  steps up to the next depth. Up-and-to-the-right means training is genuinely working.
- **Red flags:** stuck at 0 for many iterations (not learning), or bouncing down
  (instability / the net getting worse).

### `winrate/minimax_dN` — win rate vs each depth
- **Layman:** "Out of 100 games against the depth-N opponent, how many did we win?"
  (draws count as half). One line per depth on the ladder.
- **What it means:** the detail behind `max_depth_beaten` — you can watch a depth get
  "solved" as its line climbs.
- **Good behavior:** each line **climbs and crosses the 55% line** (the promotion
  threshold). Easy depths (d1, d2) cross first; harder depths (d4, d6) lag behind —
  that ordering is exactly what you want to see.
- **Red flags:** all lines flat near 50% (no progress) or drifting down.

---

## Learning metrics (the engine)

The network has two "heads": a **policy** head (which move to play) and a **value**
head (who's winning). Each has its own loss; the total is their sum.

### `loss/policy` — move-prediction loss
- **Layman:** "How well do the network's gut-feel move preferences match what the
  deeper search actually decided was good?" The search (MCTS) produces a better move
  distribution than the raw network; policy loss measures the gap.
- **What it means:** cross-entropy between the network's move probabilities and the
  search's visit distribution. Usually the larger of the two loss components (it's a
  choice over up to 65 actions; starts around ~2.0–2.3).
- **Good behavior:** **downward drift** over many iterations — the network's instincts
  are catching up to the search, so it needs less searching to play well.
- **Red flags:** rising steadily, or NaN/exploding.

### `loss/value` — win/loss-prediction loss
- **Layman:** "How well does the network guess who will win from a given position?"
  Its guess is a number from −1 (I'm losing) to +1 (I'm winning); this compares that
  guess to who *actually* won the game.
- **What it means:** mean-squared error between the predicted value and the real game
  result, from the side-to-move's perspective. A fresh random net guesses ~0, so
  against ±1 outcomes it starts near **~1.0** and should fall from there.
- **Good behavior:** **decreasing** toward a small number — the net is learning to
  read positions. (It won't reach 0: Othello has real uncertainty, and self-play
  outcomes are noisy.)
- **Red flags:** stuck at ~1.0 (learning nothing) or collapsing to ~0 almost instantly
  (a sign it's overfitting a tiny/stale set of positions rather than generalizing).

### `loss/total` — the two combined
- **Layman:** the overall "how wrong was the network" number = policy loss + value loss.
- **What it means:** the quantity training directly minimizes (plus a small hidden L2
  weight penalty that isn't in the printed number).
- **Good behavior:** **general downward trend**, allowed to be bumpy. Each iteration
  the training data changes (new self-play games), so expect wiggles — look at the
  trend across ~10 iterations, not one step.
- **Red flags:** trending up, flat-and-high forever, or NaN (usually too-high learning
  rate or a data bug).

---

## Health metrics (the plumbing)

These don't measure skill — they tell you the loop is running correctly and fast.

### `selfplay_games_per_sec` — throughput
- **Layman:** "How many self-play games are we cranking out per second?" Pure speed,
  not skill.
- **What it means:** how fast the (slow) self-play step feeds the trainer. Higher =
  more training data per hour.
- **Good behavior:** **stable and as high as your hardware allows** (≈2.1 games/sec on
  a Kaggle T4 with this config). A flat line is fine and expected.
- **Red flags:** a sudden drop mid-run (something is contending for the CPU/GPU, e.g.
  accidentally running `--workers >1` on a GPU).

### `buffer_size` — the replay buffer
- **Layman:** "How many past positions are in the pool we train from?" The bot learns
  from a rolling window of its most recent games, not just the latest batch.
- **What it means:** number of training examples currently held (each game position is
  also expanded into its 8 mirror/rotation symmetries, so this grows fast).
- **Good behavior:** **rises quickly in the first iterations, then flattens at the cap**
  (`buffer_size` in the config, e.g. 100,000). Flat-at-the-cap is normal — old games
  drop off as new ones arrive.
- **Red flags:** staying tiny (self-play not producing examples) — usually paired with
  a suspicious loss curve.

### `game_len_avg` — average game length
- **Layman:** "On average, how many moves did each self-play game last?" A full Othello
  game is about 60 half-moves (plies).
- **What it means:** mostly a sanity check on the engine and search.
- **Good behavior:** hovers in a **sensible range (~55–60 plies)** and drifts only
  gently as the bot's style changes.
- **Red flags:** wild swings or values far from ~60 — can flag a rules or search bug
  (e.g. games ending far too early).

---

## Putting it together — "is training working?"

In one glance, healthy training looks like:

1. `max_depth_beaten` a **rising staircase** (or, with eval off, Arena win rates
   climbing over time). *This is the real proof.*
2. `loss/policy` and `loss/value` **drifting down** over many iterations (bumpy is fine).
3. `selfplay_games_per_sec` **steady**, `buffer_size` **filled to its cap**,
   `game_len_avg` **near ~60**.

If (2) and (3) look right but (1) is flat, the loop is *running* but not *improving* —
usually a signal to train longer, raise self-play search (`sims_selfplay`), or adjust
the learning rate. If (1) is rising, everything's working — just keep going.
