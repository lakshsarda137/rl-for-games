# Othello RL Agent — Glossary

Key terms for building an AlphaZero-style agent (MCTS + self-play) for Othello.
Terms marked ⭐ are the load-bearing concepts.

---

## 1. The game / environment

- **Environment** — the game itself: it holds the current board and tells you the
  legal moves, the next state after a move, and whether the game is over.
- **State / position** ⭐ — a full snapshot of the game: the 8×8 board (which cells
  hold black/white/empty) plus whose turn it is.
- **Action / move** — placing a disc on a legal square (which flips the opponent's
  discs it brackets).
- **Legal moves** — in Othello you can only play where you'd flip at least one
  opponent disc. If you have none, you **pass**; if neither player can move, the game
  ends.
- **Reward** ⭐ — the outcome signal. In Othello it's sparse: **+1 win, −1 loss,
  0 draw**, given only at the end. No points along the way.
- **Terminal state** — the game is over (board full or no legal moves for either
  side); the winner is whoever has more discs.
- **Zero-sum / perfect-information** — your win is exactly the opponent's loss, and
  nothing is hidden. This is why plain MCTS works (no hidden-info tricks needed).
- **Bitboard** — representing the board as two 64-bit integers (one per color). Move
  generation and flips become fast bitwise ops — a big speed win for self-play.
- **Symmetry / data augmentation** — the Othello board has **8 symmetries**
  (4 rotations × 2 reflections). You can multiply training data 8× by rotating /
  reflecting positions.

## 2. Reinforcement-learning core

- **Agent** — the decision-maker (your net + MCTS together).
- **Policy (π)** ⭐ — the strategy: a probability distribution over which move to play
  in a given state.
- **Value (V)** ⭐ — an estimate of how good a position is: expected final outcome
  from here, in [−1, +1].
- **Self-play** ⭐ — the agent plays *against itself* to generate training games.
  No human data needed — this is the heart of AlphaZero.
- **Episode / trajectory** — one full game from start to terminal, i.e., the sequence
  of states and moves.
- **Exploration vs exploitation** — trying uncertain moves to learn vs. playing the
  current best move. MCTS + noise balances these.

## 3. The neural network ("the net") ⭐

- **Net** — the neural network that looks at a board and outputs two things at once
  (a **two-headed net**):
  - **Policy head** → prior probabilities over all moves ("which moves look
    promising").
  - **Value head** → a single number in [−1, +1] ("who's winning from here").
- **Input / feature planes** — how the board is fed to the net: typically stacked
  8×8 binary grids (my discs, opponent's discs, whose turn, etc.).
- **ResNet / convolutional layers** — the net architecture. Convolutions suit board
  games (local patterns); residual (ResNet) blocks let it go deeper without training
  problems.
- **Weights / parameters** — the numbers inside the net that get tuned during
  training.
- **Forward pass / inference** — running a board through the net to get
  (policy, value). This is what MCTS calls thousands of times.
- **Loss function** ⭐ — what training minimizes: **value loss** (predicted value vs
  actual game result) + **policy loss** (predicted policy vs the MCTS move
  distribution) + regularization.
- **Optimizer / learning rate** — how weights get updated (Adam or SGD); learning
  rate controls step size.
- **Batch / batch size** — how many positions the net processes at once. Bigger
  batches = better GPU utilization (this is where a GPU earns its keep).
- **Checkpoint** — a saved copy of the net's weights at a point in training.

## 4. MCTS (Monte Carlo Tree Search) ⭐

- **MCTS** — the search that looks ahead many moves and, guided by the net, decides
  which move to actually play. It builds a search **tree** of positions.
- **Node / edge** — a node is a position; an edge is a candidate move out of it.
- **Simulation (a.k.a. playout / iteration)** — one pass through the tree. Four
  phases: **select** a path down using PUCT → **expand** a new leaf → **evaluate** it
  with the net's value head → **back up** that value to update the path.
- **PUCT** ⭐ — the selection rule balancing "known-good" vs "worth-exploring": it
  favors moves with high **Q** (average value) *and* high **prior P** (from the net)
  *and* low **visit count N**.
- **Q-value** — the average result seen through a move so far.
- **Prior (P)** — the net's policy output, biasing the search toward promising moves
  before they're explored.
- **Visit count (N)** ⭐ — how many simulations passed through each move. After
  search, **the move visited most is the best move**, and the visit distribution
  becomes the policy training target.
- **c_puct** — the exploration constant tuning how much PUCT favors exploration.
- **Temperature (τ)** — controls move randomness from visit counts: high τ early
  (explore, varied games) → τ→0 later (play greedily).
- **Dirichlet noise** — random noise added to the root priors during self-play so the
  agent keeps trying new openings instead of collapsing to one line.

> Note: classic MCTS estimates a leaf's value with random **rollouts** to the end.
> AlphaZero replaces rollouts with the net's value head — faster and stronger.

## 5. The training loop (AlphaZero-style) ⭐

- **Iteration / generation** — one cycle of: self-play → train → evaluate. Repeated
  many times.
- **Replay buffer** — a store of recent (state, MCTS-policy, game-result) examples
  that training samples from.
- **Training step** — updating the net's weights on batches from the buffer to
  minimize the loss.
- **Arena / evaluation** — pitting the **new** net against the **old** net over many
  games.
- **Gating** — only promoting the new net if it beats the old one by a threshold
  (e.g., wins ≥ 55%). Prevents regressions.
- **Elo** — a rating number to track whether successive generations are actually
  getting stronger.

## 6. Engineering plumbing

- **Tensor** — a multi-dimensional array (the data type nets operate on).
- **Device** — where computation runs: `cpu`, `mps` (Apple Silicon), or `cuda`
  (an NVIDIA GPU).
- **Batching / vectorization** — grouping many operations to run at once; the single
  biggest practical speed lever in self-play.

---

## How it all connects, in one sentence

In each **iteration**, the **net** guides **MCTS** through many **simulations** to
pick strong moves during **self-play**; those games become training data
(state → **visit-count policy** + **game result**); the net **trains** to predict them
better; and if it beats the old net in the **arena**, it's promoted — repeat until the
**Elo** stops climbing.
