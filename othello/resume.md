# Othello AlphaZero — résumé material

A from-scratch **AlphaZero** reinforcement-learning agent that learns Othello purely
from self-play, plus the GPU-acceleration, profiling, and full-stack tooling around it.
Everything below is drawn from the real project (`othello/`) and is meant to be copied
into a résumé, LinkedIn/portfolio, or used as interview talking points.

---

## Skills / keywords (ATS-friendly)

`Python` · `PyTorch` · `NumPy` · `CUDA` · `GPU acceleration (NVIDIA T4)` ·
`Reinforcement Learning` · `Deep Learning` · `Monte-Carlo Tree Search (MCTS)` ·
`Convolutional Neural Networks (CNN)` · `Residual Networks (ResNet)` · `Self-Play` ·
`Vectorization / SIMD-style batching` · `Multiprocessing` · `FP16 / mixed precision` ·
`Performance profiling (Amdahl analysis)` · `FastAPI` · `REST APIs` ·
`Weights & Biases (W&B)` · `Kaggle GPU` · `Git` · `Unit / parity testing`

---

## Résumé bullets (project entry — copy-paste)

**AlphaZero Othello — Self-Play Deep Reinforcement Learning Engine**
*Python · PyTorch · CUDA / NVIDIA T4 GPU · NumPy · MCTS · Weights & Biases · FastAPI*

- Built an **AlphaZero** game-playing agent **from scratch** in Python/PyTorch — a
  **10-block × 128-channel residual CNN** with separate **policy and value heads**,
  guided by **PUCT Monte-Carlo Tree Search** — that learns Othello **purely from
  self-play** (no human games, no opening book), reaching a **~80% win rate vs Edax L3**
  (a world-class engine) over a **200-game** match.
- **Accelerated self-play ~5× (0.4 → 2.1 games/sec) on an NVIDIA T4 GPU** by rewriting
  the game engine and MCTS as **vectorized batched array operations** (searching a whole
  batch of games in lockstep) with **single-pass batched GPU inference** and **multi-core
  multiprocessing** — verified **bit-exact** against a reference implementation on
  thousands of positions.
- **Profiled the CPU/GPU pipeline (Amdahl-style bottleneck analysis)** to drive
  optimization decisions: measured the network forward pass at **~43%** of self-play,
  evaluated **FP16 tensor-core inference** and a device-agnostic **PyTorch on-GPU search
  port**, and made an **evidence-based decision not to ship either** after measurements
  showed the workload was kernel-launch-bound rather than compute-bound.
- **Engineered the end-to-end training + serving stack:** resume-safe checkpointing with
  **cosine learning-rate decay** across multi-session Kaggle GPU runs, **live Weights &
  Biases** metric streaming, a **FastAPI web app** (play / arena / round-robin tournament
  with live game spectating), and a **tiered, parallelized test suite** (bit-exact parity
  + perft) kept under **20 s** for a fast dev loop.

---

## One-line version

Built an **AlphaZero** game AI from scratch (PyTorch **ResNet** + **MCTS**) that learns
Othello from self-play to **~80% vs Edax L3** (200 games); **accelerated self-play ~5×
(0.4 → 2.1 games/sec) on an NVIDIA T4 GPU** via vectorization, batched CUDA inference, and
multiprocessing, backed by profiling-driven optimization and a FastAPI tournament app.

---

## Extended technical summary (portfolio / LinkedIn "About this project")

A scaled-down but faithful **AlphaZero** implementation for Othello, built end to end and
verified against reference oracles at every layer.

**The learning algorithm.** A single neural network maps a board position to *(a) a policy*
— a probability distribution over the 65 possible actions (64 squares + "pass") — and *(b) a
value* — a scalar in [-1, 1] estimating who is winning from the side-to-move's perspective.
The network is a **residual CNN** (a 3×3-conv stem → 10 residual blocks of 128 channels →
a policy head and a value head), the same family of architecture used by DeepMind's
AlphaGo Zero. At every move, **PUCT Monte-Carlo Tree Search** uses the network to look ahead
and produce a *stronger* move distribution than the raw network; the agent then trains the
network to imitate that improved search and to predict the eventual game result. Iterating
this self-play → search → train loop makes the agent bootstrap from random play to strong
play with **no human data**.

**Key ML details:** masked policy softmax over legal moves only; a combined loss of
**value MSE + policy cross-entropy + L2**; **Dirichlet root noise** and a temperature
schedule for exploration; **8-fold dihedral data augmentation** (every position expanded
into its symmetric equivalents); a rolling **replay buffer**; and **resume-safe cosine LR
decay** implemented as a pure function of the global iteration so it survives checkpoint
restarts across cloud sessions (a stateful scheduler would silently reset on resume).

**GPU / systems engineering (the part interviewers dig into).** A naïve pure-Python MCTS
self-play loop was **CPU-bound at ~0.4 games/sec** with the GPU nearly idle. I profiled it,
found the bottleneck was per-game Python overhead (not the network), and rewrote the game
engine and the tree search as **vectorized batched array operations** — a batched Othello
engine over `[B, 8, 8]` tensors (using a **directional flood-fill** to compute captures for
all boards at once) and **B MCTS trees advanced in lockstep** as flat arrays, so a whole
batch of self-play games is searched together and the network evaluates all their pending
leaf positions in **one batched forward pass** (a GPU scores 256 boards for ≈ the cost of
one). Combined with **multi-core multiprocessing**, this reached **~2.1 games/sec on an
NVIDIA T4 (~5×)**. Every fast path is **parity-tested bit-for-bit** against a simple serial
reference (perft move-generation counts 1–8, exact MCTS visit counts on hundreds of
positions, move-for-move self-play), so speed never traded away correctness.

**Measurement culture (what I'm most proud of).** I treated performance claims like
experiments. I built an opt-in **profiler** and did an **Amdahl-style analysis** before
committing to expensive optimizations: it showed the GPU network forward was **~43%** of
self-play time (an upper bound on any CPU-side rewrite) and that both a proposed **C++/CUDA
native search port** (~2.3× ceiling, large build cost) and **FP16 tensor-core inference**
(~0× — the tiny 8×8 board makes inference **kernel-launch-bound, not compute-bound**) were
**not worth it on this GPU**. Deciding *not* to build things based on measurement — and
documenting why — is a core part of the project. I also measure *strength* honestly: on a
**fixed opponent yardstick (Edax / minimax) over 40–200 games**, never on training loss
(which only measures fit to the net's own moving self-play targets).

**Full-stack tooling.** A **FastAPI** web app to play the trained bot or watch bots play,
an **Arena** (parallel N-game matches with live move-by-move spectating and pause/resume),
a **round-robin tournament** with a live standings table, checkpoint management, a live
**training-metrics dashboard**, and **Weights & Biases** streaming so I can watch remote
Kaggle GPU training in real time and pull intermediate weights to play mid-run.

---

## Interview talking points (be ready to defend these)

- **"What is AlphaZero and what did you actually build?"** — Explain the self-play →
  MCTS → train loop and that I implemented all of it from scratch: the game engine, the
  encoder, the ResNet, PUCT search, the training loop, and the eval harness.
- **"How did you get the ~80% vs Edax number?"** — Fixed-opponent match: the trained
  net (via MCTS) vs **Edax level 3** over a **200-game**, colour-alternated match, reported
  as an aggregate win rate — not a single game and not training loss. (Name the level and
  game count; a specific number is more credible than a vague one.)
- **"Where was the bottleneck and how did you find it?"** — Profiling showed CPU-bound
  per-game Python overhead, not the GPU. Fix = vectorize the engine + search into batched
  array ops and batch the network inference; result ~5× on a T4. I can walk through the
  Amdahl ceiling math.
- **"Why *didn't* you write the CUDA/C++ port or use FP16?"** — Because I measured first:
  the forward is only ~43% of self-play (caps the payoff), and the 8×8 board makes
  inference launch-bound so tensor cores don't help. Good engineering includes *not*
  building the flashy thing when the numbers say it won't pay off.
- **"How do you know your fast code is correct?"** — Bit-exact parity tests against a
  simple serial oracle at every layer (perft, MCTS visit counts, move-for-move self-play),
  plus tiered fast/slow test suites run in parallel and kept under 20 s.
