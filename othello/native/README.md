# `native/` — the C++ port of move generation and tree search

An **optional** accelerator. If it isn't built, everything falls back to NumPy
automatically and nothing breaks — it just runs slower.

```bash
python native/build.py     # from othello/ — takes ~20s, one compiler call
python run_tests.py        # tests/test_native.py pins C++ == NumPy, bit for bit
python native/bench.py     # A/B the two paths
```

## What moved to C++

| Was (Python) | Now (C++) | What it does |
|---|---|---|
| `engine/board_batched.py` | bitboard engine | `legal_move_masks`, `legal_action_masks`, `apply_moves`, `is_terminal`, `count_discs`, `winner`, `encode_batch` |
| `az/mcts_batched.py::run_batched` | scalar tree loops | PUCT select, expand, backup |

The NumPy versions are still there — renamed `_np_*` in `board_batched.py`, and
`_run_batched_numpy` in `mcts_batched.py`. They are the reference implementation
and the parity oracle, not dead code.

Two representation changes are internal only. Boards are `uint64` bitboards
inside C++ (the 8-direction flood becomes shift-and-mask on two 64-bit words
instead of array ops over `[B,8,8]`), and each game's tree is plain scalar
arrays. Everything at the Python boundary is unchanged: still `int8 [B,8,8]`
absolute colours in, still `float32 [B,65]` visit counts out. No caller changed.

## What deliberately did *not* move

- **The network.** `run_batched` takes an `evaluate` callback and calls back into
  Python once per simulation step, exactly where the NumPy loop made its batched
  net call. So this build needs **no LibTorch and no CUDA** — it is a plain CPU
  `.so` linking nothing but header-only pybind11, and the same source builds on a
  Mac and on Kaggle. It also means C++ cannot touch the GPU forward, which was
  ~43% of self-play on a T4 and is what caps the whole exercise (see below).
- **The root Dirichlet noise.** It's the only part of the search that consumes
  the NumPy RNG, so reproducing PCG64's gamma stream in C++ would be a permanent
  parity liability for no gain — it's `O(B*65)` once per search against an
  `O(sims*depth*B*65)` inner loop. Python draws it and passes the vector in.
- **`root_value` and the ×8 augmentation.** `root_value` is finished in NumPy
  from the raw root rows because it's the one float reduction whose result
  depends on summation order. The augmentation was only ~2.6% of self-play and
  isn't worth a second implementation to keep in sync.

## Bit-exact parity is the contract

Not "close" — identical. That's what lets the C++ path inherit every guarantee
the NumPy path already proved against `board_numpy` and the serial `mcts.py`.
What makes it hold:

- Every float is `float`, never `double`. `c_puct` and `dirichlet_eps` are cast
  to `float` at exactly the points NumPy casts them, and `1.0 - eps` is computed
  in `double` first because NumPy evaluates that Python-scalar expression in
  float64.
- Operator order matches NumPy's left-to-right evaluation, op for op.
- `N.sum()` is order-independent here: `N` only ever holds non-negative *integer*
  float32 values, and integer sums below 2²⁴ are exact in float32 — so NumPy's
  pairwise summation and a naive C++ loop agree bit for bit. (This is why the
  usual "reproduce NumPy's pairwise sum" trap doesn't apply.)
- `argmax` keeps NumPy's **first**-maximum tie-break.
- Leaf batches are assembled in the same order, so the network sees identical
  batches and rounds identically.
- Built with **`-ffp-contract=off`** so no FMA silently fuses a multiply-add into
  a differently-rounded result. Do not remove that flag.

`tests/test_native.py` checks all of it: every engine function, the search with
and without root noise, and a whole self-play batch compared example by example.

## Switching it off

```bash
OTHELLO_NATIVE=0 python run/train_loop.py ...   # env var, read at import
python run/train_loop.py --no-native ...        # CLI, flips it at runtime
```

Both give byte-identical games — that's the point of the parity tests. Use them
to A/B the speed-up, or to rule the port out when chasing a bug.

## On Kaggle

Build once per session, before training:

```
!cd /kaggle/working/rl-for-games/othello && python native/build.py
```

Don't ship a prebuilt `.so`: it is tied to a Python version, libc and CPU
architecture, and Kaggle images change under you. Building from source takes
~20–40s and a failure is non-fatal (you get the NumPy path and a warning line).

## What to expect

The measured T4 profile that motivated this put `net_fwd` at 43% of self-play —
the GPU forward, which C++ cannot touch. That is a hard Amdahl ceiling of **~2.3×**
at 96 games/96 sims, and higher (~2.5–3×) at `--games 200`, where the CPU-side
work grows but the net-call *count* doesn't.

So: the migrated code itself is 30–100× faster (see `bench.py --games 96`), and
the *end-to-end* self-play win is bounded well below that by everything C++ isn't
allowed to help with. Measure the real number on the T4 with:

```
python run/train_loop.py --kaggle --iterations 2 --profile             # C++
python run/train_loop.py --kaggle --iterations 2 --profile --no-native # NumPy
```

and compare the `self-play` totals. Ignore any CPU-only ratio from a laptop — on
a CPU the network dominates, so a local A/B badly understates the GPU win.
