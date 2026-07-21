"""A tiny, opt-in self-play profiler — the measurement behind the "is a C++/native
MCTS port worth it?" question.

The C++ payoff is capped by Amdahl's law: C++ can speed up the CPU-side tree search
+ game rules, but NOT the GPU network forward pass. So the decisive number is *what
fraction of self-play wall-time is the GPU net forward*. This profiler measures that
directly (instead of inferring it from a net-size sweep) by bucketing one self-play
iteration into:

    net_fwd        the raw net(x) forward pass on the GPU        (C++ CANNOT speed up)
    net_prep       encode + host<->device transfer + softmax     (mostly CPU; C++ can trim)
    tree_ops       PUCT select / expand / backup (NumPy on CPU)  (C++ TARGET)
    per_move_ops   apply_moves, masks, move sampling (NumPy CPU) (C++ TARGET)
    postproc       x8 augmentation + example/record building     (C++ TARGET)

Timing is GPU-correct: `PROF.sync()` (torch.cuda.synchronize) brackets the forward so
async kernels are attributed to net_fwd, not to whatever CPU code ran next. The
instrumentation lives in `mcts_batched.make_net_evaluator` (net_fwd/net_prep) and
`selfplay._play_batch` (search + postproc); it covers the DEFAULT NumPy array-ops path
(the training path), not the torch or multi-process variants.

Zero cost when off: every hook is guarded by `if PROF.enabled` (default False), so the
parity tests and normal runs are untouched. Enable with `--profile` on train_loop.
"""

import time


class _Prof:
    def __init__(self):
        self.enabled = False
        self.t = {}          # bucket name -> accumulated seconds
        self.c = {}          # counter name -> accumulated count
        self._sync = None    # torch.cuda.synchronize (or None on CPU)

    def configure(self, enabled, sync=None):
        self.enabled = enabled
        self._sync = sync

    def reset(self):
        """Clear buckets/counters (call at the start of each iteration)."""
        self.t, self.c = {}, {}

    def sync(self):
        if self._sync is not None:
            self._sync()

    def add(self, bucket, dt):
        self.t[bucket] = self.t.get(bucket, 0.0) + dt

    def count(self, name, k=1):
        self.c[name] = self.c.get(name, 0) + k

    def clock(self):
        return time.perf_counter()


PROF = _Prof()


def _pct(x, total):
    return f"{100.0 * x / total:5.1f}%" if total > 0 else "   -- "


def format_report(sp_total, iteration=None, device="cpu", cpu_count=None,
                  gpu_name=None):
    """Render one iteration's buckets into a decision-oriented text block.

    `sp_total` is the measured wall-time of `generate_games` for this iteration
    (the authoritative denominator); the derived buckets are pulled from PROF.
    """
    t, c = PROF.t, PROF.c
    net_fwd = t.get("net_fwd", 0.0)
    net_eval_total = t.get("net_eval_total", 0.0)
    search_total = t.get("search_total", 0.0)
    postproc = t.get("postproc", 0.0)

    net_prep = max(0.0, net_eval_total - net_fwd)          # encode + transfer + softmax
    tree_ops = max(0.0, search_total - net_eval_total)     # select/expand/backup
    per_move = max(0.0, sp_total - search_total - postproc)  # ply-level engine ops
    cpu_work = net_prep + tree_ops + per_move + postproc   # everything C++ could attack
    gpu_work = net_fwd                                     # the hard floor

    lines = []
    head = f"self-play profile" + (f" — iter {iteration}" if iteration is not None else "")
    lines.append("=" * 64)
    lines.append(f"  {head}")
    facts = f"device={device}"
    if cpu_count:
        facts += f"  vCPUs={cpu_count}"
    if gpu_name:
        facts += f"  gpu={gpu_name}"
    lines.append(f"  {facts}")
    lines.append("-" * 64)
    lines.append(f"  {'bucket':<16}{'seconds':>10}{'  share':>9}   {'speed-up-able?'}")
    rows = [
        ("net_fwd", net_fwd, "NO  (GPU forward)"),
        ("net_prep", net_prep, "some (encode/xfer)"),
        ("tree_ops", tree_ops, "YES (CPU search)"),
        ("per_move_ops", per_move, "YES (CPU rules)"),
        ("postproc", postproc, "YES (CPU augment)"),
    ]
    for name, secs, tag in rows:
        lines.append(f"  {name:<16}{secs:>10.2f}{_pct(secs, sp_total):>9}   {tag}")
    lines.append(f"  {'-'*14}")
    lines.append(f"  {'self-play':<16}{sp_total:>10.2f}{_pct(sp_total, sp_total):>9}")
    lines.append("-" * 64)

    # counts
    plies = c.get("plies_searched", 0)
    calls = c.get("fwd_calls", 0)
    pos = c.get("positions", 0)
    avg_batch = pos / calls if calls else 0.0
    lines.append(f"  net calls={calls}  positions={pos}  avg batch/call={avg_batch:.0f}"
                 f"  ply-searches={plies}")
    lines.append("-" * 64)

    # the Amdahl verdict
    lines.append(f"  CPU work (C++ target) = {_pct(cpu_work, sp_total).strip()} of self-play;"
                 f"  GPU net_fwd (hard floor) = {_pct(gpu_work, sp_total).strip()}")
    if gpu_work > 0:
        ceil_inf = sp_total / gpu_work
        lines.append(f"  HARD CEILING (all CPU work -> 0): {ceil_inf:4.1f}x")
    else:
        lines.append("  HARD CEILING: (net_fwd not measured — is this the NumPy array-ops path?)")

    # realistic scenarios: native makes CPU work Sx faster, optionally across C cores
    def speedup(native, cores):
        denom = gpu_work + cpu_work / (native * cores)
        return sp_total / denom if denom > 0 else float("inf")

    C = cpu_count or 1
    lines.append("  realistic self-play speed-up if a native port makes CPU work faster:")
    lines.append(f"    {'native x':>9}{'1 core':>9}{f'{C} cores':>10}")
    for s in (3, 5, 10):
        lines.append(f"    {s:>9}{speedup(s, 1):>8.1f}x{speedup(s, C):>9.1f}x")
    lines.append("  (multi-core column = the no-GIL lever C++ unlocks that Python multiprocess")
    lines.append("   couldn't use on a shared GPU; net_fwd is un-parallelisable on one GPU.)")
    lines.append("=" * 64)
    return "\n".join(lines)
