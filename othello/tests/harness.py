"""Shared test harness: `check()` plus a two-knob fast/full runner.

The dev loop must stay fast, so every suite splits its tests into two tiers:

  * FAST  — the default. Kept collectively well under ~20s so you can run the
            whole project's fast set after every change without thinking about it.
  * SLOW  — deep/expensive checks (deep perft, strength matches). Run only when a
            change really needs them, via `--full` or OTHELLO_FULL_TESTS=1.

Each test file calls `run(FAST, SLOW, title)` in its __main__. `run_tests.py`
runs every suite and enforces the fast-tier time budget.
"""

import os
import sys
import time

FAST_BUDGET_S = 20.0  # the whole project's fast tier should finish under this


def check(name, cond):
    print(("PASS" if cond else "FAIL") + f"  {name}")
    assert cond, name


def full_mode():
    """True when slow tests should also run (--full flag or env var)."""
    return "--full" in sys.argv or os.environ.get("OTHELLO_FULL_TESTS") == "1"


def run(fast, slow, title="suite"):
    """Run the fast tests (always) and slow tests (only in full mode), timed."""
    full = full_mode()
    tests = list(fast) + (list(slow) if full else [])
    mode = "FULL" if full else "FAST"
    print(f"### {title}  [{mode}] — {len(fast)} fast"
          f"{', ' + str(len(slow)) + ' slow' if full else ', ' + str(len(slow)) + ' slow skipped'} ###")
    t0 = time.time()
    for t in tests:
        ts = time.time()
        print(f"\n[{t.__name__}]")
        t()
        print(f"  ({time.time() - ts:.2f}s)")
    dt = time.time() - t0
    print(f"\n{title}: {len(tests)} ran in {dt:.1f}s ({mode} mode).")
    return dt
