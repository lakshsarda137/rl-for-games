"""Shared test helpers: `check()` and a fast/full runner (same pattern as othello/tests)."""

import os
import sys
import time


def check(name, cond):
    print(("PASS" if cond else "FAIL") + f"  {name}")
    assert cond, name


def full_mode():
    """True when slow tests should also run (--full flag or env var)."""
    return "--full" in sys.argv or os.environ.get("RUMMY_FULL_TESTS") == "1"


def run(fast, slow, title="suite"):
    """Run the fast tests (always) and slow tests (only in full mode), timed."""
    full = full_mode()
    tests = list(fast) + (list(slow) if full else [])
    mode = "FULL" if full else "FAST"
    print(f"### {title}  [{mode}] — {len(fast)} fast, {len(slow)} slow"
          f"{'' if full else ' skipped'} ###")
    t0 = time.time()
    for t in tests:
        ts = time.time()
        print(f"\n[{t.__name__}]")
        t()
        print(f"  ({time.time() - ts:.2f}s)")
    dt = time.time() - t0
    print(f"\n{title}: {len(tests)} ran in {dt:.1f}s ({mode} mode).")
    return dt
