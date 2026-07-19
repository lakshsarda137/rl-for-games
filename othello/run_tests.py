"""Run every test suite. Fast by default; `--full` adds the heavy tier.

    python run_tests.py            # FAST tier only — the every-change dev check
    python run_tests.py --full     # everything, no time ceiling
    python run_tests.py --serial   # one suite at a time (debug / clean interleave)

Each suite runs in its own process (isolation + clean timing) and, by default,
all suites run CONCURRENTLY — the suites are independent, so the dev loop only
waits for the slowest one, not the sum. Each suite's output is captured and
printed as a block so parallel logs stay readable. Fast mode is expected to
finish under harness.FAST_BUDGET_S (20s wall-clock); exceeding it is reported
loudly so the dev loop never silently rots.
"""

import glob
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tests"))
from harness import FAST_BUDGET_S  # noqa: E402

FULL = "--full" in sys.argv
SERIAL = "--serial" in sys.argv


def _run_suite(path, extra):
    """Run one suite in its own process; return (name, code, seconds, output)."""
    name = os.path.basename(path)
    started = time.time()
    proc = subprocess.run([sys.executable, path] + extra,
                          capture_output=True, text=True)
    return name, proc.returncode, time.time() - started, proc.stdout + proc.stderr


def main():
    suites = sorted(glob.glob(os.path.join(HERE, "tests", "test_*.py")))
    extra = ["--full"] if FULL else []

    t0 = time.time()
    if SERIAL:
        results = [_run_suite(p, extra) for p in suites]
    else:
        # Threads block on subprocess I/O, so the suites truly run in parallel;
        # capped at the CPU count so we don't oversubscribe a small machine.
        workers = min(len(suites), os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda p: _run_suite(p, extra), suites))
    total = time.time() - t0
    cpu_total = sum(dt for _, _, dt, _ in results)

    for name, _code, _dt, output in results:   # each suite's log as one block
        print(output, end="" if output.endswith("\n") else "\n")

    print("\n" + "=" * 56)
    print(f"{'SUITE':<22} {'RESULT':<8} {'TIME':>8}")
    print("-" * 56)
    for name, code, dt, _ in results:
        print(f"{name:<22} {'ok' if code == 0 else 'FAIL':<8} {dt:>7.1f}s")
    print("-" * 56)
    mode = ("FULL" if FULL else "FAST") + ("" if SERIAL else ", parallel")
    print(f"{'WALL':<22} {'':<8} {total:>7.1f}s   [{mode}]")
    if not SERIAL:
        print(f"{'(sum of suite CPU)':<22} {'':<8} {cpu_total:>7.1f}s")

    failed = [n for n, c, _, _ in results if c != 0]
    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
    if not FULL and total > FAST_BUDGET_S:
        print(f"\n⚠  FAST tier took {total:.1f}s > {FAST_BUDGET_S:.0f}s wall budget — "
              "split a slow test into its suite's SLOW list, or a suite is too heavy.")
    if not failed and (FULL or total <= FAST_BUDGET_S):
        print("\nAll good.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
