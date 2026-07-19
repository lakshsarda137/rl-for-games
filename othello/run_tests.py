"""Run every test suite. Fast by default; `--full` adds the heavy tier.

    python run_tests.py          # FAST tier only — the every-change dev check
    python run_tests.py --full   # everything, no time ceiling

Each suite runs in its own process (isolation + clean timing). Fast mode is
expected to finish under harness.FAST_BUDGET_S (20s); exceeding it is reported
loudly so the dev loop never silently rots.
"""

import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tests"))
from harness import FAST_BUDGET_S  # noqa: E402

FULL = "--full" in sys.argv


def main():
    suites = sorted(glob.glob(os.path.join(HERE, "tests", "test_*.py")))
    extra = ["--full"] if FULL else []
    results = []
    t0 = time.time()
    for path in suites:
        name = os.path.basename(path)
        started = time.time()
        proc = subprocess.run([sys.executable, path] + extra)
        results.append((name, proc.returncode, time.time() - started))
    total = time.time() - t0

    print("\n" + "=" * 56)
    print(f"{'SUITE':<22} {'RESULT':<8} {'TIME':>8}")
    print("-" * 56)
    for name, code, dt in results:
        print(f"{name:<22} {'ok' if code == 0 else 'FAIL':<8} {dt:>7.1f}s")
    print("-" * 56)
    mode = "FULL" if FULL else "FAST"
    print(f"{'TOTAL':<22} {'':<8} {total:>7.1f}s   [{mode} mode]")

    failed = [n for n, c, _ in results if c != 0]
    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
    if not FULL and total > FAST_BUDGET_S:
        print(f"\n⚠  FAST tier took {total:.1f}s > {FAST_BUDGET_S:.0f}s budget — "
              "move a slow test into its suite's SLOW list.")
    if not failed and (FULL or total <= FAST_BUDGET_S):
        print("\nAll good.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
