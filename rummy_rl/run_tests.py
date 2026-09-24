"""Run every test suite: python run_tests.py [--full]

--full adds the slow checks (large scorer parity runs and 50,000-hand bot stress runs).
"""

import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

failed = []
started = time.time()
for path in sorted(glob.glob(os.path.join(HERE, "tests", "test_*.py"))):
    name = os.path.basename(path)
    proc = subprocess.run([sys.executable, path] + sys.argv[1:], capture_output=True, text=True)
    last = (proc.stdout.strip().splitlines() or ["(no output)"])[-1]
    print(f"{'ok  ' if proc.returncode == 0 else 'FAIL'}  {name}: {last}")
    if proc.returncode != 0:
        failed.append(name)
        print(proc.stdout[-3000:] + proc.stderr[-3000:])
print(f"\n{'All suites passed' if not failed else 'Failed: ' + ', '.join(failed)} in {time.time() - started:.1f}s.")
sys.exit(1 if failed else 0)
