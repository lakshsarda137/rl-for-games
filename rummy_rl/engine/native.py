"""Loads the C++ engine (native/rummy_native.cpp), building it first if needed.

    from native import rummy_native as rn

It rebuilds automatically when the .cpp is newer than the built module, so an
edited engine is never silently stale.
"""

import importlib
import os
import sys

_NATIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "native")
sys.path.insert(0, _NATIVE_DIR)

import build as _build  # noqa: E402  (native/build.py)


def _needs_build():
    out = _build.output_path()
    return not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(_build.SOURCE)


if _needs_build():
    print("Building the C++ rummy engine (one time, about 10 seconds)...", file=sys.stderr)
    _build.build(verbose=False)

rummy_native = importlib.import_module("rummy_native")
