"""Loader + on/off switch for the compiled C++ extension (`native/`).

The extension is OPTIONAL. If it isn't built (or fails to load, or is switched
off) every caller silently falls back to the NumPy implementations, so a run
never depends on it — it just runs slower. Build it with:

    python native/build.py

Two ways to force the NumPy path, for A/B timing and for bisecting a suspected
parity bug:

    OTHELLO_NATIVE=0 python run/train_loop.py ...     # env var, read at import
    python run/train_loop.py --no-native ...          # CLI, flips it at runtime

`enabled()` is checked per call rather than bound once at import, which is what
makes the runtime switch (and the both-paths parity tests) possible; the check
is a module-global read against work measured in tens of microseconds.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_NATIVE_DIR = os.path.abspath(os.path.join(_HERE, "..", "native"))

# Bump in lockstep with `__abi__` in othello_native.cpp whenever a signature
# changes, so a stale .so is refused up front instead of crashing mid-search.
REQUIRED_ABI = 1

NATIVE = None      # the compiled module, or None
STATUS = ""        # one-line explanation of the line above
_ENABLED = os.environ.get("OTHELLO_NATIVE", "1") != "0"


def reload():
    """(Re-)attempt the import. Returns True if the extension is usable."""
    global NATIVE, STATUS
    NATIVE = None
    if _NATIVE_DIR not in sys.path:
        sys.path.insert(0, _NATIVE_DIR)
    sys.modules.pop("othello_native", None)
    try:
        import othello_native as mod
    except ImportError as exc:
        STATUS = f"not built ({exc.__class__.__name__}: {exc}) — run `python native/build.py`"
        return False
    abi = getattr(mod, "__abi__", None)
    if abi != REQUIRED_ABI:
        STATUS = (f"stale build (ABI {abi}, need {REQUIRED_ABI}) — "
                  "rerun `python native/build.py`")
        return False
    NATIVE = mod
    STATUS = f"othello_native {getattr(mod, '__version__', '?')} ({mod.__file__})"
    return True


reload()


def available():
    """True if the extension loaded, regardless of whether it's switched on."""
    return NATIVE is not None


def enabled():
    """True if the C++ path should actually be used for this call."""
    return NATIVE is not None and _ENABLED


def set_enabled(flag):
    """Turn the C++ path on/off at runtime (`--no-native`). No effect if unbuilt."""
    global _ENABLED
    _ENABLED = bool(flag)


def status():
    """One line for startup logs: which search/engine implementation is live."""
    if NATIVE is None:
        return f"engine: NumPy (C++ {STATUS})"
    if not _ENABLED:
        return f"engine: NumPy (C++ available but disabled) — {STATUS}"
    return f"engine: C++ native — {STATUS}"
