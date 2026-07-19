"""Edax wrapper — drive the strong external Othello engine as a subprocess.

Edax is the strongest open-source Othello engine. We don't reimplement it; we
compile it once and ask it for one move at a time over its text console.

Each move launches a short-lived `edax` process (measured ~70ms, dominated by
loading the 14MB eval weights). That's imperceptible for watch/play and keeps
the wrapper dead simple and robust — Edax block-buffers its stdout on a pipe and
only flushes at its `>` prompt, which makes a long-lived persistent process
fiddly for little gain here. If Edax throughput ever matters (bulk calibration),
switch to a persistent process reading raw fds.

Protocol (from the Edax source):
  * launch:  edax -level N -n T -book-usage off -verbose 0   (cwd must contain
             data/eval.dat).
  * commands: `mode 3` (so we drive every move), `setboard <64 chars> <side>`,
              `go`; then read the `Edax plays <MOVE>` line. `mode 3` is
              REQUIRED — without it `go` doesn't announce the move.

Coordinates line up exactly: Edax's square index and our move index are both
`(rank-1)*8 + file` == `row*8 + col`. We serialise the board row-major with
X=black / O=white / -=empty and parse `D3` back as `(3-1)*8 + 3 = 19`. Serialise
and parse share one convention, so it's self-consistent (validated in tests by
checking every Edax move is legal in our board across random positions).

`level` (0-60, Edax default 21) is the strength dial. Even low levels are very
strong for a human and beat our shallow minimax. Edax stays optional and behind
this module — nothing in the core loop imports it.
"""

import glob
import os
import subprocess
import sys

_HERE = os.path.dirname(__file__)
_ENGINE = os.path.join(_HERE, "..", "engine")
if _ENGINE not in sys.path:
    sys.path.insert(0, _ENGINE)

from board_numpy import BLACK, EMPTY, PASS, WHITE, legal_moves

DEFAULT_EDAX_DIR = os.path.normpath(os.path.join(_HERE, "..", "third_party", "edax"))
_GLYPH = {BLACK: "X", WHITE: "O", EMPTY: "-"}
_FILES = "ABCDEFGH"


class EdaxNotInstalled(RuntimeError):
    """Raised when the Edax binary or its eval weights can't be found."""


def _find_binary(edax_dir):
    for name in ("edax", "mEdax", "lEdax", "wEdax.exe"):
        cand = os.path.join(edax_dir, name)
        if os.path.isfile(cand):
            return cand
    hits = [h for h in sorted(glob.glob(os.path.join(edax_dir, "*[Ee]dax*")))
            if os.path.isfile(h) and os.access(h, os.X_OK)]
    return hits[0] if hits else None


def is_available(edax_dir=None):
    """True if a usable Edax install (binary + eval.dat) is present."""
    d = edax_dir or DEFAULT_EDAX_DIR
    return _find_binary(d) is not None and os.path.isfile(os.path.join(d, "data", "eval.dat"))


def _parse_move(token):
    """Algebraic Edax move (e.g. 'D3') -> our move index; anything else -> PASS."""
    token = token.strip().upper()
    if len(token) >= 2 and token[0] in _FILES and token[1] in "12345678":
        return (int(token[1]) - 1) * 8 + _FILES.index(token[0])
    return PASS  # 'PS' / pass / end-of-game marker


class EdaxEngine:
    """Config for talking to Edax; `move()` runs one search per call."""

    def __init__(self, level=6, edax_dir=None, threads=1, timeout=60.0):
        self.edax_dir = edax_dir or DEFAULT_EDAX_DIR
        self.level = level
        self.threads = threads
        self.timeout = timeout

        self.binary = _find_binary(self.edax_dir)
        if self.binary is None:
            raise EdaxNotInstalled(
                f"No Edax binary found in {self.edax_dir!r}. Build it "
                "(make -C src build ARCH=armv8-a COMP=clang OS=osx) and install "
                "the binary + data/eval.dat there.")
        if not os.path.isfile(os.path.join(self.edax_dir, "data", "eval.dat")):
            raise EdaxNotInstalled(
                f"Missing {self.edax_dir}/data/eval.dat (Edax's evaluation weights).")

    def _board_string(self, board, player):
        squares = "".join(_GLYPH[int(board[r, c])] for r in range(8) for c in range(8))
        return f"{squares} {'X' if player == BLACK else 'O'}"

    def move(self, board, player):
        """Best move Edax finds for `player` (our move index, or PASS if stuck)."""
        if not legal_moves(board, player):
            return PASS  # handle passes ourselves; never ask Edax to pass
        # No `quit`: Edax block-buffers stdout on a pipe and a `quit` exit drops
        # the buffer. Letting subprocess close stdin (EOF) flushes and exits it.
        commands = f"mode 3\nsetboard {self._board_string(board, player)}\ngo\n"
        proc = subprocess.run(
            [self.binary, "-level", str(self.level), "-n", str(self.threads),
             "-book-usage", "off", "-verbose", "0"],
            input=commands, capture_output=True, text=True,
            cwd=self.edax_dir, timeout=self.timeout)
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line.startswith("Edax plays"):
                return _parse_move(line.split()[-1])
        raise RuntimeError(
            f"Edax returned no move (exit {proc.returncode}). "
            f"stdout tail: {proc.stdout[-200:]!r}  stderr: {proc.stderr[-200:]!r}")


def edax_player(level=6, edax_dir=None, threads=1):
    """A `(board, player) -> move` function backed by Edax at the given level."""
    engine = EdaxEngine(level=level, edax_dir=edax_dir, threads=threads)
    return engine.move
