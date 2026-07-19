# Edax setup (optional external opponent)

`opponents/edax.py` drives the [Edax](https://github.com/abulmo/edax-reversi)
engine as a subprocess. Edax itself (binary + weights) lives in
`othello/third_party/edax/`, which is **gitignored** — so it is not committed and
must be built once per machine. These are the exact steps used on this repo
(Apple Silicon / arm64 macOS); adjust `ARCH` for other CPUs (`make -C src help`).

```sh
# 1. Clone and build the binary (from source, ~10s)
git clone --depth 1 https://github.com/abulmo/edax-reversi.git
mkdir -p edax-reversi/bin
make -C edax-reversi/src build ARCH=armv8-a COMP=clang OS=osx
#   -> produces edax-reversi/bin/mEdax-armv8-a  (Mach-O arm64)
#   For Intel macs use ARCH=x86-64-v3 ; see `make -C src help` for options.

# 2. Get the evaluation weights (NOT in the source repo; from a release .7z)
#    Needs a 7z extractor:  brew install sevenzip
curl -L -o eval.7z https://github.com/abulmo/edax-reversi/releases/download/v4.4/eval.7z
7zz x eval.7z          # -> data/eval.dat  (~14 MB); v4.4 weights work with this source

# 3. Install where opponents/edax.py expects it
mkdir -p othello/third_party/edax/data
cp edax-reversi/bin/mEdax-armv8-a          othello/third_party/edax/edax
cp data/eval.dat                            othello/third_party/edax/data/eval.dat
```

Verify:  `python -c "import sys; sys.path[:0]=['othello/engine','othello/opponents']; import edax; print(edax.is_available())"`
should print `True`.

## How it's driven
Each move launches a short-lived `edax` process (~70 ms, mostly weight loading)
and sends:

```
mode 3
setboard <64 chars: X=black O=white -=empty> <side: X|O>
go
```

then reads the `Edax plays <MOVE>` line. Notes learned the hard way:
- **`mode 3` is required** — without it `go` doesn't announce the move.
- **Do not send `quit`** — Edax block-buffers stdout on a pipe and a `quit` exit
  drops the buffer; closing stdin (EOF) flushes and exits cleanly instead.
- Edax's square index equals our move index (`(rank-1)*8+file == row*8+col`), so
  the mapping is direct.

## Use
```sh
python run/play_cli.py --black edax:6 --white minimax:6 --delay 0.5   # watch
python run/play_cli.py --black human --white edax:2                    # you vs Edax
```
`level` (0–60, default 21) is the strength dial. Even low levels are very strong.
