"""Build the C++ extension: `python native/build.py` (from `othello/`).

One compiler invocation, no CMake, no setuptools — the extension is a single
translation unit that links nothing but pybind11 (header-only). The PyTorch
network stays in Python and is reached through a callback, so this build needs
no LibTorch and no CUDA toolkit; it is a plain CPU .so and works identically on
a Mac and on a Kaggle GPU session.

On Kaggle, run this once per session before training (it takes ~20-40s):

    !cd /kaggle/working/rl-for-games/othello && python native/build.py

Nothing depends on it: if the extension is missing the code falls back to the
NumPy path automatically (see engine/native.py), so a failed build degrades
throughput instead of killing a run.

FLAGS THAT MATTER:
  -ffp-contract=off   forbids the compiler from fusing `a*b + c` into an FMA.
                      An FMA rounds once instead of twice, which would break
                      bit-for-bit parity with NumPy's float32 arithmetic — the
                      property tests/test_native.py and tests/test_batched.py
                      both rely on. Do not remove this.
  -O3                 the whole point. Auto-vectorisation is safe here: without
                      -ffast-math the compiler may not reorder float additions.
  (no -march=native)  deliberately portable — the .so is built on the machine
                      that runs it, and a fixed baseline keeps codegen (and so
                      float behaviour) the same across a Mac and Kaggle.
"""

import os
import subprocess
import sys
import sysconfig

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "othello_native.cpp")
MODULE = "othello_native"


def _pybind11_include():
    """pybind11's headers, installing the (pure-header, ~1MB) package if needed."""
    try:
        import pybind11
    except ImportError:
        print("pybind11 not found — installing it (header-only, no build step)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pybind11"])
        import pybind11
    return pybind11.get_include()


def _compiler():
    """The C++ compiler Python itself was built with, else the system default."""
    cxx = os.environ.get("CXX") or sysconfig.get_config_var("CXX") or "c++"
    return cxx.split()


def output_path():
    """Where the built module lands (e.g. othello_native.cpython-311-darwin.so)."""
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    return os.path.join(HERE, MODULE + suffix)


def build(verbose=True):
    out = output_path()
    cmd = _compiler() + [
        "-O3",
        "-std=c++17",
        "-fPIC",
        "-fvisibility=hidden",
        "-ffp-contract=off",          # see the module docstring — parity-critical
        "-I" + _pybind11_include(),
        "-I" + sysconfig.get_paths()["include"],
        SOURCE,
        "-o", out,
    ]
    # macOS builds extensions as bundles and resolves CPython symbols at load time;
    # Linux uses a plain shared object.
    cmd += ["-bundle", "-undefined", "dynamic_lookup"] if sys.platform == "darwin" else ["-shared"]

    if verbose:
        print("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stderr.strip():
        print(proc.stderr.strip())
    if proc.returncode != 0:
        print(f"\nBUILD FAILED (exit {proc.returncode}). "
              "Training still works — it falls back to the NumPy path.")
        return None
    if verbose:
        print(f"\nbuilt {out}")
    return out


def main():
    out = build()
    if out is None:
        sys.exit(1)
    # Import it through the normal loader so a successful build is also a
    # successful load (catches ABI / suffix / architecture mismatches now, not
    # in the middle of a training run).
    sys.path.insert(0, os.path.join(HERE, "..", "engine"))
    import native
    native.reload()
    print(native.status())
    sys.exit(0 if native.available() else 1)


if __name__ == "__main__":
    main()
