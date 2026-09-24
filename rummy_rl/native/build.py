"""Build the C++ engine: `python native/build.py` (from `rummy_rl/`).

One compiler call, no CMake: the engine is a single file that needs only
pybind11 (header-only). Same recipe as othello/native/build.py, so it builds the
same way on a Mac and on Kaggle. engine/native.py runs this automatically the
first time the module is imported, so you rarely need to run it by hand.
"""

import os
import subprocess
import sys
import sysconfig

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "rummy_native.cpp")
MODULE = "rummy_native"


def _pybind11_include():
    try:
        import pybind11
    except ImportError:
        print("pybind11 not found, installing it (header-only)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pybind11"])
        import pybind11
    return pybind11.get_include()


def output_path():
    return os.path.join(HERE, MODULE + (sysconfig.get_config_var("EXT_SUFFIX") or ".so"))


def build(verbose=True):
    out = output_path()
    cxx = (os.environ.get("CXX") or sysconfig.get_config_var("CXX") or "c++").split()
    cmd = cxx + [
        "-O3", "-std=c++17", "-fPIC", "-fvisibility=hidden",
        "-Wall", "-Wextra",
        "-I" + _pybind11_include(),
        "-I" + sysconfig.get_paths()["include"],
        SOURCE, "-o", out,
    ]
    if sys.platform == "darwin":
        # Point clang at the SDK and its C++ headers explicitly: on a macOS newer
        # than the installed Command Line Tools, clang looks for <cstddef> etc. in
        # a toolchain folder that no longer exists and never checks the SDK copy.
        sdk = subprocess.run(["xcrun", "--show-sdk-path"], capture_output=True, text=True).stdout.strip()
        if sdk:
            cmd += ["-isysroot", sdk]
            libcxx = os.path.join(sdk, "usr", "include", "c++", "v1")
            if os.path.isdir(libcxx):
                cmd += ["-isystem", libcxx]
        cmd += ["-bundle", "-undefined", "dynamic_lookup"]
    else:
        cmd += ["-shared"]
    if verbose:
        print("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stderr.strip():
        print(proc.stderr.strip())
    if proc.returncode != 0:
        raise RuntimeError(f"C++ build failed (exit {proc.returncode}); see the compiler output above")
    if verbose:
        print(f"built {out}")
    return out


if __name__ == "__main__":
    build()
