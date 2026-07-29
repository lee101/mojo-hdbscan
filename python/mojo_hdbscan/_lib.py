"""ctypes access to the compiled Mojo kernels."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "hdbscan.mojo")
LIB = os.environ.get("MOJO_HDBSCAN_LIB") or os.path.join(
    ROOT, "dist", "libmojo-hdbscan.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mhdb_mutual_reachability": ([I, I, I, I, I, I, F], None),
    "mhdb_condense_tree": ([I] * 11, I),
    "mhdb_compute_stability": ([I] * 10, None),
    "mhdb_labelling_at_cut": ([I, I, F, I, I, I, I, I, I], None),
    "mhdb_label_condensed_tree": (
        [I] * 18 + [F, I],
        None,
    ),
    "mhdb_eom_select": ([I] * 9 + [F] + [I] * 6, None),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(SRC):
        return LIB
    mojo = shutil.which("mojo")
    if mojo is None:
        raise BuildError("mojo not found; run `pixi run build` first")
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    proc = subprocess.run(
        [mojo, "build", "--emit", "shared-lib", SRC, "-o", LIB],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0:
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if array.size == 0:
        raise ValueError("cannot pass an empty buffer across the FFI boundary")
    if not array.flags.c_contiguous or not array.flags.aligned:
        raise ValueError("FFI buffers must be aligned and C-contiguous")
    address = int(array.ctypes.data)
    if address == 0:
        raise ValueError("cannot pass a null buffer across the FFI boundary")
    return address


def f64(value) -> np.ndarray:
    return np.require(value, dtype=np.float64, requirements=("C", "A"))


def i64(value) -> np.ndarray:
    return np.require(value, dtype=np.int64, requirements=("C", "A"))
