"""Reproducible timings against upstream hdbscan."""

from __future__ import annotations

import os
import platform
import statistics
import sys
import time

import numpy as np
from hdbscan import _hdbscan_reachability as upstream_reachability
from hdbscan import _hdbscan_tree as upstream_tree
from scipy.spatial.distance import pdist, squareform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojo_hdbscan as mojo


def median_time(function, repeats=7, warmups=2):
    for _ in range(warmups):
        function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def balanced_linkage(n):
    active = [(i, 1) for i in range(n)]
    rows = []
    next_node = n
    distance = 0.01
    while len(active) > 1:
        merged = []
        for index in range(0, len(active), 2):
            left, left_size = active[index]
            right, right_size = active[index + 1]
            size = left_size + right_size
            rows.append((left, right, distance, size))
            merged.append((next_node, size))
            next_node += 1
        active = merged
        distance += 0.01
    return np.asarray(rows, dtype=np.float64)


def machine():
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            cpu = next(
                line.split(":", 1)[1].strip()
                for line in handle
                if line.startswith("model name")
            )
    except (OSError, StopIteration):
        pass
    return f"{cpu}; {platform.system()} {platform.release()}; Python {platform.python_version()}"


def main():
    rng = np.random.default_rng(123)
    distances = squareform(pdist(rng.normal(size=(1200, 8))))
    hierarchy = balanced_linkage(32768)
    reference_tree = upstream_tree.condense_tree(hierarchy, 16)
    mojo_tree = mojo.condense_tree(hierarchy, 16)
    reference_stability = upstream_tree.compute_stability(reference_tree)
    mojo_stability = mojo.compute_stability(mojo_tree)

    cases = [
        (
            "mutual_reachability",
            "n=1,200",
            lambda: upstream_reachability.mutual_reachability(
                distances.copy(), 15, 1.0
            ),
            lambda: mojo.mutual_reachability(distances.copy(), 15, 1.0),
        ),
        (
            "condense_tree",
            "n=32,768",
            lambda: upstream_tree.condense_tree(hierarchy, 16),
            lambda: mojo.condense_tree(hierarchy, 16),
        ),
        (
            "compute_stability",
            f"rows={len(reference_tree):,}",
            lambda: upstream_tree.compute_stability(reference_tree),
            lambda: mojo.compute_stability(mojo_tree),
        ),
        (
            "get_clusters (EOM)",
            f"rows={len(reference_tree):,}",
            lambda: upstream_tree.get_clusters(
                reference_tree, reference_stability.copy()
            ),
            lambda: mojo.get_clusters(mojo_tree, mojo_stability.copy()),
        ),
        (
            "labelling_at_cut",
            "n=32,768",
            lambda: upstream_tree.labelling_at_cut(hierarchy, 0.12, 16),
            lambda: mojo.labelling_at_cut(hierarchy, 0.12, 16),
        ),
    ]

    print(f"Machine: {machine()}")
    print()
    print("| Operation | Input | Upstream ms | Mojo ms | Speedup |")
    print("|---|---:|---:|---:|---:|")
    for name, size, upstream_fn, mojo_fn in cases:
        upstream_seconds = median_time(upstream_fn)
        mojo_seconds = median_time(mojo_fn)
        print(
            f"| {name} | {size} | {upstream_seconds * 1000:.3f} | "
            f"{mojo_seconds * 1000:.3f} | {upstream_seconds / mojo_seconds:.2f}x |"
        )


if __name__ == "__main__":
    main()
