# mojo-hdbscan

`mojo-hdbscan` is a standalone Mojo port of the compute-heavy mutual-reachability
and condensed-tree stages from the Python
[`hdbscan`](https://github.com/scikit-learn-contrib/hdbscan) package. It provides
a small NumPy-facing Python API backed by a single compiled Mojo shared library.
This release targets 64-bit Linux and is a source distribution: a working Mojo
toolchain is required to build the shared library.

The covered functions mirror the upstream names and signatures:

- `mutual_reachability` and `sparse_mutual_reachability`
- `condense_tree` and `compute_stability`
- `get_clusters`, including EOM and leaf selection, epsilon selection,
  single-cluster selection, maximum cluster size, and reference matching
- `labelling_at_cut`
- `outlier_scores`

Single-linkage/MST construction, prediction data, branch detection, plotting,
validity indices, and the full `HDBSCAN` estimator are not included. Supply a
SciPy-format single-linkage hierarchy when using the condensed-tree API. The
package is named `mojo_hdbscan` so it can coexist with upstream `hdbscan` for
parity testing.

## Install

The checked-in Pixi environment pins the tested Mojo nightly and installs
NumPy, SciPy, pytest, and upstream `hdbscan`.

```bash
pixi install
pixi run build
pixi run test
```

The build task produces `dist/libmojo-hdbscan.so`.

To expose the package outside Pixi while working from this checkout:

```bash
pixi run python -m pip install -e . --no-deps
```

## Usage

```python
import numpy as np
from scipy.cluster.hierarchy import linkage

import mojo_hdbscan as hdbscan

points = np.array([
    [-2.1, -1.0], [-1.9, -1.2], [-2.0, -0.8],
    [ 2.0,  1.1], [ 2.2,  0.9], [ 1.8,  1.0],
])

hierarchy = linkage(points, method="single")
tree = hdbscan.condense_tree(hierarchy, min_cluster_size=2)
stability = hdbscan.compute_stability(tree)
labels, probabilities, cluster_stabilities = hdbscan.get_clusters(
    tree, stability
)

print(labels)
print(probabilities)
```

The same program is checked in and runs from the repository with
`pixi run python examples/basic.py`.

## Benchmarks

These are median wall-clock times over seven measured runs after two warmups.
Inputs and output allocation are included for both implementations; dense
mutual reachability also includes the required fresh matrix copy because both
APIs operate in place. Run only through `pixi run bench`, which takes the
machine-wide benchmark lock.

Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz; Linux 6.8.0-136-generic;
Python 3.13.14.

| Operation | Input | Upstream ms | Mojo ms | Speedup |
|---|---:|---:|---:|---:|
| mutual_reachability | n=1,200 | 36.469 | 19.971 | 1.83x |
| condense_tree | n=32,768 | 89.478 | 2.953 | 30.30x |
| compute_stability | rows=36,862 | 67.564 | 3.800 | 17.78x |
| get_clusters (EOM) | rows=36,862 | 184.934 | 18.080 | 10.23x |
| labelling_at_cut | n=32,768 | 195.544 | 5.243 | 37.29x |

The large tree-kernel gains come from caller-owned contiguous scratch buffers
and avoiding temporary Python containers. Default EOM selection is a linear
compiled pass, and point probabilities are produced alongside compiled labels;
leaf and epsilon selection retain the reference-style Python policy path.
Dense mutual reachability uses private quickselect scratch per worker above an
`n² >= 262,144` threshold and SIMD for contiguous result rows, including a
scalar remainder. Results vary by hierarchy shape and machine.

No GPU path is included; all compiled kernels run on the CPU.

## How it works

`src/hdbscan.mojo` is one compilation unit exported through a C ABI. Python
passes NumPy buffer addresses as 64-bit integers through `ctypes`; Mojo rebuilds
typed pointers with mutable origins. All allocation stays on the Python side,
so the shared library owns no cross-language memory.

Dense distance and linkage matrices are C-contiguous row-major `float64`.
Condensed structured arrays are split into contiguous `int64` parent, child,
and size buffers plus a `float64` lambda buffer at the FFI boundary. The dense
mutual-reachability kernel uses in-place quickselect for each core distance,
then applies
`max(core_distance[i], core_distance[j], distance[i, j] / alpha)`. Tree
condensation uses preallocated breadth-first queues, while union-find kernels
handle cut labelling and final point-to-cluster assignment.

The Python boundary validates shapes, linkage topology, node ranges, scalar
parameters, and structured-field dtypes before exposing an address to Mojo.
It materializes aligned C-contiguous buffers where needed and retains every
NumPy owner until the synchronous native call returns. The exported C ABI is an
internal unsafe interface; callers should use the Python API rather than call
the shared-library symbols directly.

The test suite compares every covered operation against the installed upstream
Cython implementation, including exact condensed-tree rows and behavioral
parity for EOM/leaf selection edge options.

## License

This project is MIT licensed. The upstream `hdbscan` project is BSD-3-Clause
licensed and is used as the behavioral reference in tests and benchmarks.
