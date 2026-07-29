"""Condensed-tree construction and flat cluster extraction."""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, i64, lib

CONDENSED_DTYPE = np.dtype(
    [
        ("parent", np.intp),
        ("child", np.intp),
        ("lambda_val", np.float64),
        ("child_size", np.intp),
    ]
)


def _fields(tree):
    array = np.asarray(tree)
    required = {"parent", "child", "lambda_val", "child_size"}
    if array.dtype.names is None or not required.issubset(array.dtype.names):
        raise ValueError("tree must have parent, child, lambda_val, and child_size fields")
    if array.ndim != 1:
        raise ValueError("tree must be a one-dimensional structured array")
    for name in ("parent", "child", "child_size"):
        values = array[name]
        if values.dtype.kind not in "iu":
            raise TypeError(f"tree field {name!r} must have an integer dtype")
        if values.dtype.kind == "u" and values.size and int(values.max()) > np.iinfo(np.int64).max:
            raise OverflowError(f"tree field {name!r} does not fit in int64")
    if array["lambda_val"].dtype.kind not in "fiu":
        raise TypeError("tree field 'lambda_val' must have a numeric dtype")
    return (
        i64(array["parent"]),
        i64(array["child"]),
        f64(array["lambda_val"]),
        i64(array["child_size"]),
    )


def _positive_integer(value, name):
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if converted != value or converted < 1:
        raise ValueError(f"{name} must be a positive integer")
    return converted


def _linkage(value, name):
    array = f64(value)
    if array.ndim != 2 or array.shape[1] != 4 or array.shape[0] == 0:
        raise ValueError(f"{name} must have shape (n_samples - 1, 4)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    rows = array.shape[0]
    n = rows + 1
    children = array[:, :2]
    if np.any(children != np.floor(children)):
        raise ValueError(f"{name} child ids must be integers")
    if np.any(children < 0) or np.any(children >= (n + np.arange(rows))[:, None]):
        raise ValueError(f"{name} contains an out-of-range child id")
    child_ids = children.astype(np.int64, copy=False)
    if np.any(child_ids[:, 0] == child_ids[:, 1]):
        raise ValueError(f"{name} is not a valid single-linkage tree")
    uses = np.bincount(child_ids.ravel(), minlength=2 * n - 1)
    if not np.all(uses[:-1] == 1) or uses[-1] != 0:
        raise ValueError(f"{name} is not a valid single-linkage tree")
    counts = array[:, 3]
    if (
        np.any(counts != np.floor(counts))
        or np.any(counts < 2)
        or np.any(counts > n)
        or counts[-1] != n
    ):
        raise ValueError(f"{name} contains an invalid cluster size")
    return array


def _validate_condensed(parents, children, lambdas, sizes):
    if parents.size == 0:
        raise ValueError("condensed_tree must not be empty")
    if np.any(parents < 0) or np.any(children < 0):
        raise ValueError("condensed_tree node ids must be non-negative")
    if np.any(sizes < 1):
        raise ValueError("condensed_tree child sizes must be positive")
    if np.isnan(lambdas).any() or np.any(lambdas < 0):
        raise ValueError("condensed_tree lambda values must be non-negative and not NaN")
    root = int(parents.min())
    if root < 1 or np.any(children == parents):
        raise ValueError("condensed_tree has invalid node relationships")
    max_node = max(int(parents.max()), int(children.max()))
    if max_node > 100_000_000:
        raise ValueError("condensed_tree node ids are unreasonably large")


def condense_tree(hierarchy, min_cluster_size=10):
    """Condense a SciPy-format single-linkage hierarchy."""
    hierarchy = _linkage(hierarchy, "hierarchy")
    min_cluster_size = _positive_integer(min_cluster_size, "min_cluster_size")
    rows = hierarchy.shape[0]
    n = rows + 1
    total_nodes = 2 * rows + 1
    capacity = 2 * n
    node_queue = np.empty(total_nodes, dtype=np.int64)
    subtree_queue = np.empty(total_nodes, dtype=np.int64)
    relabel = np.empty(total_nodes, dtype=np.int64)
    ignore = np.empty(total_nodes, dtype=np.uint8)
    parents = np.empty(capacity, dtype=np.int64)
    children = np.empty(capacity, dtype=np.int64)
    lambdas = np.empty(capacity, dtype=np.float64)
    sizes = np.empty(capacity, dtype=np.int64)
    count = lib().mhdb_condense_tree(
        addr(hierarchy),
        rows,
        min_cluster_size,
        addr(node_queue),
        addr(subtree_queue),
        addr(relabel),
        addr(ignore),
        addr(parents),
        addr(children),
        addr(lambdas),
        addr(sizes),
    )
    result = np.empty(count, dtype=CONDENSED_DTYPE)
    result["parent"] = parents[:count]
    result["child"] = children[:count]
    result["lambda_val"] = lambdas[:count]
    result["child_size"] = sizes[:count]
    return result


def compute_stability(condensed_tree):
    """Return excess-of-mass stability keyed by condensed cluster id."""
    parents, children, lambdas, sizes = _fields(condensed_tree)
    _validate_condensed(parents, children, lambdas, sizes)
    smallest_cluster = int(parents.min())
    largest_child = max(int(children.max()), smallest_cluster)
    largest_parent = int(parents.max())
    num_clusters = largest_parent - smallest_cluster + 1
    births = np.empty(largest_child + 1, dtype=np.float64)
    stability = np.empty(num_clusters, dtype=np.float64)
    lib().mhdb_compute_stability(
        addr(parents),
        addr(children),
        addr(lambdas),
        addr(sizes),
        parents.size,
        addr(births),
        addr(stability),
        smallest_cluster,
        largest_child,
        num_clusters,
    )
    return {
        float(cluster): float(stability[cluster - smallest_cluster])
        for cluster in range(smallest_cluster, largest_parent + 1)
    }


def _parent_children(cluster_tree):
    result = {}
    for parent, child in zip(cluster_tree["parent"], cluster_tree["child"]):
        result.setdefault(int(parent), []).append(int(child))
    return result


def _bfs(parent_to_children, root):
    result = []
    pending = [root]
    while pending:
        result.extend(pending)
        pending = [
            child
            for node in pending
            for child in parent_to_children.get(node, ())
        ]
    return result


def _leaves(parent_to_children, root):
    result = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node in parent_to_children:
            stack.extend(parent_to_children[node])
        else:
            result.append(node)
    return result


def _epsilon_search(
    leaves,
    child_lookup,
    parent_to_children,
    epsilon,
    allow_single_cluster,
    root,
):
    selected = []
    processed = set()
    for leaf in leaves:
        leaf_lambda = child_lookup[leaf][1]
        leaf_epsilon = np.inf if leaf_lambda == 0.0 else 1.0 / leaf_lambda
        if leaf_epsilon < epsilon:
            if leaf in processed:
                continue
            node = leaf
            parent = child_lookup[node][0]
            if parent == root:
                node = parent if allow_single_cluster else node
            else:
                while parent != root:
                    parent_lambda = child_lookup[parent][1]
                    parent_epsilon = (
                        np.inf if parent_lambda == 0.0 else 1.0 / parent_lambda
                    )
                    if parent_epsilon > epsilon:
                        node = parent
                        break
                    node = parent
                    parent = child_lookup[node][0]
                else:
                    node = parent if allow_single_cluster else node
            selected.append(node)
            processed.update(_bfs(parent_to_children, node)[1:])
        else:
            selected.append(leaf)
    return set(selected)


def _max_lambdas(parents, lambdas):
    deaths = np.zeros(int(parents.max()) + 1, dtype=np.float64)
    np.maximum.at(deaths, parents, lambdas)
    return deaths


def _compiled_eom_select(
    parents,
    children,
    lambdas,
    sizes,
    stability,
    max_cluster_size,
    cluster_selection_epsilon_max,
):
    root = int(parents.min())
    largest_cluster = int(parents.max())
    num_clusters = largest_cluster - root + 1
    keys = np.fromiter(stability, dtype=np.int64, count=len(stability))
    if (
        len(stability) != num_clusters
        or not np.array_equal(np.sort(keys), np.arange(root, largest_cluster + 1))
    ):
        return None

    stability_values = np.empty(num_clusters, dtype=np.float64)
    stability_values[keys - root] = np.fromiter(
        stability.values(), dtype=np.float64, count=len(stability)
    )
    original_stability = stability_values.copy()
    max_node = max(largest_cluster, int(children.max()))
    selected = np.zeros(max_node + 1, dtype=np.uint8)
    cluster_parent = np.empty(num_clusters, dtype=np.int64)
    cluster_size = np.empty(num_clusters, dtype=np.int64)
    node_epsilon = np.empty(num_clusters, dtype=np.float64)
    subtree_stability = np.empty(num_clusters, dtype=np.float64)
    blocked = np.empty(num_clusters, dtype=np.uint8)
    lib().mhdb_eom_select(
        addr(parents),
        addr(children),
        addr(lambdas),
        addr(sizes),
        parents.size,
        addr(stability_values),
        root,
        num_clusters,
        int(max_cluster_size),
        float(cluster_selection_epsilon_max),
        addr(selected),
        addr(cluster_parent),
        addr(cluster_size),
        addr(node_epsilon),
        addr(subtree_stability),
        addr(blocked),
    )
    for offset in np.flatnonzero(stability_values != original_stability):
        value = stability_values[offset]
        stability[root + offset] = float(value)
    return set(np.flatnonzero(selected[root:]) + root)


def _labels(
    parents,
    children,
    lambdas,
    clusters,
    allow_single_cluster,
    cluster_selection_epsilon,
    match_reference_implementation,
    deaths,
):
    root = int(parents.min())
    max_node = max(int(parents.max()), int(children.max()))
    selected = np.zeros(max_node + 1, dtype=np.uint8)
    cluster_labels = np.full(max_node + 1, -1, dtype=np.int64)
    selected[clusters] = 1
    cluster_labels[clusters] = np.arange(clusters.size, dtype=np.int64)
    parent = np.empty(max_node + 1, dtype=np.int64)
    rank = np.empty(max_node + 1, dtype=np.int64)
    child_lambda = np.empty(max_node + 1, dtype=np.float64)
    parent_max_lambda = np.empty(max_node + 1, dtype=np.float64)
    result = np.empty(root, dtype=np.int64)
    probabilities = np.empty(root, dtype=np.float64)
    reverse_clusters = np.empty(max(1, clusters.size), dtype=np.int64)
    reverse_clusters[: clusters.size] = clusters
    lib().mhdb_label_condensed_tree(
        addr(parents),
        addr(children),
        addr(lambdas),
        parents.size,
        root,
        max_node,
        addr(selected),
        addr(cluster_labels),
        addr(parent),
        addr(rank),
        addr(child_lambda),
        addr(parent_max_lambda),
        addr(result),
        addr(deaths),
        addr(reverse_clusters),
        addr(probabilities),
        clusters.size,
        int(allow_single_cluster),
        float(cluster_selection_epsilon),
        int(match_reference_implementation),
    )
    return result, probabilities


def get_clusters(
    tree,
    stability,
    cluster_selection_method="eom",
    allow_single_cluster=False,
    match_reference_implementation=False,
    cluster_selection_epsilon=0.0,
    max_cluster_size=0,
    cluster_selection_epsilon_max=float("inf"),
):
    """Extract flat labels, probabilities, and normalized stabilities."""
    parents, children, lambdas, sizes = _fields(tree)
    _validate_condensed(parents, children, lambdas, sizes)
    if not np.any(sizes == 1):
        raise ValueError("condensed_tree must contain point rows")
    if not hasattr(stability, "items"):
        raise TypeError("stability must be a mapping")
    num_points = int(children[sizes == 1].max()) + 1
    max_lambda = float(lambdas.max())
    deaths = _max_lambdas(parents, lambdas)
    if max_cluster_size <= 0:
        max_cluster_size = num_points + 1
    clusters = None
    if (
        cluster_selection_method == "eom"
        and not allow_single_cluster
        and cluster_selection_epsilon == 0.0
    ):
        clusters = _compiled_eom_select(
            parents,
            children,
            lambdas,
            sizes,
            stability,
            max_cluster_size,
            cluster_selection_epsilon_max,
        )

    if clusters is None:
        node_list = sorted(stability, reverse=True)
        if not allow_single_cluster:
            node_list = node_list[:-1]
        node_list = [int(node) for node in node_list]
        is_cluster = {node: True for node in node_list}
        cluster_mask = sizes > 1
        cluster_parents = parents[cluster_mask]
        cluster_children = children[cluster_mask]
        cluster_lambdas = lambdas[cluster_mask]
        cluster_sizes_array = sizes[cluster_mask]
        cluster_sizes = {
            int(child): int(size)
            for child, size in zip(cluster_children, cluster_sizes_array)
        }
        node_eps = {
            int(child): 1.0 / value
            for child, value in zip(cluster_children, cluster_lambdas)
        }
        cluster_tree = np.empty(cluster_children.size, dtype=CONDENSED_DTYPE)
        cluster_tree["parent"] = cluster_parents
        cluster_tree["child"] = cluster_children
        cluster_tree["lambda_val"] = cluster_lambdas
        cluster_tree["child_size"] = cluster_sizes_array
        parent_to_children = _parent_children(cluster_tree)
        child_lookup = {
            int(child): (int(parent), float(value))
            for parent, child, value in zip(
                cluster_parents, cluster_children, cluster_lambdas
            )
        }
        root = int(cluster_parents.min()) if cluster_parents.size else 0

        if allow_single_cluster:
            root_node = node_list[-1]
            cluster_sizes[root_node] = sum(
                cluster_sizes.get(child, 0)
                for child in parent_to_children.get(root_node, ())
            )
            with np.errstate(divide="ignore"):
                node_eps[root_node] = float(np.max(1.0 / lambdas))

        if cluster_selection_method == "eom":
            for node in node_list:
                subtree_stability = sum(
                    stability[child] for child in parent_to_children.get(node, ())
                )
                if (
                    subtree_stability > stability[node]
                    or cluster_sizes[node] > max_cluster_size
                    or node_eps[node] > cluster_selection_epsilon_max
                ):
                    is_cluster[node] = False
                    stability[node] = subtree_stability
                else:
                    for descendant in _bfs(parent_to_children, node)[1:]:
                        is_cluster[descendant] = False
            if cluster_selection_epsilon != 0.0 and cluster_children.size:
                eom_clusters = [node for node, keep in is_cluster.items() if keep]
                if len(eom_clusters) == 1 and eom_clusters[0] == root:
                    selected_clusters = eom_clusters if allow_single_cluster else []
                else:
                    selected_clusters = _epsilon_search(
                        set(eom_clusters),
                        child_lookup,
                        parent_to_children,
                        cluster_selection_epsilon,
                        allow_single_cluster,
                        root,
                    )
                selected_clusters = set(selected_clusters)
                for node in is_cluster:
                    is_cluster[node] = node in selected_clusters
        elif cluster_selection_method == "leaf":
            leaves = (
                set(_leaves(parent_to_children, root))
                if cluster_children.size
                else set()
            )
            if not leaves:
                for node in is_cluster:
                    is_cluster[node] = False
                is_cluster[int(parents.min())] = True
            selected_clusters = (
                _epsilon_search(
                    leaves,
                    child_lookup,
                    parent_to_children,
                    cluster_selection_epsilon,
                    allow_single_cluster,
                    root,
                )
                if cluster_selection_epsilon != 0.0
                else leaves
            )
            for node in is_cluster:
                is_cluster[node] = node in selected_clusters
        else:
            raise ValueError(
                f'Invalid Cluster Selection Method: {cluster_selection_method}\n'
                'Should be one of: "eom", "leaf"\n'
            )
        clusters = {node for node, keep in is_cluster.items() if keep}

    sorted_clusters = np.asarray(sorted(clusters), dtype=np.int64)
    labels, probabilities = _labels(
        parents,
        children,
        lambdas,
        sorted_clusters,
        allow_single_cluster,
        cluster_selection_epsilon,
        match_reference_implementation,
        deaths,
    )

    counts = np.bincount(labels[labels >= 0], minlength=sorted_clusters.size)
    if np.isinf(max_lambda) or max_lambda == 0.0:
        normalized = np.ones(sorted_clusters.size, dtype=np.float64)
    else:
        cluster_stability = np.fromiter(
            (stability[int(cluster)] for cluster in sorted_clusters),
            dtype=np.float64,
            count=sorted_clusters.size,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = cluster_stability / (counts * max_lambda)
        normalized[counts == 0] = 1.0
    return labels, probabilities, normalized


def labelling_at_cut(linkage, cut, min_cluster_size):
    """Label components below a single-linkage distance cut."""
    linkage = _linkage(linkage, "linkage")
    min_cluster_size = _positive_integer(min_cluster_size, "min_cluster_size")
    cut = float(cut)
    if not np.isfinite(cut):
        raise ValueError("cut must be finite")
    rows = linkage.shape[0]
    n = rows + 1
    total_nodes = 2 * rows + 1
    parent = np.empty(total_nodes, dtype=np.int64)
    rank = np.empty(total_nodes, dtype=np.int64)
    sizes = np.empty(total_nodes, dtype=np.int64)
    label_map = np.empty(total_nodes, dtype=np.int64)
    result = np.empty(n, dtype=np.int64)
    lib().mhdb_labelling_at_cut(
        addr(linkage),
        rows,
        cut,
        min_cluster_size,
        addr(parent),
        addr(rank),
        addr(sizes),
        addr(label_map),
        addr(result),
    )
    return result


def outlier_scores(tree):
    """Generate GLOSH outlier scores from a condensed tree."""
    parents, children, lambdas, _ = _fields(tree)
    _validate_condensed(parents, children, lambdas, _)
    deaths = _max_lambdas(parents, lambdas)
    root = int(parents.min())
    result = np.zeros(root, dtype=np.float64)
    for parent, child in zip(parents[::-1], children[::-1]):
        if deaths[child] > deaths[parent]:
            deaths[parent] = deaths[child]
    for parent, point, value in zip(parents, children, lambdas):
        if point >= root:
            continue
        maximum = deaths[parent]
        if maximum == 0.0 or not np.isfinite(value):
            result[point] = 0.0
        else:
            result[point] = (maximum - value) / maximum
    return result
