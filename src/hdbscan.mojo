"""Mutual-reachability and condensed-tree kernels exposed through a C ABI."""

from std.algorithm import parallelize
from std.sys.info import num_physical_cores, simd_width_of

comptime W = simd_width_of[DType.float64]()
comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime PARALLEL_MATRIX_ELEMENTS = 262144
comptime MAX_WORKERS = 8


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def bp(addr: Int) -> BPtr:
    return BPtr(unsafe_from_address=addr)


def infinity() -> Float64:
    var zero = 0.0
    return 1.0 / zero


def kth_value(values: FPtr, n: Int, kth: Int) -> Float64:
    var lo = 0
    var hi = n - 1
    while lo < hi:
        var pivot = values[(lo + hi) // 2]
        var i = lo
        var j = hi
        while i <= j:
            while values[i] < pivot:
                i += 1
            while values[j] > pivot:
                j -= 1
            if i <= j:
                var tmp = values[i]
                values[i] = values[j]
                values[j] = tmp
                i += 1
                j -= 1
        if kth <= j:
            hi = j
        elif kth >= i:
            lo = i
        else:
            return values[kth]
    return values[lo]


def mutual_reachability(
    matrix: FPtr,
    result: FPtr,
    core: FPtr,
    scratch: FPtr,
    n: Int,
    min_points: Int,
    alpha: Float64,
):
    var kth = min(min_points, n - 1)
    var workers = 1
    if n * n >= PARALLEL_MATRIX_ELEMENTS:
        workers = min(min(num_physical_cores(), MAX_WORKERS), n)

    @parameter
    def find_core(worker: Int):
        var start = worker * n // workers
        var end = (worker + 1) * n // workers
        var worker_scratch = scratch + worker * n
        for j in range(start, end):
            for i in range(n):
                worker_scratch[i] = matrix[i * n + j]
            core[j] = kth_value(worker_scratch, n, kth)

    if workers > 1:
        parallelize[find_core](workers, workers)
    else:
        find_core(0)

    for i in range(n):
        var offset = i * n
        var row_core = SIMD[DType.float64, W](core[i])
        var j = 0
        while j + W <= n:
            var values = matrix.load[width=W](offset + j) / alpha
            values = max(values, row_core)
            values = max(values, core.load[width=W](j))
            result.store(offset + j, values)
            j += W
        while j < n:
            var value = matrix[offset + j] / alpha
            value = max(value, core[i])
            result[offset + j] = max(value, core[j])
            j += 1


def emit_small_subtree(
    hierarchy: FPtr,
    start: Int,
    parent_label: Int,
    lambda_value: Float64,
    n: Int,
    queue: IPtr,
    ignore: BPtr,
    result_parent: IPtr,
    result_child: IPtr,
    result_lambda: FPtr,
    result_size: IPtr,
    pos: Int,
) -> Int:
    var head = 0
    var tail = 1
    queue[0] = Int64(start)
    var result_pos = pos
    while head < tail:
        var node = Int(queue[head])
        head += 1
        if node < n:
            result_parent[result_pos] = Int64(parent_label)
            result_child[result_pos] = Int64(node)
            result_lambda[result_pos] = lambda_value
            result_size[result_pos] = 1
            result_pos += 1
        else:
            var row = (node - n) * 4
            queue[tail] = Int64(hierarchy[row])
            queue[tail + 1] = Int64(hierarchy[row + 1])
            tail += 2
        ignore[node] = 1
    return result_pos


def condense_tree(
    hierarchy: FPtr,
    rows: Int,
    min_cluster_size: Int,
    node_queue: IPtr,
    subtree_queue: IPtr,
    relabel: IPtr,
    ignore: BPtr,
    result_parent: IPtr,
    result_child: IPtr,
    result_lambda: FPtr,
    result_size: IPtr,
) -> Int:
    var n = rows + 1
    var root = 2 * rows
    for i in range(root + 1):
        ignore[i] = 0

    var head = 0
    var tail = 1
    node_queue[0] = Int64(root)
    while head < tail:
        var node = Int(node_queue[head])
        head += 1
        if node >= n:
            var row = (node - n) * 4
            node_queue[tail] = Int64(hierarchy[row])
            node_queue[tail + 1] = Int64(hierarchy[row + 1])
            tail += 2

    relabel[root] = Int64(n)
    var next_label = n + 1
    var pos = 0
    for q in range(tail):
        var node = Int(node_queue[q])
        if ignore[node] != 0 or node < n:
            continue
        var row = (node - n) * 4
        var left = Int(hierarchy[row])
        var right = Int(hierarchy[row + 1])
        var distance = hierarchy[row + 2]
        var lambda_value = infinity()
        if distance > 0.0:
            lambda_value = 1.0 / distance
        var left_count = 1
        if left >= n:
            left_count = Int(hierarchy[(left - n) * 4 + 3])
        var right_count = 1
        if right >= n:
            right_count = Int(hierarchy[(right - n) * 4 + 3])

        if left_count >= min_cluster_size and right_count >= min_cluster_size:
            relabel[left] = Int64(next_label)
            next_label += 1
            result_parent[pos] = relabel[node]
            result_child[pos] = relabel[left]
            result_lambda[pos] = lambda_value
            result_size[pos] = Int64(left_count)
            pos += 1
            relabel[right] = Int64(next_label)
            next_label += 1
            result_parent[pos] = relabel[node]
            result_child[pos] = relabel[right]
            result_lambda[pos] = lambda_value
            result_size[pos] = Int64(right_count)
            pos += 1
        elif left_count < min_cluster_size and right_count < min_cluster_size:
            pos = emit_small_subtree(
                hierarchy, left, Int(relabel[node]), lambda_value, n,
                subtree_queue, ignore, result_parent, result_child,
                result_lambda, result_size, pos,
            )
            pos = emit_small_subtree(
                hierarchy, right, Int(relabel[node]), lambda_value, n,
                subtree_queue, ignore, result_parent, result_child,
                result_lambda, result_size, pos,
            )
        elif left_count < min_cluster_size:
            relabel[right] = relabel[node]
            pos = emit_small_subtree(
                hierarchy, left, Int(relabel[node]), lambda_value, n,
                subtree_queue, ignore, result_parent, result_child,
                result_lambda, result_size, pos,
            )
        else:
            relabel[left] = relabel[node]
            pos = emit_small_subtree(
                hierarchy, right, Int(relabel[node]), lambda_value, n,
                subtree_queue, ignore, result_parent, result_child,
                result_lambda, result_size, pos,
            )
    return pos


def compute_stability(
    parents: IPtr,
    children: IPtr,
    lambdas: FPtr,
    sizes: IPtr,
    rows: Int,
    births: FPtr,
    result: FPtr,
    smallest_cluster: Int,
    largest_child: Int,
    num_clusters: Int,
):
    for i in range(largest_child + 1):
        births[i] = infinity()
    for i in range(rows):
        var child = Int(children[i])
        if lambdas[i] < births[child]:
            births[child] = lambdas[i]
    births[smallest_cluster] = 0.0
    for i in range(num_clusters):
        result[i] = 0.0
    for i in range(rows):
        var parent = Int(parents[i])
        result[parent - smallest_cluster] += (
            (lambdas[i] - births[parent]) * Float64(sizes[i])
        )


def find_root(parent: IPtr, value: Int) -> Int:
    var node = value
    while Int(parent[node]) != node:
        node = Int(parent[node])
    var root = node
    node = value
    while Int(parent[node]) != node:
        var next_node = Int(parent[node])
        parent[node] = Int64(root)
        node = next_node
    return root


def union_nodes(parent: IPtr, rank: IPtr, a: Int, b: Int):
    var root_a = find_root(parent, a)
    var root_b = find_root(parent, b)
    if root_a == root_b:
        return
    if rank[root_a] < rank[root_b]:
        parent[root_a] = Int64(root_b)
    elif rank[root_a] > rank[root_b]:
        parent[root_b] = Int64(root_a)
    else:
        parent[root_b] = Int64(root_a)
        rank[root_a] += 1


def labelling_at_cut(
    linkage: FPtr,
    rows: Int,
    cut: Float64,
    min_cluster_size: Int,
    parent: IPtr,
    rank: IPtr,
    sizes: IPtr,
    label_map: IPtr,
    result: IPtr,
):
    var n = rows + 1
    var total_nodes = 2 * rows + 1
    for i in range(total_nodes):
        parent[i] = Int64(i)
        rank[i] = 0
        sizes[i] = 0
        label_map[i] = -1
    for row in range(rows):
        if linkage[row * 4 + 2] < cut:
            union_nodes(parent, rank, Int(linkage[row * 4]), n + row)
            union_nodes(parent, rank, Int(linkage[row * 4 + 1]), n + row)
    for point in range(n):
        var root = find_root(parent, point)
        result[point] = Int64(root)
        sizes[root] += 1
    var next_label = 0
    for root in range(total_nodes):
        if Int(sizes[root]) >= min_cluster_size:
            label_map[root] = Int64(next_label)
            next_label += 1
    for point in range(n):
        result[point] = label_map[Int(result[point])]


def label_condensed_tree(
    parents: IPtr,
    children: IPtr,
    lambdas: FPtr,
    rows: Int,
    root_cluster: Int,
    max_node: Int,
    selected: BPtr,
    cluster_labels: IPtr,
    parent: IPtr,
    rank: IPtr,
    child_lambda: FPtr,
    parent_max_lambda: FPtr,
    result: IPtr,
    deaths: FPtr,
    reverse_clusters: IPtr,
    probabilities: FPtr,
    selected_count: Int,
    allow_single_cluster: Int,
    cluster_selection_epsilon: Float64,
    match_reference: Int,
):
    for i in range(max_node + 1):
        parent[i] = Int64(i)
        rank[i] = 0
        child_lambda[i] = 0.0
        parent_max_lambda[i] = 0.0
    for i in range(rows):
        var child = Int(children[i])
        var parent_node = Int(parents[i])
        child_lambda[child] = lambdas[i]
        if lambdas[i] > parent_max_lambda[parent_node]:
            parent_max_lambda[parent_node] = lambdas[i]
        if selected[child] == 0:
            union_nodes(parent, rank, parent_node, child)

    for point in range(root_cluster):
        var cluster = find_root(parent, point)
        if cluster < root_cluster:
            result[point] = -1
        elif cluster == root_cluster:
            if selected_count == 1 and allow_single_cluster != 0 and selected[cluster] != 0:
                if cluster_selection_epsilon != 0.0:
                    if child_lambda[point] >= 1.0 / cluster_selection_epsilon:
                        result[point] = cluster_labels[cluster]
                    else:
                        result[point] = -1
                elif child_lambda[point] >= parent_max_lambda[cluster]:
                    result[point] = cluster_labels[cluster]
                else:
                    result[point] = -1
            else:
                result[point] = -1
        elif match_reference != 0:
            if child_lambda[point] > child_lambda[cluster]:
                result[point] = cluster_labels[cluster]
            else:
                result[point] = -1
        else:
            result[point] = cluster_labels[cluster]

    for point in range(root_cluster):
        var label = Int(result[point])
        if label < 0:
            probabilities[point] = 0.0
            continue
        var maximum = deaths[Int(reverse_clusters[label])]
        var value = child_lambda[point]
        if maximum == 0.0 or value != value or abs(value) == infinity():
            probabilities[point] = 1.0
        else:
            probabilities[point] = min(value, maximum) / maximum


def eom_select(
    parents: IPtr,
    children: IPtr,
    lambdas: FPtr,
    sizes: IPtr,
    rows: Int,
    stability: FPtr,
    root: Int,
    num_clusters: Int,
    max_cluster_size: Int,
    epsilon_max: Float64,
    selected: BPtr,
    cluster_parent: IPtr,
    cluster_size: IPtr,
    node_epsilon: FPtr,
    subtree_stability: FPtr,
    blocked: BPtr,
):
    for i in range(num_clusters):
        cluster_parent[i] = -1
        cluster_size[i] = 0
        node_epsilon[i] = 0.0
        subtree_stability[i] = 0.0
        blocked[i] = 0
        selected[root + i] = 0

    for i in range(rows):
        if sizes[i] <= 1:
            continue
        var child = Int(children[i])
        var offset = child - root
        cluster_parent[offset] = parents[i]
        cluster_size[offset] = sizes[i]
        if lambdas[i] == 0.0:
            node_epsilon[offset] = infinity()
        else:
            node_epsilon[offset] = 1.0 / lambdas[i]

    var offset = num_clusters - 1
    while offset > 0:
        var node = root + offset
        if (
            subtree_stability[offset] > stability[offset]
            or Int(cluster_size[offset]) > max_cluster_size
            or node_epsilon[offset] > epsilon_max
        ):
            stability[offset] = subtree_stability[offset]
        else:
            selected[node] = 1
        var parent = Int(cluster_parent[offset])
        if parent >= root:
            subtree_stability[parent - root] += stability[offset]
        offset -= 1

    for child_offset in range(1, num_clusters):
        var parent = Int(cluster_parent[child_offset])
        if parent < root:
            continue
        var parent_offset = parent - root
        if blocked[parent_offset] != 0 or selected[parent] != 0:
            blocked[child_offset] = 1
            selected[root + child_offset] = 0


@export("mhdb_mutual_reachability")
def mhdb_mutual_reachability(
    matrix: Int,
    result: Int,
    core: Int,
    scratch: Int,
    n: Int,
    min_points: Int,
    alpha: Float64,
) abi("C"):
    mutual_reachability(
        fp(matrix), fp(result), fp(core), fp(scratch), n, min_points, alpha
    )


@export("mhdb_condense_tree")
def mhdb_condense_tree(
    hierarchy: Int,
    rows: Int,
    min_cluster_size: Int,
    node_queue: Int,
    subtree_queue: Int,
    relabel: Int,
    ignore: Int,
    result_parent: Int,
    result_child: Int,
    result_lambda: Int,
    result_size: Int,
) abi("C") -> Int:
    return condense_tree(
        fp(hierarchy), rows, min_cluster_size, ip(node_queue), ip(subtree_queue),
        ip(relabel), bp(ignore), ip(result_parent), ip(result_child),
        fp(result_lambda), ip(result_size),
    )


@export("mhdb_compute_stability")
def mhdb_compute_stability(
    parents: Int,
    children: Int,
    lambdas: Int,
    sizes: Int,
    rows: Int,
    births: Int,
    result: Int,
    smallest_cluster: Int,
    largest_child: Int,
    num_clusters: Int,
) abi("C"):
    compute_stability(
        ip(parents), ip(children), fp(lambdas), ip(sizes), rows, fp(births),
        fp(result), smallest_cluster, largest_child, num_clusters,
    )


@export("mhdb_labelling_at_cut")
def mhdb_labelling_at_cut(
    linkage: Int,
    rows: Int,
    cut: Float64,
    min_cluster_size: Int,
    parent: Int,
    rank: Int,
    sizes: Int,
    label_map: Int,
    result: Int,
) abi("C"):
    labelling_at_cut(
        fp(linkage), rows, cut, min_cluster_size, ip(parent), ip(rank),
        ip(sizes), ip(label_map), ip(result),
    )


@export("mhdb_label_condensed_tree")
def mhdb_label_condensed_tree(
    parents: Int,
    children: Int,
    lambdas: Int,
    rows: Int,
    root_cluster: Int,
    max_node: Int,
    selected: Int,
    cluster_labels: Int,
    parent: Int,
    rank: Int,
    child_lambda: Int,
    parent_max_lambda: Int,
    result: Int,
    deaths: Int,
    reverse_clusters: Int,
    probabilities: Int,
    selected_count: Int,
    allow_single_cluster: Int,
    cluster_selection_epsilon: Float64,
    match_reference: Int,
) abi("C"):
    label_condensed_tree(
        ip(parents), ip(children), fp(lambdas), rows, root_cluster, max_node,
        bp(selected), ip(cluster_labels), ip(parent), ip(rank), fp(child_lambda),
        fp(parent_max_lambda), ip(result), fp(deaths), ip(reverse_clusters),
        fp(probabilities), selected_count, allow_single_cluster,
        cluster_selection_epsilon, match_reference,
    )


@export("mhdb_eom_select")
def mhdb_eom_select(
    parents: Int,
    children: Int,
    lambdas: Int,
    sizes: Int,
    rows: Int,
    stability: Int,
    root: Int,
    num_clusters: Int,
    max_cluster_size: Int,
    epsilon_max: Float64,
    selected: Int,
    cluster_parent: Int,
    cluster_size: Int,
    node_epsilon: Int,
    subtree_stability: Int,
    blocked: Int,
) abi("C"):
    eom_select(
        ip(parents), ip(children), fp(lambdas), ip(sizes), rows, fp(stability),
        root, num_clusters, max_cluster_size, epsilon_max, bp(selected),
        ip(cluster_parent), ip(cluster_size), fp(node_epsilon),
        fp(subtree_stability), bp(blocked),
    )
