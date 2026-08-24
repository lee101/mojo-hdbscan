"""Behavioral parity with the compiled upstream hdbscan package."""

import numpy as np
import pytest
from scipy.cluster.hierarchy import linkage
from scipy.sparse import lil_matrix
from scipy.spatial.distance import pdist, squareform

import mojo_hdbscan as mojo

upstream_reachability = pytest.importorskip("hdbscan._hdbscan_reachability")
upstream_tree = pytest.importorskip("hdbscan._hdbscan_tree")


@pytest.fixture(scope="module")
def hierarchy():
    rng = np.random.default_rng(12)
    points = np.vstack(
        [
            rng.normal((-2.0, -1.0), 0.35, (45, 2)),
            rng.normal((2.0, 1.0), 0.45, (40, 2)),
            rng.normal((0.0, 3.0), 0.30, (35, 2)),
        ]
    )
    return linkage(points, method="single")


@pytest.fixture(scope="module")
def trees(hierarchy):
    upstream = upstream_tree.condense_tree(hierarchy, 8)
    ours = mojo.condense_tree(hierarchy, 8)
    return upstream, ours


def assert_tree_equal(left, right):
    assert left.dtype == right.dtype
    assert left.dtype.names == right.dtype.names
    for field in left.dtype.names:
        assert np.array_equal(left[field], right[field])


def test_mutual_reachability_published_definition_vector():
    distances = np.array(
        [
            [0.0, 1.0, 4.0, 7.0],
            [1.0, 0.0, 3.0, 6.0],
            [4.0, 3.0, 0.0, 2.0],
            [7.0, 6.0, 2.0, 0.0],
        ]
    )
    expected = np.array(
        [
            [4.0, 4.0, 4.0, 7.0],
            [4.0, 3.0, 3.0, 6.0],
            [4.0, 3.0, 3.0, 6.0],
            [7.0, 6.0, 6.0, 6.0],
        ]
    )
    result = mojo.mutual_reachability(distances, min_points=2)
    assert result is distances
    assert np.array_equal(result, expected)


@pytest.mark.parametrize("min_points", [1, 5, 19, 50])
def test_mutual_reachability_parity(min_points):
    rng = np.random.default_rng(min_points)
    distances = squareform(pdist(rng.normal(size=(20, 4))))
    expected = upstream_reachability.mutual_reachability(
        distances.copy(), min_points=min_points
    )
    actual = mojo.mutual_reachability(distances.copy(), min_points=min_points)
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("alpha", [0.5, 1.0, 1.7])
def test_mutual_reachability_alpha_parity(alpha):
    rng = np.random.default_rng(4)
    distances = squareform(pdist(rng.normal(size=(31, 3))))
    expected = upstream_reachability.mutual_reachability(
        distances.copy(), min_points=6, alpha=alpha
    )
    actual = mojo.mutual_reachability(
        distances.copy(), min_points=6, alpha=alpha
    )
    assert np.array_equal(actual, expected)


def test_mutual_reachability_simd_tail_nonsymmetric():
    rng = np.random.default_rng(41)
    distances = rng.uniform(0.0, 5.0, size=(37, 37))
    np.fill_diagonal(distances, 0.0)
    expected = upstream_reachability.mutual_reachability(
        distances.copy(), min_points=7, alpha=1.3
    )
    actual = mojo.mutual_reachability(
        distances.copy(), min_points=7, alpha=1.3
    )
    assert np.array_equal(actual, expected)


def test_mutual_reachability_converts_strided_float32_input_safely():
    rng = np.random.default_rng(77)
    backing = rng.uniform(size=(19, 38)).astype(np.float32)
    distances = backing[:, ::2]
    np.fill_diagonal(distances, 0.0)
    expected_input = np.ascontiguousarray(distances, dtype=np.float64)
    expected = upstream_reachability.mutual_reachability(expected_input.copy(), 4)
    actual = mojo.mutual_reachability(distances, 4)
    assert actual.dtype == np.float64
    assert actual.flags.c_contiguous
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.empty((0, 0)), "must not be empty"),
        (np.empty((2, 3)), "square"),
        (np.array([[0.0, np.nan], [1.0, 0.0]]), "finite"),
    ],
)
def test_mutual_reachability_rejects_unsafe_inputs(matrix, message):
    with pytest.raises(ValueError, match=message):
        mojo.mutual_reachability(matrix)


@pytest.mark.parametrize("alpha", [0.0, -1.0, np.inf, np.nan])
def test_mutual_reachability_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        mojo.mutual_reachability(np.eye(3), alpha=alpha)


@pytest.mark.parametrize("n", [1499, 1500])
def test_mutual_reachability_parallel_threshold(n):
    rng = np.random.default_rng(n)
    values = rng.uniform(0.0, 5.0, size=(n, n))
    distances = (values + values.T) * 0.5
    np.fill_diagonal(distances, 0.0)
    expected = upstream_reachability.mutual_reachability(
        distances.copy(), min_points=11, alpha=1.2
    )
    actual = mojo.mutual_reachability(
        distances.copy(), min_points=11, alpha=1.2
    )
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("min_points", [64, 65])
def test_mutual_reachability_selection_cutoff(min_points):
    rng = np.random.default_rng(min_points)
    distances = rng.uniform(0.0, 5.0, size=(97, 97))
    np.fill_diagonal(distances, 0.0)
    expected = upstream_reachability.mutual_reachability(
        distances.copy(), min_points=min_points, alpha=1.1
    )
    actual = mojo.mutual_reachability(
        distances.copy(), min_points=min_points, alpha=1.1
    )
    assert np.array_equal(actual, expected)


def test_sparse_mutual_reachability_parity():
    matrix = lil_matrix((7, 7), dtype=np.float64)
    for i, j, value in [
        (0, 1, 1.0),
        (1, 0, 1.0),
        (1, 2, 2.0),
        (2, 1, 2.0),
        (2, 3, 1.5),
        (3, 2, 1.5),
        (4, 5, 0.8),
        (5, 4, 0.8),
    ]:
        matrix[i, j] = value
    expected = upstream_reachability.sparse_mutual_reachability(
        matrix.copy(), min_points=2, alpha=1.3, max_dist=9.0
    )
    actual = mojo.sparse_mutual_reachability(
        matrix.copy(), min_points=2, alpha=1.3, max_dist=9.0
    )
    assert np.array_equal(actual.toarray(), expected.toarray())


@pytest.mark.parametrize("min_cluster_size", [2, 5, 8, 20])
def test_condense_tree_parity(hierarchy, min_cluster_size):
    expected = upstream_tree.condense_tree(hierarchy, min_cluster_size)
    actual = mojo.condense_tree(hierarchy, min_cluster_size)
    assert_tree_equal(actual, expected)


def test_condense_tree_zero_distance_parity():
    points = np.array([[0.0], [0.0], [1.0], [1.0], [5.0], [5.0]])
    hierarchy = linkage(points, method="single")
    expected = upstream_tree.condense_tree(hierarchy, 2)
    actual = mojo.condense_tree(hierarchy, 2)
    assert_tree_equal(actual, expected)
    assert np.isinf(actual["lambda_val"]).any()


def test_condense_tree_accepts_strided_float32_linkage(hierarchy):
    backing = np.empty((hierarchy.shape[0], 8), dtype=np.float32)
    backing[:, ::2] = hierarchy
    actual = mojo.condense_tree(backing[:, ::2], 5)
    expected = upstream_tree.condense_tree(
        np.asarray(backing[:, ::2], dtype=np.float64), 5
    )
    assert_tree_equal(actual, expected)


@pytest.mark.parametrize(
    "bad_row",
    [
        [0.5, 1.0, 1.0, 2.0],
        [0.0, 3.0, 1.0, 2.0],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 1.0, 1.0, 3.0],
    ],
)
def test_condense_tree_rejects_malformed_linkage(bad_row):
    with pytest.raises(ValueError):
        mojo.condense_tree(np.asarray([bad_row], dtype=np.float64), 1)


def test_compute_stability_parity(trees):
    expected_tree, actual_tree = trees
    expected = upstream_tree.compute_stability(expected_tree)
    actual = mojo.compute_stability(actual_tree)
    assert set(actual) == set(expected)
    for cluster in expected:
        assert actual[cluster] == pytest.approx(expected[cluster], rel=1e-14)


@pytest.mark.parametrize("method", ["eom", "leaf"])
def test_get_clusters_parity(trees, method):
    expected_tree, actual_tree = trees
    expected_stability = upstream_tree.compute_stability(expected_tree)
    actual_stability = mojo.compute_stability(actual_tree)
    expected = upstream_tree.get_clusters(
        expected_tree, expected_stability.copy(), method
    )
    actual = mojo.get_clusters(actual_tree, actual_stability.copy(), method)
    assert np.array_equal(actual[0], expected[0])
    assert np.allclose(actual[1], expected[1], rtol=0.0, atol=0.0)
    assert np.allclose(actual[2], expected[2], rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("epsilon", [0.1, 0.5, 1.0])
def test_get_clusters_epsilon_parity(trees, epsilon):
    expected_tree, actual_tree = trees
    expected = upstream_tree.get_clusters(
        expected_tree,
        upstream_tree.compute_stability(expected_tree),
        cluster_selection_method="leaf",
        allow_single_cluster=True,
        cluster_selection_epsilon=epsilon,
    )
    actual = mojo.get_clusters(
        actual_tree,
        mojo.compute_stability(actual_tree),
        cluster_selection_method="leaf",
        allow_single_cluster=True,
        cluster_selection_epsilon=epsilon,
    )
    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])
    assert np.allclose(actual[2], expected[2], rtol=1e-14, atol=1e-14)


def test_get_clusters_selection_limits_parity(trees):
    expected_tree, actual_tree = trees
    kwargs = dict(
        cluster_selection_method="eom",
        max_cluster_size=20,
        cluster_selection_epsilon_max=0.7,
        match_reference_implementation=True,
    )
    expected = upstream_tree.get_clusters(
        expected_tree, upstream_tree.compute_stability(expected_tree), **kwargs
    )
    actual = mojo.get_clusters(
        actual_tree, mojo.compute_stability(actual_tree), **kwargs
    )
    for ours, theirs in zip(actual, expected):
        assert np.allclose(ours, theirs, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("cut", [0.15, 0.45, 1.5])
def test_labelling_at_cut_parity(hierarchy, cut):
    expected = upstream_tree.labelling_at_cut(hierarchy, cut, 5)
    actual = mojo.labelling_at_cut(hierarchy, cut, 5)
    assert np.array_equal(actual, expected)


def test_outlier_scores_parity(trees):
    expected_tree, actual_tree = trees
    expected = upstream_tree.outlier_scores(expected_tree)
    actual = mojo.outlier_scores(actual_tree)
    assert np.allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_end_to_end_condensed_tree_pipeline(hierarchy):
    expected_tree = upstream_tree.condense_tree(hierarchy, 6)
    expected = upstream_tree.get_clusters(
        expected_tree, upstream_tree.compute_stability(expected_tree)
    )
    actual_tree = mojo.condense_tree(hierarchy, 6)
    actual = mojo.get_clusters(actual_tree, mojo.compute_stability(actual_tree))
    assert_tree_equal(actual_tree, expected_tree)
    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])
    assert np.allclose(actual[2], expected[2], rtol=1e-14, atol=1e-14)


def test_invalid_cluster_selection_method(trees):
    _, tree = trees
    with pytest.raises(ValueError, match="Invalid Cluster Selection Method"):
        mojo.get_clusters(
            tree,
            mojo.compute_stability(tree),
            cluster_selection_method="invalid",
        )


def test_tree_functions_reject_empty_or_narrowed_structured_fields():
    empty = np.empty(
        0,
        dtype=[
            ("parent", np.intp),
            ("child", np.intp),
            ("lambda_val", np.float64),
            ("child_size", np.intp),
        ],
    )
    with pytest.raises(ValueError, match="empty"):
        mojo.compute_stability(empty)

    unsafe_dtype = np.dtype(
        [
            ("parent", np.float64),
            ("child", np.int64),
            ("lambda_val", np.float64),
            ("child_size", np.int64),
        ]
    )
    unsafe = np.array([(2.5, 0, 1.0, 1)], dtype=unsafe_dtype)
    with pytest.raises(TypeError, match="integer dtype"):
        mojo.compute_stability(unsafe)
