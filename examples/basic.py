"""Minimal condensed-tree pipeline using the Mojo-backed API."""

import numpy as np
from scipy.cluster.hierarchy import linkage

import mojo_hdbscan as hdbscan

points = np.array(
    [
        [-2.1, -1.0],
        [-1.9, -1.2],
        [-2.0, -0.8],
        [2.0, 1.1],
        [2.2, 0.9],
        [1.8, 1.0],
    ]
)

hierarchy = linkage(points, method="single")
tree = hdbscan.condense_tree(hierarchy, min_cluster_size=2)
stability = hdbscan.compute_stability(tree)
labels, probabilities, cluster_stabilities = hdbscan.get_clusters(tree, stability)

print(labels)
print(probabilities)
