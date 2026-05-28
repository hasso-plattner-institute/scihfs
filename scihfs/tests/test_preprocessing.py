import warnings

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from scihfs.data_utils import create_mapping_columns_to_nodes
from scihfs.helpers import get_columns_for_numpy_hierarchy
from scihfs.preprocessing import ColumnNotInHierarchyWarning, HierarchicalPreprocessor


@pytest.mark.parametrize(
    "data",
    ["data1_preprocessing", "data2_preprocessing"],
)
def test_hierarchical_preprocessor(data, request):
    data = request.getfixturevalue(data)
    X, X_transformed, hierarchy, columns, hierarchy_expected = data

    preprocessor = HierarchicalPreprocessor(hierarchy)

    preprocessor.fit(X, columns=columns)
    assert preprocessor.is_fitted_
    X = preprocessor.transform(X)
    assert np.array_equal(X, X_transformed)
    hierarchy_transformed = preprocessor.get_hierarchy()
    assert np.array_equal(hierarchy_transformed, hierarchy_expected)


def test_fit(data3_preprocessing):
    X, hierarchy, hierarchy_transformed, X_identifiers = data3_preprocessing
    preprocessor = HierarchicalPreprocessor(hierarchy)
    preprocessor.fit(X, columns=X_identifiers)
    assert preprocessor.is_fitted_
    hierarchy = preprocessor.get_hierarchy()
    assert np.equal(hierarchy.all(), hierarchy_transformed.all())


def test_adjust_node_names():
    # [4, 5, 0, 1, 3] # original node names
    # [0, 1, 2, 3, 4] # node names after transformation to numpy array
    # [2, 3, -1, 4] # mapping

    # [0, 1, 2, 3, 4, 5] # updated nodes
    # [2, 3, 5, 4] # updated mapping (without deleting or renaming)

    # [2, 3, 4, 5] # updated nodes (with deletion)
    # [2, 3, 5, 4] # updated mapping (without deleting or renaming)

    # [0, 1, 2, 3] # renamed nodes
    # [0, 1, 3, 2] # renamed nodes mapping

    X = np.zeros((4, 4))
    edges = [(4, 5), (0, 1), (0, 3), (0, 4)]
    hierarchy = nx.DiGraph(edges)
    columns = get_columns_for_numpy_hierarchy(hierarchy, X.shape[1])
    hierarchy = nx.to_numpy_array(hierarchy)
    preprocessor = HierarchicalPreprocessor(hierarchy)
    preprocessor.fit(X, columns=columns)
    preprocessor.transform(X)
    updated_columns = preprocessor.get_columns()
    assert updated_columns == [0, 1, 3, 2]


def test_columns_not_in_hierarchy_raises_warning():
    hierarchy_graph = nx.DiGraph([(0, 1)])
    hierarchy = nx.to_numpy_array(hierarchy_graph)
    estimator = HierarchicalPreprocessor(hierarchy)
    X = [[0.42, 4.2, 0.42], [4, 2, 0.42]]
    with pytest.warns(ColumnNotInHierarchyWarning):
        estimator.fit(X)


# ---------------------------------------------------------------------------
# Tests for the vectorized _propagate_ones implementation.
#
# The vectorized version is compared against a triple-nested Python reference
# oracle defined below. It is the algorithm that used to live in
# HierarchicalPreprocessor._propagate_ones; it has been moved here so the
# production class no longer carries the slow path.
# ---------------------------------------------------------------------------


def _propagate_ones_reference(hierarchy_graph, columns, X):
    """Reference 0-1 propagation. Mutates ``X`` in place and returns it.

    Equivalent to the pre-vectorization _propagate_ones loop. Kept in the
    test module so we still have an independent oracle to compare against.
    """
    nodes = list(hierarchy_graph.nodes)
    nodes.remove("ROOT")
    for node in nodes:
        column_index = columns.index(node)
        ancestor_nodes = set(nx.ancestors(hierarchy_graph, node))
        ancestor_nodes.discard("ROOT")
        for row_index, entry in enumerate(X[:, column_index]):
            if entry == 1.0:
                for ancestor in ancestor_nodes:
                    X[row_index, columns.index(ancestor)] = 1.0
    return X


def _fit_with_all_columns(hierarchy, X):
    """Fit a preprocessor where every column maps 1:1 onto a hierarchy node.

    Avoids needing _extend_dag / _shrink_dag side-effects in property tests.
    """
    n_nodes = hierarchy.shape[0]
    pre = HierarchicalPreprocessor(hierarchy)
    pre.fit(X, columns=list(range(n_nodes)))
    return pre


def _canonical_setup():
    df = pd.DataFrame(
        {
            "dog": [1, 0, 0, 1, 0],
            "cat": [0, 1, 0, 0, 0],
            "eagle": [0, 0, 1, 0, 1],
        }
    )
    graph = nx.DiGraph(
        [
            ("animal", "mammal"),
            ("animal", "bird"),
            ("animal", "fish"),
            ("mammal", "dog"),
            ("mammal", "cat"),
            ("bird", "eagle"),
            ("fish", "trout"),
        ]
    )
    columns = create_mapping_columns_to_nodes(df, graph)
    X = df.to_numpy()
    hierarchy = nx.to_numpy_array(graph)
    return X, hierarchy, columns


def test_propagate_ones_canonical_equivalence():
    X, hierarchy, columns = _canonical_setup()
    pre = HierarchicalPreprocessor(hierarchy)
    pre.fit(X, columns=columns)

    X_added = pre._add_columns(X)
    X_vec = pre._propagate_ones(X_added.copy())
    X_loop = _propagate_ones_reference(pre._hierarchy_graph, pre._columns, X_added.copy())

    expected = np.array(
        [
            [1, 0, 0, 1, 1, 0],
            [0, 1, 0, 1, 1, 0],
            [0, 0, 1, 1, 0, 1],
            [1, 0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0, 1],
        ]
    )
    assert np.array_equal(X_vec, X_loop)
    assert np.array_equal(X_vec, expected)


def test_ancestor_closure_canonical_shape_and_content():
    X, hierarchy, columns = _canonical_setup()
    pre = HierarchicalPreprocessor(hierarchy)
    pre.fit(X, columns=columns)

    # Closure indexed by COLUMN POSITION:
    #   c0 dog   -> ancestors: animal (c3), mammal (c4)
    #   c1 cat   -> ancestors: animal (c3), mammal (c4)
    #   c2 eagle -> ancestors: animal (c3), bird   (c5)
    #   c3 animal-> no ancestors
    #   c4 mammal-> animal (c3)
    #   c5 bird  -> animal (c3)
    expected = np.array(
        [
            [False, False, False, True, True, False],
            [False, False, False, True, True, False],
            [False, False, False, True, False, True],
            [False, False, False, False, False, False],
            [False, False, False, True, False, False],
            [False, False, False, True, False, False],
        ]
    )
    assert pre._ancestor_closure_.shape == (6, 6)
    assert pre._ancestor_closure_.dtype == bool
    assert np.array_equal(pre._ancestor_closure_.toarray(), expected)


def _random_tree_hierarchy(n_nodes, rng):
    """Build a random rooted tree as an int adjacency matrix.

    Node 0 is the root; every node i > 0 picks a uniformly-random parent in [0, i).
    """
    adj = np.zeros((n_nodes, n_nodes), dtype=int)
    for i in range(1, n_nodes):
        parent = int(rng.integers(0, i))
        adj[parent, i] = 1
    return adj


def _random_dag_hierarchy(n_nodes, rng, extra_edge_prob=0.15):
    """Build a random DAG (tree plus a few extra forward edges) as adjacency."""
    adj = _random_tree_hierarchy(n_nodes, rng)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if adj[i, j] == 0 and rng.random() < extra_edge_prob:
                adj[i, j] = 1
    return adj


@pytest.mark.parametrize("seed", list(range(5)))
def test_propagate_ones_equivalence_random_trees(seed):
    rng = np.random.default_rng(seed)
    n_nodes = int(rng.integers(5, 30))
    n_rows = int(rng.integers(3, 25))

    hierarchy = _random_tree_hierarchy(n_nodes, rng)
    X = rng.integers(0, 2, size=(n_rows, n_nodes), dtype=int)

    pre = _fit_with_all_columns(hierarchy, X)
    X_added = pre._add_columns(X)

    X_vec = pre._propagate_ones(X_added.copy())
    X_loop = _propagate_ones_reference(pre._hierarchy_graph, pre._columns, X_added.copy())
    assert np.array_equal(X_vec, X_loop)


@pytest.mark.parametrize("seed", list(range(5)))
def test_propagate_ones_equivalence_random_dags(seed):
    rng = np.random.default_rng(1000 + seed)
    n_nodes = int(rng.integers(5, 25))
    n_rows = int(rng.integers(3, 20))

    hierarchy = _random_dag_hierarchy(n_nodes, rng)
    X = rng.integers(0, 2, size=(n_rows, n_nodes), dtype=int)

    pre = _fit_with_all_columns(hierarchy, X)
    X_added = pre._add_columns(X)

    X_vec = pre._propagate_ones(X_added.copy())
    X_loop = _propagate_ones_reference(pre._hierarchy_graph, pre._columns, X_added.copy())
    assert np.array_equal(X_vec, X_loop)


def test_propagate_ones_empty_input():
    rng = np.random.default_rng(0)
    n_nodes = 12
    hierarchy = _random_tree_hierarchy(n_nodes, rng)
    X = np.zeros((6, n_nodes), dtype=int)

    pre = _fit_with_all_columns(hierarchy, X)
    X_added = pre._add_columns(X)
    out = pre._propagate_ones(X_added.copy())
    assert np.array_equal(out, X_added)


def test_propagate_ones_saturated_input():
    rng = np.random.default_rng(1)
    n_nodes = 12
    hierarchy = _random_tree_hierarchy(n_nodes, rng)
    X = np.ones((6, n_nodes), dtype=int)

    pre = _fit_with_all_columns(hierarchy, X)
    X_added = pre._add_columns(X)
    out = pre._propagate_ones(X_added.copy())
    assert np.array_equal(out, X_added)


def test_propagate_ones_single_node():
    hierarchy = np.zeros((1, 1), dtype=int)
    X = np.array([[0], [1]], dtype=int)
    pre = HierarchicalPreprocessor(hierarchy)
    pre.fit(X, columns=[0])

    assert pre._ancestor_closure_.shape == (1, 1)
    assert pre._ancestor_closure_.nnz == 0

    X_added = pre._add_columns(X)
    out = pre._propagate_ones(X_added.copy())
    assert np.array_equal(out, X_added)


def test_propagate_ones_multi_parent_dag():
    # Diamond: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3 (node 3 has two parents)
    hierarchy = np.zeros((4, 4), dtype=int)
    hierarchy[0, 1] = 1
    hierarchy[0, 2] = 1
    hierarchy[1, 3] = 1
    hierarchy[2, 3] = 1
    X = np.array([[0, 0, 0, 1]], dtype=int)

    pre = HierarchicalPreprocessor(hierarchy)
    pre.fit(X, columns=[0, 1, 2, 3])

    X_added = pre._add_columns(X)
    out = pre._propagate_ones(X_added.copy())
    expected = np.array([[1, 1, 1, 1]])
    assert np.array_equal(out, expected)


# ---------------------------------------------------------------------------
# Hand-crafted hierarchies that stress specific structural extremes the
# random property tests are unlikely to generate consistently.
# ---------------------------------------------------------------------------


def _long_chain(n=25):
    """Pure chain 0 -> 1 -> 2 -> ... -> n-1. Depth-dominant: every node has
    exactly one path to the root and n-1-i ancestors."""
    h = np.zeros((n, n), dtype=int)
    for i in range(n - 1):
        h[i, i + 1] = 1
    return h


def _wide_bush(k=60):
    """Root with k children, no grandchildren. Branching-dominant: closure
    has exactly k non-zeros (each leaf points to the root)."""
    n = k + 1
    h = np.zeros((n, n), dtype=int)
    for i in range(1, n):
        h[0, i] = 1
    return h


def _deep_diamond():
    """Multi-parent diamond stacked on top of a chain, so the closure has
    both >1-parent rows and a long ancestor list per leaf.

        0
       / \\
      1   2
       \\ /
        3
        |
        4
        |
        5  (multi-parent ancestor closure for 6 and 7)
        |
       / \\
      6   7
       \\ /
        8
    """
    h = np.zeros((9, 9), dtype=int)
    h[0, 1] = h[0, 2] = 1
    h[1, 3] = h[2, 3] = 1
    h[3, 4] = 1
    h[4, 5] = 1
    h[5, 6] = h[5, 7] = 1
    h[6, 8] = h[7, 8] = 1
    return h


def _multi_root_forest():
    """Two disjoint trees that are merged only by the virtual ROOT.

    Tree A: 0 -> 1 -> 2 (with 1 -> 3 sibling)
    Tree B: 4 -> 5 -> 6 (with 5 -> 7 sibling)

    The two halves must NOT share ancestors after ROOT is added (ROOT itself
    is excluded from the closure).
    """
    h = np.zeros((8, 8), dtype=int)
    h[0, 1] = 1
    h[1, 2] = 1
    h[1, 3] = 1
    h[4, 5] = 1
    h[5, 6] = 1
    h[5, 7] = 1
    return h


HAND_CRAFTED = [
    pytest.param(_long_chain(25), id="long_chain_depth_25"),
    pytest.param(_wide_bush(60), id="wide_bush_60_children"),
    pytest.param(_deep_diamond(), id="deep_diamond_multi_parent"),
    pytest.param(_multi_root_forest(), id="multi_root_forest"),
]


@pytest.mark.parametrize("hierarchy", HAND_CRAFTED)
def test_propagate_ones_complex_shapes(hierarchy):
    rng = np.random.default_rng(0)
    n_nodes = hierarchy.shape[0]
    X = rng.integers(0, 2, size=(20, n_nodes), dtype=int)

    with warnings.catch_warnings():
        # multi_root_forest legitimately warns "multiple disjoint hierarchies"
        warnings.simplefilter("ignore")
        pre = _fit_with_all_columns(hierarchy, X)

    X_added = pre._add_columns(X)
    X_vec = pre._propagate_ones(X_added.copy())
    X_ref = _propagate_ones_reference(pre._hierarchy_graph, pre._columns, X_added.copy())
    assert np.array_equal(X_vec, X_ref)


def test_propagate_ones_long_chain_explicit():
    """On a 0->1->2->3 chain, a 1 at the leaf must propagate up the full chain.

    This pins down the depth-dominant behavior with an explicit expectation,
    not just an oracle comparison.
    """
    hierarchy = _long_chain(4)
    X = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]], dtype=int)
    pre = HierarchicalPreprocessor(hierarchy)
    pre.fit(X, columns=[0, 1, 2, 3])

    X_added = pre._add_columns(X)
    out = pre._propagate_ones(X_added.copy())
    expected = np.array([[1, 1, 1, 1], [1, 1, 1, 0], [1, 0, 0, 0]])
    assert np.array_equal(out, expected)
