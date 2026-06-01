import warnings

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

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
    X = X.astype(bool)

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
    preprocessor.fit(X.astype(bool), columns=X_identifiers)
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

    X = np.zeros((4, 4), dtype=bool)
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
    X = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
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
    X = df.to_numpy().astype(bool)
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
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ],
        dtype=bool,
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
    X = rng.integers(0, 2, size=(n_rows, n_nodes)).astype(bool)

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
    X = rng.integers(0, 2, size=(n_rows, n_nodes)).astype(bool)

    pre = _fit_with_all_columns(hierarchy, X)
    X_added = pre._add_columns(X)

    X_vec = pre._propagate_ones(X_added.copy())
    X_loop = _propagate_ones_reference(pre._hierarchy_graph, pre._columns, X_added.copy())
    assert np.array_equal(X_vec, X_loop)


def test_propagate_ones_empty_input():
    rng = np.random.default_rng(0)
    n_nodes = 12
    hierarchy = _random_tree_hierarchy(n_nodes, rng)
    X = np.zeros((6, n_nodes), dtype=bool)

    pre = _fit_with_all_columns(hierarchy, X)
    X_added = pre._add_columns(X)
    out = pre._propagate_ones(X_added.copy())
    assert np.array_equal(out, X_added)


def test_propagate_ones_saturated_input():
    rng = np.random.default_rng(1)
    n_nodes = 12
    hierarchy = _random_tree_hierarchy(n_nodes, rng)
    X = np.ones((6, n_nodes), dtype=bool)

    pre = _fit_with_all_columns(hierarchy, X)
    X_added = pre._add_columns(X)
    out = pre._propagate_ones(X_added.copy())
    assert np.array_equal(out, X_added)


def test_propagate_ones_single_node():
    hierarchy = np.zeros((1, 1), dtype=int)
    X = np.array([[0], [1]], dtype=bool)
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
    X = np.array([[0, 0, 0, 1]], dtype=bool)

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
    X = rng.integers(0, 2, size=(20, n_nodes)).astype(bool)

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
    X = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]], dtype=bool)
    pre = HierarchicalPreprocessor(hierarchy)
    pre.fit(X, columns=[0, 1, 2, 3])

    X_added = pre._add_columns(X)
    out = pre._propagate_ones(X_added.copy())
    expected = np.array([[1, 1, 1, 1], [1, 1, 1, 0], [1, 0, 0, 0]])
    assert np.array_equal(out, expected)


# ---------------------------------------------------------------------------
# Bool-dtype contract tests.
#
# The preprocessor enforces bool-dtype input on both fit and transform. The
# vectorized propagation is only semantically correct for binary data; until
# a sum-propagation mode will be implemented, numeric inputs are rejected up front.
# ---------------------------------------------------------------------------


_EXPECTED_CANONICAL = np.array(
    [
        [1, 0, 0, 1, 1, 0],
        [0, 1, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 1],
        [1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 1],
    ]
)


_REJECTED_DTYPES = [np.int8, np.int32, np.int64, np.float32, np.float64]


def test_preprocessor_preserves_bool_dtype():
    """fit + transform must round-trip bool input unchanged."""
    X, hierarchy, columns = _canonical_setup()
    assert X.dtype == np.bool_

    pre = HierarchicalPreprocessor(hierarchy)
    pre.fit(X, columns=columns)
    X_pp = pre.transform(X)

    assert X_pp.dtype == np.bool_
    assert pre._ancestor_closure_.dtype == bool
    assert np.array_equal(X_pp.astype(int), _EXPECTED_CANONICAL)


@pytest.mark.parametrize("dtype", _REJECTED_DTYPES)
def test_preprocessor_rejects_non_bool_dtype(dtype):
    """fit and transform both reject non-bool dtypes with a clear message."""
    X_bool, hierarchy, columns = _canonical_setup()
    X = X_bool.astype(dtype)

    pre = HierarchicalPreprocessor(hierarchy)
    with pytest.raises(ValueError, match="bool-dtype"):
        pre.fit(X, columns=columns)

    # Fit on bool, then try to transform a non-bool array: still rejected.
    pre.fit(X_bool, columns=columns)
    with pytest.raises(ValueError, match="bool-dtype"):
        pre.transform(X)


def test_preprocessor_accepts_bool_dense():
    """Canonical example with bool dense input pins the supported contract."""
    X, hierarchy, columns = _canonical_setup()

    pre = HierarchicalPreprocessor(hierarchy)
    pre.fit(X, columns=columns)
    X_pp = pre.transform(X)

    assert X_pp.dtype == np.bool_
    assert np.array_equal(X_pp.astype(int), _EXPECTED_CANONICAL)


def test_preprocessor_accepts_bool_sparse():
    """csr_array(dtype=bool) input passes the bool-dtype check and round-trips
    through fit + transform without densifying.

    Tests whether the validator treats ``X.dtype`` uniformly for dense and sparse matrices and yields the is_fitted_ attribute.
    """
    X_dense, hierarchy, columns = _canonical_setup()
    X = sp.csr_array(X_dense)
    assert X.dtype == np.bool_

    pre = HierarchicalPreprocessor(hierarchy)
    pre.fit(X, columns=columns)
    assert pre.is_fitted_

    X_pp = pre.transform(X)
    assert isinstance(X_pp, sp.sparray)
    assert X_pp.format == "csr"
    assert X_pp.dtype == np.bool_
    assert X_pp.shape == (X_dense.shape[0], len(pre._columns))


# ---------------------------------------------------------------------------
# Sparse input support.
#
# fit + transform must work end-to-end on scipy.sparse.csr_matrix(bool) and
# return CSR(bool) output equivalent (cell-by-cell) to the dense path on the
# same input. Sparse and dense paths must never silently cross over.
# ---------------------------------------------------------------------------


def test_preprocessor_sparse_canonical_equivalence():
    """Canonical example: sparse path yields the same matrix as dense."""
    X_dense, hierarchy, columns = _canonical_setup()
    X_sparse = sp.csr_array(X_dense)

    pre_d = HierarchicalPreprocessor(hierarchy)
    pre_d.fit(X_dense, columns=columns)
    out_d = pre_d.transform(X_dense)

    pre_s = HierarchicalPreprocessor(hierarchy)
    pre_s.fit(X_sparse, columns=columns)
    out_s = pre_s.transform(X_sparse)

    assert not sp.issparse(out_d)
    assert isinstance(out_s, sp.sparray)
    assert out_s.format == "csr"
    assert out_d.dtype == np.bool_
    assert out_s.dtype == np.bool_
    assert np.array_equal(out_s.toarray(), out_d)
    assert np.array_equal(out_s.toarray().astype(int), _EXPECTED_CANONICAL)


def test_scipy_sparse_bool_matmul_preserves_bool():
    """Test whether matmul (``@``) and ``maximum()`` preserve dtype in scipy sparse arrays. This behaviour is not immediately obvious from the scipy docs at first glance, so this is just a defensive test."""
    a = sp.csr_array(np.array([[True, False], [False, True]]))
    b = sp.csr_array(np.array([[True, True], [False, True]]))
    assert (a @ b).dtype == np.bool_
    assert a.maximum(b).dtype == np.bool_


def test_preprocessor_accepts_csc_input():
    """CSC input is accepted (and normalized to csr_array on output)."""
    X_dense, hierarchy, columns = _canonical_setup()
    X_csc = sp.csc_array(X_dense)
    assert X_csc.format == "csc"

    pre = HierarchicalPreprocessor(hierarchy)
    pre.fit(X_csc, columns=columns)
    out = pre.transform(X_csc)

    assert isinstance(out, sp.sparray)
    assert out.format == "csr"
    assert out.dtype == np.bool_
    assert np.array_equal(out.toarray().astype(int), _EXPECTED_CANONICAL)


def test_preprocessor_accepts_coo_input():
    """COO input is accepted (and normalized to csr_array on output)."""
    X_dense, hierarchy, columns = _canonical_setup()
    X_coo = sp.coo_array(X_dense)
    assert X_coo.format == "coo"

    pre = HierarchicalPreprocessor(hierarchy)
    pre.fit(X_coo, columns=columns)
    out = pre.transform(X_coo)

    assert isinstance(out, sp.sparray)
    assert out.format == "csr"
    assert out.dtype == np.bool_
    assert np.array_equal(out.toarray().astype(int), _EXPECTED_CANONICAL)


def test_preprocessor_normalizes_legacy_csr_matrix_to_csr_array():
    """Legacy ``csr_matrix`` input is accepted and normalized to ``csr_array``.

    scipy encourages the use of csr_array over csr_matrix.
    Since csr_matrix is just a different representation of the same underlying sparse data, but has a different API, the preprocessor accepts csr_matrix for backward compatibility but internally converts it to csr_array. This ensures consistent behaviour for all downstream operations.
    """
    X_dense, hierarchy, columns = _canonical_setup()
    X_legacy = sp.csr_matrix(X_dense)
    assert isinstance(X_legacy, sp.spmatrix)

    pre = HierarchicalPreprocessor(hierarchy)
    pre.fit(X_legacy, columns=columns)
    out = pre.transform(X_legacy)

    assert isinstance(out, sp.sparray)
    assert not isinstance(out, sp.spmatrix)
    assert out.format == "csr"
    assert out.dtype == np.bool_
    assert np.array_equal(out.toarray().astype(int), _EXPECTED_CANONICAL)


def test_preprocessor_sparse_dense_paths_do_not_cross_format():
    """Dense in -> dense out. Sparse in -> sparse out. No silent crossover."""
    X_dense, hierarchy, columns = _canonical_setup()
    X_sparse = sp.csr_array(X_dense)

    pre_d = HierarchicalPreprocessor(hierarchy)
    pre_d.fit(X_dense, columns=columns)
    assert isinstance(pre_d.transform(X_dense), np.ndarray)

    pre_s = HierarchicalPreprocessor(hierarchy)
    pre_s.fit(X_sparse, columns=columns)
    assert sp.issparse(pre_s.transform(X_sparse))


def test_preprocessor_rejects_non_bool_sparse():
    """Sparse int input must be rejected with the same bool-dtype error."""
    X_dense, hierarchy, columns = _canonical_setup()
    X = sp.csr_array(X_dense.astype(np.int8))
    assert X.dtype == np.int8

    pre = HierarchicalPreprocessor(hierarchy)
    with pytest.raises(ValueError, match="bool-dtype"):
        pre.fit(X, columns=columns)


def test_preprocessor_rejects_int_with_binary_values():
    """Binary-valued int input is still rejected; we check the dtype, not values."""
    X_bool, hierarchy, columns = _canonical_setup()
    X = X_bool.astype(np.int8)  # values are {0, 1} but dtype is int8

    pre = HierarchicalPreprocessor(hierarchy)
    with pytest.raises(ValueError, match="bool-dtype"):
        pre.fit(X, columns=columns)


def test_preprocessor_rejects_float_with_binary_values():
    """Binary-valued float input is still rejected; we check the dtype, not values."""
    X_bool, hierarchy, columns = _canonical_setup()
    X = X_bool.astype(np.float64)  # values are {0.0, 1.0} but dtype is float64

    pre = HierarchicalPreprocessor(hierarchy)
    with pytest.raises(ValueError, match="bool-dtype"):
        pre.fit(X, columns=columns)


def test_preprocessor_rejects_non_bool_in_transform():
    """fit-on-bool then transform-on-int8 must still raise in transform."""
    X_bool, hierarchy, columns = _canonical_setup()

    pre = HierarchicalPreprocessor(hierarchy)
    pre.fit(X_bool, columns=columns)

    X_int = X_bool.astype(np.int8)
    with pytest.raises(ValueError, match="bool-dtype"):
        pre.transform(X_int)


def test_error_message_mentions_astype_bool():
    """The user-facing error must guide users to the fix for binary data."""
    X_bool, hierarchy, columns = _canonical_setup()
    X = X_bool.astype(np.int64)

    pre = HierarchicalPreprocessor(hierarchy)
    with pytest.raises(ValueError, match=r"astype\(bool\)"):
        pre.fit(X, columns=columns)


def test_preprocessor_rejects_nan():
    """validate_data's ensure_all_finite check still rejects NaN.

    NaN can only appear in float arrays, which are themselves rejected, but
    the NaN check inside validate_data fires before the bool-dtype check.
    """
    X_int, hierarchy, columns = _canonical_setup()
    X = X_int.astype(np.float64)
    X[0, 0] = np.nan

    pre = HierarchicalPreprocessor(hierarchy)
    with pytest.raises(ValueError):
        pre.fit(X, columns=columns)
