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


# ---------------------------------------------------------------------------
# DiGraph hierarchy + DataFrame X acceptance and column auto-derivation.
#
# The canonical example is reused so the new (DataFrame + DiGraph) happy path
# can be pinned cell-by-cell against the established ndarray + explicit-columns
# result in _EXPECTED_CANONICAL.
# ---------------------------------------------------------------------------


def _canonical_digraph():
    """Named DiGraph equivalent of the canonical ndarray hierarchy.

    Node insertion order is animal, mammal, bird, fish, dog, cat, eagle, trout,
    i.e. positions 4/5/6 are dog/cat/eagle -- matching nx.to_numpy_array(graph)
    used by _canonical_setup, so both paths yield identical output.
    """
    return nx.DiGraph(
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


def _canonical_dataframe():
    """Bool DataFrame matching _canonical_setup's X, with named columns."""
    return pd.DataFrame(
        {
            "dog": [1, 0, 0, 1, 0],
            "cat": [0, 1, 0, 0, 0],
            "eagle": [0, 0, 1, 0, 1],
        }
    ).astype(bool)


def test_preprocessor_accepts_digraph_hierarchy_with_ndarray_X():
    """DiGraph hierarchy + ndarray X + explicit columns == ndarray-hierarchy path."""
    X, _, columns = _canonical_setup()  # X is bool ndarray, columns == [4, 5, 6]
    graph = _canonical_digraph()

    pre = HierarchicalPreprocessor(graph)
    pre.fit(X, columns=columns)
    out = pre.transform(X)

    assert isinstance(out, np.ndarray)
    assert out.dtype == np.bool_
    assert np.array_equal(out.astype(int), _EXPECTED_CANONICAL)


def test_preprocessor_accepts_dataframe_with_ndarray_hierarchy():
    """DataFrame X + ndarray hierarchy + explicit columns; output stays ndarray."""
    _, hierarchy, columns = _canonical_setup()
    df = _canonical_dataframe()

    pre = HierarchicalPreprocessor(hierarchy)
    pre.fit(df, columns=columns)
    out = pre.transform(df)

    assert isinstance(out, np.ndarray)  # ndarray by default, not a DataFrame
    assert out.dtype == np.bool_
    assert np.array_equal(out.astype(int), _EXPECTED_CANONICAL)


def test_preprocessor_accepts_dataframe_with_digraph_hierarchy_explicit_columns():
    """DataFrame X + DiGraph hierarchy + explicit columns."""
    _, _, columns = _canonical_setup()
    df = _canonical_dataframe()
    graph = _canonical_digraph()

    pre = HierarchicalPreprocessor(graph)
    pre.fit(df, columns=columns)
    out = pre.transform(df)

    assert np.array_equal(out.astype(int), _EXPECTED_CANONICAL)


def test_preprocessor_autoderives_columns_when_dataframe_and_digraph():
    """DataFrame X + DiGraph + columns=None: auto-derive equals explicit-columns."""
    _, _, columns = _canonical_setup()
    df = _canonical_dataframe()
    graph = _canonical_digraph()

    pre_auto = HierarchicalPreprocessor(graph)
    pre_auto.fit(df)  # no columns=
    assert pre_auto.is_fitted_

    pre_explicit = HierarchicalPreprocessor(graph)
    pre_explicit.fit(df, columns=columns)

    out_auto = pre_auto.transform(df)
    out_explicit = pre_explicit.transform(df)
    assert np.array_equal(out_auto, out_explicit)
    assert np.array_equal(out_auto.astype(int), _EXPECTED_CANONICAL)


def test_preprocessor_rejects_dataframe_with_ndarray_hierarchy_no_columns():
    """DataFrame X + ndarray hierarchy + columns=None raises a clear ValueError."""
    _, hierarchy, _ = _canonical_setup()
    df = _canonical_dataframe()

    pre = HierarchicalPreprocessor(hierarchy)
    with pytest.raises(ValueError, match="Cannot auto-derive columns"):
        pre.fit(df)


# --- Input-validation edge cases -------------------------------------------


def test_dataframe_with_unknown_column():
    """A DataFrame column absent from the graph maps to -1 and warns (ROOT add)."""
    graph = _canonical_digraph()
    df = _canonical_dataframe()
    df["unicorn"] = [False, True, False, False, True]  # not a node in graph

    pre = HierarchicalPreprocessor(graph)
    with pytest.warns(ColumnNotInHierarchyWarning):
        pre.fit(df)  # auto-derive path


def test_orphan_dataframe_column_keeps_its_name_in_feature_names_out():
    """An orphan DataFrame column (absent from the graph) is labelled with its
    own feature name in get_feature_names_out, not the "x<node>" fallback."""
    graph = _canonical_digraph()
    df = _canonical_dataframe()
    df["unicorn"] = [False, True, False, False, True]

    pre = HierarchicalPreprocessor(graph)
    with pytest.warns(ColumnNotInHierarchyWarning):
        pre.fit(df)

    names = list(pre.get_feature_names_out())
    assert "unicorn" in names
    assert not any(n.startswith("x") for n in names)
    # round-trips through set_output too
    pre2 = HierarchicalPreprocessor(graph)
    pre2.set_output(transform="pandas")
    with pytest.warns(ColumnNotInHierarchyWarning):
        pre2.fit(df)
    out = pre2.transform(df)
    assert "unicorn" in out.columns


def test_orphan_ndarray_column_falls_back_to_x_node():
    """With a nameless (ndarray) X, an orphan column has no feature name to
    recover, so it falls back to the "x<node>" label."""
    # Two columns; hierarchy only knows node 0, so column 1 is an orphan.
    X = np.array([[True, False], [False, True]])
    hierarchy = nx.to_numpy_array(nx.DiGraph([(0, 1)]))  # nodes 0, 1

    pre = HierarchicalPreprocessor(hierarchy)
    with pytest.warns(ColumnNotInHierarchyWarning):
        pre.fit(X, columns=[0, -1])

    names = list(pre.get_feature_names_out())
    assert any(n.startswith("x") for n in names)


# --- Hierarchy validation error paths -------------------------------------


def test_cyclic_hierarchy_raises_value_error():
    """A hierarchy containing a cycle fails the DAG check in fit."""
    # 0 -> 1 -> 2 -> 0 is a cycle, so no node has in-degree 0; the virtual
    # ROOT connects nothing and the cycle survives -> not a DAG.
    hierarchy = nx.to_numpy_array(nx.DiGraph([(0, 1), (1, 2), (2, 0)]))
    X = np.zeros((2, 3), dtype=bool)

    pre = HierarchicalPreprocessor(hierarchy)
    with pytest.raises(ValueError, match="not a directed acyclic graph"):
        pre.fit(X, columns=[0, 1, 2])


def test_none_hierarchy_raises_type_error_in_fit():
    """hierarchy=None reaches _set_hierarchy via the preprocessor and raises.

    The preprocessor calls _fit_hierarchy directly (bypassing the base fit's
    own None guard), so the None check inside _set_hierarchy is what fires.
    """
    X = np.zeros((2, 2), dtype=bool)
    pre = HierarchicalPreprocessor(None)
    with pytest.raises(TypeError, match="Hierarchy is None but is required"):
        pre.fit(X)


def test_invalid_hierarchy_type_raises_type_error():
    """A hierarchy that is neither ndarray nor DiGraph raises TypeError."""
    X = np.zeros((2, 2), dtype=bool)
    pre = HierarchicalPreprocessor("not a graph")
    with pytest.raises(TypeError, match="must be np.ndarray or nx.DiGraph"):
        pre.fit(X)


def test_dataframe_non_bool_dtype():
    """An int-dtype DataFrame is rejected by the bool-dtype contract."""
    graph = _canonical_digraph()
    df = _canonical_dataframe().astype(int)

    pre = HierarchicalPreprocessor(graph)
    with pytest.raises(ValueError, match="bool-dtype"):
        pre.fit(df)


def test_digraph_with_integer_node_names():
    """Integer-named DiGraph + DataFrame with matching string columns.

    sklearn coerces DataFrame column labels to str in feature_names_in_, and
    auto-derivation compares node names as strings, so str column "0" matches
    integer node 0. This is the documented way to use an int-named DiGraph with
    auto-derive.
    """
    graph = nx.DiGraph([(0, 1), (0, 2)])  # int node names
    df = pd.DataFrame({"1": [True, False], "2": [False, True]}).astype(bool)

    pre = HierarchicalPreprocessor(graph)
    pre.fit(df)  # auto-derive via str(node) match
    assert pre.is_fitted_
    out = pre.transform(df)
    # node 0 is the parent of both 1 and 2; a 1 in either child propagates to it
    assert out.shape[0] == 2


def test_digraph_with_string_node_names_dataframe_with_int_columns():
    """String-named DiGraph but int-labelled DataFrame columns.

    sklearn only records feature_names_in_ for all-string columns; integer
    column labels are NOT captured, so auto-derive does not trigger and the
    preprocessor falls back to positional 1:1 mapping (the same behaviour as
    a plain ndarray X). The mismatch therefore surfaces as positional mapping
    rather than name matching -- documented limitation: use string columns.
    """
    graph = nx.DiGraph([("dog", "cat")])
    df = pd.DataFrame({0: [True, False], 1: [False, True]}).astype(bool)

    pre = HierarchicalPreprocessor(graph)
    assert not hasattr(pre, "feature_names_in_")
    pre.fit(df)  # no feature names captured -> positional fallback, no raise
    pre.fit(df)
    assert pre.is_fitted_
    assert not hasattr(pre, "feature_names_in_")


# --- Round-trip / fit-transform consistency --------------------------------


def test_dataframe_fit_dataframe_transform():
    """fit on a DataFrame then transform on a DataFrame with the same columns."""
    df = _canonical_dataframe()
    graph = _canonical_digraph()

    pre = HierarchicalPreprocessor(graph)
    pre.fit(df)
    # feature names are preserved through fit (single validate_data)
    assert list(pre.feature_names_in_) == ["dog", "cat", "eagle"]

    out = pre.transform(df)
    assert np.array_equal(out.astype(int), _EXPECTED_CANONICAL)


def test_dataframe_transform_mismatched_columns_raises():
    """Transforming a DataFrame whose columns differ from fit raises ValueError."""
    df = _canonical_dataframe()
    graph = _canonical_digraph()

    pre = HierarchicalPreprocessor(graph)
    pre.fit(df)

    df_wrong = df.rename(columns={"dog": "wolf"})
    with pytest.raises(ValueError, match="feature names"):
        pre.transform(df_wrong)


def test_preprocessor_accepts_polars_dataframe_with_digraph():
    """polars DataFrame works via sklearn's interchange-protocol plumbing.

    scihfs imports no DataFrame library directly; polars support is inherited
    from scikit-learn (feature names via __dataframe__, conversion via
    np.asarray). Skipped when polars is not installed (no hard dependency).
    """
    pl = pytest.importorskip("polars")
    _, _, _ = _canonical_setup()
    graph = _canonical_digraph()
    df = pl.DataFrame(
        {
            "dog": [True, False, False, True, False],
            "cat": [False, True, False, False, False],
            "eagle": [False, False, True, False, True],
        }
    )

    pre = HierarchicalPreprocessor(graph)
    pre.fit(df)
    out = pre.transform(df)
    out = out.toarray() if sp.issparse(out) else np.asarray(out)
    assert np.array_equal(out.astype(int), _EXPECTED_CANONICAL)


# --- get_feature_names_out / set_output ---------------------


def test_get_feature_names_out_named_hierarchy():
    """DiGraph hierarchy: names_out are the node names in _columns order."""
    df = _canonical_dataframe()
    graph = _canonical_digraph()

    pre = HierarchicalPreprocessor(graph)
    pre.fit(df)

    names = pre.get_feature_names_out()
    assert len(names) == len(pre._columns)
    # data columns first, then the ancestor columns added during fit
    assert list(names[:3]) == ["dog", "cat", "eagle"]
    assert set(names) == {"dog", "cat", "eagle", "animal", "mammal", "bird"}


def test_get_feature_names_out_unnamed_hierarchy():
    """ndarray hierarchy: names_out are the ORIGINAL node indices (traceable).

    dog/cat/eagle are original adjacency indices 4/5/6; the ancestor columns
    added during fit are animal/mammal/bird = original indices 0/1/2. The names
    must reflect those original indices, not the post-shrink renumbering.
    """
    X, hierarchy, columns = _canonical_setup()  # columns == [4, 5, 6]

    pre = HierarchicalPreprocessor(hierarchy)
    pre.fit(X, columns=columns)

    names = pre.get_feature_names_out()
    assert len(names) == len(pre._columns)
    assert list(names[:3]) == ["4", "5", "6"]
    assert set(names) == {"4", "5", "6", "0", "1", "2"}


def test_get_feature_names_out_ndarray_matches_digraph_positions():
    """ndarray index names line up position-for-position with DiGraph names."""
    X, hierarchy, columns = _canonical_setup()
    graph = _canonical_digraph()

    pre_arr = HierarchicalPreprocessor(hierarchy)
    pre_arr.fit(X, columns=columns)
    pre_graph = HierarchicalPreprocessor(graph)
    pre_graph.fit(_canonical_dataframe())

    # Same column order; ndarray gives original indices, DiGraph gives names.
    # index 4 <-> "dog", 5 <-> "cat", 6 <-> "eagle", 0 <-> "animal", ...
    assert list(pre_arr.get_feature_names_out()) == ["4", "5", "6", "0", "1", "2"]
    assert list(pre_graph.get_feature_names_out()) == [
        "dog",
        "cat",
        "eagle",
        "animal",
        "mammal",
        "bird",
    ]


def test_set_output_pandas():
    """set_output('pandas') makes transform return a labelled DataFrame."""
    df = _canonical_dataframe()
    graph = _canonical_digraph()

    pre = HierarchicalPreprocessor(graph)
    pre.set_output(transform="pandas")
    pre.fit(df)
    out = pre.transform(df)

    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == list(pre.get_feature_names_out())
    assert list(out.columns[:3]) == ["dog", "cat", "eagle"]
    assert np.array_equal(out.to_numpy().astype(int), _EXPECTED_CANONICAL)


def test_set_output_polars():
    """set_output('polars') returns a labelled polars DataFrame (no extra code).

    polars output is inherited from scikit-learn's registered PolarsAdapter;
    scihfs imports no polars. Skipped when polars is not installed.
    """
    pl = pytest.importorskip("polars")
    df = _canonical_dataframe()
    graph = _canonical_digraph()

    pre = HierarchicalPreprocessor(graph)
    pre.set_output(transform="polars")
    pre.fit(df)
    out = pre.transform(df)

    assert isinstance(out, pl.DataFrame)
    assert out.columns == list(pre.get_feature_names_out())
    assert out.columns[:3] == ["dog", "cat", "eagle"]
    assert np.array_equal(out.to_numpy().astype(int), _EXPECTED_CANONICAL)


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


# ---------------------------------------------------------------------------
# _find_missing_columns / _adjust_node_names edge cases (permanent) and
# node-attribute survival across shrink + relabel.
# ---------------------------------------------------------------------------


def _bare_preprocessor(columns, hierarchy_graph):
    """Construct an unfitted preprocessor with the two attributes the
    fit-path helpers read/write, so they can be exercised in isolation."""
    pre = HierarchicalPreprocessor(None)
    pre._columns = columns
    pre._hierarchy_graph = hierarchy_graph
    return pre


def test_find_missing_columns_empty_columns_adds_all_non_root():
    graph = nx.DiGraph([("ROOT", "a"), ("a", "b"), ("a", "c")])
    pre = _bare_preprocessor([], graph)
    pre._find_missing_columns()
    assert pre._columns == ["a", "b", "c"]  # graph node order, ROOT excluded


def test_find_missing_columns_superset_columns_unchanged():
    graph = nx.DiGraph([("ROOT", "a"), ("a", "b")])
    columns = ["a", "b", "extra"]  # already a superset of the graph's real nodes
    pre = _bare_preprocessor(list(columns), graph)
    pre._find_missing_columns()
    assert pre._columns == columns  # nothing to add


def test_adjust_node_names_root_and_single_node():
    graph = nx.DiGraph([("ROOT", "a")])
    pre = _bare_preprocessor(["a"], graph)
    pre._adjust_node_names()
    assert pre._columns == [0]
    assert set(pre._hierarchy_graph.nodes()) == {"ROOT", 0}


def test_node_attributes_survive_shrink_and_relabel():
    """Every surviving node keeps ORIGINAL_NODE_IDENTIFIER through the full fit.

    Uses the canonical DiGraph + DataFrame: fish/trout are a dead branch
    (pruned by _shrink_dag) and animal/mammal/bird are missing ancestors
    (added by _extend_dag/_find_missing_columns), so all three rewrites run.
    get_feature_names_out / set_output rely on this attribute surviving the
    prune + relabel.
    """
    from scihfs.selectors.base import ORIGINAL_NODE_IDENTIFIER

    df = _canonical_dataframe()
    graph = _canonical_digraph()

    pre = HierarchicalPreprocessor(graph)
    pre.fit(df)

    for node, data in pre._hierarchy_graph.nodes(data=True):
        if node == "ROOT":
            continue
        assert ORIGINAL_NODE_IDENTIFIER in data, f"node {node} lost its attribute"
