from fractions import Fraction

import networkx as nx
import numpy as np
import pytest
from info_gain.info_gain import info_gain, info_gain_ratio

from scihfs.helpers import (
    add_virtual_root_node,
    compute_aggregated_values,
    get_relevance,
    shrink_dag,
)
from scihfs.metrics import gain_ratio, information_gain


def test_shrink_dag():
    edges = [(0, 1), (0, 2), (0, 4), (3, 4), (3, 5), (6, 1), (6, 4)]
    graph = nx.DiGraph(edges)
    relevant_nodes = [1]
    nodes_to_remove = [2, 3, 4, 5]

    assert len(graph.nodes()) == 7
    graph = shrink_dag(relevant_nodes, graph)
    assert len(graph.nodes()) == 3
    assert all(node not in graph.nodes() for node in nodes_to_remove)


# ---------------------------------------------------------------------------
# shrink_dag: edge cases (permanent) pinning the new ancestor-walk behaviour.
# ---------------------------------------------------------------------------


def _rooted_dag(edges):
    """Build a DiGraph from edges and attach the virtual ROOT above sources."""
    return add_virtual_root_node(nx.DiGraph(edges))


def test_shrink_dag_in_place_mutation_and_return_identity():
    """shrink_dag mutates its input AND returns the same object. This behaviour will be removed in the future."""
    graph = _rooted_dag([(0, 1), (0, 2)])
    result = shrink_dag([1], graph)
    assert result is graph
    assert 2 not in graph.nodes()


def test_shrink_dag_empty_identifiers_keeps_root_only():
    """No relevant identifiers: every real node is a dead branch; ROOT survives."""
    graph = _rooted_dag([(0, 1), (0, 2), (1, 3)])
    shrink_dag([], graph)
    assert set(graph.nodes()) == {"ROOT"}


def test_shrink_dag_all_nodes_identifiers_prunes_nothing():
    """Every node is relevant: output equals input (no node removed)."""
    graph = _rooted_dag([(0, 1), (0, 2), (1, 3)])
    before = set(graph.nodes())
    shrink_dag([0, 1, 2, 3], graph)
    assert set(graph.nodes()) == before


def test_shrink_dag_single_root_only_passthrough():
    """A graph that is just ROOT passes through untouched."""
    graph = nx.DiGraph()
    graph.add_node("ROOT")
    shrink_dag([], graph)
    assert set(graph.nodes()) == {"ROOT"}


def test_shrink_dag_keeps_interior_identifier_and_its_subtree_ancestors():
    """An interior (non-leaf) relevant node survives along with all its ancestors.

    nx.ancestors excludes the node itself, so the union with relevant_nodes
    is what keeps the node when it happens to be a leaf; here we also
    confirm an interior relevant node keeps its ancestor chain up to ROOT.
    """
    graph = _rooted_dag([(0, 1), (1, 2), (1, 3), (0, 4)])
    shrink_dag([1], graph)
    # 1 (relevant) + 0 (ancestor) + ROOT survive; 2, 3 (descendants) and the
    # unrelated branch 4 are pruned.
    assert set(graph.nodes()) == {"ROOT", 0, 1}


def test_relevance(lazy_data2):
    small_DAG, train_x_data, train_y_data, test_x_data, test_y_data = lazy_data2
    results = [Fraction(1, 2), Fraction(8, 9), 2, 0]
    for node_idx in range(len(small_DAG)):
        value = get_relevance(train_x_data, train_y_data, node_idx)
        assert value == results[node_idx]


def test_information_gain(data2):
    X, y, _, _ = data2
    ig = information_gain(X, y)
    ig_expected = [round(info_gain(X[:, i], y), 6) for i in range(len(X))]
    assert ig == ig_expected


def test_gain_ratio(data2):
    X, y, _, _ = data2
    gr = gain_ratio(X, y)
    gr_expected = [info_gain_ratio(X[:, i], y) for i in range(len(X))]
    assert gr == gr_expected


@pytest.mark.parametrize(
    "data, result",
    [
        ("data1", "result_aggregated1"),
        ("data2", "result_aggregated2"),
    ],
)
def test_compute_aggregated_values(data, result, request):
    data = request.getfixturevalue(data)
    result = request.getfixturevalue(result)
    X, _, hierarchy, columns = data
    # Contract: the input is bool (binary features). Aggregation must COUNT the
    # 'True' values per subtree, returning a compact uint32 count array.
    assert X.dtype == np.bool_
    hierarchy = add_virtual_root_node(nx.DiGraph(hierarchy))
    X_transformed = compute_aggregated_values(X, hierarchy, columns)
    assert X_transformed.dtype == np.uint32
    assert np.array_equal(X_transformed, result)
    # The bool input is left untouched (a fresh integer working copy is built).
    assert X.dtype == np.bool_
