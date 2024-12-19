from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from hfs.helpers import (
    add_virtual_root_node,
    compute_aggregated_values,
    connect_dag,
    get_relevance,
    shrink_dag,
)
from hfs.metrics import gain_ratio, information_gain


def test_shrink_dag():
    edges = [(0, 1), (0, 2), (0, 4), (3, 4), (3, 5), (6, 1), (6, 4)]
    graph = nx.DiGraph(edges)
    x_identifiers = [1]
    nodes_to_remove = [2, 3, 4, 5]

    assert len(graph.nodes()) == 7
    graph = shrink_dag(x_identifiers, graph)
    assert len(graph.nodes()) == 3
    assert all(node not in graph.nodes() for node in nodes_to_remove)


def test_connect_dag(lazy_data4):
    small_DAG, big_DAG = lazy_data4
    graph = nx.DiGraph(big_DAG)
    x_identifiers = [0, 1, 2, 5, 6, 7, 8]
    graph = connect_dag(hierarchy=graph, x_identifiers=x_identifiers)
    new_graph = nx.DiGraph([(0, 1), (0, 2), (1, 6), (1, 5), (1, 7), (0, 7), (5, 8)])
    assert nx.is_isomorphic(graph, new_graph)


def test_relevance(lazy_data2):
    small_DAG, train_x_data, train_y_data, test_x_data, test_y_data = lazy_data2
    results = [Fraction(1, 2), Fraction(8, 9), 2, 0]
    for node_idx in range(len(small_DAG)):
        value = get_relevance(train_x_data, train_y_data, node_idx)
        assert value == results[node_idx]


def test_information_gain(data2, result_ig_values2):
    X, y, _, _ = data2
    ig = information_gain(X, y)
    assert ig == result_ig_values2


def test_gain_ratio(data2, result_gr_values2):
    X, y, _, _ = data2
    gr = gain_ratio(X, y)
    assert gr == result_gr_values2


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
    hierarchy = add_virtual_root_node(nx.DiGraph(hierarchy))
    X_transformed = compute_aggregated_values(X, hierarchy, columns)
    assert np.array_equal(X_transformed, result)
