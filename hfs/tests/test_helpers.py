import os
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from ..helpers import (
    compute_aggregated_values,
    connect_dag,
    create_hierarchy,
    getRelevance,
    shrink_dag,
)
from ..metrics import gain_ratio, information_gain


def test_shrink_dag():
    dirname = os.path.dirname(__file__)
    nodes = np.load(os.path.join(dirname, "../data/nodes_go.npy"))
    graph = nx.read_gml(os.path.join(dirname, "../data/go_digraph.gml"))

    nonexist_nodes = [
        "GO:2001301",
        "GO:2001302",
        "GO:2001303",
        "GO:2001304",
        "GO:2001305",
        "GO:2001306",
        "GO:2001307",
        "GO:2001308",
        "GO:2001309",
        "GO:2001310",
        "GO:2001311",
        "GO:2001312",
        "GO:2001313",
        "GO:2001314",
        "GO:2001315",
        "GO:2001316",
        "GO:2001092",
        "GO:2001094",
        "GO:2001106",
        "GO:2001107",
    ]  # 4 nodes that are leaves

    x_identifiers = np.setdiff1d(nodes, nonexist_nodes)

    # four nodes in non_existing_nodes can be removed as they are leaves
    assert (
        len(
            [
                x
                for x in nonexist_nodes
                if graph.out_degree(x) == 0 and graph.in_degree(x) == 1
            ]
        )
        == 4
    )

    # test removal of the nodes
    assert len(graph.nodes()) == 43008
    graph = shrink_dag(x_identifiers, graph)
    assert len(graph.nodes()) == 43004
    for node in ["GO:2001092", "GO:2001094", "GO:2001106", "GO:2001107"]:
        assert not (node in graph.nodes())


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
        value = getRelevance(train_x_data, train_y_data, node_idx)
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
    hierarchy = create_hierarchy(nx.DiGraph(hierarchy))
    X_transformed = compute_aggregated_values(X, hierarchy, columns)
    assert np.array_equal(X_transformed, result)
