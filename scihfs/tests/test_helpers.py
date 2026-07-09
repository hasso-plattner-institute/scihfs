from collections import Counter
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest
from scipy.stats import entropy

from scihfs.helpers import (
    add_virtual_root_node,
    compute_aggregated_values,
    get_relevance,
    shrink_dag,
)
from scihfs.metrics import conditional_mutual_information, gain_ratio, information_gain

# ---------------------------------------------------------------------------
# Independent reference oracles for the information-gain and gain ratio metrics, reproducing the
# algorithm of the (now-removed) unmaintained ``info_gain`` package. Kept local
# to the test so it pins scihfs.metrics against an external definition.

# Original code: Copyright (c) 2018 Thijsvanede under MIT License.


def _reference_information_gain(examples: np.ndarray, attribute: np.ndarray) -> float:
    examples_entropy = entropy(list(Counter(examples).values()))
    conditional_entropy = 0.0
    for value in set(attribute):
        subset = [e for e, a in zip(examples, attribute) if a == value]
        probability = len(subset) / len(examples)
        conditional_entropy += probability * entropy(list(Counter(subset).values()))
    return examples_entropy - conditional_entropy


def _reference_information_gain_ratio(feature: np.ndarray, target: np.ndarray) -> float:
    intrinsic_value = entropy(list(Counter(feature).values()))
    if intrinsic_value == 0:
        return 0.0
    return _reference_information_gain(feature, target) / intrinsic_value


# ---------------------------------------------------------------------------


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
    ig_expected = [
        round(_reference_information_gain(X[:, i], y), 6) for i in range(len(X))
    ]
    assert ig == ig_expected


def test_gain_ratio(data2):
    X, y, _, _ = data2
    gr = gain_ratio(X, y)
    # Oracle is the scipy/Counter reference above -- deliberately independent of
    # the sklearn.mutual_info_score used in gain_ratio, so this is a genuine
    # cross-check. The two implementations agree only to floating-point noise.
    gr_expected = [_reference_information_gain_ratio(X[:, i], y) for i in range(len(X))]
    assert gr == pytest.approx(gr_expected)


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


# ---------------------------------------------------------------------------
# External oracle for the implementation of conditional mutual
# information is the actively maintained ``dit`` package
# (dev-only dependency). dit requires Python 3.11+, so the
# test is skipped on older interpreters.
# ---------------------------------------------------------------------------


def test_conditional_mutual_information(data2):
    dit = pytest.importorskip("dit")
    from dit.multivariate import coinformation

    def reference_cmi(x, y, z):
        """I(X;Y|Z) in bits via dit, from the empirical (ML) joint distribution."""
        outcome_counts = Counter(zip(x.tolist(), y.tolist(), z.tolist()))
        distribution = dit.Distribution(
            list(outcome_counts), [c / len(x) for c in outcome_counts.values()]
        )
        return coinformation(distribution, [[0], [1]], [2])

    # All ordered feature pairs of the real fixture data, conditioned on the target
    # (i == j included: CMI(X;X|Z) is the conditional entropy H(X|Z)).
    X, y, _, _ = data2
    cases = [(X[:, i], X[:, j], y) for i in range(X.shape[1]) for j in range(X.shape[1])]

    # Degenerate inputs: constants, identical/complementary features, single sample.
    ones = np.ones(10, dtype=int)
    alternating = np.array([0, 1] * 5)
    cases += [
        (ones, ones, ones),
        (alternating, alternating, ones),
        (alternating, 1 - alternating, ones),
        (alternating, ones, alternating),
        (np.array([0]), np.array([1]), np.array([2])),
    ]

    # Seeded random draws over varying sample counts, alphabet sizes, and dtypes.
    rng = np.random.default_rng(42)
    for _ in range(25):
        n = int(rng.integers(2, 60))
        node1 = rng.integers(0, int(rng.integers(2, 5)), n)
        node2 = rng.integers(0, int(rng.integers(2, 5)), n)
        target = rng.integers(0, int(rng.integers(2, 4)), n)
        if rng.random() < 0.5:
            node1 = (node1 % 2).astype(bool)
            node2 = (node2 % 2).astype(bool)
        cases.append((node1, node2, target))

    for node1, node2, target in cases:
        expected = reference_cmi(node1, node2, target)
        assert conditional_mutual_information(node1, node2, target) == pytest.approx(
            expected, abs=1e-12
        )
