import networkx as nx
import numpy as np
import pandas as pd
import pytest

from scihfs.helpers import get_columns_for_numpy_hierarchy
from scihfs.selectors import SHSELSelector


@pytest.mark.parametrize(
    "data, result",
    [
        ("data1", "result_shsel1"),
        ("data2", "result_shsel2"),
        ("data3", "result_shsel3"),
        ("data1_2", "result_shsel1"),
    ],
)
def test_SHSEL_selection(data, result, request):
    data = request.getfixturevalue(data)
    result = request.getfixturevalue(result)
    X, y, hierarchy, columns = data
    expected, support = result
    selector = SHSELSelector(hierarchy)
    selector.fit(X, y, columns)
    X = selector.transform(X)
    assert np.array_equal(X, expected)

    support_mask = selector.get_support()
    assert np.array_equal(support_mask, support)


@pytest.mark.parametrize(
    "data, result",
    [
        ("data_shsel_selection", "result_shsel_selection"),
        ("data1", "result_shsel1"),
        ("data1_2", "result_shsel1"),
    ],
)
def test_SHSEL_selection_with_initial_selection(data, result, request):
    data = request.getfixturevalue(data)
    result = request.getfixturevalue(result)
    X, y, hierarchy, columns = data
    expected, support = result
    selector = SHSELSelector(hierarchy, similarity_threshold=0.8)
    selector.fit(X, y, columns)
    X = selector.transform(X)
    assert np.array_equal(X, expected)

    support_mask = selector.get_support()
    assert np.array_equal(support_mask, support)


# HFE extension disabled; the corresponding tests are
# commented out (with their result_shsel_hfe* fixtures in conftest.py) and
# kept for re-enablement when the extension returns.
# @pytest.mark.parametrize(
#     "data, result",
#     [
#         ("data1", "result_shsel_hfe1"),
#         ("data2", "result_shsel_hfe2"),
#         ("data4", "result_shsel_hfe4"),
#     ],
# )
# def test_leaf_filtering(data, result, request):
#     data = request.getfixturevalue(data)
#     result = request.getfixturevalue(result)
#     X, y, hierarchy, columns = data
#     expected, support = result
#     selector = SHSELSelector(
#         hierarchy, use_hfe_extension=True, relevance_metric="Correlation"
#     )
#     selector.fit(X, y, columns)
#     X = selector.transform(X)
#     assert np.array_equal(X, expected)
#
#     support_mask = selector.get_support()
#     assert np.array_equal(support_mask, support)
#
#
# def test_fail_on_invalid_relevance_metric(data1):
#     X, y, _, _ = data1
#     selector = SHSELSelector(use_hfe_extension=True, relevance_metric="IG")
#     with pytest.raises(ValueError):
#         selector.fit(X, y)


def test_SHSEL_ig_relevance_is_normalized(
    data_shsel_normalization, result_shsel_normalization
):
    # SHSEL normalizes the IG relevance to [0, 1] (RapidMiner-style) so that the
    # absolute similarity_threshold matches Ristoski & Paulheim (2014).
    X, y, hierarchy, columns = data_shsel_normalization
    expected, support = result_shsel_normalization

    selector = SHSELSelector(hierarchy, similarity_threshold=0.7)
    selector.fit(X, y, columns)

    # The relevance is normalized: the most relevant feature has relevance 1.0.
    assert max(selector._relevance_values.values()) == pytest.approx(1.0)

    assert np.array_equal(selector.transform(X), expected)
    assert np.array_equal(selector.get_support(), support)


def test_SHSEL_ig_average_modes(
    data_shsel_ig_average,
    result_shsel_ig_average_full_path,
    result_shsel_ig_average_survivors,
):
    # The pruning per-path average can be taken over the whole path ("full_path",
    # the paper's Algorithm 2 line 8 and the default) or over surviving features
    # only ("survivors_only"). Nodes 2 and 3 are removed in initial selection;
    # counting their IG in node 0's path averages drops node 0 (full_path), while
    # ignoring them keeps it (survivors_only).
    X, y, hierarchy, columns = data_shsel_ig_average
    full_expected, full_support = result_shsel_ig_average_full_path
    surv_expected, surv_support = result_shsel_ig_average_survivors

    default = SHSELSelector(hierarchy, similarity_threshold=0.6)
    default.fit(X, y, columns)
    assert default.ig_average == "full_path"
    assert np.array_equal(default.transform(X), full_expected)
    assert np.array_equal(default.get_support(), full_support)

    survivors = SHSELSelector(
        hierarchy, similarity_threshold=0.6, ig_average="survivors_only"
    )
    survivors.fit(X, y, columns)
    assert np.array_equal(survivors.transform(X), surv_expected)
    assert np.array_equal(survivors.get_support(), surv_support)

    assert not np.array_equal(default.get_support(), survivors.get_support())


def test_SHSEL_rejects_unknown_ig_average(data2):
    X, y, hierarchy, columns = data2
    selector = SHSELSelector(hierarchy, ig_average="bogus")
    with pytest.raises(ValueError, match="ig_average"):
        selector.fit(X, y, columns)


def test_SHSEL_threshold_none_resolves_per_metric(data_shsel_normalization):
    # similarity_threshold=None resolves to the paper's metric-specific default:
    # 0.99 for information gain, 0.6 for correlation. An explicit value is kept.
    X, y, hierarchy, columns = data_shsel_normalization

    ig = SHSELSelector(hierarchy)
    ig.fit(X, y, columns)
    assert ig._effective_threshold == 0.99

    correlation = SHSELSelector(hierarchy, relevance_metric="Correlation")
    correlation.fit(X, y, columns)
    assert correlation._effective_threshold == 0.6

    explicit = SHSELSelector(hierarchy, similarity_threshold=0.85)
    explicit.fit(X, y, columns)
    assert explicit._effective_threshold == 0.85


def test_SHSEL_pruning_toggle(data_shsel_ig_average):
    # pruning=False applies only the initial selection (the paper's initialSHSEL);
    # the default pruning=True also runs Algorithm 2 (pruneSHSEL), removing more.
    X, y, hierarchy, columns = data_shsel_ig_average

    no_prune = SHSELSelector(hierarchy, similarity_threshold=0.6, pruning=False)
    no_prune.fit(X, y, columns)
    assert np.array_equal(no_prune.get_support(), np.array([True, True, False, False]))

    pruned = SHSELSelector(hierarchy, similarity_threshold=0.6)
    pruned.fit(X, y, columns)
    assert pruned.pruning is True
    assert np.array_equal(pruned.get_support(), np.array([False, True, False, False]))

    # Pruning never selects more features than the initial selection alone.
    assert pruned.get_support().sum() <= no_prune.get_support().sum()


def test_SHSEL_selection_correlation_on_bool():
    """The non-HFE (Pearson) Correlation path works on bool input."""
    edges = [(0, 1), (1, 2), (0, 3)]
    hierarchy = nx.to_numpy_array(nx.DiGraph(edges))
    columns = get_columns_for_numpy_hierarchy(nx.DiGraph(edges), 4)
    X = np.array(
        [[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 0, 1], [1, 1, 0, 0], [0, 0, 0, 0]],
        dtype=bool,
    )
    y = np.array([1, 0, 0, 1, 0])

    selector = SHSELSelector(
        hierarchy, relevance_metric="Correlation", similarity_threshold=0.8
    )
    selector.fit(X, y, columns)
    X_transformed = selector.transform(X)

    assert np.array_equal(selector.get_support(), np.array([False, True, True, True]))
    expected = np.array(
        [[1, 1, 0], [1, 0, 1], [0, 0, 1], [1, 0, 0], [0, 0, 0]], dtype=bool
    )
    assert np.array_equal(X_transformed, expected)


@pytest.mark.parametrize(
    "data, result",
    [
        ("data1", "result_shsel1"),
        ("data2", "result_shsel2"),
        ("data3", "result_shsel3"),
        ("data1_2", "result_shsel1"),
    ],
)
def test_SHSEL_autoderives_columns_from_dataframe(data, result, request):
    """DataFrame X + named DiGraph + columns=None equals the explicit-columns run.

    The fixtures' ndarray hierarchies have the integer positions 0..n-1 as
    nodes, so naming each DataFrame column after its mapped node must
    reproduce the explicit ``columns`` mapping through the auto-derive path.
    """
    X, y, hierarchy, columns = request.getfixturevalue(data)
    expected, support = request.getfixturevalue(result)
    graph = nx.from_numpy_array(hierarchy, create_using=nx.DiGraph)
    df = pd.DataFrame(X, columns=[str(node) for node in columns])

    selector = SHSELSelector(graph)
    selector.fit(df, y)  # no columns=

    assert selector.get_columns() == list(columns)
    assert np.array_equal(selector.transform(df), expected)
    assert np.array_equal(selector.get_support(), support)
