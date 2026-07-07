import numpy as np
import pytest

from scihfs.selectors import GreedyTopDownSelector


@pytest.mark.parametrize(
    "data, result",
    [("data2", "result_gtd_selection2"), ("data2_1", "result_gtd_selection2_1")],
)
def test_greedy_top_down_selection(data, result, request):
    data = request.getfixturevalue(data)
    result = request.getfixturevalue(result)
    X, y, hierarchy, columns = data
    expected, support = result
    selector = GreedyTopDownSelector(hierarchy)
    selector.fit(X, y, columns)
    X = selector.transform(X)
    assert np.array_equal(X, expected)

    support_mask = selector.get_support()
    assert np.array_equal(support_mask, support)


def test_greedy_top_down_default_heuristic_is_gain_ratio(data2, result_gtd_selection2):

    X, y, hierarchy, columns = data2
    expected, support = result_gtd_selection2

    default = GreedyTopDownSelector(hierarchy)
    default.fit(X, y, columns)
    assert default.heuristic_function == "GR"
    assert np.array_equal(default.transform(X), expected)
    assert np.array_equal(default.get_support(), support)

    explicit = GreedyTopDownSelector(hierarchy, heuristic_function="GR")
    explicit.fit(X, y, columns)
    assert np.array_equal(explicit.get_support(), default.get_support())


def test_greedy_top_down_information_gain_heuristic(
    data_gtd_heuristic, result_gtd_heuristic_gr, result_gtd_heuristic_ig
):
    # On data_gtd_heuristic the two metrics rank the parent/child oppositely, so
    # the heuristic_function switch produces genuinely different selections.
    X, y, hierarchy, columns = data_gtd_heuristic
    gr_expected, gr_support = result_gtd_heuristic_gr
    ig_expected, ig_support = result_gtd_heuristic_ig

    gr_selector = GreedyTopDownSelector(hierarchy, heuristic_function="GR")
    gr_selector.fit(X, y, columns)
    assert np.array_equal(gr_selector.transform(X), gr_expected)
    assert np.array_equal(gr_selector.get_support(), gr_support)

    ig_selector = GreedyTopDownSelector(hierarchy, heuristic_function="IG")
    ig_selector.fit(X, y, columns)
    assert np.array_equal(ig_selector.transform(X), ig_expected)
    assert np.array_equal(ig_selector.get_support(), ig_support)

    # The switch must actually change the outcome here.
    assert not np.array_equal(gr_selector.get_support(), ig_selector.get_support())


def test_greedy_top_down_rejects_unknown_heuristic(data2):
    X, y, hierarchy, columns = data2
    selector = GreedyTopDownSelector(hierarchy, heuristic_function="nope")
    with pytest.raises(ValueError, match="heuristic_function"):
        selector.fit(X, y, columns)


def test_greedy_top_down_selection_dag(data2_2, result_gtd_selection2_2):
    # data2_2 is a DAG (node 3 has parents 1 and 4). Under the gain-ratio
    # heuristic, node 3 outranks its ancestor 4, so 4 is pruned; both traversal
    # modes therefore converge on the same selection {2, 3}.
    X, y, hierarchy, columns = data2_2
    expected, support = result_gtd_selection2_2

    selector = GreedyTopDownSelector(hierarchy)
    selector.fit(X, y, columns)
    assert np.array_equal(selector.transform(X), expected)
    assert np.array_equal(selector.get_support(), support)

    selector2 = GreedyTopDownSelector(hierarchy, iterate_first_level=False)
    selector2.fit(X, y, columns)
    assert np.array_equal(selector2.transform(X), expected)
    assert np.array_equal(selector2.get_support(), support)
