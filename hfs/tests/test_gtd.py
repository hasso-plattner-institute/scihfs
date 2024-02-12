import numpy as np
import pytest

from hfs.selectors import GreedyTopDownSelector

from .fixtures.fixtures import (
    data2,
    data2_1,
    data2_2,
    result_gtd_selection2,
    result_gtd_selection2_1,
    result_gtd_selection2_2,
)


@pytest.mark.parametrize(
    "data, result",
    [(data2(), result_gtd_selection2()), (data2_1(), result_gtd_selection2_1())],
)
def test_greedy_top_down_selection(data, result):
    X, y, hierarchy, columns = data
    expected, support = result
    selector = GreedyTopDownSelector(hierarchy)
    selector.fit(X, y, columns)
    X = selector.transform(X)
    assert np.array_equal(X, expected)

    support_mask = selector.get_support()
    assert np.array_equal(support_mask, support)


@pytest.mark.parametrize(
    "data, result_redundant, result_not_redundant",
    [(data2_2(), result_gtd_selection2_1(), result_gtd_selection2_2())],
)
def test_greedy_top_down_selection_dag(data, result_redundant, result_not_redundant):
    X, y, hierarchy, columns = data
    expected_redundant, support_redundant = result_redundant
    selector = GreedyTopDownSelector(hierarchy)
    selector.fit(X, y, columns)
    X_transformed = selector.transform(X)
    assert np.array_equal(X_transformed, expected_redundant)

    support_mask = selector.get_support()
    assert np.array_equal(support_mask, support_redundant)

    expected_not_redundant, support_not_redundant = result_not_redundant
    selector2 = GreedyTopDownSelector(hierarchy, iterate_first_level=False)
    selector2.fit(X, y, columns)
    X_transformed2 = selector2.transform(X)
    assert np.array_equal(X_transformed2, expected_not_redundant)

    support_mask2 = selector2.get_support()
    assert np.array_equal(support_mask2, support_not_redundant)
