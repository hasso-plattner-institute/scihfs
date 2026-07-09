import networkx as nx
import numpy as np
import pandas as pd
import pytest

from scihfs.selectors import TSELSelector


@pytest.mark.parametrize(
    "data, result",
    [
        ("data1", "result_tsel1"),
        ("data2", "result_tsel2"),
        ("data3", "result_tsel3"),
        ("data1_2", "result_tsel1"),
    ],
)
def test_TSEL_selection(data, result, request):
    data = request.getfixturevalue(data)
    result = request.getfixturevalue(result)
    X, y, hierarchy, columns = data
    expected, support = result
    selector = TSELSelector(hierarchy)
    selector.fit(X, y, columns)
    X = selector.transform(X)
    assert np.array_equal(X, expected)

    support_mask = selector.get_support()
    assert np.array_equal(support_mask, support)


def test_TSEL_autoderives_columns_from_dataframe(data1, result_tsel1):
    """Smoke test: DataFrame X + named DiGraph + columns=None equals the
    explicit-columns run (see the SHSEL test suite for the parametrized version)."""
    X, y, hierarchy, columns = data1
    expected, support = result_tsel1
    graph = nx.from_numpy_array(hierarchy, create_using=nx.DiGraph)
    df = pd.DataFrame(X, columns=[str(node) for node in columns])

    selector = TSELSelector(graph)
    selector.fit(df, y)  # no columns=

    assert np.array_equal(selector.transform(df), expected)
    assert np.array_equal(selector.get_support(), support)
