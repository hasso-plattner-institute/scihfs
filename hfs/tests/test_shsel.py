import numpy as np
import pytest

from hfs.selectors import SHSELSelector


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


@pytest.mark.parametrize(
    "data, result",
    [
        ("data1", "result_shsel_hfe1"),
        ("data2", "result_shsel_hfe2"),
        ("data4", "result_shsel_hfe4"),
    ],
)
def test_leaf_filtering(data, result, request):
    data = request.getfixturevalue(data)
    result = request.getfixturevalue(result)
    X, y, hierarchy, columns = data
    expected, support = result
    selector = SHSELSelector(
        hierarchy, use_hfe_extension=True, relevance_metric="Correlation"
    )
    selector.fit(X, y, columns)
    X = selector.transform(X)
    assert np.array_equal(X, expected)

    support_mask = selector.get_support()
    assert np.array_equal(support_mask, support)


def test_fail_on_invalid_relevance_metric(data1):
    X, y, _, _ = data1
    selector = SHSELSelector(use_hfe_extension=True, relevance_metric="IG")
    with pytest.raises(ValueError):
        selector.fit(X, y)
