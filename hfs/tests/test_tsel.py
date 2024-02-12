import numpy as np
import pytest

from hfs.selectors import TSELSelector

from .fixtures.fixtures import (
    data1,
    data1_2,
    data2,
    data3,
    result_tsel1,
    result_tsel2,
    result_tsel3,
)


@pytest.mark.parametrize(
    "data, result",
    [
        (data1(), result_tsel1()),
        (data2(), result_tsel2()),
        (data3(), result_tsel3()),
        (data1_2(), result_tsel1()),
    ],
)
def test_TSEL_selection(data, result):
    X, y, hierarchy, columns = data
    expected, support = result
    selector = TSELSelector(hierarchy)
    selector.fit(X, y, columns)
    X = selector.transform(X)
    assert np.array_equal(X, expected)

    support_mask = selector.get_support()
    assert np.array_equal(support_mask, support)
