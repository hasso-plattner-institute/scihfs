import pytest

from hfs.selectors import EagerHierarchicalFeatureSelector

from .fixtures.fixtures import wrong_hierarchy_X, wrong_hierarchy_X1


@pytest.mark.parametrize(
    "data",
    [wrong_hierarchy_X(), wrong_hierarchy_X1()],
)
def test_EagerHierarchicalFeatureSelector(data):
    X, hierarchy, columns = data
    selector = EagerHierarchicalFeatureSelector(hierarchy)
    with pytest.warns(UserWarning):
        selector.fit(X, columns=columns)
