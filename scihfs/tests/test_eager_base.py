"""Tests for the eager selector base class (EagerHierarchicalFeatureSelector)."""

import numpy as np
import pytest

from scihfs.selectors import EagerHierarchicalFeatureSelector, HillClimbingSelector
from scihfs.tests._estimators import _MinimalEagerSelector


@pytest.mark.parametrize(
    "abstract_class",
    [EagerHierarchicalFeatureSelector, HillClimbingSelector],
)
def test_abstract_bases_cannot_be_instantiated(abstract_class):
    with pytest.raises(TypeError, match="abstract"):
        abstract_class()


@pytest.mark.parametrize(
    "data, expected_warning",
    [
        # X has a column without a corresponding node in the hierarchy.
        ("wrong_hierarchy_X", "columns in X need to be mapped"),
        # The hierarchy has nodes without a corresponding column in X.
        ("wrong_hierarchy_X1", "hierarchy should not include any"),
    ],
)
def test_check_hierarchy_X_warns(data, expected_warning, request):
    X, hierarchy, columns = request.getfixturevalue(data)
    y = np.zeros(X.shape[0], dtype=int)
    selector = _MinimalEagerSelector(hierarchy)
    with pytest.warns(UserWarning, match=expected_warning):
        selector.fit(X, y, columns=columns)
