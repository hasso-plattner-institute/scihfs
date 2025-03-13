import pytest

from hfs.selectors.base import HierarchicalEstimator


def test_empty_hierarchy_raises_error():
    with pytest.raises(TypeError):
        estimator = HierarchicalEstimator()
        estimator.fit(None)
