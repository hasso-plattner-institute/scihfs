import networkx as nx
import pytest
from sklearn.utils.estimator_checks import check_estimator

from hfs import (
    EagerHierarchicalFeatureSelector,
    HierarchicalEstimator,
    HierarchicalPreprocessor,
)
from hfs.selectors import (
    HIP,
    HNB,
    MR,
    RNB,
    BottomUpSelector,
    GreedyTopDownSelector,
    HNBs,
    SHSELSelector,
    TopDownSelector,
    TSELSelector,
)


@pytest.mark.parametrize(
    "estimator",
    [
        TSELSelector,
        HierarchicalEstimator,
        EagerHierarchicalFeatureSelector,
        HierarchicalPreprocessor,
        TopDownSelector,
        SHSELSelector,
        HNB,
        HNBs,
        RNB,
        MR,
        HIP,
        BottomUpSelector,
        GreedyTopDownSelector,
    ],
)
def test_all_estimators(estimator):
    hierarchy_graph = nx.DiGraph()
    adj_matrix = nx.to_numpy_array(hierarchy_graph)
    return check_estimator(estimator(adj_matrix))
