import networkx as nx
import pytest
from sklearn.utils.estimator_checks import check_estimator

from scihfs import (
    EagerHierarchicalFeatureSelector,
    HierarchicalEstimator,
    HierarchicalPreprocessor,
)
from scihfs.selectors import (
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
        pytest.param(
            HierarchicalPreprocessor,
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "HierarchicalPreprocessor now requires bool-dtype input; "
                    "sklearn's check_estimator suite feeds float arrays. "
                    "Pending sum-propagation mode that will reopen numeric inputs."
                ),
            ),
        ),
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
