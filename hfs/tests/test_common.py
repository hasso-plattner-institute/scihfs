import pytest
from sklearn.utils.estimator_checks import check_estimator

from hfs import (
    EagerHierarchicalFeatureSelector,
    HierarchicalEstimator,
    HierarchicalPreprocessor,
)

from ..selectors.gtd import GreedyTopDownSelector
from ..selectors.hiertan import HierTan
from ..selectors.hill_climbing import BottomUpSelector, TopDownSelector
from ..selectors.hip import HIP
from ..selectors.hnb import HNB
from ..selectors.hnbs import HNBs
from ..selectors.mr import MR
from ..selectors.rnb import RNB
from ..selectors.shsel import SHSELSelector
from ..selectors.tsel import TSELSelector


@pytest.mark.parametrize(
    "estimator",
    [
        TSELSelector(),
        HierarchicalEstimator(),
        EagerHierarchicalFeatureSelector(),
        HierarchicalPreprocessor(),
        TopDownSelector(),
        SHSELSelector(),
        HNB(),
        HNBs(),
        RNB(),
        MR(),
        HIP(),
        BottomUpSelector(),
        GreedyTopDownSelector(),
        HierTan(),
    ],
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
