"""Estimator groupings used across the tests in this repository.

In order to route the estimators to the correct tests, they are sorted based on
their properties.

- ``ALL_ESTIMATORS`` - they expose the full sklearn estimator surface, and can
    be run through the sklearn conformance test suite. (Union of the eager and
    lazy selectors, plus the base classes and the preprocessor.)
- ``EAGER_SELECTORS`` - single pass fit().
- ``LAZY_SELECTORS`` - fit_selector() with both X_train and X_test to fit per
    test instance.

Future work: Divide by filter/wrapper/embedded feature selection methods when
the embedded methods also inherit from ClassifierMixin.
"""

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
    TAN,
    BottomUpSelector,
    GreedyTopDownSelector,
    HieAODE,
    HNBs,
    SHSELSelector,
    TopDownSelector,
    TSELSelector,
)

ALL_ESTIMATORS = [
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
]

EAGER_SELECTORS = [
    SHSELSelector,
    TSELSelector,
    GreedyTopDownSelector,
    TopDownSelector,
    BottomUpSelector,
]

LAZY_SELECTORS = [HIP, HNB, HNBs, RNB, MR, TAN, HieAODE]
