"""Estimator groupings used across the tests in this repository.

In order to route the estimators to the correct tests, they are sorted based on
their properties.

- ``ALL_ESTIMATORS`` - they expose the full sklearn estimator surface, and can
    be run through the sklearn conformance test suite. (Union of the eager and
    lazy selectors, plus the base classes and the preprocessor. The abstract
    eager base is represented by the minimal concrete stub
    ``_MinimalEagerSelector``.)
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


class _MinimalEagerSelector(EagerHierarchicalFeatureSelector):
    """Minimal concrete eager selector for testing the abstract eager base.

    ``EagerHierarchicalFeatureSelector`` cannot be instantiated directly
    (``_select`` is abstract); this stub implements the hook as "select
    nothing" so the base's estimator surface can still be exercised.
    Might be moved to the eager base's test module or removed entirely in the future.
    """

    def _select(self, X, y):
        pass


ALL_ESTIMATORS = [
    TSELSelector,
    HierarchicalEstimator,
    _MinimalEagerSelector,
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
