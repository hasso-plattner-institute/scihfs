import networkx as nx
import numpy as np
import pytest

from scihfs.tests._estimators import EAGER_SELECTORS, LAZY_SELECTORS

# ---------------------------------------------------------------------------
# TEMPORARY FILE CONTENT WARNING:
#
# At the moment, the scope of this file is only to test the rejection of
# non-bool-dtype input to the selectors. All sklearn-related tests are in
# test_sklearn_conformance.py for now.
#
# In the future, this file should contain all those tests that apply to
# ALL estimators alike. Right now, a lot of these are scattered throughout
# the respective test files for each estimator.
# ---------------------------------------------------------------------------

_HIERARCHY = nx.to_numpy_array(nx.DiGraph([(0, 1), (1, 2), (0, 3)]))
_X_BOOL = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 0]], dtype=bool)
_Y = np.array([0, 1, 0, 1])
_REJECTED_DTYPES = [np.int8, np.int32, np.int64, np.float32, np.float64]


@pytest.mark.parametrize("dtype", _REJECTED_DTYPES)
@pytest.mark.parametrize("Selector", EAGER_SELECTORS)
def test_eager_selector_rejects_non_bool_X(Selector, dtype):
    selector = Selector(_HIERARCHY)
    with pytest.raises(ValueError, match="bool-dtype"):
        selector.fit(_X_BOOL.astype(dtype), _Y)


@pytest.mark.parametrize("dtype", _REJECTED_DTYPES)
@pytest.mark.parametrize("Selector", LAZY_SELECTORS)
def test_lazy_selector_fit_rejects_non_bool_X(Selector, dtype):
    selector = Selector(_HIERARCHY)
    with pytest.raises(ValueError, match="bool-dtype"):
        selector.fit(_X_BOOL.astype(dtype), _Y)


@pytest.mark.parametrize("dtype", _REJECTED_DTYPES)
@pytest.mark.parametrize("Selector", LAZY_SELECTORS)
def test_lazy_selector_fit_selector_rejects_non_bool_X_test(Selector, dtype):
    selector = Selector(_HIERARCHY)
    with pytest.raises(ValueError, match="bool-dtype"):
        selector.fit_selector(X_train=_X_BOOL, y_train=_Y, X_test=_X_BOOL.astype(dtype))


@pytest.mark.parametrize("dtype", _REJECTED_DTYPES)
@pytest.mark.parametrize("Selector", LAZY_SELECTORS)
def test_lazy_selector_fit_selector_rejects_non_bool_X_train(Selector, dtype):
    selector = Selector(_HIERARCHY)
    with pytest.raises(ValueError, match="bool-dtype"):
        selector.fit_selector(X_train=_X_BOOL.astype(dtype), y_train=_Y, X_test=_X_BOOL)
