import networkx as nx
import numpy as np
import pytest

from scihfs.tests._estimators import EAGER_SELECTORS, LAZY_SELECTORS

# ---------------------------------------------------------------------------
# TEMPORARY FILE CONTENT WARNING:
#
# At the moment, the scope of this file is only to test the rejection of
# non-bool-dtype X input and non-binary y input to the selectors. All sklearn-
# related tests are in test_sklearn_conformance.py for now.
#
# In the future, this file should contain all those tests that apply to
# ALL estimators alike. Right now, a lot of these are scattered throughout
# the respective test files for each estimator.
# ---------------------------------------------------------------------------

_HIERARCHY = nx.to_numpy_array(nx.DiGraph([(0, 1), (1, 2), (0, 3)]))
_X_BOOL = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 0]], dtype=bool)
_Y = np.array([0, 1, 0, 1])
_Y_MULTICLASS = np.array([0, 1, 2, 0])
# Binary targets whose two labels are NOT {0, 1}: an offset ({1, 2}) and a
# signed ({-1, 1}) encoding. type_of_target reports both as "binary", so they
# clear the multiclass gate and must be rejected by the tighter {0, 1} check.
_NON_ZERO_ONE_BINARY = [np.array([1, 2, 1, 2]), np.array([-1, 1, -1, 1])]
# type_of_target reports a single-class y as "binary" as well, and {0} / {1}
# even pass the {0, 1} membership check -- so this needs its own gate.
_SINGLE_CLASS = [
    np.array([0, 0, 0, 0]),
    np.array([1, 1, 1, 1]),
    np.array([True, True, True, True]),
]
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
def test_lazy_selector_predict_rejects_non_bool_X(Selector, dtype):
    selector = Selector(_HIERARCHY).fit(_X_BOOL, _Y)
    with pytest.raises(ValueError, match="bool-dtype"):
        selector.predict(_X_BOOL.astype(dtype))


@pytest.mark.parametrize("Selector", EAGER_SELECTORS)
def test_eager_selector_rejects_non_binary_y(Selector):
    selector = Selector(_HIERARCHY)
    with pytest.raises(ValueError, match="binary target"):
        selector.fit(_X_BOOL, _Y_MULTICLASS)


@pytest.mark.parametrize("Selector", LAZY_SELECTORS)
def test_lazy_selector_rejects_non_binary_y(Selector):
    selector = Selector(_HIERARCHY)
    with pytest.raises(ValueError, match="binary target"):
        selector.fit(_X_BOOL, _Y_MULTICLASS)


@pytest.mark.parametrize("y", _NON_ZERO_ONE_BINARY)
@pytest.mark.parametrize("Selector", EAGER_SELECTORS)
def test_eager_selector_rejects_non_zero_one_binary_y(Selector, y):
    selector = Selector(_HIERARCHY)
    with pytest.raises(ValueError, match="labelled 0 and 1"):
        selector.fit(_X_BOOL, y)


@pytest.mark.parametrize("y", _NON_ZERO_ONE_BINARY)
@pytest.mark.parametrize("Selector", LAZY_SELECTORS)
def test_lazy_selector_rejects_non_zero_one_binary_y(Selector, y):
    selector = Selector(_HIERARCHY)
    with pytest.raises(ValueError, match="labelled 0 and 1"):
        selector.fit(_X_BOOL, y)


@pytest.mark.parametrize("y", _SINGLE_CLASS)
@pytest.mark.parametrize("Selector", EAGER_SELECTORS)
def test_eager_selector_rejects_single_class_y(Selector, y):
    selector = Selector(_HIERARCHY)
    with pytest.raises(ValueError, match="only one class"):
        selector.fit(_X_BOOL, y)


@pytest.mark.parametrize("y", _SINGLE_CLASS)
@pytest.mark.parametrize("Selector", LAZY_SELECTORS)
def test_lazy_selector_rejects_single_class_y(Selector, y):
    selector = Selector(_HIERARCHY)
    with pytest.raises(ValueError, match="only one class"):
        selector.fit(_X_BOOL, y)


@pytest.mark.parametrize("Selector", LAZY_SELECTORS)
def test_lazy_selector_locks_classes_to_zero_one(Selector):
    # The {0, 1} contract guarantees classes_ == [0, 1] (sorted), so downstream
    # class-ordered machinery (e.g. predict_proba) can rely on it.
    selector = Selector(_HIERARCHY).fit(_X_BOOL, _Y)
    assert np.array_equal(selector.classes_, np.array([0, 1]))
