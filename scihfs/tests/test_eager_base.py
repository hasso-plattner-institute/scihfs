"""Tests for the eager selector base class (EagerHierarchicalFeatureSelector)."""

import warnings

import networkx as nx
import numpy as np
import pandas as pd
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


# ---------------------------------------------------------------------------
# _validate_hyperparameters: no-op hook, called first thing in fit.
# ---------------------------------------------------------------------------


class _RaisingHyperparameterSelector(_MinimalEagerSelector):
    """A stub whose hyperparameter validation always fails.

    _MinimalEagerSelector itself has no hyperparameters to validate; real
    examples of the override exist on BottomUpSelector's k (see
    test_hill_climbing.py) and SHSELSelector's similarity_threshold (see
    test_shsel.py), but neither is convenient to drive through this
    minimal-selector test setup, so the override is defined locally here.
    """

    def _validate_hyperparameters(self):
        raise ValueError("bad hyperparameter")


def test_validate_hyperparameters_default_is_a_noop():
    """The base HierarchyMixin implementation rejects nothing."""
    X = np.zeros((2, 2), dtype=bool)
    hierarchy = nx.to_numpy_array(nx.DiGraph([(0, 1)]))
    selector = _MinimalEagerSelector(hierarchy)
    selector.fit(X, np.array([0, 1]))
    assert selector.is_fitted_


def test_validate_hyperparameters_runs_before_hierarchy_none_check():
    """An invalid hyperparameter is reported even when hierarchy is also None.

    Hyperparameters are known from __init__ alone, so validating them first
    means this failure is reported instead of the (also true, but less
    specific) "hierarchy is required" complaint.
    """
    X = np.zeros((2, 2), dtype=bool)
    selector = _RaisingHyperparameterSelector(None)
    with pytest.raises(ValueError, match="bad hyperparameter"):
        selector.fit(X, np.array([0, 1]))


def test_validate_hyperparameters_runs_before_data_validation():
    """An invalid hyperparameter is reported even when X is nonsense.

    X here is 1-D, which validate_data would reject with an unrelated
    "Expected 2D array" error -- proving the hyperparameter check runs first
    and validate_data is never reached.
    """
    hierarchy = nx.to_numpy_array(nx.DiGraph([(0, 1)]))
    selector = _RaisingHyperparameterSelector(hierarchy)
    with pytest.raises(ValueError, match="bad hyperparameter"):
        selector.fit(np.zeros(4), np.array([0, 1]))


@pytest.mark.filterwarnings("ignore:Hierarchy consists of multiple")
@pytest.mark.filterwarnings("ignore:.*hold no True value")
@pytest.mark.parametrize(
    "data",
    ["wrong_hierarchy_X", "wrong_hierarchy_X1"],
    ids=["column-without-node", "node-without-column"],
)
def test_check_hierarchy_X_raises_on_mismatch(data, request):
    # A column<->node mismatch is now rejected rather than warned: the eager
    # (like the lazy) HFS methods are strict consumers of aligned input (only
    # the HierarchicalPreprocessor tolerates -- and fixes -- a mismatch).
    X, hierarchy, columns = request.getfixturevalue(data)
    # y is irrelevant here, but must still hold both classes to clear the
    # target check that precedes the alignment check in fit.
    y = np.arange(X.shape[0]) % 2
    selector = _MinimalEagerSelector(hierarchy)
    with pytest.raises(ValueError, match="not aligned"):
        selector.fit(X, y, columns=columns)


# --- DataFrame auto-derive on the eager path --------------------------------
#
# Every node of the graph (insertion order: A=0, B=1, C=2, D=3) is backed by a
# DataFrame column, so _check_hierarchy_X stays silent. The DataFrame lists
# the columns in shuffled order so a positional mapping would be wrong.


def _named_graph():
    return nx.DiGraph([("A", "B"), ("B", "C"), ("A", "D")])


def _named_dataframe():
    return pd.DataFrame(
        {
            "C": [1, 0, 0, 1],
            "A": [1, 1, 0, 1],
            "D": [0, 1, 0, 0],
            "B": [1, 0, 0, 1],
        }
    ).astype(bool)


_y = np.array([1, 0, 0, 1])


def test_autoderives_columns_from_dataframe():
    """DataFrame X + named DiGraph + columns=None derives the mapping by name."""
    selector = _MinimalEagerSelector(_named_graph())
    selector.fit(_named_dataframe(), _y)
    assert selector.get_columns() == [2, 0, 3, 1]


def test_ndarray_X_keeps_positional_mapping():
    """Plain ndarray X + columns=None still maps positionally (no auto-derive).

    The formats are mixed here -- a nameless X against a named hierarchy --
    so the positional reading is announced rather than assumed silently.
    """
    X = _named_dataframe().to_numpy()
    selector = _MinimalEagerSelector(_named_graph())
    with pytest.warns(UserWarning, match="by position"):
        selector.fit(X, _y)
    assert not hasattr(selector, "feature_names_in_")
    assert selector.get_columns() == [0, 1, 2, 3]


def test_no_positional_warning_with_explicit_columns():
    """An explicit columns mapping confirms the order, so nothing is assumed."""
    X = _named_dataframe().to_numpy()
    selector = _MinimalEagerSelector(_named_graph())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        selector.fit(X, _y, columns=[0, 1, 2, 3])
    assert selector.get_columns() == [0, 1, 2, 3]


def test_no_positional_warning_for_self_naming_digraph():
    """Node names that already are their own positions add no information."""
    hierarchy = nx.DiGraph([(0, 1), (1, 2), (0, 3)])
    X = _named_dataframe().to_numpy()
    selector = _MinimalEagerSelector(hierarchy)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        selector.fit(X, _y)
    assert selector.get_columns() == [0, 1, 2, 3]


def test_no_positional_warning_for_adjacency_hierarchy():
    """An adjacency matrix has no node names to ignore in the first place."""
    hierarchy = nx.to_numpy_array(_named_graph())
    X = _named_dataframe().to_numpy()
    selector = _MinimalEagerSelector(hierarchy)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        selector.fit(X, _y)
    assert selector.get_columns() == [0, 1, 2, 3]


def test_orphan_dataframe_column_raises():
    """A DataFrame column without a matching node raises on the eager path.

    Unlike the HierarchicalPreprocessor (which adds the orphan under ROOT
    and warns), a selector cannot extend the hierarchy.
    """
    df = _named_dataframe()
    df["unicorn"] = [False, True, False, False]
    selector = _MinimalEagerSelector(_named_graph())
    with pytest.raises(ValueError, match="no matching node"):
        selector.fit(df, _y)


def test_dataframe_with_adjacency_hierarchy_and_no_columns_raises():
    """DataFrame X + adjacency-matrix hierarchy + columns=None cannot auto-derive."""
    hierarchy = nx.to_numpy_array(_named_graph())
    selector = _MinimalEagerSelector(hierarchy)
    with pytest.raises(ValueError, match="Cannot auto-derive columns"):
        selector.fit(_named_dataframe(), _y)


def _colliding_graph():
    """Nodes 1, "a", "1", "b": the two "1"s are distinct nodes, one name."""
    return nx.DiGraph([(1, "a"), ("1", "b")])


def test_colliding_node_names_raise_on_autoderive():
    """Nodes sharing a string form cannot be matched against column labels.

    Previously the later node won the lookup and the earlier one silently
    became unreachable, surfacing (if at all) as a confusing alignment error
    about a hierarchy node without a data column.
    """
    df = pd.DataFrame({"1": [1, 0, 1, 1], "a": [0, 1, 0, 1], "b": [1, 1, 0, 0]}).astype(
        bool
    )
    selector = _MinimalEagerSelector(_colliding_graph())
    with pytest.raises(ValueError, match="unique when compared as strings"):
        selector.fit(df, _y)


def test_colliding_node_names_accepted_with_explicit_columns():
    """Explicit columns skip the name matching, so the collision is harmless.

    This is the escape hatch the error message points to.
    """
    df = pd.DataFrame(
        {
            "one": [1, 0, 1, 1],
            "a": [0, 1, 0, 1],
            "1": [1, 1, 0, 0],
            "b": [0, 0, 1, 1],
        }
    ).astype(bool)
    selector = _MinimalEagerSelector(_colliding_graph())
    selector.fit(df, _y, columns=[0, 1, 2, 3])
    assert selector.get_columns() == [0, 1, 2, 3]


def test_duplicate_dataframe_column_names_raise():
    """Duplicate DataFrame column names are rejected on the auto-derive path.

    Either caught by scihfs's duplicate-mapping guard or already during
    validate_data (on scikit-learn >= 1.9); both raise a ValueError.
    """
    df = pd.DataFrame(
        np.array([[1, 0, 1], [0, 1, 1]], dtype=bool), columns=["A", "A", "B"]
    )
    selector = _MinimalEagerSelector(nx.DiGraph([("A", "B")]))
    with pytest.raises(ValueError, match="Duplicate|unique column names"):
        selector.fit(df, np.array([1, 0]))


# --- DataFrame output tests (get_feature_names_out / set_output) -------------------


class _SelectNodesACSelector(EagerHierarchicalFeatureSelector):
    """Concrete stub whose _select always keeps hierarchy nodes A (0) and C (2)."""

    def _select(self, X, y):
        self.selected_features_ = [0, 2]


def test_get_feature_names_out_from_dataframe():
    """Output of get_feature_names_out are the selected DataFrame
    column names, in column order.

    SelectorMixin's default filters ``feature_names_in_`` by the support
    mask; the shuffled frame proves the names follow the DataFrame's column
    order (C, A), not the hierarchy's node order (A, C).
    """
    selector = _SelectNodesACSelector(_named_graph())
    selector.fit(_named_dataframe(), _y)
    assert list(selector.get_feature_names_out()) == ["C", "A"]


@pytest.mark.filterwarnings("ignore:The hierarchy is an nx.DiGraph")
def test_get_feature_names_out_from_ndarray():
    """Without input feature names the x<position> fallback names are used."""
    selector = _SelectNodesACSelector(_named_graph())
    selector.fit(_named_dataframe().to_numpy(), _y)
    assert list(selector.get_feature_names_out()) == ["x0", "x2"]


def test_set_output_pandas_on_selector():
    """set_output('pandas') makes a selector's transform return a DataFrame."""
    df = _named_dataframe()
    selector = _SelectNodesACSelector(_named_graph())
    selector.set_output(transform="pandas")
    selector.fit(df, _y)
    out = selector.transform(df)

    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["C", "A"]
    assert list(out.index) == list(df.index)
    assert np.array_equal(out.to_numpy(), df[["C", "A"]].to_numpy())

    # fit_transform runs through the same output wrapping.
    refit = _SelectNodesACSelector(_named_graph()).set_output(transform="pandas")
    pd.testing.assert_frame_equal(refit.fit_transform(df, _y), out)
