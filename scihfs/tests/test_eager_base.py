"""Tests for the eager selector base class (EagerHierarchicalFeatureSelector)."""

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
    """Plain ndarray X + columns=None still maps positionally (no auto-derive)."""
    X = _named_dataframe().to_numpy()
    selector = _MinimalEagerSelector(_named_graph())
    selector.fit(X, _y)
    assert not hasattr(selector, "feature_names_in_")
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
