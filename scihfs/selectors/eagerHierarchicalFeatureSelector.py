"""
Base class for estimators for eager hierarchical feature selection.
"""

import warnings
from abc import abstractmethod

import numpy as np
from sklearn.feature_selection import SelectorMixin

from scihfs.selectors import HierarchicalEstimator


class EagerHierarchicalFeatureSelector(SelectorMixin, HierarchicalEstimator):
    """Abstract base class for eager feature selectors using hierarchical data.

    Eager selectors are supervised: ``fit`` requires a target ``y``.
    Subclasses implement their selection algorithm in ``_select``, which
    runs after the single input-validation pass and the hierarchy setup
    and fills ``representatives_`` with the names of all hierarchy nodes
    that are left after feature selection.
    """

    def __init__(self, hierarchy: np.ndarray = None):
        """Initializes an EagerHierarchicalFeatureSelector.

        Parameters
        ----------
        hierarchy : np.ndarray, scipy.sparse array/matrix or nx.DiGraph
                    The hierarchy graph, given either as a dense adjacency
                    matrix (``np.ndarray``), a sparse adjacency matrix
                    (``scipy.sparse``), or as directly as digraph (``networkx.DiGraph``, with optional node names that can match the columns in X).
        """
        super().__init__(hierarchy)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        return tags

    def _fit(self, X, y):
        """Runs the eager feature selection on the validated X and y.

        Checks the hierarchy against X, resets ``representatives_`` and
        delegates to the subclasses' ``_select``.
        """
        self._check_hierarchy_X()

        # self.representatives_ includes all node names for selected nodes.
        # self._columns maps them to the respective column in X.
        self.representatives_ = []
        self._select(X, y)

    @abstractmethod
    def _select(self, X, y):
        """The subclass's feature selection algorithm.

        Called with the validated X and y after the hierarchy setup, and
        expected to fill ``self.representatives_`` with the names of the
        selected hierarchy nodes.
        """

    def _get_support_mask(self):
        # Implements _get_support_mask method from SelectorMixin to
        # indicate the selected features from X.
        representatives_indices = [
            self._column_index(node) for node in self.representatives_
        ]
        return np.asarray(
            [
                True if index in representatives_indices else False
                for index in range(self.n_features_in_)
            ]
        )

    def _check_hierarchy_X(self):
        not_in_hierarchy = [
            feature_index
            for feature_index in range(self.n_features_in_)
            if feature_index not in self._columns
        ]
        if not_in_hierarchy:
            warning_missing_nodes = """All columns in X need to be mapped
             to a node in the hierarchy. If columns=None the corresponding
             node's name is the same as the column's index in the dataset.
             Otherwise, it is indicated by the value in self._columns at
             the columns' index."""
            warnings.warn(warning_missing_nodes)

        nodes = list(self._hierarchy_graph.nodes())
        nodes.remove("ROOT")
        not_in_dataset = [node for node in nodes if node not in self._columns]
        if not_in_dataset:
            warning_missing_columns = """The hierarchy should not include any
             nodes that are not mapped to a column in the dataset by the
             columns parameter"""
            warnings.warn(warning_missing_columns)
