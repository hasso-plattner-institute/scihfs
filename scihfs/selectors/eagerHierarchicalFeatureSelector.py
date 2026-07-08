"""
Base class for estimators for eager hierarchical feature selection.
"""

import warnings

import numpy as np
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import validate_data

from scihfs.selectors import HierarchicalEstimator


class EagerHierarchicalFeatureSelector(SelectorMixin, HierarchicalEstimator):
    """Base class for eager feature selectors using hierarchical data."""

    def __init__(self, hierarchy: np.ndarray = None):
        """Initializes an EagerHierarchicalFeatureSelector.

        Parameters
        ----------
        hierarchy : np.ndarray, scipy.sparse array/matrix or nx.DiGraph
                    The hierarchy graph, given either as a dense adjacency
                    matrix (``np.ndarray``), a sparse adjacency matrix
                    (``scipy.sparse``), or as directly as digraph (``networkx.DiGraph``, with optional node names that can match the columns in X)."""
        super().__init__(hierarchy)

    def fit(self, X, y=None, columns=None):
        """Fitting function that sets representatives.

        After fitting representatives should include the names of all
        nodes from the hierarchy that are left after feature selection.
        The number of columns in X and the number of nodes in the hierarchy
        are expected to be the same and each column should be mapped to
        exactly one node in the hierarchy with the columns parameter.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int. Not needed for all estimators.
        columns: list or None, length n_features
            The mapping from the hierarchy graph's nodes to the columns in X.
            A list of ints. If this parameter is None the columns in X and
            the corresponding nodes in the hierarchy are expected to be in the
            same order.

        Returns
        -------
        self : object
            Returns self.
        """

        super().fit(X, y, columns)
        self._check_hierarchy_X()

        # self.representatives_ includes all node names for selected nodes.
        # self._columns maps them to the respective column in X.
        self.representatives_ = []
        self._fit(X, y)

        return self

    def _fit(self, X, y):
        """Hook for the subclasses' feature selection algorithms.

        Called by ``fit`` with the validated X and y after the hierarchy
        setup, and expected to fill ``self.representatives_`` with the
        names of the selected hierarchy nodes. The base implementation
        here thus intentionally selects nothing.
        """
        pass

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
