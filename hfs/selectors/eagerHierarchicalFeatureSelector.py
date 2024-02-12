"""
Base class for estimators for eager hierarchical feature selection.
"""
import warnings

import numpy as np
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_array

from hfs.selectors import HierarchicalEstimator


class EagerHierarchicalFeatureSelector(SelectorMixin, HierarchicalEstimator):
    """Base class for eager feature selectors using hierarchical data."""

    def __init__(self, hierarchy: np.ndarray = None):
        """Initializes an EagerHierarchicalFeatureSelector.

        Parameters
        ----------
        hierarchy : np.ndarray
                    The hierarchy graph as an adjacency matrix."""
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

        return self

    def transform(self, X):
        """Reduce X to the selected features.

        Extends the transform method from SelectorMixin. Only selected
        columns from X are in the output dataset.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X : array of shape [n_samples, n_selected_features]
            The input samples with only the selected features.
        """
        X = check_array(X, dtype=None, accept_sparse="csr")
        if self.n_features_in_ != X.shape[1]:
            raise ValueError("X has a different shape than during fitting.")
        return super().transform(X)

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

        nodes = list(self._hierarchy.nodes())
        nodes.remove("ROOT")
        not_in_dataset = [node for node in nodes if node not in self._columns]
        if not_in_dataset:
            warning_missing_columns = """The hierarchy should not include any
             nodes that are not mapped to a column in the dataset by the
             columns parameter"""
            warnings.warn(warning_missing_columns)
