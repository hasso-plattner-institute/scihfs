"""
Base class for Sklearn compatible estimators using hierarchical data.
"""

import networkx as nx
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

from hfs.helpers import add_virtual_root_node


class HierarchicalEstimator(TransformerMixin, BaseEstimator):
    """Base class for estimators using hierarchical data.

    The HierarchicalEstimator implements scikit-learn's BaseEstimator and
    TransformerMixin interfaces. It can be used as a base class for feature
    selection classes or data preprocessors that use hierarchical data.
    """

    def __init__(self, hierarchy: np.ndarray = None):
        """Initializes a HierarchicalEstimator.

        Parameters
        ----------
        hierarchy : np.ndarray
                    The hierarchy graph as an adjacency matrix."""
        self.hierarchy = hierarchy

    def fit(self, X, y=None, columns=None):
        """Fitting function that prepares the hierarchy and _columns parameter.

        The hierarchy is transformed to a nx.DiGraph with a virtual root node
        named "ROOT" that connects all parts of the graph to one component.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or None
            The target values. Only necessary for some estimators.
        columns: list or None
            The mapping from the hierarchy graph's nodes to the columns in X.
            A list of ints. If this parameter is None the columns in X and
            the corresponding nodes in the hierarchy are expected to be in the
            same order.

        Raises
        ------
        TypeError
            If the passed hierarchy is None.

        Returns
        -------
        self : object
            Returns self.
        """
        if self.hierarchy is None:
            raise TypeError("Hierarchy is None but is required.")
        X = check_array(X, accept_sparse=True)

        self.n_features_in_ = X.shape[1]
        if columns:
            self._columns = columns
        else:
            self._columns = list(range(self.n_features_in_))

        self._set_hierarchy()
        self._check_dag()

        return self

    def transform(self, X):
        """Reduce X to the selected features.

        Extend this methods to actually transform the dataset.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X : array of shape (n_samples, n_selected_features)
            The input samples with only the selected features.
        """
        X = check_array(X, dtype=None, accept_sparse="csr")

        if self.n_features_in_ != X.shape[1]:
            raise ValueError("X has a different shape than during fitting.")

        return X

    def get_columns(self):
        """Get mapping from the dataset's columns to the hierarchy's nodes.

        Returns
        -------
        columns : list of shape n_features
                The value at index i is the name of the corresponding node in the
                hierarchy graph for columns i in the dataset.
        """
        return self._columns

    def _check_dag(self):
        """Checks if the hierarchy graph is a directed acyclic graph.

        Raises
        ------
        ValueError
            If the hierarchy graph is not a directed acyclic graph.
        """
        if not nx.is_directed_acyclic_graph(self._hierarchy_graph):
            raise ValueError("The hierarchy graph is not a directed acyclic graph.")

    def _set_hierarchy(self):
        """
        Assign hierarchy graph to self._hierarchy_graph
        after adding ROOT node to connect components.
        """
        hierarchy_graph = nx.from_numpy_array(self.hierarchy, create_using=nx.DiGraph)
        # Add "ROOT" node and connect components if there are multiple
        self._hierarchy_graph = add_virtual_root_node(hierarchy_graph)

    def _column_index(self, node):
        # Get the corresponding column index for a node in the hierarchy.
        return self._columns.index(node)
