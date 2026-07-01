"""
Base class for Sklearn compatible estimators using hierarchical data.
"""

import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from scihfs.helpers import (
    _check_unique_column_mappings,
    add_virtual_root_node,
    check_bool_dtype,
)

# Node-attribute key for recording a node's original identity (name or index) in the hierarchy graph. Left untouched by all operations on the graph.
# Consumed by get_feature_names_out to map back to original node names / indexes.
ORIGINAL_NODE_IDENTIFIER = "_scihfs_original_node_identifier"


class HierarchicalEstimator(TransformerMixin, BaseEstimator):
    """Base class for estimators using hierarchical data.

    The HierarchicalEstimator implements scikit-learn's BaseEstimator and
    TransformerMixin interfaces. It can be used as a base class for feature
    selection classes or data preprocessors that use hierarchical data.

    ..note:: This estimator currently supports only bool-dtype input.
    Non-binary (numeric) inputs raise ``ValueError``. Sum-propagation
    for numeric features could be a future enhancement.
    """

    def __init__(self, hierarchy=None):
        """Initializes a HierarchicalEstimator.

        Parameters
        ----------
        hierarchy : np.ndarray, scipy.sparse array/matrix or nx.DiGraph
                    The hierarchy graph, given either as a dense adjacency
                    matrix (``np.ndarray``), a sparse adjacency matrix
                    (``scipy.sparse``), or as directly as digraph (``networkx.DiGraph``, with optional node names that can match the columns in X).
                    Any ``scipy.sparse`` format (``csr_array``, ``csr_matrix``,
                    ``coo_array``, ...) is accepted and converted internally.
                    Note: ``None`` is accepted for scikit-learn ``clone()``
                    compatibility but raises ``TypeError`` in ``fit``."""
        self.hierarchy = hierarchy

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags

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
        ValueError
            If X is not bool-dtype. Numerical inputs may be supported in the future.

        Returns
        -------
        self : object
            Returns self.
        """
        if self.hierarchy is None:
            raise TypeError("Hierarchy is None but is required.")
        X = validate_data(self, X, accept_sparse=True)
        check_bool_dtype(X)
        self._fit_hierarchy(columns)

        return self

    def _fit_hierarchy(self, columns):
        """Set ``_columns`` and build ``_hierarchy_graph`` from a validated X.

        Split out from ``fit`` so that subclasses performing their own input
        validation (e.g. ``HierarchicalPreprocessor``, which must read
        ``feature_names_in_`` before any second ``validate_data`` call would
        drop it) can reuse the hierarchy setup without re-validating X.

        Assumes ``validate_data`` has already run and set ``n_features_in_``.

        Parameters
        ----------
        columns : list or None
            The mapping from the hierarchy graph's nodes to the columns in X.
            If None, positional 1:1 ordering is assumed.
        """
        if columns:
            self._columns = columns
        else:
            self._columns = list(range(self.n_features_in_))

        # Check whether there are duplicate column mappings (not dependent of the input route).
        # Placed at this point because unique column<->node mappings are required for the downstream processing and no such validation has been performed yet (specifically not in sklearn.utils.validation.validate_data).
        _check_unique_column_mappings(self._columns)

        self._set_hierarchy()
        self._check_dag()

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
        check_is_fitted(self)
        X = validate_data(self, X, accept_sparse="csr", reset=False)

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
        """Build ``self._hierarchy_graph`` from ``self.hierarchy``.

        The ``hierarchy`` parameter is accepted in three formats:

        - ``np.ndarray``: interpreted as an adjacency matrix; nodes are the
          integer row/column positions.
        - ``scipy.sparse`` array/matrix: interpreted as a sparse adjacency
          matrix; nodes are the integer row/column positions.
        - ``nx.DiGraph``: nodes may carry arbitrary (e.g. string) names. They
          are relabelled to integer positions ``0..n-1`` (preserving node
          order) so the rest of the pipeline keeps operating on integer node
          names.

        In all three cases each node is stamped with an
        ``ORIGINAL_NODE_IDENTIFIER`` attribute recording its original identity --
        the integer row/column index for an adjacency matrix, or the node name for a ``DiGraph``.

        The hierarchy is stored as a purely structural DAG: only edge
        *presence* is kept, never magnitude (edge weights are dropped).

        The user's original ``DiGraph`` is never mutated: ``relabel_nodes`` is
        called with ``copy=True`` and the virtual ROOT is added to the copy.

        After dispatch the virtual "ROOT" node is added to connect components.

        Raises
        ------
        TypeError
            If ``hierarchy`` is None or is none of ``np.ndarray``,
            ``scipy.sparse``, or ``nx.DiGraph``.
        """
        if self.hierarchy is None:
            raise TypeError("Hierarchy is None but is required.")
        if isinstance(self.hierarchy, np.ndarray):
            hierarchy_graph = nx.from_numpy_array(self.hierarchy, create_using=nx.DiGraph)
            # Adjacency nodes already ARE their original integer indices.
            original_identifiers = {
                node_index: node_index for node_index in hierarchy_graph.nodes
            }
        elif sp.issparse(self.hierarchy):
            hierarchy_graph = nx.from_scipy_sparse_array(
                self.hierarchy, create_using=nx.DiGraph
            )
            # Adjacency nodes already ARE their original integer indices.
            original_identifiers = {
                node_index: node_index for node_index in hierarchy_graph.nodes
            }
        elif isinstance(self.hierarchy, nx.DiGraph):
            node_names = list(self.hierarchy.nodes)
            mapping = {
                node_name: position for position, node_name in enumerate(node_names)
            }
            hierarchy_graph = nx.relabel_nodes(self.hierarchy, mapping, copy=True)
            original_identifiers = {
                position: node_name for position, node_name in enumerate(node_names)
            }
        else:
            raise TypeError(
                f"hierarchy must be np.ndarray, scipy.sparse or nx.DiGraph, "
                f"got {type(self.hierarchy)}."
            )
        nx.set_node_attributes(
            hierarchy_graph, original_identifiers, ORIGINAL_NODE_IDENTIFIER
        )
        # The hierarchy is purely meant as structural information.
        # Thus, any edge weights from the user input are dropped.
        # This also ensures that to_adjacency_matrix returns a boolean matrix, which is expected by the rest of the pipeline.
        for _, _, edge_data in hierarchy_graph.edges(data=True):
            edge_data.pop("weight", None)
        # Add "ROOT" node and connect components if there are multiple
        self._hierarchy_graph = add_virtual_root_node(hierarchy_graph)

    def _column_index(self, node):
        # Get the corresponding column index for a node in the hierarchy.
        return self._columns.index(node)
