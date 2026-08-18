"""
Base class for Sklearn compatible estimators using hierarchical data.
"""

import warnings

import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from scihfs.helpers import (
    _check_unique_column_mappings,
    add_virtual_root_node,
    check_adjacency_matrix_values,
    check_binary_target,
    check_bool_dtype,
    check_digraph_edge_weights,
    check_square_adjacency_matrix,
    check_unique_node_names,
    warn_on_all_false_columns,
)

# Node-attribute key for recording a node's original identity (name or index) in the hierarchy graph. Left untouched by all operations on the graph.
# Consumed by get_feature_names_out to map back to original node names / indexes.
ORIGINAL_NODE_IDENTIFIER = "_scihfs_original_node_identifier"


class HierarchyMixin:
    """Hierarchy-core mixin for handling the ``hierarchy`` parameter.

    Holds everything needed to turn the ``hierarchy`` constructor argument into
    a validated ``_hierarchy_graph`` plus the column->node mapping, *without*
    committing to any scikit-learn fit/transform/predict contract. It carries
    no ``TransformerMixin``/``ClassifierMixin`` surface and defines no ``fit``.

    Estimators combine it with ``BaseEstimator`` and the mixin matching their
    role:

    - ``HierarchicalEstimator(TransformerMixin, HierarchyMixin, BaseEstimator)``
      -- the transformer fit-template used by the preprocessor and the eager
      selectors.
    - The lazy selectors pair it with a classifier surface directly.

    Two things share the word "validation", and only one lives here.
    *Hierarchy* validation is this mixin's job: format dispatch
    (``_set_hierarchy``), acyclicity (``_check_dag``), and the unique/orphan
    column->node mapping (``_auto_derive_columns`` /
    ``_handle_orphan_features``). X *data* validation (dtype, finiteness,
    shape) is deliberately not: it is the host estimator's boundary
    responsibility, performed once per public method via ``validate_data``.
    That split is intentional -- data validation is role-specific (whether
    ``y`` is required and whether sparse is accepted come from the host's
    tags) and fires again at ``transform`` / ``predict`` time, neither of
    which this mixin sees. These methods here therefore consume only the
    *products* of that validation: ``self.n_features_in_`` (and, for DataFrame
    input, ``self.feature_names_in_``), set by the host before
    ``_fit_hierarchy`` runs.
    """

    def __init__(
        self, hierarchy: np.ndarray | sp.sparray | sp.spmatrix | nx.DiGraph | None = None
    ):
        """Initializes a HierarchyMixin.

        Parameters
        ----------
        hierarchy : np.ndarray, scipy.sparse array/matrix or nx.DiGraph
                    The hierarchy graph, given either as a dense adjacency
                    matrix (``np.ndarray``), a sparse adjacency matrix
                    (``scipy.sparse``), or as directly as digraph (``networkx.DiGraph``, with optional node names that can match the columns in X).
                    Any ``scipy.sparse`` format (``csr_array``, ``csr_matrix``,
                    ``coo_array``, ...) is accepted and converted internally.
                    Node names may be of any hashable type, but they are
                    matched against the DataFrame column labels -- and
                    reported back -- by their ``str()`` form. They therefore
                    have to be unique as strings: a hierarchy holding both
                    ``1`` and ``"1"`` is rejected when the mapping is derived
                    from column names.
                    Note: ``None`` is accepted for scikit-learn ``clone()``
                    compatibility but raises ``TypeError`` in ``fit``."""
        self.hierarchy = hierarchy

    def _validate_hyperparameters(self):
        """Hook for a subclass to validate its own constructor hyperparameters.

        Cheap and thus called before actual dataset and hierarchy validations.

        No-op here in the base class, but overridden in all HFS methods
        subclasses that have hyperparameters manually set by the user --
        using ``sklearn.utils.validation.check_scalar``, which raises
        ``TypeError`` for a wrong type and ``ValueError`` for an out-of-range
        value.
        """

    def _fit_hierarchy(self, columns):
        """Set ``_columns`` and build ``_hierarchy_graph`` from a validated X.

        Called from the template ``fit`` after its single validation pass.
        Assumes ``validate_data`` has already run and set ``n_features_in_``
        (and ``feature_names_in_`` when X was a DataFrame).

        Parameters
        ----------
        columns : list or None
            The mapping from the hierarchy graph's nodes to the columns in X.
            If None, the mapping is auto-derived from the feature names when
            X was a DataFrame (see ``_auto_derive_columns``); otherwise
            positional 1:1 ordering is assumed (see
            ``_warn_on_positional_fallback``).
        """
        if columns is None and getattr(self, "feature_names_in_", None) is not None:
            columns = self._auto_derive_columns()
        if columns:
            self._columns = columns
        else:
            self._warn_on_positional_fallback()
            self._columns = list(range(self.n_features_in_))

        # Check whether there are duplicate column mappings (not dependent of the input route).
        # Placed at this point because unique column<->node mappings are required for the downstream processing and no such validation has been performed yet (specifically not in sklearn.utils.validation.validate_data).
        _check_unique_column_mappings(self._columns)

        self._set_hierarchy()
        self._check_dag()

    def _warn_on_positional_fallback(self):
        """Warn when a named DiGraph hierarchy is mapped by position.

        Reached when neither the ``columns`` have been passed nor DataFrame
        feature names are available. In that case, the columns of X are assumed to
        be in the hierarchy's node order. For an adjacency matrix that is the only
        possible reading -- the nodes are their own indices. A ``DiGraph`` however
        carries node names, and those are silently ignored here: the input formats
        are mixed (nameless X, named hierarchy) and only one of them can be
        honoured.

        Node names that already equal their own position carry no information
        beyond the order, so those stay silent.
        """
        if not isinstance(self.hierarchy, nx.DiGraph):
            return
        nodes = list(self.hierarchy.nodes)
        if nodes == list(range(len(nodes))):
            return
        preview = f"{nodes[:5]}" + (", ..." if len(nodes) > 5 else "")
        warnings.warn(
            f"The hierarchy is an nx.DiGraph with node names, but X carries no "
            f"column names and no columns mapping was passed, so the columns of "
            f"X are mapped to the hierarchy nodes by position: {preview}. "
            f"Pass X as a DataFrame whose (string) column labels match the node "
            f"names to map by name instead, or pass the columns mapping "
            f"explicitly to confirm the positional order."
        )

    def _auto_derive_columns(self):
        """Derive the column->node mapping from DataFrame feature names.

        Called from ``_fit_hierarchy`` when X was passed to ``fit`` as a
        ``DataFrame`` (so ``feature_names_in_`` is set) and ``columns``
        was not supplied. Each captured feature name is looked up against the
        hierarchy's node names; names with no matching node map to ``-1`` and
        are handed to the ``_handle_orphan_features`` hook, which decides the
        subclass's orphan policy (the base raises, only the HierarchicalPreprocessor
        subclass does not).

        Node names are compared as strings because scikit-learn coerces
        DataFrame column labels to ``str`` in ``feature_names_in_``. This lets
        a ``DiGraph`` with integer node names still match a DataFrame whose
        (string) column labels equal those integers.

        Returns
        -------
        columns : list of int
            The derived column->node-position mapping, in feature order.

        Raises
        ------
        ValueError
            If the hierarchy is not a ``DiGraph`` – an adjacency matrix, given
            as np.ndarray or scipy.sparse, has no node names to match
            against). Also if two node names coincide once coerced to ``str``
            (see ``check_unique_node_names``).
        """
        if not isinstance(self.hierarchy, nx.DiGraph):
            raise ValueError(
                "Cannot auto-derive columns from DataFrame feature names "
                "because the hierarchy is an adjacency matrix (np.ndarray or "
                "scipy.sparse) without node names. Either pass hierarchy as "
                "nx.DiGraph with named nodes, or supply columns explicitly."
            )
        nodes = list(self.hierarchy.nodes)
        # Only relevant on this path: without the matching by name, nodes that
        # share a string form are perfectly valid.
        check_unique_node_names(nodes)
        name_to_position = {str(node): i for i, node in enumerate(nodes)}
        columns = [name_to_position.get(str(name), -1) for name in self.feature_names_in_]
        orphan_names = [
            str(name) for name, node in zip(self.feature_names_in_, columns) if node == -1
        ]
        if orphan_names:
            self._handle_orphan_features(orphan_names)
        return columns

    def _handle_orphan_features(self, orphan_names):
        """Hook deciding the policy for orphaned DataFrame feature names.

        Called from ``_auto_derive_columns`` when at least one feature name has
        no matching node in the hierarchy (mapped to ``-1``). The base
        implementation raises because the estimators cannot handle unmapped
        columns; ``HierarchicalPreprocessor`` overrides this to tolerate them
        (its ``_extend_dag`` later adds the orphans under ROOT).

        Parameters
        ----------
        orphan_names : list of str
            The DataFrame feature names without a matching hierarchy node.

        Raises
        ------
        ValueError
            Always (in the base implementation).
        """
        raise ValueError(
            f"The following DataFrame columns have no matching node in the "
            f"hierarchy: {orphan_names}. Use the HierarchicalPreprocessor to "
            "add them to the hierarchy, or supply the columns mapping "
            "explicitly."
        )

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
        ValueError
            If an ``np.ndarray`` or ``scipy.sparse`` hierarchy is not a
            (2-D, square) adjacency matrix, or if the hierarchy carries edge
            weights other than 1 (in any of the three formats).
        """
        if self.hierarchy is None:
            raise TypeError("Hierarchy is None but is required.")
        if isinstance(self.hierarchy, np.ndarray):
            check_square_adjacency_matrix(self.hierarchy)
            check_adjacency_matrix_values(self.hierarchy)
            hierarchy_graph = nx.from_numpy_array(self.hierarchy, create_using=nx.DiGraph)
            # Adjacency nodes already ARE their original integer indices.
            original_identifiers = {
                node_index: node_index for node_index in hierarchy_graph.nodes
            }
        elif sp.issparse(self.hierarchy):
            check_square_adjacency_matrix(self.hierarchy)
            check_adjacency_matrix_values(self.hierarchy)
            hierarchy_graph = nx.from_scipy_sparse_array(
                self.hierarchy, create_using=nx.DiGraph
            )
            # Adjacency nodes already ARE their original integer indices.
            original_identifiers = {
                node_index: node_index for node_index in hierarchy_graph.nodes
            }
        elif isinstance(self.hierarchy, nx.DiGraph):
            check_digraph_edge_weights(self.hierarchy)
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
        # The hierarchy is purely meant as structural information, so besides
        # accepting no edge weight information previously, even all remaining
        # weights of ``1`` at this point are dropped. This is not a processing
        # necessity, but a memory optimization.
        for _, _, edge_data in hierarchy_graph.edges(data=True):
            edge_data.pop("weight", None)
        # Add "ROOT" node and connect components if there are multiple
        self._hierarchy_graph = add_virtual_root_node(hierarchy_graph)

    def _column_index(self, node):
        # Get the corresponding column index for a node in the hierarchy.
        return self._columns.index(node)

    def _reject_column_node_mismatch(self):
        """Raise if the hierarchy nodes and data columns are not in bijection.

        Necessary for all HFS methods (eager and lazy), but not for the preprocessor
        (which aligns them automatically).

        Raises
        ------
        ValueError
            If any hierarchy node has no data column, or any data column has no
            hierarchy node.
        """
        nodes = set(self._hierarchy_graph.nodes()) - {"ROOT"}
        mapped = set(self._columns)
        nodes_without_column = nodes - mapped
        # A data column is unmapped if it has no entry in _columns or its entry
        # points at a node absent from the hierarchy graph.
        columns_without_node = [
            column
            for column in range(self.n_features_in_)
            if column >= len(self._columns) or self._columns[column] not in nodes
        ]
        if not nodes_without_column and not columns_without_node:
            return
        node_names = [
            self._hierarchy_graph.nodes[node].get(ORIGINAL_NODE_IDENTIFIER, node)
            for node in sorted(nodes_without_column)
        ]
        raise ValueError(
            "Hierarchy and data columns are not aligned: "
            f"hierarchy node(s) with no data column: {node_names}; "
            f"data column(s) with no hierarchy node: {columns_without_node}. Every "
            "node must map to exactly one column and vice versa. Use the "
            "HierarchicalPreprocessor to align them, or pass a matching "
            "``columns`` mapping."
        )

    def _relabel_hierarchy_to_columns(self):
        """Relabel ``_hierarchy_graph`` from node positions to column indices.

        The output from this method is a copy of the hierarchy graph with the
        nodes relabelled to the corresponding column indices in the dataset.

        Running this method is a prerequisite for the lazy HFS methods, which
        index ``x_row[node]`` directly (= without ``_column_index`` lookup).
        (The eager HFS methods translate node -> column on demand via
        ``_column_index``.)
        """
        self._reject_column_node_mismatch()
        self._hierarchy_graph.remove_node("ROOT")
        node_to_column = {
            position: column for column, position in enumerate(self._columns)
        }
        self._hierarchy_graph = nx.relabel_nodes(self._hierarchy_graph, node_to_column)


class HierarchicalEstimator(TransformerMixin, HierarchyMixin, BaseEstimator):
    """Base class for estimators using hierarchical data.

    The HierarchicalEstimator combines scikit-learn's ``TransformerMixin`` and
    ``BaseEstimator`` with :class:`HierarchyMixin` from scihfs (the hierarchy core).
    It is the transformer fit-template for the data preprocessor and the eager
    feature selectors, with single input-validation pass, the hierarchy setup,
    and a ``_fit`` hook.

    ..note:: This estimator currently supports only bool-dtype input.
    Non-binary (numeric) inputs raise ``ValueError``. Sum-propagation
    for numeric features could be a future enhancement.
    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags

    def fit(self, X, y=None, columns=None):
        """Template fitting function shared by all estimators of this family.

        X (and y, when given) are validated exactly once on this path, so
        ``feature_names_in_`` captured from a DataFrame input survives
        fitting. The hierarchy is then transformed to a nx.DiGraph with a
        virtual root node named "ROOT" that connects all parts of the graph
        to one component, and the subclass's ``_fit`` hook is run.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or None
            The target values. Required by the supervised subclasses (the
            eager selectors); ignored by the purely structural transformers.
        columns: list or None
            The mapping from the hierarchy graph's nodes to the columns in X.
            A list of ints. If this parameter is None the columns in X and
            the corresponding nodes in the hierarchy are expected to be in
            the same order -- unless X was passed as a DataFrame and the
            hierarchy is a named ``nx.DiGraph``; then the mapping is
            auto-derived from the feature names (see
            ``_auto_derive_columns``).

        Raises
        ------
        TypeError
            If the passed hierarchy is None, or if a constructor hyperparameter
            has the wrong type.
        ValueError
            If a constructor hyperparameter has an invalid value; if X is not
            bool-dtype (numerical inputs may be supported in the future); if y is None on a
            supervised subclass (selectors); if y is not a binary target
            labelled 0 and 1 (or False and True) with both classes present;
            or if the column->node mapping cannot be auto-derived for a
            DataFrame X (adjacency-matrix hierarchy without node names, or
            feature names with no matching node -- see
            ``_handle_orphan_features``).

        Warns
        -----
        UserWarning
            If a column of X holds no True value (see
            ``warn_on_all_false_columns``), if the column->node mapping falls
            back to positional order (see ``_warn_on_positional_fallback``),
            or if the hierarchy consists of multiple components (see
            ``add_virtual_root_node``).

        Returns
        -------
        self : object
            Returns self.
        """
        self._validate_hyperparameters()
        if self.hierarchy is None:
            raise TypeError("Hierarchy is None but is required.")
        if y is None:
            # validate_data raises here for supervised subclasses
            # (target_tags.required) and validates X alone otherwise.
            X = validate_data(self, X, y, accept_sparse=True)
        else:
            X, y = validate_data(self, X, y, accept_sparse=True)
            check_binary_target(y)
        check_bool_dtype(X)
        self._fit_hierarchy(columns)
        warn_on_all_false_columns(X, getattr(self, "feature_names_in_", None))
        self._fit(X, y)

        self.is_fitted_ = True
        return self

    def _fit(self, X, y):
        """Hook for the subclass's fitting logic.

        Called by ``fit`` with the validated X and y after the hierarchy
        setup. The base implementation does nothing.
        """

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
