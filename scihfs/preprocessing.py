"""
Sklearn compatible estimators for preprocessing hierarchical data.
"""

from __future__ import annotations

import warnings

import networkx as nx
import numpy as np
import scipy.sparse as sp
from networkx.algorithms.dag import ancestors
from sklearn.utils.validation import check_is_fitted, validate_data

from scihfs.helpers import check_bool_dtype, shrink_dag
from scihfs.selectors import HierarchicalEstimator
from scihfs.selectors.base import ORIGINAL_NODE_IDENTIFIER


class ColumnNotInHierarchyWarning(UserWarning):
    """Warning raised when columns in X do not have a corresponding node in the hierarchy.

    This warning is just informational and the issue is automatically handled by adding a
    new node directly under the root, ensuring the column is mapped appropriately.
    """

    pass


class HierarchicalPreprocessor(HierarchicalEstimator):
    """Estimator for preprocessing hierarchical data for feature selection.

    The hierarchical feature selectors expect the input data and the
    hierarchy graph to conform to certain pre-conditions.
    This preprocessor prepares the data and graph for the feature
    selection.

    This preprocessor currently supports only bool-dtype input.
    Non-binary (numeric) inputs raise ``ValueError``. Sum-propagation
    for numeric features is planned as a future enhancement.
    """

    def __init__(self, hierarchy=None):
        """Initializes a HierarchicalPreprocessor.

        Parameters
        ----------
        hierarchy : np.ndarray or nx.DiGraph
                    The hierarchy graph. Either an adjacency matrix
                    (``np.ndarray``), whose nodes are the integer column
                    positions, or a ``networkx.DiGraph`` whose nodes carry
                    names (typically strings matching the DataFrame columns).
                    Note: ``None`` is accepted for scikit-learn ``clone()`` compatibility but raises ``TypeError`` in ``fit``.

        .. note::
            When the hierarchy is a named ``DiGraph`` and X is passed
            to ``fit`` as a pandas ``DataFrame``, the column-to-node
            mapping is derived automatically from the feature names,
            so the ``columns`` argument of ``fit`` can be omitted.

        """
        self.hierarchy = hierarchy

    def fit(self, X, y=None, columns=None):
        """
        Sets the parameters for data transformation and prepares hierarchy.

        Following conditions need to be fulfilled for the feature selection algorithms:

        - Every node in the hierarchy graph should be mapped to one column in the dataset,
          and every column in the dataset should have a corresponding node in the hierarchy.

        - For binary data, if a feature has the value 1, all of its descendants in the
          hierarchy should also have the value 1.

        To achieve these conditions, missing columns are added to the hierarchy,
        and unnecessary nodes are removed. The `self._columns` parameter is adjusted so
        that it can be used to add additional columns to the dataset in the `transform`
        method.

        After fitting, the dataset can be transformed with the `transform` method, and
        the updated hierarchy and column mapping can be retrieved with `get_hierarchy`
        and `get_columns`.

        Parameters
        ----------
        X : {array-like, sparse matrix, DataFrame}, shape (n_samples, n_features)
            The training input samples. Must be bool-dtype. Non-binary
            (numeric) inputs raise ``ValueError`` until the planned
            sum-propagation mode will be implemented. A DataFrame (pandas, or
            any library scikit-learn recognises through the dataframe
            interchange protocol (up to v1.8.x) or the narwhals library (as of 1.9.0) such as polars) is accepted: its column names
            are captured as ``feature_names_in_`` and used to auto-derive
            ``columns`` (see below). No DataFrame library is imported directly;
            support comes entirely through scikit-learn's input plumbing.

        y : None
            This transformer does not require a target variable, but the pipeline API
            requires this parameter.

        columns : list or None, length n_features
            The mapping from the columns in X to the hierarchy graph's nodes. If this
            parameter is None, the columns in X and the corresponding nodes in the
            hierarchy are expected to be in the same order -- EXCEPT when X is a
            DataFrame and ``hierarchy`` is an ``nx.DiGraph``: then the mapping is
            derived automatically by matching feature names to node names, and passing
            ``columns`` separately is unnecessary. Passing a DataFrame with ``columns=None``
            while ``hierarchy`` is an adjacency ndarray raises ``ValueError`` (there
            are no node names to match against).

        Returns
        -------
        self : object
            Returns self.
        """

        X = validate_data(self, X, accept_sparse="csr")
        if sp.issparse(X):
            X = sp.csr_array(X)
        check_bool_dtype(X)

        if columns is None and getattr(self, "feature_names_in_", None) is not None:
            # For the case that X was a DataFrame (and feature_names_in_ has been set automatically in validate_data): derive the column->node mapping from the captured feature names.
            # Done here (and not in the base class) because
            # auto-derivation is preprocessor-specific FOR NOW – this behaviour might change in the future.
            columns = self._auto_derive_columns()

        # Reuse the base hierarchy setup WITHOUT a second validate_data call:
        # re-validating would drop feature_names_in_ (needed by
        # get_feature_names_out / set_output) and is redundant work here.
        self._fit_hierarchy(columns)

        self._columns = [
            column if column < len(self.hierarchy) else -1 for column in self._columns
        ]

        self._extend_dag()
        self._shrink_dag()
        self._find_missing_columns()
        self._adjust_node_names()
        self._build_ancestor_closure()
        self.is_fitted_ = True
        return self

    def _auto_derive_columns(self):
        """Derive the column->node mapping from DataFrame feature names.

        Called from ``fit`` when X is a pandas ``DataFrame`` (so
        ``feature_names_in_`` is set) and ``columns`` was not supplied. Each
        captured feature name is looked up against the hierarchy's node names;
        names with no matching node map to ``-1`` (handled downstream by
        ``_extend_dag``, which adds them under ROOT with a warning).

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
            If the hierarchy is not a ``DiGraph`` (an adjacency ndarray has no
            node names to match against).
        """
        if not isinstance(self.hierarchy, nx.DiGraph):
            raise ValueError(
                "Cannot auto-derive columns from DataFrame feature names "
                "because hierarchy is an ndarray (no node names). Either pass "
                "hierarchy as nx.DiGraph with named nodes, or supply columns "
                "explicitly."
            )
        nodes = list(self.hierarchy.nodes)
        name_to_position = {str(node): i for i, node in enumerate(nodes)}
        return [name_to_position.get(str(name), -1) for name in self.feature_names_in_]

    def transform(self, X):
        """Transforms dataset to fulfill conditions for feature selection.

        After transformation, if a feature is 1, all of its ancestors in the hierarchy are 1 as well.
        Missing columns are added to the dataset.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples. Must be bool-dtype. Non-binary (numeric)
            inputs raise ``ValueError`` until the planned sum-propagation
            mode will be implemented.

        Returns
        -------
        X_ : array of shape (n_samples, n_selected_features)
            The transformed dataset.
        """
        # Check is fit had been called
        check_is_fitted(self, "is_fitted_")

        # Input validation.
        # Any sparse input (csr/csc/coo, matrix or array) is accepted, and then normalized
        # to CSR by validate_data. To provide consistent behaviour downstream, any output is then coerced to the recommended csr_array
        # type so the rest of the transform pipeline sees only one single sparse type with a predictable behaviour.
        X = validate_data(self, X, accept_sparse="csr", reset=False)
        if sp.issparse(X):
            X = sp.csr_array(X)
        check_bool_dtype(X)

        X_ = self._add_columns(X)
        X_ = self._propagate_ones(X_)
        return X_

    def get_hierarchy(self):
        """Get the transformed hierarchy graph.

        Raises
        ----------
        RuntimeError
            If the method is called before fit has been called.
            In this case the hierarchy graph has not been updated yet.
        """
        if self.is_fitted_:
            output_hierarchy = self._hierarchy_graph
            output_hierarchy.remove_node("ROOT")
            return nx.to_numpy_array(self._hierarchy_graph)
        else:
            raise RuntimeError("Instance has not been fitted.")

    def get_feature_names_out(self, input_features=None):
        """Map each output column to its hierarchy node name.

        Enables ``set_output(transform="pandas")`` (or ``"polars"``) so that
        ``transform`` returns a labelled DataFrame. The output has one name per
        entry of ``self._columns`` (in order), i.e. including the ancestor and
        any auto-added columns appended during ``fit``.

        Names are recovered from the ``ORIGINAL_NODE_IDENTIFIER`` node attribute
        stamped in ``_set_hierarchy`` / ``_extend_dag`` (which survives the
        relabel/shrink/adjust passes). This traces each output column back to its
        original identity: the node name for a ``DiGraph`` hierarchy, the original
        integer index for an adjacency-matrix hierarchy (NOT the post-shrink
        renumbering), or the input feature name for an orphan column added under
        ROOT (when X was a DataFrame). Only orphan columns from a nameless X
        (e.g. a plain ndarray) lack the attribute and fall back to ``"x<node>"``.

        Parameters
        ----------
        input_features : array-like of str or None
            Unused; present for scikit-learn API compatibility.

        Returns
        -------
        feature_names_out : ndarray of str, shape (n_output_features,)
        """
        check_is_fitted(self, "is_fitted_")
        names = []
        for node in self._columns:
            data = self._hierarchy_graph.nodes[node]
            if ORIGINAL_NODE_IDENTIFIER in data:
                name = data[ORIGINAL_NODE_IDENTIFIER]
            else:
                name = f"x{node}"
            names.append(str(name))
        return np.asarray(names, dtype=object)

    def _extend_dag(self):
        """Adds missing nodes to the hierarchy graph.

        For columns that don't have a corresponding node in the hierarchy a
        node is added right under the "ROOT" node.
        We then update the columns mapping to include the new nodes.
        If a node in the hierarchy has a name conflict with a column in the
        dataset we add a node with the next available id.
        """
        # Subtract 1 because the "ROOT" node is included in the total count,
        # but the other N-1 nodes are indexed starting from 0
        next_available_node_id = len(self._hierarchy_graph.nodes) - 1
        columns_without_node = []

        for column_index, column_mapping in enumerate(self._columns):
            if column_mapping == -1:  # no corresponding node yet
                columns_without_node.append(column_index)
                if column_index in self._hierarchy_graph.nodes:
                    # column_index has name conflict with an existing node
                    # so we add a node with next available id
                    new_node = next_available_node_id
                    next_available_node_id += 1
                else:
                    # directly add the column as a node under "ROOT"
                    new_node = column_index
                self._hierarchy_graph.add_edge("ROOT", new_node)
                self._columns[column_index] = new_node
                # If X was a DataFrame, stamp the input feature name on this
                # orphan node so get_feature_names_out can label it properly
                # instead of falling back to "x<node>".
                if getattr(self, "feature_names_in_", None) is not None:
                    self._hierarchy_graph.nodes[new_node][ORIGINAL_NODE_IDENTIFIER] = str(
                        self.feature_names_in_[column_index]
                    )

        # Warn user for all columns that were not in hierarchy
        if columns_without_node:
            warning_missing_nodes = f"""The following columns in X
             do not have a corresponding node in the hierarchy: {columns_without_node}.
             A node was added for it under ROOT."""
            warnings.warn(warning_missing_nodes, ColumnNotInHierarchyWarning)

    def _shrink_dag(self):
        """Irrelevant nodes are removed from the hierarchy graph.

        Nodes are considered irrelevant if they do not have a corresponding
        column in the input dataframe and don't have any children. These
        features would always be 0 in the dataset and, therefore, do not
        contain any necessary information.
        """
        node_identifier = self._columns
        digraph = self._hierarchy_graph
        self._hierarchy_graph = shrink_dag(node_identifier, digraph)

    def _find_missing_columns(self):
        """Finds nodes for which a column needs to be added to the dataset.

        These node names are added to self._columns and the corresponding
        columns will be added in the transform method.
        """
        missing_nodes = [
            node
            for node in self._hierarchy_graph.nodes
            if node not in self._columns and node != "ROOT"
        ]
        self._columns.extend(missing_nodes)

    def _add_columns(self, X):
        """Adds missing columns to the dataset.

        Missing columns are added and all values are set to 0. Sparse inputs
        are padded with a sparse zero block via ``scipy.sparse.hstack``
        dense inputs use a single ``np.concatenate``.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        X_ : array of shape [n_samples, n_new_features]
            The dataset with the added columns. Output format matches input
            format (CSR in -> CSR out; dense in -> dense out).
        """
        num_rows, num_columns = X.shape
        n_extra = len(self._columns) - num_columns
        if n_extra <= 0:
            return X
        if sp.issparse(X):
            padding = sp.csr_array((num_rows, n_extra), dtype=X.dtype)
            return sp.hstack([X, padding], format="csr")
        padding = np.zeros((num_rows, n_extra), dtype=X.dtype)
        return np.concatenate([X, padding], axis=1)

    def _build_ancestor_closure(self):
        """Precompute the ancestor closure matrix indexed by column position.

        For columns i and j, ``self._ancestor_closure_[i, j]`` is True iff the
        node mapped to column j is an ancestor (in ``self._hierarchy_graph``)
        of the node mapped to column i. The virtual "ROOT" node is excluded
        since it has no corresponding column.

        Under the assumption that ancestor closures of ontologies are very sparse, this matrix is stored as a ``scipy.sparse.csr_array`` of dtype ``bool`` – primarily to save memory (but also to speed up the matmul in ``_propagate_ones``).
        """
        n_cols = len(self._columns)
        rows: list[int] = []
        cols: list[int] = []
        node_to_col = {node: i for i, node in enumerate(self._columns)}

        for col_i, node in enumerate(self._columns):
            for anc in ancestors(self._hierarchy_graph, node):
                if anc == "ROOT":
                    continue
                rows.append(col_i)
                cols.append(node_to_col[anc])

        data = np.ones(len(rows), dtype=bool)
        self._ancestor_closure_ = sp.csr_array(
            (data, (rows, cols)), shape=(n_cols, n_cols), dtype=bool
        )

    def _propagate_ones(self, X):
        """Update the dataset to fulfill the 0-1-propagation rule.

        If a feature in the dataset is 1, all of its ancestors in the
        sample are set to 1. Sparse inputs stay sparse end-to-end (the
        intermediate ``X @ closure`` product is sparse, and elementwise OR
        is realised via ``sparse.maximum``); dense inputs use ``|``.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X : array of shape [n_samples, n_new_features]
            The dataset with updated feature values. Output format matches
            input format (CSR in -> CSR out; dense in -> dense out).
        """
        # Cast X to bool so scipy's sparse matmul keeps OR-style semantics
        # (numeric X would otherwise return path counts, not a bool mask).
        X_bool = X.astype(bool, copy=False)
        propagation = X_bool @ self._ancestor_closure_
        if sp.issparse(X):
            return X_bool.maximum(propagation).tocsr()
        return X_bool | propagation

    def _adjust_node_names(self):
        """Adjust node names in hierarchy and _columns.

        When nodes are removed from the hierarchy graph the mapping in
        self._columns is not correct anymore after the hierarchy graph
        is transformed to a numpy.ndarray and back again. However, this
        transformation needs to be performed to ouput the hierarchy.
        Therefore the node names need to be adjusted.
        """
        nodes = list(self._hierarchy_graph.nodes())
        nodes.remove("ROOT")
        self._columns = [nodes.index(node_name) for node_name in self._columns]
        mapping = {node_name: nodes.index(node_name) for node_name in nodes}
        self._hierarchy_graph = nx.relabel_nodes(self._hierarchy_graph, mapping)
