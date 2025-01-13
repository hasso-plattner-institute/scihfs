"""
SHSEL Feature Selector.
"""

import statistics

import numpy as np
from scipy import sparse
from sklearn.utils.validation import check_X_y

from hfs.helpers import compute_aggregated_values, get_leaves, get_paths
from hfs.metrics import information_gain, pearson_correlation
from hfs.selectors import EagerHierarchicalFeatureSelector


class SHSELSelector(EagerHierarchicalFeatureSelector):
    """SHSEL feature selection method for hierarchical features.

    This feature selection method was proposed by Ristoski and Paulheim
    in 2014. The features are selected by removing features with
    parents that have a similar relevance and removing features with
    lower than average information gain for each path from leaf to
    root.
    This Selector also implements the hierarchical feature
    engineering (HFE) extension proposed by Oudah and Henschel in
    2018.
    """

    def __init__(
        self,
        hierarchy: np.ndarray = None,
        relevance_metric: str = "IG",
        similarity_threshold=0.99,
        use_hfe_extension=False,
        preprocess_numerical_data=False,
    ):
        """Initializes a SHSELSelector.

        Parameters
        ----------
        hierarchy : np.ndarray
                    The hierarchy graph as an adjacency matrix.
        relevance_metric : str
                    The relevance metric to use in the initial selection
                    stage of the algorithm. The options ore "IG" for
                    information gain and "Correlation". Default is IG.
        similarity_threshold : float
                    The similarity threshold to use in the initial selection
                    stage of the algorithm. This can be a number between
                    0 an 1. Default is 0.99.
        use_hfe_extension : bool
                    If True the HFE algorithm proposed by Oudah and Henschel is
                    used. Set relevance_metric to "Correlation" when using this
                    extension. Default is False.
        preprocess_numerical_data : False
                    If True the data is preprocessed by adding up the child values.
                    This method is used in the HFE extension algorithm which
                    expects numerical data. If binary data is used it is
                    recommended to set this parameter to False. Default is False.

        """
        super().__init__(hierarchy)
        self.relevance_metric = relevance_metric
        self.similarity_threshold = similarity_threshold
        self.use_hfe_extension = use_hfe_extension
        self.preprocess_numerical_data = preprocess_numerical_data

    def fit(self, X, y, columns=None):
        """Fitting function that sets self.representatives\_.

        The number of columns in X and the number of nodes in the hierarchy
        are expected to be the same and each column should be mapped to
        exactly one node in the hierarchy with the columns parameter.
        After fitting self.representatives\_ includes the names of all
        nodes from the hierarchy that are left after feature selection.
        The features are selected by removing features with
        parents that have a similar relevance and removing features with
        lower than average information gain for each path from leaf to
        root.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
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
        # Input validation
        if self.use_hfe_extension and self.relevance_metric != "Correlation":
            raise ValueError(
                "When using the HFE extension the relevance_metric should be 'Correlation'."
            )
        X, y = check_X_y(X, y, accept_sparse=True)
        if sparse.issparse(X):
            X = X.tocsr()
        super().fit(X, y, columns)

        # Feature Selection Algorithm
        self._calculate_ig_relevance(X, y)
        self._fit(X)

        self.is_fitted_ = True
        return self

    def _fit(self, X):
        """The feature selection algorithm."""
        if self.preprocess_numerical_data:
            X = self._preprocess(X)
        paths = get_paths(self._hierarchy_graph, reverse=True)
        self._inital_selection(paths, X)
        self._pruning(paths)
        if self.use_hfe_extension:
            self._leaf_filtering()

    def _inital_selection(self, paths, X):
        """First part of the feature selection algorithm."""
        nodes_to_remove = set()

        for path in paths:
            # If the relevance is similar to the parents relevance, the child is removed
            for index, node in enumerate(path):
                parent_node = path[index + 1]
                if parent_node == "ROOT":
                    break
                if self.relevance_metric == "IG":
                    similarity = 1 - abs(
                        self._relevance_values[parent_node] - self._relevance_values[node]
                    )
                else:
                    similarity = pearson_correlation(
                        X[:, self._columns.index(parent_node)],
                        X[:, self._columns.index(node)],
                    )
                if similarity >= self.similarity_threshold:
                    nodes_to_remove.add(node)

        self.representatives_ = [
            feature for feature in self._columns if feature not in nodes_to_remove
        ]

    def _pruning(self, paths):
        """Second part of the feature selection algorithm"""
        updated_representatives = []

        for path in paths:
            path.remove("ROOT")
            average_relevance = statistics.mean(
                [
                    self._relevance_values[node]
                    for node in path
                    if node in self.representatives_
                ]
            )
            average_relevance = round(average_relevance, 6)
            for node in path:
                if (
                    node in self.representatives_
                    and self._relevance_values[node] >= average_relevance
                ):
                    if (
                        self.use_hfe_extension is False
                        or self._relevance_values[node] > 0.0
                    ):
                        updated_representatives.append(node)

        self.representatives_ = list(set(updated_representatives))  # remove duplicates

    def _calculate_ig_relevance(self, X, y):
        values = information_gain(X, y)
        self._relevance_values = dict(zip(self._columns, values))

    def _preprocess(self, X):
        """Preprocess numerical data by summing up child values.

        This is part of the HFE extension and only makes sense for
        numerial data and not for binary data.
        """
        return compute_aggregated_values("ROOT", X, self._hierarchy_graph, self._columns)

    def _leaf_filtering(self):
        """Filtering representatives by removing leaves with low relevance.

        This is part of the HFE extension proposed by Oudah and Henschel.
        """
        average_ig = statistics.mean(
            [self._relevance_values[node] for node in self.representatives_]
        )

        leaves = self._get_leaves_in_incomplete_paths()

        nodes_to_remove = [
            leaf
            for leaf in leaves
            if self._relevance_values[leaf] < average_ig
            or self._relevance_values[leaf] == 0
        ]
        updated_representatives = [
            node for node in self.representatives_ if node not in nodes_to_remove
        ]
        self.representatives_ = updated_representatives

    def _get_leaves_in_incomplete_paths(self):
        """Select leaves of incomplete paths (part of HFE extension)"""
        leaves = [
            leaf
            for leaf in get_leaves(self._hierarchy_graph)
            if leaf in self.representatives_
        ]

        paths = get_paths(self._hierarchy_graph)
        max_path_len = max([len(path) for path in paths])
        selected_leaves = []
        for leaf in leaves:
            for path in paths:
                if leaf in path and len(path) != max_path_len:
                    selected_leaves.append(leaf)
        return selected_leaves
