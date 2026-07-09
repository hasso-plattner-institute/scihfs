"""
SHSEL Feature Selector.
"""

import statistics

import networkx as nx
import numpy as np
import scipy.sparse as sp

# The HFE extension from Oudah and Henschel (2018) is commented out below, as it requires numerical features in the input (currently only bool supported).
# When it is restored, also restore the `compute_aggregated_values` and `get_leaves` imports.
from scihfs.helpers import get_paths
from scihfs.metrics import information_gain, pearson_correlation
from scihfs.selectors import EagerHierarchicalFeatureSelector


class SHSELSelector(EagerHierarchicalFeatureSelector):
    """SHSEL feature selection method for hierarchical features.

    This feature selection method was proposed by Ristoski and Paulheim
    in 2014. The features are selected by removing features with
    parents that have a similar relevance and removing features with
    lower than average information gain for each path from leaf to
    root.

    The hierarchical feature engineering (HFE) extension proposed by
    Oudah and Henschel (2018) is temporarily disabled. It requires
    numerical input which is currently not supported.
    Corresponding code is retained (but commented out).
    """

    def __init__(
        self,
        hierarchy: np.ndarray | sp.sparray | sp.spmatrix | nx.DiGraph | None = None,
        relevance_metric: str = "IG",
        similarity_threshold=None,
        pruning: bool = True,
        ig_average: str = "full_path",
        # HFE extension disabled:
        # use_hfe_extension=False,
        # preprocess_numerical_data=False,
    ):
        """Initializes a SHSELSelector.

        Parameters
        ----------
        hierarchy : np.ndarray, scipy.sparse array/matrix or nx.DiGraph
                    The hierarchy graph. See ``HierarchicalEstimator.__init__``
                    for the accepted formats.
        relevance_metric : str
                    The relevance metric to use in the initial selection
                    stage of the algorithm. The options ore "IG" for
                    information gain and "Correlation". Default is IG.
        similarity_threshold : float or None
                    The similarity threshold to use in the initial selection
                    stage of the algorithm, a number between 0 and 1. If None
                    (the default), a metric-specific default is used: 0.99 for
                    the "IG" (information gain) metric and 0.6 for "Correlation" according to the original paper.
                    The IG metric is normalized to the [0,1] interval, so it
                    is independent of the logarithm base of the IG calculation.
        pruning : bool
                    Whether to run the pruning stage (Algorithm 2) after the
                    initial selection (Algorithm 1). If False, only the initial
                    selection is applied (the paper's initialSHSEL); if True both stages run (pruneSHSEL).
                    Default is True (pruneSHSEL).
        ig_average : str
                    How the per-path information-gain average is computed in the
                    pruning stage (Algorithm 2). "full_path" averages over every
                    node on the path, including nodes already removed in the
                    initial selection.
                    "survivors_only" averages over the surviving features from Algorithm 1 only.
                    In both cases only features surviving the initial selection
                    in Algorithm 1 can be subsequently selected in Algorithm 2.
                    This parameter is only relevant if pruning=True, and has been
                    introduced here because the original paper's Algorithm 2 textual
                    and pseudocode descriptions are divergent on this point.
                    Default is "full_path".

        Notes
        -----
        The hierarchical feature engineering (HFE) extension proposed
        by Oudah and Henschel (2018) is temporarily disabled. It
        requires numerical input which is currently not supported.
        Corresponding code is retained though (but commented out),
        such as the following two parameters, which are not active:

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
        self.pruning = pruning
        self.ig_average = ig_average
        # HFE extension disabled:
        # self.use_hfe_extension = use_hfe_extension
        # self.preprocess_numerical_data = preprocess_numerical_data

    def _select(self, X, y):
        """The actual SHSEL feature selection algorithm."""
        # HFE extension disabled:
        # if self.use_hfe_extension and self.relevance_metric != "Correlation":
        #     raise ValueError(
        #         "When using the HFE extension the relevance_metric should be 'Correlation'."
        #     )
        if sp.issparse(X):
            X = X.tocsc()
        self._calculate_ig_relevance(X, y)
        # HFE extension disabled:
        # if self.preprocess_numerical_data:
        #     X = self._preprocess(X)
        paths = get_paths(self._hierarchy_graph, reverse=True)
        self._inital_selection(paths, X)
        if self.pruning:
            self._pruning(paths)
        # HFE extension disabled:
        # if self.use_hfe_extension:
        #     self._leaf_filtering()

    def _inital_selection(self, paths, X):
        """First part of the feature selection algorithm."""
        if self.similarity_threshold is None:
            self._effective_threshold = 0.99 if self.relevance_metric == "IG" else 0.6
        else:
            self._effective_threshold = self.similarity_threshold
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
                if similarity >= self._effective_threshold:
                    nodes_to_remove.add(node)

        self.representatives_ = [
            feature for feature in self._columns if feature not in nodes_to_remove
        ]

    def _pruning(self, paths):
        """Second part of the feature selection algorithm"""
        if self.ig_average not in ("full_path", "survivors_only"):
            raise ValueError(
                f"Unknown ig_average {self.ig_average!r}; "
                'expected "full_path" or "survivors_only".'
            )
        updated_representatives = []

        for path in paths:
            path.remove("ROOT")
            if self.ig_average == "full_path":
                average_nodes = path
            else:
                average_nodes = [node for node in path if node in self.representatives_]
            average_relevance = statistics.mean(
                [self._relevance_values[node] for node in average_nodes]
            )
            average_relevance = round(average_relevance, 6)
            for node in path:
                if (
                    node in self.representatives_
                    and self._relevance_values[node] >= average_relevance
                ):
                    # HFE extension disabled:
                    # if (
                    #     self.use_hfe_extension is False
                    #     or self._relevance_values[node] > 0.0
                    # ):
                    updated_representatives.append(node)

        self.representatives_ = list(set(updated_representatives))  # remove duplicates

    def _calculate_ig_relevance(self, X, y):
        values = information_gain(X, y)
        # Normalize the relevance to [0, 1] by dividing by the maximum, matching
        # the paper's use of RapidMiner's normalized information-gain weights.
        max_relevance = max(values)
        if max_relevance > 0:
            values = [round(value / max_relevance, 6) for value in values]
        self._relevance_values = dict(zip(self._columns, values))

    # The following are all HFE extension methods.
    # Note: `_preprocess` below additionally has an argument-
    # order bug (passing "ROOT" as X); fix when re-enabling HFE.

    # def _preprocess(self, X):
    #     """Preprocess numerical data by summing up child values.
    #
    #     This is part of the HFE extension and only makes sense for
    #     numerial data and not for binary data.
    #     """
    #     return compute_aggregated_values(
    #         "ROOT", X, self._hierarchy_graph, self._columns
    #     )
    #
    # def _leaf_filtering(self):
    #     """Filtering representatives by removing leaves with low relevance.
    #
    #     This is part of the HFE extension proposed by Oudah and Henschel.
    #     """
    #     average_ig = statistics.mean(
    #         [self._relevance_values[node] for node in self.representatives_]
    #     )
    #
    #     leaves = self._get_leaves_in_incomplete_paths()
    #
    #     nodes_to_remove = [
    #         leaf
    #         for leaf in leaves
    #         if self._relevance_values[leaf] < average_ig
    #         or self._relevance_values[leaf] == 0
    #     ]
    #     updated_representatives = [
    #         node for node in self.representatives_ if node not in nodes_to_remove
    #     ]
    #     self.representatives_ = updated_representatives
    #
    # def _get_leaves_in_incomplete_paths(self):
    #     """Select leaves of incomplete paths (part of HFE extension)"""
    #     leaves = [
    #         leaf
    #         for leaf in get_leaves(self._hierarchy_graph)
    #         if leaf in self.representatives_
    #     ]
    #
    #     paths = get_paths(self._hierarchy_graph)
    #     max_path_len = max([len(path) for path in paths])
    #     selected_leaves = []
    #     for leaf in leaves:
    #         for path in paths:
    #             if leaf in path and len(path) != max_path_len:
    #                 selected_leaves.append(leaf)
    #     return selected_leaves
