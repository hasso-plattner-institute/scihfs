"HNB feature selection"

import networkx as nx
import numpy as np
import scipy.sparse as sp

from .lazyHierarchicalFeatureSelector import LazyHierarchicalFeatureSelector


class HNB(LazyHierarchicalFeatureSelector):
    """HNB (Hierarchy Based Redundant Attribute Removal Naive Bayes) classifier from Wan & Freitas, 2013.

    Selects the top-k non-redundant features in descending order of their
    relevance, with redundancy removed along each path.
    """

    def __init__(
        self,
        hierarchy: np.ndarray | sp.sparray | sp.spmatrix | nx.DiGraph | None = None,
        k=0,
    ):
        """Initializes a HNB-Selector.

        Parameters
        ----------
        hierarchy : np.ndarray, scipy.sparse array/matrix or nx.DiGraph
            The hierarchy graph. See ``HierarchicalEstimator.__init__``
            for the accepted formats.
        k : int
            The numbers of features to select.
        """
        super().__init__(hierarchy)
        self.k = k

    def _fit(self, X, y):
        self._compute_relevance(X, y)
        self._sort_relevance()

    def _select_features_per_instance(self, x_row):
        return self._get_top_k(self._get_nonredundant_features_relevance(x_row))
