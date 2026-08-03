from .lazyHierarchicalFeatureSelector import LazyHierarchicalFeatureSelector


class HNBs(LazyHierarchicalFeatureSelector):
    """HNB-s (Hierarchy Based Redundant Attribute Removal Naive Bayes without Selection Step) classifier proposed by Wan & Freitas, 2013.

    Selects the non-redundant features such that redundancy along each path is
    removed, using the per-node relevance.
    """

    def _fit(self, X, y):
        self._compute_relevance(X, y)

    def _select_features_per_instance(self, x_row):
        return self._get_nonredundant_features_relevance(x_row)
